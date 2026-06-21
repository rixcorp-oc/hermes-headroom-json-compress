"""Unit tests for the headroom-json-compress plugin callback.

Run from the headroom-poc repo with the .venv active (so `headroom` imports):

    source .venv/bin/activate
    python -m pytest hermes-plugin/headroom-json-compress/test_plugin.py -v

The compression happy-path test exercises the REAL headroom lib (no mock) so
we validate the actual single-string API contract, not our assumptions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Load the plugin module by path (its dir name has a hyphen, so it's not a
# normal importable package name).
_PLUGIN_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "headroom_json_compress_plugin", _PLUGIN_DIR / "__init__.py"
)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Reset all plugin env vars before each test; enable by default."""
    for k in (
        "HERMES_HEADROOM_JSON_COMPRESS",
        "HEADROOM_COMPRESS_TOOLS",
        "HEADROOM_COMPRESS_SHADOW_TOOLS",
        "HEADROOM_COMPRESS_MIN_BYTES",
        "HEADROOM_COMPRESS_LOG",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_HEADROOM_JSON_COMPRESS", "1")
    monkeypatch.setenv("HEADROOM_COMPRESS_MIN_BYTES", "200")
    # The allow-list is empty by default now (a tool on no list is out of
    # scope), so put search_files on the INJECT list for the gate tests that
    # exercise the compression happy-path.
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "search_files")


def _big_json_array(n=300):
    """A structured payload that compresses well (repeated keys factor out)."""
    rows = [
        {
            "id": f"svc-{i:04d}",
            "cpu": round(i % 100 + 0.5, 1),
            "mem_mb": (i * 37) % 8000,
            "owner": ["team-a", "team-b", "team-c"][i % 3],
            "region": ["us-east-1", "eu-west-1", "ap-southeast-2"][i % 3],
            "status": "running" if i % 4 else "degraded",
        }
        for i in range(n)
    ]
    return json.dumps(rows, indent=2)


# --------------------------------------------------------------------------
# Master switch
# --------------------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_HEADROOM_JSON_COMPRESS", raising=False)
    out = plugin.compress_tool_result(tool_name="search_files", result=_big_json_array())
    assert out is None


def test_disabled_explicit_zero(monkeypatch):
    monkeypatch.setenv("HERMES_HEADROOM_JSON_COMPRESS", "0")
    out = plugin.compress_tool_result(tool_name="search_files", result=_big_json_array())
    assert out is None


# --------------------------------------------------------------------------
# Allow-list gate
# --------------------------------------------------------------------------

def test_tool_not_in_allowlist_passthrough():
    out = plugin.compress_tool_result(tool_name="web_extract", result=_big_json_array())
    assert out is None


def test_tool_in_default_allowlist_compresses():
    out = plugin.compress_tool_result(tool_name="search_files", result=_big_json_array())
    assert isinstance(out, str)
    assert len(out) < len(_big_json_array())


def test_custom_allowlist(monkeypatch):
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "read_file, my_tool")
    # search_files now excluded
    assert plugin.compress_tool_result(tool_name="search_files", result=_big_json_array()) is None
    # my_tool now included
    out = plugin.compress_tool_result(tool_name="my_tool", result=_big_json_array())
    assert isinstance(out, str)


# --------------------------------------------------------------------------
# Type / JSON / size gates (all fail open => None)
# --------------------------------------------------------------------------

def test_non_str_result_passthrough():
    assert plugin.compress_tool_result(tool_name="search_files", result={"not": "a string"}) is None
    assert plugin.compress_tool_result(tool_name="search_files", result=None) is None
    assert plugin.compress_tool_result(tool_name="search_files", result=12345) is None


def test_below_size_floor_passthrough(monkeypatch):
    monkeypatch.setenv("HEADROOM_COMPRESS_MIN_BYTES", "100000")
    out = plugin.compress_tool_result(tool_name="search_files", result=_big_json_array())
    assert out is None


def test_non_json_passthrough():
    big_text = "this is a plain log line\n" * 200  # well over the floor, not JSON
    out = plugin.compress_tool_result(tool_name="search_files", result=big_text)
    assert out is None


def test_malformed_json_passthrough():
    broken = '[{"id": "x", "v": 1}, {"id": "y"' + (" " * 1000)  # truncated JSON
    out = plugin.compress_tool_result(tool_name="search_files", result=broken)
    assert out is None


def test_json_scalar_passthrough():
    # Valid JSON but not a list/dict — a bare quoted string padded over the floor.
    scalar = json.dumps("x" * 2000)
    out = plugin.compress_tool_result(tool_name="search_files", result=scalar)
    assert out is None


# --------------------------------------------------------------------------
# Losslessness — the core safety property
# --------------------------------------------------------------------------

def test_compression_is_row_lossless():
    payload = _big_json_array(300)
    out = plugin.compress_tool_result(tool_name="search_files", result=payload)
    assert isinstance(out, str)
    # Every original id must survive in the compressed output.
    ids = [r["id"] for r in json.loads(payload)]
    missing = [i for i in ids if i not in out]
    assert missing == [], f"{len(missing)} ids dropped by compression: {missing[:5]}"


def test_output_is_smaller():
    payload = _big_json_array(300)
    out = plugin.compress_tool_result(tool_name="search_files", result=payload)
    assert isinstance(out, str)
    assert len(out) < len(payload)


# --------------------------------------------------------------------------
# Cache invariant — compress-once must be byte-stable for a fixed input
# in a fresh process (the property prompt caching depends on).
# --------------------------------------------------------------------------

def test_first_op_deterministic_in_subprocess():
    """A fresh interpreter compressing a fixed payload as its first op must
    produce identical bytes across runs. (In-process repeat drift is expected
    and irrelevant — we never re-compress a frozen result.)"""
    import subprocess
    import textwrap

    payload = _big_json_array(300)
    pf = _PLUGIN_DIR / "_test_payload.json"
    pf.write_text(payload)
    try:
        script = textwrap.dedent(f"""
            import importlib.util, hashlib, os
            os.environ["HERMES_HEADROOM_JSON_COMPRESS"] = "1"
            os.environ["HEADROOM_COMPRESS_MIN_BYTES"] = "200"
            spec = importlib.util.spec_from_file_location("p", r"{_PLUGIN_DIR / '__init__.py'}")
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            raw = open(r"{pf}").read()
            out = m.compress_tool_result(tool_name="search_files", result=raw)
            print(hashlib.sha256(out.encode()).hexdigest())
        """)
        hashes = set()
        for _ in range(3):
            r = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            assert r.returncode == 0, f"subprocess failed: {r.stderr[-500:]}"
            hashes.add(r.stdout.strip().splitlines()[-1])
        assert len(hashes) == 1, f"cross-process bytes not stable: {hashes}"
    finally:
        pf.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Per-tool shadow (monitor) mode — observe without injecting
# --------------------------------------------------------------------------

def _capture_log():
    """Return (handler, records_list) capturing the plugin logger's messages."""
    import logging

    records: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Cap()
    plugin.logger.addHandler(h)
    plugin.logger.setLevel(logging.INFO)
    return h, records


def test_shadow_tool_monitors_never_injects(monkeypatch):
    """A tool on HEADROOM_COMPRESS_SHADOW_TOOLS runs the full pipeline but
    always returns None, and logs 'shadow-would-compress' with a ratio."""
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "")  # nothing injecting
    monkeypatch.setenv("HEADROOM_COMPRESS_SHADOW_TOOLS", "discord")
    h, records = _capture_log()
    try:
        out = plugin.compress_tool_result(tool_name="discord", result=_big_json_array())
    finally:
        plugin.logger.removeHandler(h)
    assert out is None
    assert any("shadow-would-compress" in m for m in records), records


def test_inject_tool_compresses(monkeypatch):
    """A tool on HEADROOM_COMPRESS_TOOLS (and not shadowed) is injected."""
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "discord")
    monkeypatch.setenv("HEADROOM_COMPRESS_SHADOW_TOOLS", "")
    out = plugin.compress_tool_result(tool_name="discord", result=_big_json_array())
    assert isinstance(out, str)
    assert len(out) < len(_big_json_array())


def test_shadow_wins_on_overlap(monkeypatch):
    """A tool present in BOTH lists is monitored, never injected (shadow wins).
    This is the safe promotion path — staged in shadow until removed from it."""
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "discord")
    monkeypatch.setenv("HEADROOM_COMPRESS_SHADOW_TOOLS", "discord")
    h, records = _capture_log()
    try:
        out = plugin.compress_tool_result(tool_name="discord", result=_big_json_array())
    finally:
        plugin.logger.removeHandler(h)
    assert out is None, "shadow must win over inject when a tool is on both lists"
    assert any("shadow-would-compress" in m for m in records), records


def test_tool_on_neither_list_out_of_scope(monkeypatch):
    """A tool on no list is untouched — no compression attempt."""
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "discord")
    monkeypatch.setenv("HEADROOM_COMPRESS_SHADOW_TOOLS", "process")
    assert plugin.compress_tool_result(tool_name="web_extract", result=_big_json_array()) is None


def test_master_switch_kills_monitoring(monkeypatch):
    """The master kill-switch disables monitoring too — off => total no-op even
    for a shadow-listed tool, with no log line emitted."""
    monkeypatch.delenv("HERMES_HEADROOM_JSON_COMPRESS", raising=False)
    monkeypatch.setenv("HEADROOM_COMPRESS_SHADOW_TOOLS", "discord")
    h, records = _capture_log()
    try:
        out = plugin.compress_tool_result(tool_name="discord", result=_big_json_array())
    finally:
        plugin.logger.removeHandler(h)
    assert out is None
    assert records == [], f"master switch off must emit nothing, got: {records}"


def test_master_switch_kills_injection(monkeypatch):
    """The master kill-switch disables injection too — off => no compression."""
    monkeypatch.delenv("HERMES_HEADROOM_JSON_COMPRESS", raising=False)
    monkeypatch.setenv("HEADROOM_COMPRESS_TOOLS", "discord")
    assert plugin.compress_tool_result(tool_name="discord", result=_big_json_array()) is None


# --------------------------------------------------------------------------
# register() contract
# --------------------------------------------------------------------------

def test_register_wires_the_hook():
    captured = {}

    class FakeCtx:
        def register_hook(self, name, cb):
            captured[name] = cb

    plugin.register(FakeCtx())
    assert "transform_tool_result" in captured
    assert captured["transform_tool_result"] is plugin.compress_tool_result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
