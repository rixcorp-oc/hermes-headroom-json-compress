"""Unit tests for the headroom-windowed-ccr plugin.

Run from the plugin dir's parent so `import headroom_windowed_ccr` resolves,
or via the test runner at the bottom which injects the path.

Covers:
  - master switch + per-tool list gating (inject / shadow / out-of-scope)
  - fail-open guards (non-str, below floor, not-smaller)
  - real compress + store round-trip + marker present (live headroom)
  - shadow tool monitors but never replaces
  - apply_window: grep / lines / head / tail / full / bounded-fallback / cap
  - retrieve_window handler: valid round-trip, invalid hash, graceful miss
  - tool schema shape
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Load the plugin module by path (dir name has hyphens -> not importable直接).
PLUGIN = Path(__file__).resolve().parent / "__init__.py"
spec = importlib.util.spec_from_file_location("headroom_windowed_ccr", PLUGIN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def big_log(n=600):
    lines = [f"[{i:04d}] PASS tests/test_mod_{i%80}.py::test_case_{i} ({i%40+2}ms)" for i in range(n)]
    lines[317] = "[0317] FAIL tests/test_payments.py::test_refund_rounding - AssertionError: expected 1000 got 999"
    lines.append("=== 599 passed, 1 failed in 84.21s ===")
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("HERMES_WINDOWED_CCR", "WINDOWED_CCR_TOOLS", "WINDOWED_CCR_SHADOW_TOOLS",
              "WINDOWED_CCR_MIN_BYTES", "WINDOWED_CCR_TTL_SECONDS", "WINDOWED_CCR_LOG"):
        monkeypatch.delenv(k, raising=False)
    yield


def enable(monkeypatch, inject="terminal", shadow=""):
    monkeypatch.setenv("HERMES_WINDOWED_CCR", "1")
    if inject:
        monkeypatch.setenv("WINDOWED_CCR_TOOLS", inject)
    if shadow:
        monkeypatch.setenv("WINDOWED_CCR_SHADOW_TOOLS", shadow)


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------

def test_master_switch_off_is_noop(monkeypatch):
    monkeypatch.setenv("WINDOWED_CCR_TOOLS", "terminal")  # listed but master off
    assert mod.compress_log_result("terminal", {}, big_log()) is None


def test_tool_not_in_any_list_is_noop(monkeypatch):
    enable(monkeypatch, inject="terminal")
    assert mod.compress_log_result("web_extract", {}, big_log()) is None


def test_non_str_result_fails_open(monkeypatch):
    enable(monkeypatch, inject="terminal")
    assert mod.compress_log_result("terminal", {}, {"not": "a string"}) is None


def test_below_size_floor_fails_open(monkeypatch):
    enable(monkeypatch, inject="terminal")
    monkeypatch.setenv("WINDOWED_CCR_MIN_BYTES", "100000")
    assert mod.compress_log_result("terminal", {}, big_log()) is None


# --------------------------------------------------------------------------
# Real compression + store round-trip (live headroom)
# --------------------------------------------------------------------------

def test_inject_compresses_and_marks(monkeypatch):
    enable(monkeypatch, inject="terminal")
    log = big_log()
    out = mod.compress_log_result("terminal", {"command": "pytest"}, log)
    assert out is not None, "should compress a big log"
    assert len(out) < len(log), "must be smaller"
    assert mod.HASH_RE.search(out), "must contain a retrieval marker"


def test_compressed_original_roundtrips_via_store(monkeypatch):
    enable(monkeypatch, inject="terminal")
    log = big_log()
    out = mod.compress_log_result("terminal", {}, log)
    h = mod.HASH_RE.search(out).group(1)
    # retrieve_window should pull the planted needle back out
    res = json.loads(mod.retrieve_window_handler({"hash": h, "mode": "grep", "pattern": "FAIL"}))
    assert res["matched"] is True
    assert "test_payments.py" in res["snippet"]
    assert res["fell_back"] is False


def test_shadow_monitors_but_never_replaces(monkeypatch):
    enable(monkeypatch, inject="", shadow="terminal")
    out = mod.compress_log_result("terminal", {}, big_log())
    assert out is None, "shadow tool must never replace the result"


def test_shadow_wins_on_overlap(monkeypatch):
    enable(monkeypatch, inject="terminal", shadow="terminal")
    out = mod.compress_log_result("terminal", {}, big_log())
    assert out is None, "tool in both lists -> shadow wins -> no replace"


def test_ttl_restore_sets_configured_ttl(monkeypatch):
    enable(monkeypatch, inject="terminal")
    monkeypatch.setenv("WINDOWED_CCR_TTL_SECONDS", "1800")
    log = big_log()
    out = mod.compress_log_result("terminal", {}, log)
    h = mod.HASH_RE.search(out).group(1)
    store = mod._get_store()
    status = store.get_entry_status(h)
    # hook must have re-stored under the configured 1800s TTL, not the 300s default
    assert status["ttl_seconds"] == 1800
    assert status["status"] == "available"


# --------------------------------------------------------------------------
# apply_window
# --------------------------------------------------------------------------

def test_grep_returns_match_with_context():
    log = big_log()
    # case-insensitive grep: "FAIL" matches the failure line AND "1 failed" in
    # the summary -> 2 matches. Both are legitimate hits.
    snip, meta = mod.apply_window(log, {"mode": "grep", "pattern": "FAIL", "context": 2})
    assert meta["matched"] and meta["n_matches"] == 2
    assert "test_payments.py" in snip
    assert meta["n_lines_returned"] <= mod.MAX_CONTEXT_LINES
    # a more specific pattern isolates the single failure line
    snip1, meta1 = mod.apply_window(log, {"mode": "grep", "pattern": "refund_rounding", "context": 0})
    assert meta1["n_matches"] == 1 and "test_payments.py" in snip1


def test_grep_miss_bounded_fallback():
    log = big_log()
    snip, meta = mod.apply_window(log, {"mode": "grep", "pattern": "ZZZ_NO_MATCH"})
    assert meta["matched"] is False and meta["fell_back"] is True
    assert meta["n_lines_returned"] <= mod.FALLBACK_LINES
    assert "bounded fallback" in snip


def test_cap_enforced_on_huge_match():
    # every line matches -> must clip to MAX_CONTEXT_LINES
    log = "\n".join(f"ERROR line {i}" for i in range(500))
    snip, meta = mod.apply_window(log, {"mode": "grep", "pattern": "ERROR", "context": 0})
    assert meta["n_lines_returned"] == mod.MAX_CONTEXT_LINES


def test_head_tail_lines_modes():
    log = "\n".join(f"line{i}" for i in range(100))
    h, hm = mod.apply_window(log, {"mode": "head", "n": 3})
    assert hm["n_lines_returned"] == 3 and "line0" in h
    t, tm = mod.apply_window(log, {"mode": "tail", "n": 2})
    assert tm["n_lines_returned"] == 2 and "line99" in t
    l, lm = mod.apply_window(log, {"mode": "lines", "start": 10, "end": 13})
    assert lm["matched"] and "line9" in l  # 1-indexed start=10 -> line idx 9


def test_full_returns_everything():
    log = big_log()
    snip, meta = mod.apply_window(log, {"mode": "full"})
    assert snip == log and meta["mode"] == "full"


# --------------------------------------------------------------------------
# retrieve_window handler edge cases
# --------------------------------------------------------------------------

def test_retrieve_invalid_hash():
    res = json.loads(mod.retrieve_window_handler({"hash": "not-a-hash!", "mode": "grep"}))
    assert res["error"] == "invalid_hash"


def test_retrieve_graceful_miss_on_unknown_hash():
    # valid-shaped hash that was never stored
    res = json.loads(mod.retrieve_window_handler({"hash": "deadbeefdeadbeefdeadbeef", "mode": "grep", "pattern": "x"}))
    assert res["error"] == "expired_or_missing"
    assert "re-run" in res["message"].lower()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_schema_shape():
    s = mod.RETRIEVE_WINDOW_SCHEMA
    assert s["name"] == "retrieve_window"
    assert s["parameters"]["required"] == ["hash", "mode"]
    assert set(s["parameters"]["properties"]["mode"]["enum"]) == {"grep", "lines", "head", "tail", "full"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
