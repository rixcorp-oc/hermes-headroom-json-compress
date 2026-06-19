"""headroom-json-compress plugin — lossless JSON tool-result compression.

Wires ONE behaviour:

* ``transform_tool_result`` hook — when an allow-listed tool returns a
  confidently-structured JSON string above a size floor, reformat it into
  Headroom's compact table-schema form (header line + data rows) via
  ``smart_crush_tool_output``. Typically 55-73% smaller with zero rows
  dropped (the reform factors out repeated keys / indentation; it does NOT
  elide rows). The compressed string replaces what the model sees.

Why a hook and not a core edit or a context engine?
  * ``transform_tool_result`` fires exactly ONCE, at append time, before the
    result enters conversation context — the precise "freeze" boundary.
    DB-loaded history never re-enters it, so the compressed bytes stay
    byte-identical across turns and prompt caching is preserved (the
    compress-once-and-freeze invariant proven in the POC).
  * It's a supported public plugin API, so it survives harness upgrades —
    no fork of run_agent.py / model_tools.py.
  * It's a *general* plugin hook, so it does NOT collide with the
    single-active ContextEngine slot (e.g. an LCM engine can run alongside).

Safety posture — FAIL OPEN. Anything unexpected (non-str result, non-JSON,
below size floor, tool not allow-listed, compression error, output not
actually smaller) returns ``None`` => Hermes keeps the original result
untouched. Compression can never corrupt or lose a tool result.

OFF by default. Enable with ``HERMES_HEADROOM_JSON_COMPRESS=1``.

Config (env):
  HERMES_HEADROOM_JSON_COMPRESS   master switch (unset/0 => no-op)
  HEADROOM_COMPRESS_SHADOW        "1" => compute + log the compression decision
                                  but DO NOT replace the result (always return
                                  None). Lets you observe ratios on real traffic
                                  for days before flipping injection on. Implies
                                  logging on. Overrides the master switch for the
                                  injection decision (shadow never injects).
  HEADROOM_COMPRESS_TOOLS         comma list of tool names (default: search_files)
  HEADROOM_COMPRESS_MIN_BYTES     size floor in bytes (default: 800)
  HEADROOM_COMPRESS_LOG           "1" => emit a per-call decision line to the logger

Dependency: headroom-ai==0.26.0 (bare install, no ML extras). The JSON path
loads only the Rust _core.abi3.so — zero torch/transformers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default allow-list: start narrow with the biggest, cleanest structured win.
_DEFAULT_TOOLS = "search_files"
_DEFAULT_MIN_BYTES = 800

# CCR (retrieval/cache markers) must be OFF — we want pure lossless table
# reform, never a "[N items compressed... Retrieve more: hash=...]" marker or
# any in-process retrieval store. Resolved lazily so importing this module
# never hard-requires headroom (keeps Hermes load resilient if the dep is
# missing — the hook just no-ops).
_ccr_off = None
_crush = None
_import_error: Optional[str] = None


def _ensure_headroom() -> bool:
    """Lazy-import the headroom single-string API. Returns True on success.

    Importing here (not at module top) means a missing/broken headroom install
    degrades to a silent no-op instead of breaking plugin discovery.
    """
    global _crush, _ccr_off, _import_error
    if _crush is not None:
        return True
    if _import_error is not None:
        return False
    try:
        from headroom.transforms.smart_crusher import (  # type: ignore
            smart_crush_tool_output,
            CCRConfig,
        )
        _crush = smart_crush_tool_output
        _ccr_off = CCRConfig(
            enabled=False,
            inject_retrieval_marker=False,
            inject_tool=False,
            inject_system_instructions=False,
        )
        return True
    except Exception as e:  # ImportError or any init failure
        _import_error = str(e)
        logger.warning(
            "headroom-json-compress: headroom import failed (%s); hook is a "
            "no-op. Install the pinned dep into the Hermes venv: "
            "`uv pip install --python <venv>/bin/python -r "
            "plugins/headroom-json-compress/requirements.txt`",
            _import_error,
        )
        return False


# --------------------------------------------------------------------------
# Config helpers (read fresh each call so flags can flip without restart)
# --------------------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("HERMES_HEADROOM_JSON_COMPRESS", "").lower() in {
        "1", "true", "yes", "on",
    }


def _shadow_enabled() -> bool:
    return os.environ.get("HEADROOM_COMPRESS_SHADOW", "").lower() in {
        "1", "true", "yes", "on",
    }


def _log_enabled() -> bool:
    # Shadow mode implies logging — observing without a log is pointless.
    if _shadow_enabled():
        return True
    return os.environ.get("HEADROOM_COMPRESS_LOG", "").lower() in {
        "1", "true", "yes", "on",
    }


def _allow_tools() -> frozenset[str]:
    raw = os.environ.get("HEADROOM_COMPRESS_TOOLS", _DEFAULT_TOOLS)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def _min_bytes() -> int:
    raw = os.environ.get("HEADROOM_COMPRESS_MIN_BYTES", "")
    try:
        return int(raw) if raw else _DEFAULT_MIN_BYTES
    except ValueError:
        return _DEFAULT_MIN_BYTES


def _log_decision(tool: str, orig: int, out: Optional[int], strategy: str, reason: str) -> None:
    if not _log_enabled():
        return
    if out is None:
        logger.info(
            "headroom-json-compress: tool=%s orig=%dB decision=passthrough reason=%s",
            tool, orig, reason,
        )
    else:
        ratio = (1 - out / orig) if orig else 0.0
        logger.info(
            "headroom-json-compress: tool=%s orig=%dB out=%dB ratio=%.3f strategy=%s decision=%s",
            tool, orig, out, ratio, strategy, reason,
        )


# --------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------

def compress_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """transform_tool_result callback.

    Returns a compressed replacement string, or None to leave the result
    unchanged. Every guard returns None (fail open).

    Shadow mode (HEADROOM_COMPRESS_SHADOW=1): the full gate + compress + log
    pipeline runs, but the function ALWAYS returns None — it measures what
    compression *would* do on real traffic without changing any result.
    """
    # 1. Master switch. In shadow mode we proceed even when the master switch
    #    is off (the point is to observe before enabling injection).
    if not _enabled() and not _shadow_enabled():
        return None

    # 2. Allow-list gate.
    if tool_name not in _allow_tools():
        return None

    # 3. Type gate — only string results are compressible here.
    if not isinstance(result, str):
        _log_decision(tool_name, 0, None, "", "non-str-result")
        return None

    orig_len = len(result)

    # 4. Size floor.
    if orig_len < _min_bytes():
        _log_decision(tool_name, orig_len, None, "", "below-size-floor")
        return None

    # 5. JSON gate — must parse to a list or dict (structured payload).
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        _log_decision(tool_name, orig_len, None, "", "not-json")
        return None
    if not isinstance(parsed, (list, dict)):
        _log_decision(tool_name, orig_len, None, "", "json-not-list-or-dict")
        return None

    # 6. Compress once (fail open on any error).
    if not _ensure_headroom():
        _log_decision(tool_name, orig_len, None, "", "headroom-unavailable")
        return None
    try:
        compressed, was_modified, strategy = _crush(
            result,
            config=None,
            ccr_config=_ccr_off,
            with_compaction=True,
        )
    except Exception as e:
        logger.debug("headroom-json-compress: crush error on %s: %s", tool_name, e)
        _log_decision(tool_name, orig_len, None, "", f"crush-error:{type(e).__name__}")
        return None

    # 7. Only replace if it actually shrank and was modified.
    if not was_modified or not isinstance(compressed, str):
        _log_decision(tool_name, orig_len, None, strategy or "", "not-modified")
        return None
    if len(compressed) >= orig_len:
        _log_decision(tool_name, orig_len, len(compressed), strategy or "", "not-smaller")
        return None

    # Shadow mode: log what we WOULD have done, but never inject.
    if _shadow_enabled() and not _enabled():
        _log_decision(tool_name, orig_len, len(compressed), strategy or "", "shadow-would-compress")
        return None

    _log_decision(tool_name, orig_len, len(compressed), strategy or "", "ok")
    return compressed


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", compress_tool_result)
