"""headroom-windowed-ccr plugin — windowed CCR for logs / structured-text.

The COMPANION to headroom-json-compress. Where that plugin does lossless,
no-retrieval table-reform of JSON, THIS plugin handles the log / terminal /
structured-text path that RUN 4 validated: compress the big text blob to a
compact view + a `hash=` retrieval marker, stash the original in Headroom's
compression store, and register a `retrieve_window` tool so the model can pull
back a PARTIAL extract (grep / line-range / head / tail) instead of the whole
original.

Why a separate plugin (not an extension of headroom-json-compress)?
  * Different engine: JSON path is ``smart_crush_tool_output`` (Rust table
    reform, NEVER compresses logs — verified ``modified=False`` on a log).
    Log path is ``UniversalCompressor(content_type=LOG, ccr_enabled=True)``
    (entropy-preserving + CCR marker, 74.6% on a 600-line log — verified).
  * Different contract: JSON is lossless one-shot (no retrieval). Logs are
    LOSSY-with-recovery (the marker + retrieve_window tool ARE the design).
  * Isolation: shipping this separately means the live JSON plugin is never
    touched, and the windowed path can be flagged/rolled back on its own.

The RUN 4 result this implements (3-run mean, Opus): windowed-CCR on logs =
90% accuracy, -34% input tokens vs raw, ~27% cheaper, 0 fallback dumps —
reversing RUN 2's full-original token-tax inversion (-6.9% / +10%).

Two moving parts
----------------
1. ``transform_tool_result`` hook (``compress_log_result``): an allow-listed
   tool returns a big LOG/TEXT string -> UniversalCompressor compresses it,
   the original lands in ``get_compression_store()`` keyed by a hash, and the
   compact view + ``[... hash=...]`` marker replaces what the model sees.
   Frozen once at append (cache-safe — same compress-once invariant as JSON).
2. ``retrieve_window`` tool (registered via ``register_tool``): the model
   calls it with the hash + a string-op (grep/lines/head/tail/full). The
   handler resolves the hash against the store and returns only the matched
   span +/- context, hard-capped at MAX_CONTEXT_LINES, with a BOUNDED
   fallback on a pattern miss (never the whole blob), and a GRACEFUL miss
   message if the original was evicted (TTL) or the store lost it on restart.

Safety posture — FAIL OPEN. Any unexpected condition in the hook (non-str,
too small, compressor didn't shrink, error) returns ``None`` => Hermes keeps
the original result untouched. The retrieve tool fails SOFT (returns a
human-readable "re-run the tool" message, never raises).

Config (env, read fresh each call so flags flip without restart EXCEPT the
tool registration itself, which happens once at plugin load):
  HERMES_WINDOWED_CCR            master kill-switch for the compress hook
                                 (unset/0 => hook is a total no-op; the
                                 retrieve_window tool stays registered but
                                 simply won't be offered markers to act on).
  WINDOWED_CCR_TOOLS             comma list of tools to COMPRESS+mark (inject).
                                 Default empty. Ships staged to log tools only.
  WINDOWED_CCR_SHADOW_TOOLS      comma list to MONITOR (compute + log the
                                 would-compress decision, return None — never
                                 replace). Shadow wins on overlap.
  WINDOWED_CCR_MIN_BYTES         size floor in bytes (default 2000 — logs are
                                 big; small outputs aren't worth a retrieve
                                 round-trip).
  WINDOWED_CCR_TTL_SECONDS       store TTL for the cached original (default
                                 1800 = 30m; generous so a later-turn retrieve
                                 still resolves).
  WINDOWED_CCR_LOG               "1" => per-call decision line (forced on when
                                 any shadow tool is configured).

Dependency: headroom-ai==0.26.0 (same pin as headroom-json-compress), PLUS the
log compressor's runtime. UNLIKE the JSON path (pure Rust, zero ML deps), the
LOG path (UniversalCompressor) needs the Kompress backend, which requires
``onnxruntime`` AND ``transformers`` (NOT torch) and a one-time ~2MB
ModernBERT-base download into the HF cache. Without that backend the compressor
returns nothing and this plugin fails OPEN (every result passes through
untouched, and no shadow decisions are logged). Install the LOG-path runtime:
``pip install onnxruntime transformers`` (transformers pins tokenizers<=0.22.x;
that range still satisfies litellm's tokenizers>=0.21,<1.0).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Both lists empty by default — a tool on no list is out of scope. Logs only
# when staged (NOT web_extract — RUN 4 proved prose emits no usable marker).
_DEFAULT_TOOLS = ""
_DEFAULT_SHADOW_TOOLS = ""
_DEFAULT_MIN_BYTES = 2000
_DEFAULT_TTL_SECONDS = 1800

# Windowing limits (carried over verbatim from the RUN 4 harness that scored
# 90% / -34% so production behaviour matches the validated eval).
MAX_CONTEXT_LINES = 40   # hard cap on lines one windowed retrieve may return
DEFAULT_CONTEXT = 5      # +/- N lines around each grep match
FALLBACK_LINES = 30      # bounded fallback on a pattern miss (NEVER whole blob)

HASH_RE = re.compile(r"hash=([0-9a-f]{6,32})")

# Lazy headroom handles (import failure => silent no-op, never breaks load).
_UniversalCompressor = None
_UniversalCompressorConfig = None
_ContentType = None
_get_store = None
_import_error: Optional[str] = None


def _ensure_headroom() -> bool:
    """Lazy-import the headroom log-compression + store APIs."""
    global _UniversalCompressor, _UniversalCompressorConfig, _ContentType
    global _get_store, _import_error
    if _UniversalCompressor is not None:
        return True
    if _import_error is not None:
        return False
    try:
        from headroom.compression import (  # type: ignore
            UniversalCompressor,
            UniversalCompressorConfig,
            ContentType,
        )
        from headroom.cache.compression_store import get_compression_store  # type: ignore
        _UniversalCompressor = UniversalCompressor
        _UniversalCompressorConfig = UniversalCompressorConfig
        _ContentType = ContentType
        _get_store = get_compression_store
        return True
    except Exception as e:  # ImportError or any init failure
        _import_error = str(e)
        logger.warning(
            "headroom-windowed-ccr: headroom import failed (%s); compress hook "
            "is a no-op and retrieve_window will report unavailable. Install "
            "the pinned dep into the Hermes venv: `uv pip install --python "
            "<venv>/bin/python -r plugins/headroom-windowed-ccr/requirements.txt`",
            _import_error,
        )
        return False


# --------------------------------------------------------------------------
# Config helpers (read fresh each call)
# --------------------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("HERMES_WINDOWED_CCR", "").lower() in {
        "1", "true", "yes", "on",
    }


def _inject_tools() -> frozenset[str]:
    raw = os.environ.get("WINDOWED_CCR_TOOLS", _DEFAULT_TOOLS)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def _shadow_tools() -> frozenset[str]:
    raw = os.environ.get("WINDOWED_CCR_SHADOW_TOOLS", _DEFAULT_SHADOW_TOOLS)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def _min_bytes() -> int:
    raw = os.environ.get("WINDOWED_CCR_MIN_BYTES", "")
    try:
        return int(raw) if raw else _DEFAULT_MIN_BYTES
    except ValueError:
        return _DEFAULT_MIN_BYTES


def _ttl_seconds() -> int:
    raw = os.environ.get("WINDOWED_CCR_TTL_SECONDS", "")
    try:
        return int(raw) if raw else _DEFAULT_TTL_SECONDS
    except ValueError:
        return _DEFAULT_TTL_SECONDS


def _log_enabled() -> bool:
    if _shadow_tools():
        return True
    return os.environ.get("WINDOWED_CCR_LOG", "").lower() in {
        "1", "true", "yes", "on",
    }


def _log_decision(tool: str, orig: int, out: Optional[int], reason: str) -> None:
    if not _log_enabled():
        return
    if out is None:
        logger.info(
            "headroom-windowed-ccr: tool=%s orig=%dB decision=passthrough reason=%s",
            tool, orig, reason,
        )
    else:
        ratio = (1 - out / orig) if orig else 0.0
        logger.info(
            "headroom-windowed-ccr: tool=%s orig=%dB out=%dB ratio=%.3f decision=%s",
            tool, orig, out, ratio, reason,
        )


# --------------------------------------------------------------------------
# The compress hook
# --------------------------------------------------------------------------

def compress_log_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """transform_tool_result callback for the LOG/TEXT path.

    Returns a compressed replacement string (compact view + retrieval marker),
    or None to leave the result unchanged. Every guard returns None (fail open).
    """
    # 1. Master kill-switch.
    if not _enabled():
        return None

    # 2. Scope gate — tool must be in at least one list.
    inject = _inject_tools()
    shadow = _shadow_tools()
    if tool_name not in inject and tool_name not in shadow:
        return None
    is_shadow = tool_name in shadow  # shadow wins on overlap

    # 3. Type gate.
    if not isinstance(result, str):
        _log_decision(tool_name, 0, None, "non-str-result")
        return None

    orig_len = len(result)

    # 4. Size floor.
    if orig_len < _min_bytes():
        _log_decision(tool_name, orig_len, None, "below-size-floor")
        return None

    # 5. Headroom available?
    if not _ensure_headroom():
        _log_decision(tool_name, orig_len, None, "headroom-unavailable")
        return None

    # 6. Compress once via the LOG path (fail open on any error).
    try:
        uc = _UniversalCompressor(
            config=_UniversalCompressorConfig(ccr_enabled=True)
        )
        res = uc.compress(result, content_type=_ContentType.LOG)
    except Exception as e:
        logger.debug("headroom-windowed-ccr: compress error on %s: %s", tool_name, e)
        _log_decision(tool_name, orig_len, None, f"compress-error:{type(e).__name__}")
        return None

    compressed = getattr(res, "compressed", None)
    ccr_key = getattr(res, "ccr_key", None)

    # 7. Only replace if it shrank AND produced a retrieval marker (a marker
    #    with no recoverable original would be a one-way lossy cut — refuse).
    if not isinstance(compressed, str) or len(compressed) >= orig_len:
        _log_decision(tool_name, orig_len, None, "not-smaller")
        return None
    if not ccr_key or not HASH_RE.search(compressed):
        _log_decision(tool_name, orig_len, None, "no-retrieval-marker")
        return None

    # 8. Confirm the original is actually recoverable from the store BEFORE we
    #    replace what the model sees (defends the fidelity guarantee). If the
    #    compressor didn't populate the store, fail open.
    #
    #    UniversalCompressor.compress() stores with the store's DEFAULT ttl
    #    (300s). That's too short for a marker the model may act on several
    #    turns later, so we re-store the original under the SAME hash with our
    #    configured TTL (verified: store.store(explicit_hash=, ttl=) round-trips
    #    and refreshes the expiry). Best-effort: a re-store failure leaves the
    #    300s entry in place — still recoverable, just sooner-expiring — so we
    #    do NOT fail open on it.
    try:
        store = _get_store()
        entry = store.retrieve(ccr_key)
        if entry is None or getattr(entry, "original_content", None) != result:
            _log_decision(tool_name, orig_len, None, "store-roundtrip-failed")
            return None
        try:
            store.store(result, compressed, ttl=_ttl_seconds(), explicit_hash=ccr_key)
        except Exception as e:
            logger.debug("headroom-windowed-ccr: ttl re-store best-effort failed: %s", e)
    except Exception as e:
        logger.debug("headroom-windowed-ccr: store verify error on %s: %s", tool_name, e)
        _log_decision(tool_name, orig_len, None, f"store-error:{type(e).__name__}")
        return None

    # Shadow tool: log what we WOULD have done, but never inject.
    if is_shadow:
        _log_decision(tool_name, orig_len, len(compressed), "shadow-would-compress")
        return None

    _log_decision(tool_name, orig_len, len(compressed), "ok")
    return compressed


# --------------------------------------------------------------------------
# Windowing engine (verbatim logic from the RUN 4 harness — apply_window)
# --------------------------------------------------------------------------

def apply_window(original: str, params: dict) -> tuple[str, dict]:
    """Run the requested string op over the original. Returns (snippet, meta).

    meta: {mode, matched, n_lines_returned, fell_back, n_matches}
    """
    lines = original.split("\n")
    total = len(lines)
    mode = (params.get("mode") or "grep").lower()

    def clip(idxs):
        keep = sorted(set(i for i in idxs if 0 <= i < total))
        if not keep:
            return None, 0
        if len(keep) > MAX_CONTEXT_LINES:
            keep = keep[:MAX_CONTEXT_LINES]
        out = []
        prev = None
        for i in keep:
            if prev is not None and i > prev + 1:
                out.append(f"... [lines {prev+2}-{i} omitted] ...")
            out.append(f"{i+1:>5}: {lines[i]}")
            prev = i
        return "\n".join(out), len(keep)

    if mode == "full":
        return original, {"mode": "full", "matched": True,
                          "n_lines_returned": total, "fell_back": False}

    if mode == "head":
        n = min(int(params.get("n", 10)), MAX_CONTEXT_LINES)
        snip, k = clip(range(0, n))
        return (snip or "(empty)"), {"mode": "head", "matched": True,
                                      "n_lines_returned": k, "fell_back": False}

    if mode == "tail":
        n = min(int(params.get("n", 10)), MAX_CONTEXT_LINES)
        snip, k = clip(range(total - n, total))
        return (snip or "(empty)"), {"mode": "tail", "matched": True,
                                      "n_lines_returned": k, "fell_back": False}

    if mode == "lines":
        start = max(1, int(params.get("start", 1))) - 1
        end = min(total, int(params.get("end", start + 10)))
        snip, k = clip(range(start, end))
        if snip is None:
            return "(no lines in range)", {"mode": "lines", "matched": False,
                                            "n_lines_returned": 0, "fell_back": False}
        return snip, {"mode": "lines", "matched": True,
                      "n_lines_returned": k, "fell_back": False}

    # mode == grep (default)
    pattern = params.get("pattern") or ""
    ctx = int(params.get("context", DEFAULT_CONTEXT))
    try:
        rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    except re.error:
        rx = None
    hit_idxs = [i for i, ln in enumerate(lines) if rx and rx.search(ln)] if rx else []

    if hit_idxs:
        keep = set()
        for i in hit_idxs:
            for j in range(i - ctx, i + ctx + 1):
                keep.add(j)
        snip, k = clip(keep)
        return snip, {"mode": "grep", "matched": True, "n_lines_returned": k,
                      "fell_back": False, "n_matches": len(hit_idxs)}

    # PATTERN MISS -> bounded fallback (head of file), NEVER the whole blob.
    snip, k = clip(range(0, FALLBACK_LINES))
    return (snip + "\n... [pattern did not match; showing bounded fallback. "
            "Retry with a different pattern or mode=tail for a summary] ..."), \
           {"mode": "grep", "matched": False, "n_lines_returned": k,
            "fell_back": True, "n_matches": 0}


# --------------------------------------------------------------------------
# The retrieve_window tool
# --------------------------------------------------------------------------

RETRIEVE_WINDOW_SCHEMA = {
    "name": "retrieve_window",
    "description": (
        "Retrieve a PARTIAL extract from an original tool output that was "
        "compressed to save tokens. When a tool result shows a marker like "
        "'[N items compressed to M. Retrieve more: hash=abc123...]', call this "
        "with that hash to pull back ONLY the lines you need — not the whole "
        "original. Choose a mode:\n"
        "- grep: return lines matching `pattern` (regex), each with +/- "
        "`context` lines. PREFERRED — use a specific pattern like 'FAIL|ERROR'.\n"
        "- lines: return lines `start`..`end` (1-indexed).\n"
        "- head: first `n` lines.\n"
        "- tail: last `n` lines (e.g. a summary line).\n"
        "- full: everything (last resort; expensive — defeats the point).\n"
        f"A single retrieve returns at most {MAX_CONTEXT_LINES} lines unless "
        "mode=full. If your pattern misses, you get a bounded fallback (not the "
        "whole file) — retry with a better pattern or mode=tail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "The hash from the compression marker (hex string).",
            },
            "mode": {
                "type": "string",
                "enum": ["grep", "lines", "head", "tail", "full"],
                "description": "Retrieval strategy. Prefer 'grep' with a pattern.",
            },
            "pattern": {
                "type": "string",
                "description": "Regex for mode=grep, e.g. 'FAIL|ERROR|Traceback'.",
            },
            "context": {
                "type": "integer",
                "description": f"+/- lines around each grep match (default {DEFAULT_CONTEXT}).",
            },
            "start": {"type": "integer", "description": "First line, 1-indexed (mode=lines)."},
            "end": {"type": "integer", "description": "Last line, 1-indexed (mode=lines)."},
            "n": {"type": "integer", "description": "Line count (mode=head/tail)."},
        },
        "required": ["hash", "mode"],
    },
}


def _valid_hash(h: Any) -> bool:
    return isinstance(h, str) and bool(re.fullmatch(r"[0-9a-f]{6,32}", h.lower()))


def retrieve_window_handler(args: dict, **_: Any) -> str:
    """Resolve a compression hash to a windowed snippet. Fails SOFT (never raises).

    Graceful-miss contract: if the original was evicted (TTL) or
    lost on restart, return a clear "re-run the tool" message instead of an
    error or a fabricated answer.
    """
    import json

    h = (args or {}).get("hash", "")
    if not _valid_hash(h):
        return json.dumps({
            "error": "invalid_hash",
            "message": "Provide the hex hash from a compression marker like "
                       "'Retrieve more: hash=abc123...'.",
        })

    if not _ensure_headroom():
        return json.dumps({
            "error": "unavailable",
            "message": "Compression store unavailable (headroom not importable). "
                       "Re-run the original tool to get the full output.",
        })

    try:
        store = _get_store()
        entry = store.retrieve(h.lower())
    except Exception as e:
        logger.debug("headroom-windowed-ccr: retrieve store error: %s", e)
        entry = None

    if entry is None:
        # Graceful miss — evicted by TTL, lost on restart, or never stored.
        return json.dumps({
            "error": "expired_or_missing",
            "message": "That compressed original is no longer cached (it may "
                       "have expired or the session restarted). Re-run the "
                       "original tool to regenerate the output.",
            "hash": h,
        })

    original = getattr(entry, "original_content", None)
    if not isinstance(original, str):
        return json.dumps({
            "error": "corrupt_entry",
            "message": "Cached entry could not be read. Re-run the original tool.",
            "hash": h,
        })

    snippet, meta = apply_window(original, args or {})
    return json.dumps({
        "hash": h,
        "mode": meta.get("mode"),
        "matched": meta.get("matched"),
        "fell_back": meta.get("fell_back"),
        "lines_returned": meta.get("n_lines_returned"),
        "n_matches": meta.get("n_matches"),
        "total_lines": len(original.split("\n")),
        "snippet": snippet,
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def register(ctx) -> None:
    # The compress hook.
    ctx.register_hook("transform_tool_result", compress_log_result)
    # The retrieval tool. Always registered so a marker minted earlier in a
    # session can always be acted on; the master switch only gates the HOOK.
    try:
        ctx.register_tool(
            name="retrieve_window",
            toolset="windowed_ccr",
            schema=RETRIEVE_WINDOW_SCHEMA,
            handler=retrieve_window_handler,
            description="Retrieve a partial extract from a compressed tool output.",
            emoji="🪟",
        )
    except Exception as e:  # never break plugin load on a registration hiccup
        logger.warning("headroom-windowed-ccr: retrieve_window registration failed: %s", e)
