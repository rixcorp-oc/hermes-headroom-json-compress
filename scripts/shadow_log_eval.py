#!/usr/bin/env python3
"""Analyze headroom-json-compress shadow-mode decision logs.

The plugin emits one INFO line per in-scope tool call to the Hermes logger,
e.g.:

  headroom-json-compress: tool=discord orig=27047B out=7602B ratio=0.719 \
      strategy=table decision=shadow-would-compress
  headroom-json-compress: tool=search_files orig=1115B decision=passthrough \
      reason=not-json

This script parses those lines from one or more log files and produces a
per-tool verdict so you can decide which tools to PROMOTE from
HEADROOM_COMPRESS_SHADOW_TOOLS into HEADROOM_COMPRESS_TOOLS.

What it reports per tool:
  * calls seen, and how they split across decisions/passthrough reasons
  * for would-compress calls: count, mean/median ratio, total bytes saved,
    p25/p75 ratio spread
  * a verdict (PROMOTE / MARGINAL / SKIP) from two thresholds:
      - a minimum mean ratio (default 0.30)
      - a minimum hit-rate = would-compress / total in-scope calls (default 0.50)
    A tool that rarely produces JSON (low hit-rate) is a poor injection
    target even if the rare hits compress well, because most calls just burn
    the gate.

Usage:
  shadow_log_eval.py                      # scans ~/.hermes/logs/agent.log
  shadow_log_eval.py --log path [path...] # explicit files (supports .gz)
  shadow_log_eval.py --since 2026-06-20   # only lines on/after this date
  shadow_log_eval.py --min-ratio 0.35 --min-hit-rate 0.6
  shadow_log_eval.py --json               # machine-readable summary
  shadow_log_eval.py --csv out.csv        # per-tool table as CSV

Exit code is 0 always (reporting tool); parse/IO errors print to stderr.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

# Matches the plugin's two log shapes. The leading timestamp is the standard
# Hermes log prefix "YYYY-MM-DD HH:MM:SS,mmm" — optional so raw lines parse too.
_TS = r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
_COMPRESS_RE = re.compile(
    r"headroom-json-compress: tool=(?P<tool>\S+) orig=(?P<orig>\d+)B "
    r"out=(?P<out>\d+)B ratio=(?P<ratio>[\d.]+) strategy=(?P<strategy>\S*) "
    r"decision=(?P<decision>\S+)"
)
_PASS_RE = re.compile(
    r"headroom-json-compress: tool=(?P<tool>\S+) orig=(?P<orig>\d+)B "
    r"decision=passthrough reason=(?P<reason>\S+)"
)
_LINE_TS_RE = re.compile(_TS)

DEFAULT_LOG = os.path.expanduser("~/.hermes/logs/agent.log")


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _line_date(line: str) -> Optional[str]:
    m = _LINE_TS_RE.search(line)
    if not m:
        return None
    return m.group("ts")[:10]  # YYYY-MM-DD


class ToolStats:
    __slots__ = ("ratios", "saved", "would", "passthrough", "other_decisions")

    def __init__(self):
        self.ratios: list[float] = []
        self.saved = 0          # bytes saved on would-compress calls
        self.would = 0          # would-compress / ok count
        self.passthrough: dict[str, int] = defaultdict(int)  # reason -> n
        self.other_decisions: dict[str, int] = defaultdict(int)  # e.g. not-smaller

    @property
    def in_scope_calls(self) -> int:
        return self.would + sum(self.passthrough.values()) + sum(self.other_decisions.values())

    @property
    def hit_rate(self) -> float:
        n = self.in_scope_calls
        return self.would / n if n else 0.0


def parse(paths: list[str], since: Optional[str]) -> dict[str, ToolStats]:
    stats: dict[str, ToolStats] = defaultdict(ToolStats)
    for path in paths:
        try:
            fh = _open(path)
        except OSError as e:
            print(f"warn: cannot open {path}: {e}", file=sys.stderr)
            continue
        with fh:
            for line in fh:
                if "headroom-json-compress:" not in line:
                    continue
                if since:
                    d = _line_date(line)
                    if d and d < since:
                        continue
                mc = _COMPRESS_RE.search(line)
                if mc:
                    tool = mc.group("tool")
                    orig = int(mc.group("orig"))
                    out = int(mc.group("out"))
                    ratio = float(mc.group("ratio"))
                    decision = mc.group("decision")
                    st = stats[tool]
                    if decision in ("shadow-would-compress", "ok"):
                        st.would += 1
                        st.ratios.append(ratio)
                        st.saved += max(0, orig - out)
                    else:  # e.g. not-smaller (has out= but wasn't a win)
                        st.other_decisions[decision] += 1
                    continue
                mp = _PASS_RE.search(line)
                if mp:
                    stats[mp.group("tool")].passthrough[mp.group("reason")] += 1
    return stats


def verdict(st: ToolStats, min_ratio: float, min_hit_rate: float) -> str:
    if st.would == 0:
        return "SKIP (no compressible calls)"
    mean = statistics.fmean(st.ratios)
    hr = st.hit_rate
    if mean >= min_ratio and hr >= min_hit_rate:
        return "PROMOTE"
    if mean >= min_ratio and hr >= min_hit_rate * 0.5:
        return "MARGINAL (good ratio, low hit-rate)"
    if mean >= min_ratio:
        return "MARGINAL (good ratio, rarely fires)"
    return "SKIP (ratio below threshold)"


def human_bytes(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(v) < 1024:
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}TB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", default=None,
                    help=f"log file(s) or globs (default: {DEFAULT_LOG}, plus rotated .1/.gz)")
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                    help="only consider lines on/after this date")
    ap.add_argument("--min-ratio", type=float, default=0.30,
                    help="min mean compression ratio to PROMOTE (default 0.30)")
    ap.add_argument("--min-hit-rate", type=float, default=0.50,
                    help="min would-compress / in-scope-calls to PROMOTE (default 0.50)")
    ap.add_argument("--json", action="store_true", help="emit JSON summary")
    ap.add_argument("--csv", metavar="FILE", help="write per-tool table to CSV")
    args = ap.parse_args(argv)

    if args.log:
        paths: list[str] = []
        for p in args.log:
            paths.extend(glob.glob(os.path.expanduser(p)) or [os.path.expanduser(p)])
    else:
        # default: agent.log + any rotated siblings
        base = DEFAULT_LOG
        paths = [base] + sorted(glob.glob(base + ".*"))
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            paths = [base]

    stats = parse(paths, args.since)

    if not stats:
        print("No headroom-json-compress decision lines found.", file=sys.stderr)
        print(f"Scanned: {', '.join(paths)}", file=sys.stderr)
        print("Is the plugin armed (HERMES_HEADROOM_JSON_COMPRESS=1) with tools in "
              "HEADROOM_COMPRESS_SHADOW_TOOLS, and HEADROOM_COMPRESS_LOG=1?", file=sys.stderr)
        return 0

    rows = []
    for tool in sorted(stats, key=lambda t: -stats[t].saved):
        st = stats[tool]
        mean = statistics.fmean(st.ratios) if st.ratios else 0.0
        median = statistics.median(st.ratios) if st.ratios else 0.0
        p25 = (statistics.quantiles(st.ratios, n=4)[0] if len(st.ratios) >= 4 else mean)
        p75 = (statistics.quantiles(st.ratios, n=4)[2] if len(st.ratios) >= 4 else mean)
        rows.append({
            "tool": tool,
            "in_scope_calls": st.in_scope_calls,
            "would_compress": st.would,
            "hit_rate": round(st.hit_rate, 3),
            "mean_ratio": round(mean, 3),
            "median_ratio": round(median, 3),
            "p25_ratio": round(p25, 3),
            "p75_ratio": round(p75, 3),
            "bytes_saved": st.saved,
            "passthrough": dict(st.passthrough),
            "other_decisions": dict(st.other_decisions),
            "verdict": verdict(st, args.min_ratio, args.min_hit_rate),
        })

    if args.json:
        print(json.dumps({"scanned": paths, "since": args.since,
                          "thresholds": {"min_ratio": args.min_ratio, "min_hit_rate": args.min_hit_rate},
                          "tools": rows}, indent=2))
        return 0

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tool", "in_scope_calls", "would_compress", "hit_rate",
                        "mean_ratio", "median_ratio", "p25_ratio", "p75_ratio",
                        "bytes_saved", "verdict"])
            for r in rows:
                w.writerow([r["tool"], r["in_scope_calls"], r["would_compress"],
                            r["hit_rate"], r["mean_ratio"], r["median_ratio"],
                            r["p25_ratio"], r["p75_ratio"], r["bytes_saved"], r["verdict"]])
        print(f"wrote {args.csv}")

    # Human-readable report
    print(f"Headroom shadow-log analysis  (scanned {len(paths)} file(s)"
          f"{', since ' + args.since if args.since else ''})")
    print(f"Thresholds: mean ratio >= {args.min_ratio:.0%}, hit-rate >= {args.min_hit_rate:.0%}\n")
    hdr = f"{'tool':<18}{'calls':>6}{'hits':>6}{'hit%':>7}{'mean':>7}{'med':>7}{'p25-p75':>12}{'saved':>10}  verdict"
    print(hdr)
    print("-" * len(hdr))
    total_saved = 0
    for r in rows:
        total_saved += r["bytes_saved"]
        spread = f"{r['p25_ratio']:.0%}-{r['p75_ratio']:.0%}"
        print(f"{r['tool']:<18}{r['in_scope_calls']:>6}{r['would_compress']:>6}"
              f"{r['hit_rate']:>6.0%} {r['mean_ratio']:>6.0%} {r['median_ratio']:>6.0%}"
              f"{spread:>12}{human_bytes(r['bytes_saved']):>10}  {r['verdict']}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<18}{'':>6}{'':>6}{'':>7}{'':>7}{'':>7}{'':>12}{human_bytes(total_saved):>10}")

    # Passthrough breakdown (why calls didn't compress) — the actionable detail
    any_pass = any(r["passthrough"] or r["other_decisions"] for r in rows)
    if any_pass:
        print("\nPassthrough / non-win reasons (why in-scope calls weren't compressed):")
        for r in rows:
            detail = {**r["passthrough"], **r["other_decisions"]}
            if detail:
                parts = ", ".join(f"{k}={v}" for k, v in sorted(detail.items(), key=lambda x: -x[1]))
                print(f"  {r['tool']:<18} {parts}")
        print("\nNote: 'not-json' on a tool means its output often isn't a JSON "
              "string (e.g. text/grep mode) — a poor injection target regardless "
              "of ratio. 'not-smaller'/'below-size-floor' = payloads too small to win.")

    print("\nPromote a PROMOTE-verdict tool by moving it from "
          "HEADROOM_COMPRESS_SHADOW_TOOLS into HEADROOM_COMPRESS_TOOLS in "
          "~/.hermes/.env, then restart Hermes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
