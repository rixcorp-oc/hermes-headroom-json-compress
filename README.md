# headroom-json-compress

A [Hermes Agent](https://hermes-agent.nousresearch.com/docs) **general plugin** that
losslessly compresses confidently-structured JSON tool results before they enter
the model's context, using the `transform_tool_result` hook.

It is a thin adapter over the JSON/structured path of
[Headroom](https://github.com/chopratejas/headroom) (the context-compression
library, Apache-2.0), wired into Hermes in the one place that keeps prompt
caching intact.

> Written up here: **[Ozempic for the context window](https://intelligent-machines-blog.pages.dev/blog/context-ozempic-losslessly-slimming-tool-output/)** —
> the eval that justified it (including why the lossy-text + retrieval path was
> rejected), the context-window anatomy, and the one design rule that makes it
> safe to turn on.

## What it does

When an in-scope tool returns a JSON string above a size floor, the plugin
reformats it into Headroom's compact table-schema form (a header line declaring
the columns + one row per record). Repeated keys and indentation are factored
out; **no rows are dropped**. Typically ~55–73% byte reduction with 100% row
retention (verified 301/301, 402/402 in the POC).

The compressed string replaces what the model sees for that tool result. It is
compressed **exactly once, at append time**, then frozen into history and never
re-compressed — so the conversation prefix stays byte-identical across turns and
prompt caching is preserved.

## Why a hook (not a core edit or a context engine)

- `transform_tool_result` fires once, at the append boundary, before the result
  enters context — the exact freeze point. History loaded from the DB never
  re-enters it, so the compress-once-and-freeze cache invariant holds for free.
- It's a supported public plugin API → survives harness upgrades, no fork.
- It's a *general* hook, so it does **not** collide with the single-active
  ContextEngine slot (an LCM engine can run alongside it).

## Safety: fail open

Every guard returns `None` (= leave the result untouched). Non-string result,
non-JSON, JSON scalar, below the size floor, tool out of scope (on neither
list), headroom unavailable, compression error, or output that isn't actually
smaller → all pass the original result through unchanged. Compression can never
corrupt or lose a tool result.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `HERMES_HEADROOM_JSON_COMPRESS` | *(unset = OFF)* | Master kill-switch. Set `1` to arm the plugin. When off, **both** injection and monitoring are a total no-op regardless of list contents. |
| `HEADROOM_COMPRESS_TOOLS` | *(empty)* | Comma-separated list of tools to **inject** (replace with compressed output). |
| `HEADROOM_COMPRESS_SHADOW_TOOLS` | *(empty)* | Comma-separated list of tools to **monitor**: run the full gate + compress + log pipeline and emit `decision=shadow-would-compress ratio=…`, but never replace the result. Implies logging on. |
| `HEADROOM_COMPRESS_MIN_BYTES` | `800` | Size floor; smaller results pass through. |
| `HEADROOM_COMPRESS_LOG` | *(unset)* | `1` → emit a per-call decision line to the logger. (Forced on whenever any shadow tools are configured.) |

### Per-tool two-list model

A tool's behaviour is decided by which list it's on:

- on `HEADROOM_COMPRESS_TOOLS` → **inject** (compressed result replaces the original)
- on `HEADROOM_COMPRESS_SHADOW_TOOLS` → **monitor** (logged, never replaced)
- on **both** → **shadow wins** (monitor only — you can't accidentally inject a tool you're still observing)
- on **neither** → **out of scope** (untouched)

**OFF by default** (both lists empty + master switch unset). Recommended rollout:

1. **Arm + shadow:** set `HERMES_HEADROOM_JSON_COMPRESS=1` and add your
   candidate tools to `HEADROOM_COMPRESS_SHADOW_TOOLS` (e.g.
   `session_search,delegate_task,discord,process,todo`). The plugin logs
   `decision=shadow-would-compress` lines with real ratios on live tool
   output, but changes nothing. Watch for a few days.
2. **Promote per tool:** for each tool clearing a worthwhile ratio, move it
   from `HEADROOM_COMPRESS_SHADOW_TOOLS` into `HEADROOM_COMPRESS_TOOLS`. Since
   shadow wins on overlap, you can leave it in both briefly and it stays in
   monitor mode until you remove it from the shadow list.
3. Keep the master switch as your one-line kill-switch for incidents.

### Evaluating shadow output

`scripts/shadow_log_eval.py` parses the plugin's decision lines out of the
Hermes log and gives a per-tool verdict so you can decide what to promote:

```
python scripts/shadow_log_eval.py                 # scans ~/.hermes/logs/agent.log
python scripts/shadow_log_eval.py --since 2026-06-21
python scripts/shadow_log_eval.py --min-ratio 0.35 --min-hit-rate 0.6
python scripts/shadow_log_eval.py --json          # machine-readable
python scripts/shadow_log_eval.py --csv out.csv
```

It reports, per tool: in-scope calls, hit-rate (would-compress / in-scope
calls), mean/median/p25-p75 compression ratio, total bytes saved, and a
verdict (PROMOTE / MARGINAL / SKIP). Two thresholds drive the verdict — a
minimum mean ratio (default 30%) AND a minimum hit-rate (default 50%) — so a
tool whose output is only occasionally large JSON is flagged MARGINAL even if
those rare hits compress well. A passthrough breakdown shows *why* calls
didn't compress (`not-json`, `below-size-floor`, `not-smaller`), which tells
you whether a tool is structurally a bad target (mostly `not-json` = it emits
text, not JSON) or just needs a lower size floor.

#### Automating the evaluation (optional weekly cron)

If you run Hermes Agent, you can have the analyzer run itself on a schedule and
report verdicts instead of remembering to check. Drop a small wrapper in
`~/.hermes/scripts/` that emits the JSON summary:

```bash
#!/bin/bash
# ~/.hermes/scripts/headroom_shadow_weekly.sh
exec "$HOME/.hermes/hermes-agent/venv/bin/python" \
  "$HOME/.hermes/scripts/shadow_log_eval.py" \
  --since 2026-06-21 --json     # set --since to when your shadow run started
```

Then create a cron job (e.g. Mondays 08:00) whose `script` is that wrapper and
whose prompt asks the agent to turn the injected JSON into a short briefing —
per-tool verdicts, which tools are ready to promote, and the likely reason for
any SKIP/MARGINAL. The wrapper's stdout is fed to the job as context, so the
agent only reasons over the compact summary, not the raw log. Keep the
`--since` date aligned with the start of your shadow window so old lines don't
skew the numbers.

## Install

1. Dependency (bare, no ML extras — the JSON path is the Rust `_core.abi3.so`,
   zero torch/transformers):

   ```
   pip install headroom-ai==0.26.0
   ```

   Pin it. Version bumps can change compression bytes → re-run the determinism
   and cache-invariant validation gates before upgrading.

2. Copy this directory into your Hermes plugins tree:

   ```
   cp -r hermes-headroom-json-compress ~/.hermes/plugins/headroom-json-compress
   ```

   (Or wherever your Hermes install discovers plugins — see the Hermes docs.)

## Tests

```
source .venv/bin/activate     # an env with headroom-ai==0.26.0 installed
python -m pytest test_plugin.py -v
```

20 tests: master kill-switch (inject + monitor), per-tool inject/shadow lists,
shadow-wins-on-overlap, out-of-scope passthrough, type/JSON/size gates,
row-losslessness, output-shrinks, cross-process determinism, and the
`register()` contract.

## Credits

- [Headroom](https://github.com/chopratejas/headroom) by Tejas Chopra — the
  underlying context-compression library (Apache-2.0). This plugin uses only its
  structured JSON path (`smart_crush_tool_output`).
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) by Nous Research —
  the host agent and its `transform_tool_result` plugin hook.

## License

Apache-2.0. See [LICENSE](./LICENSE).
