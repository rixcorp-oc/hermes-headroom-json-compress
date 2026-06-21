# hermes-headroom-plugin

Two [Hermes Agent](https://hermes-agent.nousresearch.com/docs) plugins that shrink
tool-result tokens before they enter the model's context, using the
`transform_tool_result` hook. Both are thin adapters over
[Headroom](https://github.com/chopratejas/headroom) (the context-compression
library, Apache-2.0), wired into Hermes in the one place that keeps prompt
caching intact (compress once, at append, then freeze).

They cover two different shapes of tool output, with two different contracts:

| Plugin | Handles | Contract | ML deps |
|--------|---------|----------|---------|
| [`plugins/json-compress`](./plugins/json-compress) | confidently-structured **JSON** (`search_files`, record arrays) | **lossless** table-reform, no retrieval | **none** (pure Rust) |
| [`plugins/windowed-ccr`](./plugins/windowed-ccr) | big **LOG / terminal / structured-text** blobs | **lossy-with-recovery**: compact view + `hash=` marker + a `retrieve_window` tool that returns a *partial* extract (grep/lines/head/tail) | onnxruntime + transformers (no torch) |

> Written up here: **[Ozempic for the context window](https://intelligent-machines-blog.pages.dev/blog/context-ozempic-losslessly-slimming-tool-output/)** —
> the eval that justified both paths (including why the *full-original* lossy-text +
> retrieval path was rejected, and how making retrieval **deterministic and partial**
> brought the log path back), the context-window anatomy, and the design rules that
> make them safe to turn on.

## The two paths, and why they're separate plugins

The JSON path and the log path use **different Headroom engines** and have
**different correctness contracts**, so they ship as two independent plugins:

- **JSON** is `smart_crush_tool_output` — a Rust table-reform that hoists repeated
  keys into a header and emits one compact line per row. It is **lossless** (no row
  is dropped) and needs **no retrieval** — there is never anything to call back for.
  It loads zero ML libraries. *(`smart_crush_tool_output` compresses JSON only — it
  verifiably leaves logs untouched, which is exactly why logs need a second plugin.)*

- **LOG** is `UniversalCompressor(content_type=LOG, ccr_enabled=True)` — an
  entropy-preserving compressor that emits a compact view plus a `hash=` marker and
  stashes the original in Headroom's compression store. It is **lossy**, so it ships
  with a `retrieve_window` tool: when the model needs detail it didn't keep, it pulls
  back a **bounded partial extract** (the matched span ±5 lines, hard-capped at 40
  lines) rather than the whole original. That "partial, deterministic recovery" is the
  design — it's what makes the log path a net token *win* instead of the token *tax* a
  full-original retrieve incurs.

Keeping them separate means the live, battle-tested JSON plugin is never touched when
the newer log path changes, and each can be flagged or rolled back on its own.

## Shared invariants (both plugins)

- **Compress once, at append, then freeze.** The hook fires at the append boundary, so
  the compressed bytes are frozen into history and never re-compressed. The conversation
  prefix stays byte-identical across turns → prompt caching is preserved. A compressor
  that re-ran over history every turn would shred the cache and cost more than it saves.
- **Fail open, every gate.** Any unexpected condition (wrong type, too small, parse/compress
  error, backend unavailable, output not actually smaller) returns `None` → Hermes keeps
  the original result untouched. Compression can never corrupt or lose a tool result.
- **Per-tool two-list control.** Each plugin reads an INJECT list and a SHADOW list.
  A tool on the inject list is replaced with the compact view; a tool on the shadow list
  is *monitored* (the full pipeline runs and logs a `would-compress` decision, but the
  result is never replaced); **shadow wins on overlap** (you can't accidentally inject a
  tool you're still observing); a tool on neither list is out of scope. Off by default.
- **General hook, not a context engine.** `transform_tool_result` is a supported public
  plugin API, so neither plugin forks the harness, and neither collides with the
  single-active ContextEngine slot (an LCM engine can run alongside both).

## Repo layout

```
plugins/
  json-compress/      # lossless JSON table-reform (transform_tool_result)
    __init__.py
    plugin.yaml
    requirements.txt
    test_plugin.py
  windowed-ccr/       # windowed CCR for logs (hook + retrieve_window tool)
    __init__.py
    plugin.yaml
    requirements.txt
    test_windowed_ccr.py
scripts/
  shadow_log_eval.py  # shared analyzer — parses BOTH plugins' shadow decision lines
```

## Install

Each plugin is self-contained. Copy the one(s) you want into your Hermes plugins tree
and install its dependency. See each plugin's README for specifics:

- [`plugins/json-compress/README.md`](./plugins/json-compress/README.md) — zero ML deps.
- [`plugins/windowed-ccr/README.md`](./plugins/windowed-ccr/README.md) — needs the LOG-path
  runtime (`onnxruntime` + `transformers`, **not** torch).

## Evaluating shadow output (both plugins)

`scripts/shadow_log_eval.py` parses the decision lines that *either* plugin emits in
shadow mode out of the Hermes log and gives a per-tool, per-plugin verdict so you can
decide what to promote from SHADOW → INJECT:

```
python scripts/shadow_log_eval.py                      # scans ~/.hermes/logs/agent.log, both plugins
python scripts/shadow_log_eval.py --plugin windowed    # one plugin only
python scripts/shadow_log_eval.py --since 2026-06-21
python scripts/shadow_log_eval.py --json               # machine-readable
python scripts/shadow_log_eval.py --csv out.csv
```

It reports, per (plugin, tool): in-scope calls, hit-rate, mean/median/p25-p75
compression ratio, total bytes saved, and a PROMOTE / MARGINAL / SKIP verdict against
two thresholds (min mean ratio 30%, min hit-rate 50%). A passthrough breakdown shows
*why* in-scope calls didn't compress (`not-json`, `below-size-floor`, `not-smaller`,
`no-retrieval-marker`, `store-roundtrip-failed`), and the footer prints the exact
SHADOW→INJECT env-var pair to flip for each plugin present.

#### Automating the evaluation (optional weekly cron)

If you run Hermes Agent, you can have the analyzer run itself on a schedule and report
verdicts instead of remembering to check. Drop a small wrapper in `~/.hermes/scripts/`
that emits the JSON summary:

```bash
#!/bin/bash
# ~/.hermes/scripts/headroom_shadow_weekly.sh
exec "$HOME/.hermes/hermes-agent/venv/bin/python" \
  "$HOME/.hermes/scripts/shadow_log_eval.py" \
  --since 2026-06-21 --json     # set --since to when your shadow run started
```

Then create a cron job (e.g. Mondays 08:00) whose `script` is that wrapper and whose
prompt asks the agent to turn the injected JSON into a short briefing. The wrapper's
stdout is fed to the job as context, so the agent reasons over the compact summary,
not the raw log.

## Credits

- [Headroom](https://github.com/chopratejas/headroom) by Tejas Chopra — the underlying
  context-compression library (Apache-2.0). The JSON plugin uses its structured path
  (`smart_crush_tool_output`); the windowed plugin uses its `UniversalCompressor` LOG
  path + compression store.
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) by Nous Research — the host
  agent and its `transform_tool_result` plugin hook.

## License

Apache-2.0. See [LICENSE](./LICENSE).
