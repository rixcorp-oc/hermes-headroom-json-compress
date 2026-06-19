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

When an allow-listed tool returns a JSON string above a size floor, the plugin
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
non-JSON, JSON scalar, below the size floor, tool not allow-listed, headroom
unavailable, compression error, or output that isn't actually smaller → all
pass the original result through unchanged. Compression can never corrupt or
lose a tool result.

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

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `HERMES_HEADROOM_JSON_COMPRESS` | *(unset = OFF)* | Master switch. Set `1` to enable injection. |
| `HEADROOM_COMPRESS_SHADOW` | *(unset)* | `1` → observe-only: run the full gate + compress + log pipeline but NEVER replace a result. Implies logging on. Use this on real traffic for a few days before enabling injection. |
| `HEADROOM_COMPRESS_TOOLS` | `search_files` | Comma-separated allow-list of tool names. |
| `HEADROOM_COMPRESS_MIN_BYTES` | `800` | Size floor; smaller results pass through. |
| `HEADROOM_COMPRESS_LOG` | *(unset)* | `1` → emit a per-call decision line to the logger. |

**OFF by default.** Recommended rollout:

1. **Shadow run:** set `HEADROOM_COMPRESS_SHADOW=1` (leave the master switch
   unset). The plugin logs `decision=shadow-would-compress` lines with real
   ratios on live tool output, but changes nothing. Watch for a few days.
2. **Enable injection:** set `HERMES_HEADROOM_JSON_COMPRESS=1` with the default
   `search_files` allow-list (biggest, cleanest win). If both flags are set,
   injection wins.
3. Widen `HEADROOM_COMPRESS_TOOLS` as confidence grows. Keep the kill-switch.

## Tests

```
source .venv/bin/activate     # an env with headroom-ai==0.26.0 installed
python -m pytest test_plugin.py -v
```

18 tests: master switch, allow-list, type/JSON/size gates, row-losslessness,
output-shrinks, shadow mode, cross-process determinism, and the `register()`
contract.

## Credits

- [Headroom](https://github.com/chopratejas/headroom) by Tejas Chopra — the
  underlying context-compression library (Apache-2.0). This plugin uses only its
  structured JSON path (`smart_crush_tool_output`).
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) by Nous Research —
  the host agent and its `transform_tool_result` plugin hook.

## License

Apache-2.0. See [LICENSE](./LICENSE).
