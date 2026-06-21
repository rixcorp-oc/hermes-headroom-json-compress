# headroom-windowed-ccr

A [Hermes Agent](https://hermes-agent.nousresearch.com/docs) **general plugin** that
compresses big **log / terminal / structured-text** tool results before they enter the
model's context, and gives the model a `retrieve_window` tool to pull back a **bounded,
partial extract** of the original when it needs detail the compact view didn't keep.

It is the companion to [`json-compress`](../json-compress). Where that plugin does
*lossless* table-reform of JSON and never needs to retrieve anything, this one handles
the path JSON can't: prose-shaped logs, where compression is necessarily **lossy** and
the safety net is a deterministic partial-recovery tool.

> Part of [**hermes-headroom-plugin**](../../).
>
> Written up here: **[Ozempic for the context window](https://intelligent-machines-blog.pages.dev/blog/context-ozempic-losslessly-slimming-tool-output/)** —
> including why the *full-original* CCR path was rejected (it cost more than not
> compressing at all), and how making retrieval **partial and deterministic** reversed
> that result.

## What it does

When an in-scope tool returns a big LOG/TEXT string above a size floor:

1. **Compress hook** (`transform_tool_result`): Headroom's
   `UniversalCompressor(content_type=LOG, ccr_enabled=True)` compresses the blob to a
   compact view plus a `[... hash=…]` retrieval marker, and stashes the **original** in
   Headroom's compression store keyed by that hash. The compact view + marker replaces
   what the model sees, frozen once at append (same cache-safe compress-once invariant as
   the JSON plugin). **The store round-trip is verified before the result is replaced** —
   if the plugin can't mint a marker it can recover, it fails open and leaves the original.
2. **`retrieve_window` tool**: when the model needs detail it didn't keep, it calls this
   with the hash plus a string-op — `grep` / `lines` / `head` / `tail` / `full`. The
   handler resolves the hash against the store and returns **only the matched span ±5
   lines**, hard-capped at **40 lines**. A pattern miss returns a **bounded 30-line
   fallback** (never the whole blob). If the original was evicted (TTL) or lost on
   restart, it returns a graceful "re-run the tool" message — never an error, never a
   fabricated answer.

The point is the *partial* extract. A full-original retrieve pays to un-compress the
entire blob, which (per the eval in the post) makes lossy compression a net token *loss*
for agent log-spelunking, because retrieval is the common case, not the rare one. Handing
back only the windowed span keeps the retrieve cost tiny (~450–650 bytes/call in the eval)
and turns the path into a net win: **90% accuracy, −34% input tokens vs raw, ~27%
cheaper, 0 fallback dumps** (RUN 4, 3-run mean, Opus-class).

## ⚠️ Dependencies: NOT a bare install (unlike json-compress)

This is the one place the two plugins differ materially. The JSON path is pure Rust and
loads zero ML libraries. **The LOG path does not.** Headroom's `UniversalCompressor`
runs through its Kompress backend, which requires:

- `onnxruntime` **and** `transformers` (both — `onnxruntime` alone is not enough), **but
  not** torch; and
- a one-time **~2MB ModernBERT-base** download into the Hugging Face cache on first
  compress.

```
pip install headroom-ai==0.26.0 onnxruntime transformers
```

Notes:

- **No torch.** `transformers` here is used only for its tokenizer/config utilities; the
  ONNX runtime does the inference. The heavy `torch` stack is not pulled in.
- **tokenizers downgrade is safe for litellm.** `transformers` resolves `tokenizers` to
  `<=0.22.x`; that still satisfies litellm's `tokenizers>=0.21,<1.0`, so a Hermes install
  using litellm keeps working. Verified.
- **Without the backend the plugin is inert, not broken.** If `onnxruntime`/`transformers`
  are missing, the compressor returns nothing, the hook **fails open** (every result
  passes through untouched), and — importantly — **no shadow decisions are logged**. If
  you've staged this in shadow mode and see an empty analyzer report, check this first.

## Safety: fail open (hook) / fail soft (tool)

- The compress hook returns `None` on any unexpected condition (non-string, below floor,
  compressor unavailable or didn't shrink, store round-trip unverifiable) → Hermes keeps
  the original result untouched. Compression can never corrupt or lose a result.
- The `retrieve_window` tool never raises: a missing/expired/unknown hash returns a
  human-readable "re-run the original tool" message rather than an error or a guess.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `HERMES_WINDOWED_CCR` | *(unset = OFF)* | Master kill-switch for the compress hook. When off, the hook is a total no-op; the `retrieve_window` tool stays registered but is simply never offered a marker to act on. |
| `WINDOWED_CCR_TOOLS` | *(empty)* | Comma-separated list of tools to **compress + mark** (inject). |
| `WINDOWED_CCR_SHADOW_TOOLS` | *(empty)* | Comma-separated list of tools to **monitor**: run the full gate + compress + store round-trip + log a `decision=shadow-would-compress` line, but never replace the result. Implies logging on. |
| `WINDOWED_CCR_MIN_BYTES` | `2000` | Size floor; smaller results pass through (logs are big — small outputs aren't worth a retrieve round-trip). |
| `WINDOWED_CCR_TTL_SECONDS` | `1800` | Store TTL for the cached original (30m; generous so a later-turn retrieve still resolves). |
| `WINDOWED_CCR_LOG` | *(unset)* | `1` → emit a per-call decision line. (Forced on whenever any shadow tools are configured.) |

### Per-tool two-list model

Same semantics as the JSON plugin: a tool on `WINDOWED_CCR_TOOLS` is **injected**, a tool
on `WINDOWED_CCR_SHADOW_TOOLS` is **monitored** (logged, never replaced), **shadow wins on
overlap**, a tool on neither list is **out of scope**. Off by default.

Recommended target for the inject/shadow lists is `terminal` and other **log emitters**
(builds, test runs, script output). Do **not** add `web_extract` / free prose: the eval
showed Headroom's prose path emits no usable `hash=` marker, so there is nothing to window
against (leave prose to LCM-style externalization instead).

Rollout mirrors the JSON plugin: arm with `HERMES_WINDOWED_CCR=1`, stage candidate tools in
`WINDOWED_CCR_SHADOW_TOOLS`, watch the analyzer's ratios, then move proven tools into
`WINDOWED_CCR_TOOLS`. Evaluate with the shared
[`../../scripts/shadow_log_eval.py`](../../scripts/shadow_log_eval.py)
(`--plugin windowed`).

## Install

1. Dependency (see the dependencies note above — this one is **not** bare):

   ```
   pip install headroom-ai==0.26.0 onnxruntime transformers
   ```

   Pin `headroom-ai`. Version bumps can change compressed bytes **and the hash scheme**,
   which breaks both prompt-cache prefix stability and any markers minted under the old
   version — treat any upgrade as a re-validation event.

2. Copy this directory into your Hermes plugins tree:

   ```
   cp -r plugins/windowed-ccr ~/.hermes/plugins/headroom-windowed-ccr
   ```

## Tests

```
source .venv/bin/activate     # an env with headroom-ai==0.26.0 + onnxruntime + transformers
python -m pytest test_windowed_ccr.py -v
```

17 tests: gating (master switch, inject/shadow lists, shadow-wins-on-overlap, size floor,
out-of-scope), real compress + store round-trip, shadow-never-replaces, the windowing
modes (grep/lines/head/tail/full), ±5 context, the 40-line cap, the bounded 30-line
fallback on a pattern miss, graceful soft-miss on an invalid/missing/evicted hash, TTL
wiring, and the tool schema. (The compress + round-trip tests require the LOG-path runtime
above; without it they fail with the Kompress backend error.)

## Credits

- [Headroom](https://github.com/chopratejas/headroom) by Tejas Chopra — the underlying
  context-compression library (Apache-2.0). This plugin uses its
  `UniversalCompressor` LOG path and `compression_store`.
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) by Nous Research — the host
  agent, its `transform_tool_result` hook, and its tool-registration API.

## License

Apache-2.0. See [LICENSE](../../LICENSE).
