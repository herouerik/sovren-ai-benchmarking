# Handover — Benchmarking work for the GPU server agent

**From:** M4 (192.168.68.106) benchmarking machine, 2026-08-15
**To:** GPU server (192.168.68.115) agent — takes over benchmark execution
**Status:** Unified 6× P100 pool is LIVE. First harness smoke tests on the pool
revealed model-serving issues that need hands-on server access to resolve.

---

## Why this is handed over

The benchmark harness runs from the M4 over HTTP to the GPU server's Ollama
(port 11434). That remote execution mode is working, but several models on the
unified pool return **empty content via `/api/chat`** or **hang**, and the
diagnosis needs direct server access (ollama logs, service restarts, possibly
template/model re-pulls). The M4 machine cannot fix those from here — that's
why execution is being handed to the GPU server agent which has system access.

## Current state (verified this session)

- `ollama-unified.service` active, CUDA_VISIBLE_DEVICES=1-6 (6× P100), Ollama
  0.32.9. See `GPU_SERVER_UNIFIED_POOL_SPEC.md` for the live setup and its
  corrections (service names, env file, ctx ceilings — all real now).
- All models in `config-gpu-unified.yaml` are pulled and present on the server.
- Harness (`run_benchmark.py`) installed and imports cleanly on the M4
  (added `mpmath>=1.3.0` to `requirements.txt` — new vendored BFCL env dep).

## Problems found — need your investigation

Smoke-tested `--benchmarks speed` and direct streaming probes on the unified pool:

1. **`hf.co/bartowski/Muse-Glimmer-30B-GGUF:Q4_K_M` — empty content via
   `/api/chat`.** The model generates tokens (`eval_count: 3`) but returns
   `message.content: ""` both streamed and non-streamed. `/api/generate`
   returns only `" to=self"`. The ONYX ATEM chat template emits a
   `to=<recipient>` prefix then stops at its own stop tokens
   (`<|start|>`, `<|message|>`, `<|eot|>`) — the response is being truncated
   by the template's stop-word list. Needs a template/tokenizer fix or a
   different way to drive this model (possibly `recipient` handling in the
   request, or overriding `stop`).
2. **`qwen3.6-128k:latest` — empty content.** Same symptom via `/api/chat`
   (and `/api/generate`), although this model benchmarked fine on this pool
   earlier in the week (has mmlu/gsm8k/humaneval scores in `merged.summary.json`).
   May be an eviction/OOM-then-emptystring artifact rather than a template issue.
3. **`llama4:scout` and `deepseek-r1:70b` — hang / empty** on first probe;
   `curl` to `/api/generate` returned nothing within ~100s. Possibly long
   cold-load times (first load after service start) — needs a warmup loop and
   a longer timeout to confirm, or a real hang worth investigating.
4. **`gemma4:31b` works** — 8.8 tok/s on the `speed` probe; 2/4 probes failed
   (likely TTFT/cold-load related, not model-broken). This is the baseline for
   "healthy model on this pool."

Pool state note: `api/ps` showed **zero models resident** after the probes —
either keep_alive isn't holding or everything got evicted. Worth confirming
`OLLAMA_KEEP_ALIVE=24h` is honored and whether load order matters.

## What the benchmark run needs (once serving is healthy)

From the M4 (or from the server with the harness installed):

```bash
.venv/bin/python run_benchmark.py --config config-gpu-unified.yaml \
    --benchmarks mmlu gsm8k humaneval mbpp spider philosophical \
    --n-samples 20
```

Models to benchmark (config `models:` block, all pulled):
`deepseek-r1:70b`, `MichelRosselli/GLM-4.5-Air:Q3_K_M`, `gemma4:31b`,
`qwen3-coder-next:sovereign-128k`, `llama4:scout`, `qwen3.6-128k:latest`,
`hf.co/bartowski/Muse-Glimmer-30B-GGUF:Q4_K_M`, `qwen3-coder:30b-sovereign`.

Already-benchmarked (in `results/merged.summary.json`, GPU pools):
`qwen3-coder-next:sovereign-128k` (mmlu/gsm8k/humaneval),
`qwen3-coder:30b-sovereign` (mmlu/gsm8k/humaneval),
`qwen3.6-128k:latest` (mmlu/gsm8k/humaneval). The unified-pool run can use
`--baseline results/merged.summary.json` to merge and preserve those.

Speed caveat (see `docs/ROADMAP.md`): decode TPS is comparable only within one
Ollama version / machine. All unified-pool runs are new-version (0.32.9) —
their speed numbers should NOT be blended with the older M4 or pool-split GPU
numbers in the same column without a note.

## Files in this repo

| File | Purpose |
|---|---|
| `GPU_SERVER_UNIFIED_POOL_SPEC.md` | Live unified-pool setup + corrections (written by GPU server agent) |
| `config-gpu-unified.yaml` | Benchmark config for the unified pool (source of truth for ctx ceilings) |
| `config-gpu-server.yaml` | Old pool-a/b split configs (keep for rollback) |
| `results/merged.summary.json` | 23-model aggregate (M4 + GPU) for dashboard merging |
| `merge_summaries.py` | Merge script for dashboard (M4 + GPU side by side) |
| `docs/ROADMAP.md` | Harness analysis + BFCL status (v0.3 scope) |
| `requirements.txt` | Now includes `mpmath>=1.3.0` |

## Open questions for the GPU server agent

1. Is Muse-Glimmer's chat template fixable on this Ollama (stop-word override,
   recipient injection) or should it be excluded from the run?
2. Is `qwen3.6-128k` empty-content a flake or a regression on 0.32.9?
3. Are `llama4:scout`/`deepseek-r1:70b` just slow cold-loads? Please warm them
   before the run and report measured first-load times.
4. After a healthy run, regenerate the merged dashboard so M4 + GPU unified
   rows render side by side.
