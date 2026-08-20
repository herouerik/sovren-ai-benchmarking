# Roadmap and open analysis

Working notes for what this harness measures well, where it is weak, and what
should come next. Written at the end of the v0.2 work (2026-08-13).

## Machine roles

| Machine | Role |
|---|---|
| M4 Pro / 48 GB | **Reference performance machine.** Model evaluation only — speed numbers are only comparable when produced here |
| Private laptop | Development of the harness itself |

Speed figures (`decode_tps`) are hardware-bound and only comparable within a
single machine. Accuracy figures are portable. Any dashboard that mixes runs
from different machines must say so — `hardware` is recorded on every result
record for exactly this reason.

Do not run anything else heavy on the reference machine during a benchmark.
We saw a process competing for CPU dropping decode from ~23 tok/s to ~2 tok/s —
a 10× collapse that invalidates every speed number in the affected window.
Detection recipe: normalise each sample's `decode_tps` against that model's own
median and bucket by hour; a contended window shows a median ratio well under
1.0, while a healthy one sits at 0.97–1.02.

---

## Current state (v0.2)

Seven benchmarks: MMLU, ARC, GSM8K, HumanEval, MBPP, Spider, philosophical
(LLM-as-judge). See the README release notes for what v0.2 fixed.

These are well-established academic benchmarks, but they are 2018–2021 vintage
and were designed to discriminate between models far weaker than the ones now
running locally. That shows up in the data.

### Benchmark saturation — measured, not assumed

From the 16-model v2 baseline, counting how many *items* every model answers
correctly (such an item contributes nothing to ranking):

| benchmark | median score | spread | items all 16 pass | informative |
|---|---|---|---|---|
| **arc** | 95% | 25 pts | **13 of 20** | 35% |
| **gsm8k** | 96% | 32 pts | **13 of 25** | 48% |
| humaneval | 75% | 50 pts | 6 of 20 | 65% |
| mbpp | 60% | 25 pts | 2 of 20 | 75% |
| mmlu | 75% | 50 pts | 3 of 20 | 85% |

ARC is effectively a 7-question test and GSM8K a 12-question one; the rest of
the runtime buys no discrimination. Note a saturated benchmark still has value
as a floor check — it catches a broken or badly quantised model — it just
cannot rank the top of the field.

### Comparability rules learned the hard way

- **Speed** is comparable only within one Ollama version, one machine, and one
  measurement method. `tok_per_sec` (end-to-end) and `decode_tps` (generation
  only) differ by 3–5× on short answers; never mix them in one column.
- **Accuracy** is comparable only at a fixed `sample_seed`, since the seed
  determines which questions are asked.
- **Thinking on/off** changes scores substantially (+7 points overall for
  Muse Glimmer, ~0 for Qwen 3.6 35B). Always compare like with like; the
  harness records `think` per sample so this cannot be lost.

---

## v0.3 — proposed work

### A. Restore discrimination

Two approaches, only one of which adds information.

1. **Harder content** (preferred, keeps external comparability):
   - **MMLU-Pro** — drop-in replacement for MMLU, 10 options, far less saturated
   - **EvalPlus** (HumanEval+ / MBPP+) — ~80× more test cases on the same
     problems, typically 10–20 points lower
   - **MATH** or **AIME** in place of / alongside GSM8K
2. **Difficulty-filtered sampling** (cheap, uses data already collected):
   compute per-item pass rate across the model fleet and sample only items in a
   discriminating band (e.g. 0.15–0.85). Caveat: selecting items using the same
   models that are then scored is circular, and the resulting scores are no
   longer comparable to published numbers. Useful for internal ranking, not for
   external claims.

Do **not** rescale scores (z-scores, rank normalisation) to spread the field.
It changes the picture without adding information, and makes a one-sample
difference look dramatic.

Also worth adding: a per-benchmark **discrimination diagnostic** on the
dashboard, so a saturated column is visibly flagged rather than silently
misleading.

### B. Anchoring to widely published numbers

Motivation: leaderboards such as [Artificial Analysis](https://artificialanalysis.ai/)
publish composite indices (AA Intelligence Index, GDPVal, CursorBench,
APEX-*, AA-Briefcase, Harvey LAB). It would be useful to know roughly where
local models sit relative to those.

**Most of those cannot be reproduced here.** They are either proprietary
datasets or vendor-run harnesses; the index is a composite of *their* runs, not
a downloadable benchmark. Quoting their numbers beside ours would imply a
comparability that does not exist.

Worse, even for *open* benchmarks our numbers will not match published ones:
prompt format, answer extraction, shot count, sample size and thinking mode
each move scores by 10+ points. We saw a 20-point swing from an extraction fix
alone during v0.2.

**The sound way to anchor is a reference model run through this harness.**
Point the existing OpenAI-compatible client at one cloud model, run the
identical suite, and record it as a calibration row. Every variable is then
held constant, and the statement becomes "on our harness, model X scores N,
our best local model scores M" — which is defensible. The `judge` config
already supports OpenAI-compatible endpoints, so the client plumbing exists.

### C. Candidate benchmarks, by value

| benchmark | why | effort |
|---|---|---|
| **MMLU-Pro** | Fixes the most saturated column; widely reported | drop-in |
| **GPQA Diamond** | Graduate-level science, 198 items, strongly discriminating | small |
| **IFEval** | Instruction following, *programmatic* scoring (no judge needed) | cheap |
| **BFCL** | Function calling / tool use — see below | medium |
| **EvalPlus** | Harder versions of benchmarks already run | drop-in |
| **LiveCodeBench** | Time-versioned, contamination-resistant coding | medium |
| **SWE-bench Verified** | The benchmark that matters for agentic coding | heavy (repos + containers) |

The first three fit the existing MCQ / exact-match machinery and are mostly a
`load_samples()` + `score()` implementation each.

### D. The biggest gap: tool use

The models now being evaluated are marketed as **agentic** — Muse Glimmer is
explicitly an agentic model, Qwen 3.6 emphasises agentic coding. This suite
measures no function calling, no multi-step tool loops, and no failure
recovery. That is the axis the vendors' claims rest on and the one the
dashboard is silent about.

BFCL is the cheapest entry point. A local agentic-loop benchmark (tool call →
observation → recovery from a bad call) would be more representative of the
actual use case — routing work to local models — but is a larger build.

### E. BFCL build scope

Written 2026-08-15, after scoping against the current benchmark harness code.

**Status (2026-08-15, same day): Phase 1 done, Phase 2 done for Long-Context.**
Both landed and were verified against a real model (qwen3.6-128k), not just
imported cleanly. One correction to the plan below: `bfcl-eval` was never
actually used — attempting the install showed it pulls in sglang/vllm/
cuda-bindings (multiple GB) as hard dependencies just to reach its dataset
and checker utilities. Both phases instead use `huggingface_hub` (already a
dependency) to pull BFCL's dataset JSON directly, plus a local reimplementation
of the AST/state-diff scoring logic — see `benchmarks/bfcl.py` and
`benchmarks/bfcl_multi_turn.py` module docstrings for the full reasoning.
Phase 2's stateful env classes are vendored (Apache 2.0, attributed in
`benchmarks/bfcl_multi_turn_envs/NOTICE.md`) rather than imported from that
package for the same reason. Base/Missing-Functions/Missing-Parameters
multi-turn subcategories share the same engine and are not yet wired up —
straightforward follow-on (new `BENCHMARK_REGISTRY` entry, different category
constant) once there's a reason to prioritize them over Long-Context.

Real cost data point: one Long-Context sample (4 turns, 28 tool calls)
against qwen3.6-128k took ~450-500s on the unified GPU pool. Left disabled
by default in `config.yaml`; enable deliberately with a small `n_samples`.

BFCL V4 (current, last leaderboard update Jul 2026) splits into single-turn
categories (Non-Live/Live × single/multiple/parallel/parallel-multiple) and an
800-example multi-turn suite with four subcategories: Base, Missing-Functions,
Missing-Parameters, and **Long-Context** (state-tracking over extended dialogue
histories — directly relevant to evaluating large-context models, see the GPU
unified-pool work).

Official tooling is the `bfcl-eval` PyPI package, but its own execution
pipeline assumes sglang/vllm-served models — not a drop-in for this harness.
Plan: use `bfcl-eval` as a **library** (dataset + ground-truth + AST/state-diff
checkers only), keep `OllamaClient` as the execution driver — the same pattern
`benchmarks/sql.py` already uses (HF `datasets` for data, `harness/sandbox.py`
for our own execution/scoring, not Spider's reference implementation).

**Phase 1 — single-turn AST categories (medium, matches existing architecture)**

- `harness/client.py`: extend `complete_native()` (client.py:72) to accept a
  `tools=` param in the request body and parse `message.tool_calls` from the
  streamed `/api/chat` response (client.py:135 currently only reads
  `thinking`/`content`). Ollama's native API already supports this; it's just
  unused today.
- New `benchmarks/bfcl.py`, shaped like `SpiderBenchmark` (sql.py:65):
  `load_samples()` pulls from `bfcl_eval`'s dataset, `score()` calls
  `bfcl_eval`'s AST-checker against the parsed `tool_calls` rather than
  hand-rolling parameter matching.
- Registry entry in `run_benchmark.py`'s `BENCHMARK_REGISTRY` (line 45), plus a
  config group entry following the existing `config.yaml` group-name pattern.
- New dependency: `bfcl-eval` (pip) — justified; not reasonably reimplementable
  from stdlib, same bar as the existing `datasets` dependency.

**Phase 2 — multi-turn/agentic (heavy, the part that actually stresses large
context and big models)**

- `BaseBenchmark.run()` (base.py:102) is hard single-shot — one prompt in, one
  response out. Multi-turn needs a genuinely different loop: model → tool_call
  → execute against BFCL's stateful mock classes (in-memory Python objects
  simulating a file system, trading account, etc., vendored from `bfcl_eval`)
  → append tool result as a `tool`-role message → repeat until final answer or
  turn budget. This is a new `run_multi_turn()` path, not a `BaseBenchmark`
  subclass tweak.
- Scoring is state-diff (does the mock object's final state match ground
  truth) plus optional call-path checking — different shape from every
  existing `score()` in this repo.
- **Prioritize the Long-Context subcategory first within Phase 2** — it is the
  closest match to what actually differentiates large-context models; the
  other three subcategories mostly test correctness, not context capacity.

**Not recommended for v1:** BFCL's "Live" / real-API-executable categories —
flaky by design (real external services), same reasoning as why this doc
doesn't chase live leaderboard parity elsewhere (see §B).

**Sequencing note:** Phase 1 alone will not show a large-context advantage —
it's still single-shot, same structural limitation as the current suite, just
harder questions. Phase 2's Long-Context subcategory is the one actually worth
building to answer "do large models / large contexts differentiate on complex
tasks?"; Phase 1 is worth doing regardless since it is cheap and fixes the
tool-use blind spot generally.

---

## M4 factorial study — COMPLETE (2026-08-18)

Results in `docs/FINDINGS-qwen-factorial.md` and
`docs/FINDINGS-qwen-evalplus.md`. Host held fixed so model effect and build
effect separate cleanly; the host dimension needs the GPU server and is **not**
covered by this.

**Verdict: every comparison is statistically indistinguishable.** Across 400
pooled items per arm (plus 200 on EvalPlus), all four arms land at 78.0–79.8%:

| comparison | delta | p |
|---|---|---|
| model effect, MLX fixed | +0.5 | 0.932 |
| model effect, GGUF fixed | +0.5 | 0.930 |
| build effect on 3.8 | −1.2 | 0.728 |
| build effect on 3.6 | −1.2 | 0.730 |

This retires three readings the n=20 dashboard appeared to support: that 3.8 is
worse than 3.6, that MLX is "20 points smarter" than GGUF, and that 3.8 is
uniformly 3× faster.

**Speed is the real difference, and it is asymmetric:** MLX is 3.50× faster on
qwen3.8 (31.3 vs 8.9 tok/s) but only 1.10× on qwen3.6 (13.6 vs 12.4). Any
"MLX is faster" claim must name the model.

Why it exists: every comparison available in the dashboard moved several
factors at once (model, quantisation+engine, host), so "3.8 is worse than 3.6"
and "MLX is 20 points better than GGUF" were both being read off data that
could not support them. At n=20 a single sample is 5 points.

**Arms** (all `think:false`, n=100, M4 only):

| arm | build |
|---|---|
| `qwen3.6:27b` | GGUF / Q4_K_M |
| `qwen3.6:27b-mlx` | MLX / nvfp4 |
| `qwen3.8:27b` | GGUF / Q4_K_M |
| `qwen3.8:27b-mlx` | MLX / nvfp4 |

**Phases:** 1+2 humaneval/mbpp/spider/bfcl at n=100 -> 3 BFCL multi-turn at n=5
-> 4 EvalPlus at n=100 -> 5 coverage fill (mmlu/arc/gsm8k/philosophical) for
the two newly pulled builds -> cell-level merge into `merged.summary.json`.

Phase 5 exists because the two new builds would otherwise land in the dashboard
with only 4 of 8 benchmarks, and `overall` averages over whatever is present —
so their rank would not be comparable to any other row.

**Findings land in** `docs/FINDINGS-qwen-factorial.md` and
`docs/FINDINGS-qwen-evalplus.md`, generated by `tools/factorial_effects.py`.

**Early result, already clear:** the build effect on qwen3.6 is ~nil
(317/400 GGUF vs 312/400 MLX). That means the 80% vs 60% bfcl gap between the
M4 MLX and P100 GGUF builds of qwen3.8 was *not* a build effect — it was small
samples and/or host/config differences.

### If the run is interrupted

A host **suspend wedges an in-flight request**: the process stays alive,
blocked, 0% CPU, no model loaded, and no log output — past the 330s read
timeout, which never fires because the socket blocks rather than erroring. It
does not recover. Kill it and resume with `--baseline <newest results json>`;
completed models are checkpointed and carry over.

Checkpoints are per *model*, so the model in flight loses whatever benchmarks
it had already finished (300 samples, twice now). See the promoted item below.

---

## GPU-server full 7-benchmark sweep — done, 2 models excluded (2026-08-18)

The unified-pool GPU server now has full mmlu/arc/gsm8k/humaneval/mbpp/spider/
philosophical coverage (matching the M4's n_samples and judge model) for 7 of
9 configured models: `GLM-4.5-Air`, `gemma4:31b`, `qwen3-coder-next:sovereign-
128k`, `qwen3.6-128k`, `qwen3-coder:30b-sovereign`, `Qwen3.8-27B`, and
`Muse-Glimmer-30B`. Merged into `merged.summary.json` / `report_final.html`
(cell-level merge, so each row keeps its own host's data).

**`llama4:scout` and `deepseek-r1:70b` are deliberately excluded** — both were
already the two models flagged as troublesome (deepseek-r1: 0% bfcl, ~4.5
tok/s, lowest value to gate the sweep behind; llama4:scout: the model that
spans all 6 GPUs at ~94% VRAM, previously correlated with PCIe correctable bus
errors on GPU5's riser). Reordering the sweep to run qwen models first (per
explicit decision, since the two problem models kept blocking the others)
worked for getting the other 7 done, but `llama4:scout` itself was then killed
3 consecutive times — always mid-run (never during model load, even after
pre-warming it with a direct API call first), never for any other model, no
OOM/reboot/contention found in the logs available without sudo (no dmesg
access on this box). Root cause still unconfirmed. `llama4:scout` has partial
coverage (mmlu, arc, bfcl only); `deepseek-r1:70b` was never reached this
session and still only has its earlier bfcl score. Not retried further —
if either model matters enough to chase, it needs sudo/dmesg access to check
PCIe AER counters during a live run, which wasn't available here.

---

## qwen3.8:27b — GPU server vs M4, capability assessment (2026-08-19)

Written on request, comparing the GPU server's GGUF run against the M4's
three variants (GGUF, MLX no-think, MLX +think) on TPS, TTFT, and swap/
spillover risk. All numbers pulled from `merged.summary.json` and, for TTFT,
the GPU server's raw per-sample file (`results/full_gpu_pool_FINAL.json`) —
the M4's raw records aren't committed, only its aggregate summary, so TTFT
comparison is one-sided; noted rather than guessed at.

### Throughput (decode tok/s)

| variant | host | tok/s |
|---|---|---|
| GGUF Q4_K_M | GPU server | 8.09 |
| GGUF Q4_K_M | M4 | 8.53 |
| MLX nvfp4, no-think | M4 | 31.6 |
| MLX nvfp4, +think | M4 | 29.8 |

Raw GGUF throughput is a wash between the two hosts. **MLX is the actual
speed story, and it is M4-only** — ~3.7–3.9x faster than GGUF on the same
model, matching the factorial study's finding that this speedup is specific
to qwen3.8 (not qwen3.6, see the M4 study above). The GPU server has no MLX
path (Apple-only framework); its ceiling on this model is the ~8 tok/s GGUF
number unless a different serving stack were introduced (untested).

### TTFT

GPU server, by category (median / max, n=20-25 except philosophical n=10):

| benchmark | median | max |
|---|---|---|
| mmlu | 3.9s | 119.7s |
| arc | 4.2s | 4.7s |
| gsm8k | 4.1s | 4.8s |
| humaneval | 4.9s | 5.9s |
| mbpp | 3.7s | 4.2s |
| spider | 3.7s | 4.3s |
| philosophical | 254.0s | 639.7s |

Fast and boring everywhere except `philosophical`, which runs with
`think: true` and where the harness marks TTFT at the first *content*
token — so 254-640s there is 12,000-31,000 characters of internal
chain-of-thought before any visible answer, not slow prefill. Worth knowing
so a `philosophical` outlier isn't misread as a hardware problem.

### Swap / spillover risk

| variant | host | aborted categories | sample exposure |
|---|---|---|---|
| GGUF Q4_K_M | GPU server | 0 of 8 | 20-25/category |
| GGUF Q4_K_M | M4 | 2 of 10 (bfcl, humaneval_plus) | 100/category |
| MLX, no-think | M4 | 1 of 10 (mbpp_plus) | 100/category |
| MLX, +think | M4 | 1 of 10 (bfcl) — stale, pre-dates the guard fix | 20/category |

Not apples-to-apples: the M4 study ran 100 samples/category on the harder
benchmarks vs. the GPU server's 20, so it has had ~5x the exposure to
whatever trips this. Per the earlier guard investigation, these are not real
OS-level swap events on the M4 — a 27B@Q4 model is ~15GB, nowhere near
exhausting 48GB — they are genuine sustained throughput collapses (0.4-0.6
tok/s for ~340s) specific to qwen3.8 on certain BFCL/EvalPlus prompts,
mislabeled "swap" by the guard's default `kind`. **This is a model-behavior
risk that would plausibly also show up on the GPU server at 100-sample
exposure** — its clean run so far looks more like "hasn't hit it yet" than
"immune." (The GPU server's own OS-swap corroboration is real and unaffected
by the cross-host sensor fix above, which only concerns the M4-*remote*
case — a genuine GPU-side swap would be correctly detected.)

### The capability gap that matters most: context ceiling

| variant | native ctx | effective ctx |
|---|---|---|
| GPU server (GGUF) | 262144 | **262144** — full offload confirmed |
| M4 (GGUF) | 262144 | 32768 |
| M4 (MLX) | 262144 | 32768 |

The GPU server runs qwen3.8 at its full native context with full GPU-layer
offload (layer-count confirmed, not just "loads without erroring" — see the
bisection methodology at the top of `config-gpu-unified.yaml`). The M4 is
capped at **8x less** usable context, because 48GB unified memory has to
hold weights + KV cache + OS overhead simultaneously, and KV cache at long
context is what blows the budget, not the weights.

### Assessment

- **Speed:** a wash on GGUF (~8 tok/s either host). MLX on the M4 is the
  real lever (~30 tok/s) if throughput is what's being optimized for — the
  GPU server has no equivalent path for this model today.
- **Context:** GPU server wins decisively, 8x the usable context — arguably
  a bigger practical differentiator than the speed numbers for anything
  agentic or long-document.
- **Reliability:** the GPU server's clean record is real but under-tested
  relative to the M4's exposure; not yet provably safer, just less-exercised.
  The failure mode looks like a qwen3.8 quirk that travels with the model,
  not a hardware-fit problem specific to either host.
- **Bottom line:** GPU-server-hosted qwen3.8 for agentic work trades MLX's
  raw speed for 8x the context headroom, at GGUF-level throughput. If
  throughput matters more than context depth, M4+MLX is meaningfully
  faster — but don't expect that speed edge to double as a reliability
  edge; the abort risk looks orthogonal to which host runs it.

---

## Exhaustive GPU-server sweep (n=100) — 3 models done, 2 infra bugs found (2026-08-19/20)

Bumped `humaneval`/`mbpp`/`spider`/`bfcl` to n=100 and added EvalPlus
(`humaneval_plus`/`mbpp_plus`, n=100) for `qwen3.8:27b`, `qwen3.6-128k`, and
`gemma4:31b` on the GPU server — matching the M4 factorial study's depth so
these are now fair, high-confidence comparisons rather than the earlier
n=20 shallow numbers. `muse-glimmer` was in scope too but its results are
**not** merged — see below.

### Bug 1: Spider was scoring string-match only on the GPU server

Every GPU-server model's `spider` score has been wrong since the first
sweep — `data/spider/database/` never existed on this machine, so scoring
silently fell back to normalised string matching instead of real SQL
execution accuracy. Found while investigating a 57-point spider gap
between GPU-server qwen3.8 (15.0) and M4 qwen3.8 (72.0) on what should be
the same benchmark; every other core category matched within noise,
isolating it to this one data gap.

The original download source (a Google Drive zip ID in
`prefetch_datasets.py`) is dead — both `gdown` and a direct curl hit a
404/permission error, not a transient failure. Re-sourced from
[`prem-research/spider`](https://huggingface.co/datasets/prem-research/spider)
on HuggingFace, which mirrors the same `database/<db_id>/<db_id>.sqlite`
layout (169 dbs, full coverage of the 20 validation `db_id`s this
benchmark needs) via `huggingface_hub` — already a dependency, no new pip
package. Re-pointing `prefetch_datasets.py` at this mirror instead of the
dead Google Drive link is still a TODO; for now the data is in place
(`data/spider/database/`, gitignored) and confirmed working.

All 7 already-benchmarked GPU-server models were re-run on spider (n=20)
against the restored databases: scores jumped from 0-15% to 35-75%.
`evalplus/humanevalplus` and `evalplus/mbppplus` were also prefetched
(previously only cached on the M4).

### Bug 2: Muse Glimmer's GPU-server pull is broken — every score was fake

`hf.co/bartowski/Muse-Glimmer-30B-GGUF:Q4_K_M`'s scores on the GPU server
were **0% across mmlu/arc/gsm8k/humaneval/mbpp/bfcl, a fake 10% on spider,
and a fake 67.2% on philosophical** — every single one generated from a
completely empty model response (`completion_tokens: 3`, `response: ""`).
Reproduced live: any prompt to this model, with or without a system
message, at any context length, returns `eval_count: 3` and empty content,
`done_reason: "stop"` — not a timeout, not a swap abort, a clean stop that
happens instantly.

**Root cause:** `ollama show`'s `parameters` for this pull include `stop
"<|start|>"` and `stop "<|message|>"` as bare stop strings. This model
uses a Harmony-style multi-channel output format
(`<|start|>assistant<|channel|>analysis<|message|>...thinking...
<|channel|>final<|message|>...answer...<|eot|>`), where `<|message|>` is a
**structural marker that appears multiple times within one well-formed
response**, not a turn boundary. The moment the model emits its first
channel header, Ollama's stop-matching kills generation — before any
content, thinking or otherwise. Confirmed by overriding the stop list in a
raw `/api/chat` call (`"stop": ["<|start|>user<|message|>", "<|eot|>"]`):
the model then produces a real, coherent `analysis`-channel stream. Ollama
also reports this model "does not support thinking" via `/api/show`
capabilities — almost certainly the same kind of metadata gap already
documented for qwen3.8's missing `tools` capability, not a real
limitation, since the Harmony template clearly has a thinking channel.

**Fix path (not yet applied):** re-create this model via a custom
Modelfile that keeps only genuine turn-boundary stops (e.g.
`<|start|>user<|message|>`, `<|eot|>`) and drops the bare `<|start|>` /
`<|message|>` entries, then re-run its whole benchmark set from scratch —
every existing score for this model, on this machine, is invalid.
Excluded from the dashboard entirely rather than left showing the fake
numbers: `results/merged.summary.json` has an empty `scores` entry for
this model (renders as untested/pending, not as a 0% capability score).

### Bug 3 (harness-side, general): judge doesn't reject empty responses

Found while diagnosing Bug 2's fake philosophical score: `llm_judge()` in
`harness/judge.py` sent Muse Glimmer's empty response straight to the
judge model, which invented a plausible-looking 0.64-0.72 score across ten
questions rather than recognising there was nothing to judge. This isn't
Muse-Glimmer-specific — **any** model producing an empty or failed
response on a judge-scored benchmark would get the same silent score
fabrication instead of a correct near-zero. Fixed: `llm_judge()` now
short-circuits to a `0.0` score (skipping the judge call entirely) when
the response is empty or whitespace-only. `llm_judge_ensemble()` calls
`llm_judge()` per model, so the fix covers both paths with one change.

### Fair GPU-server-vs-M4-MLX comparison, now at matched depth

| model | host | humaneval | mbpp | spider | bfcl | humaneval+ | mbpp+ |
|---|---|---|---|---|---|---|---|
| qwen3.8:27b (GGUF) | GPU server | 95.0 | 68.0 | 74.0 | 75.0 | 90.0 | 61.0 |
| qwen3.6-128k (GGUF) | GPU server | 93.0 | 70.0 | 71.0 | 88.0 | 88.0 | 59.0 |
| gemma4:31b (GGUF) | GPU server | 95.0 | 71.0 | 75.0 | 85.0 | 91.0 | 65.0 |

All n=100, matching the M4 study's depth. `muse-glimmer` excluded — see
Bug 2. `bfcl` numbers reflect Bug 4's fix (below) — the values in this
table are already corrected, not the ones originally reported.

**Follow-up, not done yet:** fix Muse Glimmer's Modelfile and re-run its
full benchmark set; re-point `prefetch_datasets.py` at the working Spider
mirror; fold these into the `overall` ranking (now correctly excluding
EvalPlus/BFCL per the earlier fix) and compare directly against
`qwen3.6:35b-mlx`, `qwen3.8:27b-mlx`, `gemma4:31b-mlx` on the M4 side.

### Bug 4: BFCL's "float"/"tuple" types broke tool-call requests

Flagged by the earlier assessment above: qwen3.8's GPU-server `bfcl` was
55% vs 81% on both M4 builds — a 26-point gap not explained by anything
already fixed. Root cause: `_bfcl_type_to_json_schema_type()` in
`benchmarks/bfcl.py` only converted BFCL's `"dict"`→`"object"` and
`"any"`→`"string"`, leaving `"float"` and `"tuple"` (not valid JSON Schema
types) to pass through unconverted — affecting ~23% of samples across all
4 categories. Array-typed properties' `items` sub-schema wasn't converted
at all, the same gap one level deeper.

**Why only qwen3.8 showed it:** confirmed live that all three models
(qwen3.8, qwen3.6-128k, gemma4:31b) received the identical malformed
schema on the identical sample (`multiple_129`, seed-matched across all
three), but Ollama's qwen3.8/qwen35 backend hard-rejects an unrecognized
schema type with an HTTP 400, while gemma4 and qwen3.6 tolerate it and
attempt an answer anyway — sometimes still wrong in a different way
(qwen3.6-128k passed `annual_rate: "5"` as a string instead of a number
for that same sample, a related but distinct type-coercion issue the
schema fix also resolves).

**Why this was invisible instead of a clear error:** `harness/client.py`'s
`complete_chat` never checked the HTTP status code on the stream response.
Ollama's 400 error body (`{"error": {...}}`, no `message`/`done` keys) was
parsed the same as a normal empty chunk, so the call fell through to
reporting a fabricated `completion_tokens: 1`, `error: None` — an actual
API rejection silently indistinguishable from "the model just didn't
answer." All 28 of qwen3.8's "no tool call attempted" bfcl failures had
`prompt_tokens: 0`, the tell that the request never actually reached the
model. Fixed: a non-2xx response now raises with the real status and body,
surfaced through the existing `error` field.

**Result after both fixes, n=100 re-run:** qwen3.8 55%→75% (0 schema
rejections left, vs 28 before), qwen3.6-128k 82%→88%, gemma4:31b unchanged
at 85% (it was already working around the bug, just less visibly). The
qwen3.8-vs-M4 bfcl gap shrank from 26 points to 6 — likely genuine
model/sampling variance now, not an infra bug.

---

## Next up — do these in this order (2026-08-18)

State at handover: 33 models / 10 benchmarks / 7215 samples across both hosts,
no merge conflicts, dashboard regenerated.

### 1. Fix `overall` before adding any new benchmark — DONE (2026-08-18, GPU server)

`overallOf()` averaged over whichever benchmarks a row happened to have,
which unfairly tanked the 4 models carrying EvalPlus (5–7 points lower by
design on the same problems): qwen3.8:27b landed at 86.3 vs gemma4:31b-mlx's
88.3, despite the two being statistically indistinguishable per this same
study.

**Fixed:** `overall` now averages a fixed `CORE_BENCHES` set (mmlu, arc,
gsm8k, humaneval, mbpp, spider, philosophical) only. BFCL, BFCL-MT, EvalPlus,
and any future optional benchmark (LiveCodeBench) render as their own
columns but never enter the average. qwen3.8:27b now correctly lands at
89.7, ahead of gemma4:31b-mlx's 88.8. A model with partial core coverage
(e.g. `llama4:scout` at mmlu+arc only) now shows an "n/7 core" flag next to
its overall score instead of silently presenting a 2-benchmark average as
equivalent to a 7-benchmark one. Display-only change in
`scoring/benchmark_dashboard.html`, no re-running needed.

**Found while fixing this:** the GPU-server sweep merge (`7c48c75`) had
silently dropped `bfcl` scores for all 8 models it touched. The pre-existing
bfcl-only entries were still under the old host label ("GPU Server (6x P100,
unified pool)"), so `tools/merge_summaries.py --cell-level` read it as a
cross-host conflict and fell back to whole-entry replacement — correctly
recorded in `merge_conflicts` inside the output JSON, but that key was never
printed to console, so nobody saw it (the "no merge conflicts" note above,
from immediately after, was accurate for *that* merge but the data was
already gone from the one before it). All 8 scores restored from the
pre-merge summary; the CLI now prints `merge_conflicts` so a silent
whole-entry replacement can't happen unnoticed again.

### 2. Clear two stale swap flags

`qwen3.8:27b-mlx +think` (bfcl) and `qwen3.6:35b-mlx +think` (mbpp) carry 💀
markers from runs made **before** the two guard fixes below. Minutes to re-run;
they are artifacts, not model behaviour.

Not stale: `qwen3.8:27b` (bfcl, humaneval_plus) and `qwen3.8:27b-mlx`
(mbpp_plus). Those four aborts have a different signature — sustained
0.4–0.6 tok/s across ~340s rather than 0.0 with an absurd ratio — so they are
genuinely slow calls the guard correctly killed. 4 aborts in ~1800 samples.

### 3. BFCL multi-turn — zero coverage

`bfcl_multi_turn_long_context` ran once with the broken guard, **all 5 samples
aborted**, and the results were discarded. No model has it. Needs a deliberate
run now the guard is fixed; at ~450–500s per sample budget it explicitly rather
than folding it into a sweep.

### 4. EvalPlus across the wider fleet, or accept it as an optional column

Only the 4 study arms have it. Either run it fleet-wide (~20 models × 200
samples) or rely on the `overall` fix in item 1 and treat it as supplementary.

### 5. Then LiveCodeBench (below)

---

## Guard false positives — fixed 2026-08-18, worth knowing

Found because qwen3.8 was failing *every* BFCL sample with swap aborts on a
machine with **zero swap activity** (`swapouts +0 pages` over 10s).

Two distinct bugs, both in `harness/`:

1. **Empty trailing window read as collapse.** A finished generation whose
   stream is still open has no tokens in the trailing window, and the ratio
   read that as catastrophic decode collapse — producing
   `12 -> 0.0 tok/s, collapsed 11739695046x`. A window with fewer than two
   tokens is now treated as silence, which `token_stall_seconds` owns. Real
   thrash keeps producing tokens slowly.
2. **Watchdog outlived the generation.** It now disarms on the `done` chunk.
   The guard's job ends when the model stops generating; waiting for the server
   to drain the stream is not thrash.

Effect: `qwen3.8:27b` bfcl went 74/100 (10 spurious aborts) → 81/100 (2).

**Diagnostic recipe** for "is this swap or is it slow": check
`swapouts` delta over ~10s (zero ⇒ not thrashing), then read
`~/.ollama/logs/server.log` for `Prompt processing progress` — steady linear
progress is healthy, however slow it feels. A 85k-token prompt spends ~22
minutes in prefill at ~64 tok/s before emitting a single token, which is
inherent, not a fault. Note a second request to a busy model **queues** and
looks like a hang.

---

## Host identity — declare it, never infer it

`execution.host_label` (or `BENCH_HOST_LABEL`) sets a machine's published
identity. Added because IP-based inference broke twice:

- The M4 changed network (192.168.68.106 → 172.30.185.105), so every row it
  produced landed as "unregistered host" and the cross-host merge guard
  correctly refused to combine them with the same machine's earlier rows —
  silently dropping mmlu/arc/gsm8k/philosophical from two models.
- The GPU sweep arrived labelled "GPU Server (6x P100, unified pool)" while
  this side had renamed it "i9 GPU server", splitting one machine across two
  names and generating eight bogus conflicts.

**TODO on the GPU server:** add `execution.host_label: "i9 GPU server"` to
`config-gpu-unified.yaml`. Until it declares the label, the next sweep
reintroduces the old name.

Labels are deliberately generic — they are published in a public repo.
`platform.node()` is no longer recorded or used as a fallback anywhere.

---

## LiveCodeBench — unblocked, implementation remaining

Wanted because it is widely published (so numbers are externally recognisable)
and contamination-resistant via time-versioned releases, unlike HumanEval/MBPP.

**The blocker is solved.** `load_dataset("livecodebench/code_generation_lite")`
fails — it is a script-based dataset and modern `datasets` refuses those. But
the repo also ships plain JSONL, so bypass the script entirely:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("livecodebench/code_generation_lite", "test.jsonl",
                    repo_type="dataset")     # test.jsonl .. test6.jsonl = v1..v6
```

Verified: `test.jsonl` is **400 problems, ~1.2 GB**. Fields:
`question_id, question_title, question_content, starter_code, platform,
difficulty, contest_date, public_test_cases, private_test_cases, metadata`.

**What still needs building** — this is more than a `load_samples()`/`score()`
pair, which is why it was not bolted on mid-study:

1. **Test-case decoding.** `public_test_cases` is a JSON *string*;
   `private_test_cases` is base64 + zlib (`eJx` prefix). Both need decoding
   before use.
2. **stdin/stdout execution.** Most problems are competitive-programming style:
   feed `input`, capture stdout, compare exactly. `harness/sandbox.py`'s
   `execute_python(code, test_code)` only appends assertions — it has no stdin
   path. Needs a sibling runner.
3. **Two problem shapes.** Codeforces-style (stdin/stdout) vs LeetCode-style
   (`starter_code`, call a method). Detect via `starter_code` being non-empty.
4. **Per-problem time limits**, and a much lower default than 10s is unwise —
   these are algorithmic problems, not one-liners.
5. **Release selection.** Pick a version deliberately (`test.jsonl` = v1) and
   record it, since the whole point is time-versioning against contamination.

Disk note: the M4 already has the 1.2 GB v1 file in the HF cache.

---

## Dashboard file names — consolidate (needs both machines idle)

There are two dashboard files doing two different jobs under names that say
neither, and it has already caused a silently stale bookmark.

| file | role | tracked |
|---|---|---|
| `results/report.html` | whatever *this machine* last generated or ran | ignored |
| `results/report_final.html` | the shared, merged, cross-machine view | tracked |

The split was accidental: `report.html` was the original and only name, written
automatically by every run. `report_final.html` began life as a stable output
name for a multi-pass chain script, which then `cp`'d it over `report.html`;
the GPU server reasonably adopted it as the committed artifact since it is the
one that survives a pull.

The consequence is that `report.html` — the natural thing to bookmark — is
gitignored, so a `git pull` updates `report_final.html` while the bookmark
silently keeps showing an older, single-machine view.

**Proposed:**

- `results/dashboard.html` — tracked, merged, the only file to bookmark.
- `results/report_<run_id>.html` — every live run writes here, *unconditionally*
  (today it only diverts when `--baseline` is absent).
- Retire `report.html` and `report_final.html`.

**Do not "fix" this by symlinking `report.html` → `report_final.html`.**
`run_benchmark.py` writes `report.html` directly during a run whenever
`--baseline` is used; through a symlink that write follows the link and
silently overwrites the committed merged dashboard with a mid-run,
single-machine view. It looks fine until it is pushed.

Coordination: this renames files the GPU server also writes and commits, so it
has to land on both sides together — the rename, the `run_benchmark.py` write
path, the `.gitignore` rules, and the `report_final.html` references in
`tools/merge_summaries.py` and `tools/export_model_scoreboard.py`. Deferred
until neither machine is mid-sweep, otherwise the next
`generate_report.py --output results/report_final.html` on the other side
recreates the old name.

---

## Smaller open items

- **Per-benchmark checkpointing — promoted.** Checkpoints are written after each
  *model*, so a run killed mid-model loses every benchmark that model had
  already finished. This has now cost 300 samples twice (both times a host
  suspend). It is a small change — write the checkpoint inside the benchmark
  loop rather than after it — and it is the highest-value item in this list.
- `--benchmarks` accepts registry keys (`spider`) but `config.yaml` uses group
  names (`sql`). The README's own incremental-run examples use `--benchmarks sql`,
  which argparse rejects. Accept both vocabularies.
- EvalPlus datasets (`evalplus/humanevalplus`, `evalplus/mbppplus`) must be
  fetched online once before any run, because runs force `HF_DATASETS_OFFLINE=1`
  when `.datasets_ready` exists. Cached on the M4; not yet on the GPU server.
- A host suspend can wedge an in-flight streaming request indefinitely — the
  read blocks rather than timing out, so neither the 330s read timeout nor the
  guard's stall detector fires. Worth a wall-clock ceiling per sample as a
  last-resort backstop, distinct from the guard's adaptive signals.
- Generated dashboards and `merged.summary.json` conflict in git when two
  machines regenerate them from the same base — confirmed by simulation, and
  guaranteed rather than unlucky because the models array is score-sorted, so
  inserting one model rewrites the file (adding two rows changed 214 lines).
  Resolve by regenerating, never by hand-merging: take either side, re-run
  `tools/merge_summaries.py` over the per-machine summaries, then
  `generate_report.py --from-summary`. The per-run summaries themselves have
  distinct filenames and merge cleanly as pure additions.
- `Baseline: N results loaded` prints `len()` of the wrapped
  `{metadata, results}` dict, so it always reports 2. Cosmetic but confusing.
- Checkpoints are written after each *model*, so a model killed mid-run loses
  its completed benchmarks. Per-benchmark checkpointing would be cheap.
- VRAM estimates are omitted when model metadata lacks attention shape (both
  MLX conversions). Could be derived from the MLX config instead.
- The philosophical judge is a single local model (`llama3.1:8b`). An ensemble
  would reduce single-judge bias; the code path already exists.

---

## §E addendum — BFCL moved to v4, and `irrelevance` added (2026-08-20)

Two corrections to §E as written, both from checking the data rather than the
version numbers.

**1. `bfcl-eval` is no longer a heavy dependency.** §E rejected it because it
"pulls in sglang/vllm/cuda-bindings as hard dependencies, several GB". At
2026.3.23 the wheel is 1.9 MB and `requires_dist` is `requests, tqdm, numpy,
pandas, huggingface_hub, pydantic, python-dotenv, tree_sitter{,-java,-javascript},
openai, mistralai, anthropic, cohere` — no CUDA, no serving stack. Still four
vendor SDKs this harness never calls, so the data is extracted from the wheel
by `prefetch_datasets.download_bfcl_v4()` rather than taken as a runtime
dependency; the local AST checker in `benchmarks/bfcl.py` stays.

**2. v4 is not on HuggingFace.** `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
carries `BFCL_v3_*` and nothing else. The v4 dataset ships only inside the
`bfcl-eval` wheel. "Switch to v4" is therefore a data-sourcing change, not a
filename change.

### The v3 -> v4 move cost nothing

ID-aligned diff of every field across the four non-live AST categories
(1000 items):

| category | items | question diffs | schema diffs | ground-truth diffs |
|---|---|---|---|---|
| simple | 400 | 0 | 0 | 1 |
| multiple | 200 | 0 | 0 | 0 |
| parallel | 200 | 2 | 0 | 2 |
| parallel_multiple | 200 | 0 | 0 | 1 |

v4's non-live AST set is v3 with four corrected ground truths, all genuine
bugs: `simple_363` omitted the `restaurant_search.` namespace, so a model
calling the function correctly scored wrong; `parallel_multiple_141` expected
the month `"Febuary"`.

None of the four corrected items fall inside the seed-42 n=100 draw, so **every
`bfcl` score already in `results/` stands** — verified two ways: the v4 draw is
byte-identical to the stored v3 draw including order, and re-running needle2
under v4 reproduced 41.0% exactly.

v4 renamed the Python AST set `simple` -> `simple_python`. The category name is
kept as `simple` and IDs are canonicalised back (`simple_python_7` ->
`simple_7`) so stratify buckets, the seeded selection, and per-sample record
IDs all stay comparable with the existing runs. `_load_category` now also sorts
by numeric ID: both v3 and v4 ship ascending, but `select_samples` shuffles
each stratum in arrival order, so an upstream reordering would silently change
which items every model is scored on.

### New: `bfcl_irrelevance`

240 non-live requests that no available function can satisfy; a pass means the
model called **nothing**. This closes a real blind spot — nothing else in the
suite penalises over-calling, so a model that fires a tool at every prompt
scored identically to one with judgment.

Kept as its own benchmark rather than a fifth category in `bfcl`, because
folding it in would redefine a metric already published for seven models at
n=100.

The system prompt spells out the opt-out explicitly ("if none of the available
functions can satisfy the request, do not call any function"). Without that
line the item is a trick question and the score measures prompt wording rather
than judgment.

**Still not v4-comparable.** The published v4 leaderboard figure is a weighted
aggregate over every category including live and web_search. The honest label
for this harness remains "BFCL v4 non-live AST" (+ irrelevance, reported
separately) — not "BFCL v4".

### Remaining v4 categories, ranked

- `simple_java` / `simple_javascript` — cheap items; AST-comparing non-Python
  literals is what `tree_sitter` is for. Medium value given Python coding is
  covered elsewhere.
- `multi_turn_base` / `miss_func` / `miss_param` — a `BENCHMARK_REGISTRY` entry
  each on the engine `bfcl_multi_turn.py` already implements. Code cost is near
  zero; the blocker is the measured ~450-500s per sample.
- `live_*`, `web_search` — skip. Real external services break the
  temperature-0.0 determinism the rest of the suite is built on. §E's original
  reasoning holds harder for v4.
- `memory`, `format_sensitivity` — new in v4, no track record. Defer.
