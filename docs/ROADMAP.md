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

## Smaller open items

- `--benchmarks` accepts registry keys (`spider`) but `config.yaml` uses group
  names (`sql`). The README's own incremental-run examples use `--benchmarks sql`,
  which argparse rejects. Accept both vocabularies.
- `Baseline: N results loaded` prints `len()` of the wrapped
  `{metadata, results}` dict, so it always reports 2. Cosmetic but confusing.
- Checkpoints are written after each *model*, so a model killed mid-run loses
  its completed benchmarks. Per-benchmark checkpointing would be cheap.
- VRAM estimates are omitted when model metadata lacks attention shape (both
  MLX conversions). Could be derived from the MLX config instead.
- The philosophical judge is a single local model (`llama3.1:8b`). An ensemble
  would reduce single-judge bias; the code path already exists.
