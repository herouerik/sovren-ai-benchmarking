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
