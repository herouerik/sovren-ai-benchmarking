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

## In flight on the M4 (2026-08-17) — handover state

A qwen3.6 vs qwen3.8 x MLX vs GGUF study is running unattended on the M4. Host
is held fixed so model effect and build effect separate cleanly; the host
dimension needs the GPU server and is **not** covered by this.

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
