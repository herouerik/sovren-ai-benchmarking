<img width="1509" height="771" alt="image" src="https://github.com/user-attachments/assets/011f3b95-7d8c-4b19-8e30-8103d96257ca" />


# sovren-ai-benchmarking

A self-contained benchmarking harness for local LLMs served via e.g. [Ollama](https://ollama.com) (other servers work too). Runs a suite of standard and custom benchmarks against any model available at your sovereign LLM serving endpoint, scores them, and produces a comparative summary across models.

Built to work fully offline once datasets are cached.

---

## Structure

```
sovren-ai-benchmark/
├── run_benchmark.py        ← single entry point for everything
├── prefetch_datasets.py    ← one-time dataset download/caching
├── config.example.yaml     ← copy to config.yaml; what to run, against which models
├── benchmarks/             ← one file per category
├── harness/                ← shared infrastructure
├── scoring/                ← result display and analysis
├── tools/                  ← standalone scripts (context-perf probe, external reference,
│                              multi-machine summary merging)
├── data/                   ← dataset definitions and cached third-party reference figures
└── results/                ← JSON output from each run
```

---

## Benchmark categories

### 1. MMLU — Massive Multitask Language Understanding (reasoning)
Multiple choice questions across 57 academic subjects. Config selects a subset of subjects; defaults cover logic, algebra, philosophy, mathematics, and fallacies. The model picks A/B/C/D and is scored by exact match.

**Tells you:** how broadly knowledgeable is this model across academic domains?

**Source:** [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) on HuggingFace — Hendrycks et al., 2020. [Paper](https://arxiv.org/abs/2009.03300).

---

### 2. ARC — AI2 Reasoning Challenge (reasoning)
Harder science multiple choice. Same format as MMLU. Uses the `ARC-Challenge` split which filters for questions that simple retrieval methods fail on.

**Tells you:** can the model reason through multi-step factual problems?

**Source:** [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) on HuggingFace — Clark et al., 2018. [Paper](https://arxiv.org/abs/1803.05457).

---

### 3. GSM8K — Grade School Math (problem solving)
1319 grade-school arithmetic word problems. The model must show its work and end its response with `#### <number>`. A regex extractor pulls the final number and compares it to the ground truth.

**Tells you:** can the model follow a chain of arithmetic reasoning to a correct conclusion?

**Source:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) on HuggingFace — Cobbe et al., 2021. [Paper](https://arxiv.org/abs/2110.14168).

---

### 4. HumanEval + MBPP (Mostly Basic Python Problems) — Python coding
Two standard coding benchmarks. The model generates a Python function; the harness writes it to a temp file and executes it in a subprocess against bundled unit tests. `pass@1` only — one attempt, pass or fail, no partial credit. Never uses `exec()`.

- **HumanEval** — 164 hand-written Python function stubs from OpenAI. Problems are well-specified and curated; this is the cleaner end of the coding evaluation spectrum. [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) — Chen et al., 2021. [Paper](https://arxiv.org/abs/2107.03374).
- **MBPP** — ~400 crowd-sourced Python problems, `sanitized` split. Broader and more varied in spec quality than HumanEval, drawn from a wider range of contributors — which makes them harder to overfit to, and complements HumanEval by probing the noisier, real-world end of the coding distribution. [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) — Austin et al., 2021. [Paper](https://arxiv.org/abs/2108.07732).

**Tells you:** does the code actually run and pass tests?

**`humaneval_plus` / `mbpp_plus`** — the same problems and the same prompts (the model is never told it is being graded more strictly), but scored against [EvalPlus](https://github.com/evalplus/evalplus)'s test suites, extended roughly 80x over the originals (Liu et al., NeurIPS 2023. [Paper](https://arxiv.org/abs/2305.01210)). The plain suites accept solutions that are wrong on edge cases — empty inputs, boundary values, type surprises — so scores run high and compress at the top; these are a drop-in harder replacement rather than a bespoke filter, and stay directly comparable to a widely published leaderboard. Because the prompt is unchanged, the gap between a model's `humaneval`/`mbpp` score and its `_plus` score is a clean read on its edge-case failure rate. `mbpp_plus` tends to compress scores into a narrower band than `humaneval_plus`, so read it as a capability ceiling this whole class of model runs into rather than a ranking — `humaneval_plus` is the more discriminating of the two coding columns.

---

### 5. Spider — SQL generation
Natural language questions mapped to SQL queries over real multi-table relational schemas. If the Spider SQLite database files are present locally, the harness executes both the predicted and ground-truth SQL and compares result sets (execution accuracy). Otherwise it falls back to normalised string match.

**Tells you:** can the model translate natural language intent into correct, executable SQL?

**Source:** [xlangai/spider](https://huggingface.co/datasets/xlangai/spider) on HuggingFace — Yu et al., 2018. [Paper](https://arxiv.org/abs/1809.08887).

**Getting the SQLite databases (recommended):** The HuggingFace dataset only ships the questions and reference SQL — not the actual database files. Without them, scoring falls back to normalised string match, which is too strict and causes most correct-but-differently-phrased queries to score zero. For proper execution scoring, download the Spider v1.0 zip manually and pass it to `prefetch_datasets.py`:

```bash
# 1. Download Spider v1.0 from https://yale-lily.github.io/spider
#    ("Download Spider v1.0" button → Google Drive zip, ~100 MB)

# 2. Point prefetch_datasets.py at the downloaded zip
python prefetch_datasets.py --spider-zip ~/Downloads/spider.zip
```

This extracts the 20 validation databases to `data/spider/database/` and all subsequent runs use execution-based scoring automatically. The database files are gitignored.

---

### 6. Philosophical discussion (LLM-as-judge)
Ten curated open-ended questions: free will, justice, moral realism, suffering, technology & freedom, lying, obligations to future generations, machine understanding, meaning through suffering, and knowledge & power. No ground truth exists — the rubric is the signal. A judge model scores each response 1–5 on five rubric axes: depth of reasoning, coherence, acknowledgment of multiple perspectives, originality of insight, and clarity of expression. The mean judge score, divided by 5.0, becomes the benchmark score.

**Tells you:** how well does the model reason through open-ended, ambiguous problems with no single correct answer?

**Method:** LLM-as-judge is a widely used evaluation pattern for open-ended generation. See [Zheng et al., 2023 — MT-Bench](https://arxiv.org/abs/2306.05685) for the canonical reference. The prompts and rubric in this repo are original.

The judge is configured via `judge.provider` in `config.yaml` — see the [Judge configuration](#judge-configuration) section for available backends.

---

### 7. BFCL (Berkeley Function-Calling Leaderboard) — function calling / tool use

**`bfcl`** — the non-live, single-turn AST categories: `simple` (one function,
no decoys), `multiple` (right function among distractors), `parallel`
(several calls to the same function in one turn), and `parallel_multiple`
(several calls across different functions). The model gets a request plus a
set of function schemas and must emit the right call(s). Scored by BFCL's
AST-match rule — function name plus every ground-truth parameter drawn from
an acceptable-value set, not exact string match — reimplemented locally in
`benchmarks/bfcl.py`.

**`bfcl_irrelevance`** — the inverse. 240 requests that *no* available function
can satisfy; a pass means the model called nothing and answered in plain text
instead. Nothing else in this suite penalises over-calling, so without this
category a model that fires a tool at every prompt scores identically to one
with judgment — tool selection and tool restraint turn out to be separable
skills that rank models differently. The system prompt states the opt-out
explicitly, so this measures judgment rather than whether the model guessed
declining was allowed. The characteristic failure is topical adjacency —
reaching for a distance calculator when asked how long a drive takes, or a
date lookup when asked a casualty count — a tool that matches the subject but
cannot answer the question, which is exactly what a schema check passes and a
result check catches.

**Tells you:** can this model be trusted inside an agent loop — both to call
the right tool, and to decline when no tool fits?

**Source:** BFCL **v4**, from the [`bfcl-eval`](https://pypi.org/project/bfcl-eval/)
wheel. The HuggingFace dataset repo still carries only the v3 files, so
`prefetch_datasets.py` extracts the v4 JSONs from the wheel into
`data/bfcl_v4/` rather than taking `bfcl-eval` as a runtime dependency.
[Gorilla / Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).

> **Not leaderboard-comparable.** The published BFCL v4 figure is a weighted
> aggregate over every category, including the live and web-search sets this
> harness deliberately skips (real external services would break the
> temperature-0.0 determinism everything else here relies on). The honest label
> for these numbers is "BFCL v4 non-live AST", with irrelevance reported
> separately.

Multi-turn/agentic categories are a separate benchmark
(`bfcl_multi_turn_long_context`), disabled by default — one sample runs 4 turns
and ~28 tool calls, measured at ~450–500s. It runs a multi-turn tool-call loop
against a live, stateful simulated environment (file system, trading bot,
vehicle controls, etc.), with extra state padding injected specifically to
stress tracking across a long dialogue history: the model calls a function,
gets a real result back, and continues for several turns. Scored by comparing
the environment's final state after the model's run against the same
environment driven by the ground-truth call sequence — not by inspecting
individual calls. This is the benchmark most likely to actually separate
large-context models from each other; everything else here is closer to
single-shot.

---

### 8. Speed — latency and throughput probes

Four scripted probes, run `n_runs` times each with the median reported to reduce
cold-start noise. Unlike the benchmarks above, there is no ground truth — this
measures how fast, not how correct.

| Probe | Isolates |
|---|---|
| `ttft_baseline` | Time-to-first-token, via a short prompt with a 1-word answer |
| `decode_throughput` | Decode tok/s, via a short prompt forced into a long output |
| `prefill_speed` | Prefill/encode speed, via a long (~350-token) prompt with a short answer |
| `realistic_task` | A medium prompt + medium output, typical of an agentic code call |

**Tells you:** is this model fast enough to be worth its accuracy, independent of
the per-benchmark timings already recorded on every other run?

---

## How a run works

```
run_benchmark.py
  └─ loads config.yaml
  └─ for each model (variant):
       └─ for each benchmark:
            └─ load_samples()    — dataset from HuggingFace (cached after first pull)
            └─ select_samples()  — seeded, shuffled, stratified across subjects
            └─ for each sample:
                 └─ complete_native() — streaming POST to /api/chat, watched by
                                        the guard, which can abort it mid-flight
                 └─ score() — exact match / code execution / LLM judge
            └─ print live pass rate
       └─ unload the model, then checkpoint results to disk
  └─ save results/<timestamp>.json
  └─ regenerate results/report.html
  └─ print Rich summary tables
```

Temperature is `0.0` for all deterministic benchmarks (MCQ, math, coding, SQL) so runs are reproducible — verified identical across repeated runs, down to per-sample token counts. The philosophical judge also runs at `0.0` — subjectivity is in the rubric, not the sampling.

Benchmark inference uses Ollama's **native `/api/chat`** over plain HTTP. The
OpenAI-compatible `/v1` endpoint is not used for it, because `/v1` silently
ignores the `think` parameter — and whether a model reasons before answering
changes its scores substantially, so it has to be explicit rather than an
implicit default that shifts with the Ollama version. Native also reports
authoritative prefill and decode timings, and supports `keep_alive` for
deterministic unloading between models. The OpenAI SDK is still used for cloud
judges.

Each result records what it was measured under — `think`, `ttft`, `decode_tps`,
token counts, `ollama_version`, git SHA, and hardware — so runs from different
setups are never silently compared.

---

## Memory swap detection

When a model is too large for the available RAM (or KV cache fills available memory during a long run), inference slows dramatically as the OS pages memory to disk — **swap thrashing**. To prevent a single swapping model from wasting hours, the harness detects swap and aborts the remaining benchmarks for that model.

The guard aborts **mid-call**, not after. A thrashing generation runs effectively
forever, so classifying it once it returns saves nothing — the stream is closed
as soon as the pattern is recognised.

### Detection signals

Detection is **timing-primary**: inter-token intervals inside a single
generation. That is the only signal that works everywhere, because the
underlying fault differs by platform — unified-memory swap on Apple Silicon,
VRAM exhaustion and CPU offload on a Linux GPU box, host swap on a CPU-only
machine. All three show up as decode falling off a cliff; only some appear in
swap counters.

| # | Phase | Condition | Meaning |
|---|---|---|---|
| 1 | **Prefill** | Sustained swap-out above `swap_bytes_per_sec` for `swap_sustain_seconds` | The machine is thrashing before generation even starts |
| 2 | **Prefill** | No token at all within `ttft_ceiling_seconds` | Backstop only — a long TTFT is legitimate (cold load, extended thinking) |
| 3 | **Decode** | No token for `token_stall_seconds` | Absolute silence no healthy generation exhibits |
| 4 | **Decode** | Median of the last `window_tokens` gaps exceeds `decode_degradation_factor ×` the median of the first `baseline_tokens` | The rate this call itself established has collapsed — the classic swap pattern |
| 5 | **Decode** | Sustained rate below `min_decode_tps` | *Not* thrash: the model runs fine, just too slowly to be worth benchmarking here |

Signals 3–5 count **thinking tokens as liveness**. Chain-of-thought streams in a
separate channel from the answer, so a long thinking phase would otherwise look
like a stalled call.

A trip is graded by the OS sensor. **Hard** (pressure corroborated) means the
model does not fit — its remaining benchmarks are skipped. **Soft** (timing
alone) costs one sample and the run continues. Signal 5 reports as `too_slow`
rather than `swap_abort`, so "too slow to bother" is never mistaken for a
memory fault.

### Configuration

```yaml
execution:
  memory_guard:
    enabled: true                 # false disables detection entirely
    poll_interval: 1.0
    ttft_ceiling_seconds: 300     # backstop; cold load and thinking are legitimate
    token_stall_seconds: 20
    decode_degradation_factor: 10 # collapse vs this call's own established rate
    baseline_tokens: 16
    window_tokens: 8
    swap_bytes_per_sec: 33554432  # 32 MB/s sustained swap-out = real thrash
    swap_sustain_seconds: 5
    min_decode_tps: 0.0           # 0 disables the throughput floor
    min_decode_tps_after_tokens: 32
```

### Portability

| | macOS | Linux | Other |
|---|---|---|---|
| Swap counters | `vm_stat` (Swapins/Swapouts) | `/proc/vmstat` (`pswpin`/`pswpout`) | degrades to timing-only |

Adding a platform is one `PressureSensor` subclass in `harness/pressure.py`.
Nothing else changes, and an unknown platform still gets full timing detection.

### GPU vs CPU

The old guard needed different absolute thresholds per machine
(`max_baseline_seconds` of 30 on GPU, 300 on CPU) because it timed whole
samples, so prompt length and answer length polluted the signal. Signal 4 is
relative to the rate each call establishes for itself, so **the same config
works on both**. A 7B model steady at 2 tok/s on a CPU laptop never trips it;
a swap event does. Median-of-window comparison also absorbs the 2–5× per-sample
jitter that CPU scheduling produces on its own.

The one genuinely machine-specific setting is `min_decode_tps`, and that is by
design — it encodes what *you* consider too slow to be worth measuring, not
what is broken. Leave it at 0 on a CPU box, where slow is expected rather than
a fault.

### Per-model context limits

Reducing `num_ctx` (the KV cache length) reduces peak memory usage and speeds up prompt processing — especially valuable for CPU inference where memory bandwidth is the bottleneck. Set per-model in the `models` list:

```yaml
models:
  - model: qwen2.5:7b
    ctx: 16384          # Half the default 32K — saves ~2 GB KV cache
  - model: deepseek-r1:7b
    ctx: 8192           # Thinking models benefit most (less prompt to reprocess)
```

Models listed as plain strings (without a `model:` key) use `ollama.default_ctx`. The global default is set via `ollama.default_ctx` in `config.yaml`.

---

## Thinking mode

Models that declare a `thinking` capability reason before answering, and recent
Ollama enables it by default. It changes results substantially in both
directions — it can turn a wrong answer right, or burn the entire token budget
before the answer starts — so it is an explicit, recorded dimension here rather
than an implicit default.

By default only the philosophical benchmark thinks (extended reasoning is the
point there); everything else runs with it off. Override per benchmark with
`think:` in its config block, or per model:

```yaml
models:
  - model: muse-glimmer:30b-mlx     # thinking off
    ctx: 32768
    think: false
  - model: muse-glimmer:30b-mlx     # same model, thinking on
    ctx: 32768
    think: true                     # → dashboard row "muse-glimmer:30b-mlx +think"
```

The two appear as **separate dashboard rows**, so the delta reads straight off
the table. Set `label:` to name a variant yourself. Without distinct labels the
second run would silently replace the first, since results merge on
`(model, benchmark)`.

Thinking uses `ollama.think_max_tokens` (default 16384) instead of
`max_tokens`, because chain-of-thought is spent before the answer begins — at a
2048 budget, thinking models fail by exhaustion rather than incapability. Each
result stores `think` and `reasoning_chars`, so a thinking run is never
mistaken for a non-thinking one.

---

## Two measurements the benchmark grid does not cover

Both ship as first-class parts of the harness — a tool, a data file, a generator flag and a
dashboard panel each. Neither is merged into the benchmark grid, because neither is the
same kind of number.

### At-context throughput — `tools/measure_context_perf.py`

The `speeds` column is decode tok/s on benchmark prompts, which are short. Agentic use
runs at a much larger context, where both prefill and decode are slower — by factors that
differ per architecture, so a short-prompt ranking need not hold. Prefill matters as much
as decode: a harness that compacts invalidates the KV prefix, so every compaction pays a
full-window prefill rather than an incremental one.

```sh
python3 tools/measure_context_perf.py --config config.yaml \
    --context-tokens 9000 --host "<host label>" --out data/context_perf.json
```

`data/context_perf.json` is gitignored — a per-machine measurement, same category as
`config.yaml`. Picked up automatically, or pass `--perf <path>`. Renders as the
**AT-CONTEXT THROUGHPUT** sidebar panel.

Read the module docstring before changing defaults; it records the three traps the
implementation guards against (a resident model ignores a changed `num_ctx`; thinking
models spend the output budget on reasoning; too short a generation measures startup
rather than decode).

### External reference figures

Published third-party scores, carried for calibration only and rendered in their own tab —
excluded from `OVERALL`, from ranking, and from every model row.

| file | written by | contains |
|---|---|---|
| `data/external_reference.yaml` | hand | which benchmarks to carry, labels, aliases, `column` flags, sources |
| `data/external_scores.yaml` | the refresher | figures, providers, licences, provenance, retrieval date |

```sh
python3 tools/fetch_external_reference.py --dry-run   # show what would change
python3 tools/fetch_external_reference.py             # rewrite the scores file only
```

The refresher rewrites the scores file wholesale and never opens the definitions file, so
the refresh is code-only and cannot destroy hand-written prose. Committed rather than
fetched at report time: a run must not depend on the network, and a published number that
moves should move in a reviewable diff.

**What's tracked today.** `data/external_reference.yaml` currently declares SWE-bench
Verified, SWE-bench Pro, SWE-bench Multilingual, LiveCodeBench, and Terminal-Bench 2.0.
Coverage is intentionally partial: SWE-bench Verified's public leaderboard is
frontier-hosted models only, so it reads as a reference *band* rather than a
like-for-like comparison; LiveCodeBench carries open-weight rows, so it's the one place
the external and local lists genuinely overlap and can estimate a quantisation penalty
on your own hardware. Adding another published benchmark (e.g. `osWorldVerified`,
`browseComp`, `arcAgi2`, `hle`) is a data-only change — a new block in that file, no code.

**Two source types.** `slug` reads a leaderboard mirror; `sources.hf_cards` reads Hugging
Face model cards. Cards are the source that covers open-weight models, since leaderboard
views tend to list only the largest hosted ones. Inspect a card before adding it:

```sh
python3 tools/hf_model_cards.py <org>/<model> --all   # every benchmark label on the card
python3 tools/hf_model_cards.py <org>/<model>         # only ones with declared aliases
```

Card parsing is shape-based rather than markup-based, because card formats vary widely
between publishers: every table is reduced to text, then orientation is decided by testing
which axis matches a declared benchmark alias. **`aliases` is therefore load-bearing — a
benchmark with none declared is invisible to the card harvester** even when a card reports
it.

**Provenance.** Each score is tagged `SELF` (the card's own model), `3RD` (reported on
another vendor's card) or `EXT` (leaderboard mirror). Self-reported supersedes third-party
for the same model. None of the three is independent verification; the tag records whose
claim it is. Rows are deduplicated by model family, since publishers spell the same weights
differently.

**Matching** is family-level, never tag-level, and rejects a containment match whose
leftover is only a tier or size marker (`plus`, `max`, `<n>b`, …) — a generic local tag can
otherwise inherit a much larger product's score. Benchmark variants (`Verified`, `Pro`,
`Multilingual`) are separate keys and must not be read as each other.

### Where the external figures appear

**Its own tab.** The tables are long enough to fill a small screen by themselves, so they
sit behind an `EXTERNAL REFERENCE` tab beside `LOCAL BENCHMARKS`. The tab hides itself when
no external data is loaded.

**Optionally as an inline column.** Set `column: true` on a benchmark and it also renders
in the main table, immediately left of `OVERALL` — adjacent for comparison, outside the
measured block, and drawn unmistakably differently: no heatmap fill, dashed rules, italic
figure and a provenance tag. It sorts like any other column; the default sort stays
`OVERALL`. Where several are flagged, the first declared sits closest to `OVERALL`.

A flagged column renders **only if at least one benchmarked model matches** a row in that
benchmark, so a benchmark can be flagged now and appear later when a model you run enters
its published list.

## Usage

```bash
# Set up (first time only)
cd sovren-ai-benchmark
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download benchmark datasets once (cached locally, no HF requests after this)
python prefetch_datasets.py

# Optional: add Spider SQLite databases for execution-based SQL scoring
# Download Spider v1.0 from https://yale-lily.github.io/spider then:
python prefetch_datasets.py --spider-zip ~/Downloads/spider.zip

# See available local models
python run_benchmark.py --list-models

# Quick sanity check (3 samples, fast)
python run_benchmark.py --models llama3.2:3b --n-samples 3

# Compare two coding models head to head
python run_benchmark.py --models devstral-small-2 qwen3-coder --benchmarks humaneval mbpp --n-samples 50

# Philosophical evaluation
python run_benchmark.py --models gemma4:31b-mlx qwen3:32b --benchmarks philosophical

# Full run — all models and benchmarks from config.yaml
python run_benchmark.py
```

Edit `config.yaml` to change which models are included, how many samples per benchmark, and which judge model to use.

> **Judge configuration:** The philosophical benchmark uses an LLM-as-judge to
> score open-ended responses. You can choose between four judge backends by
> setting `judge.provider` in `config.yaml`. All config keys live at the same
> level under `judge`; only the ones relevant to the selected provider are read.
>
> | `provider` | Config key | Behaviour | Party |
> |---|---|---|---|
> | `opencode` | `cloud_model` | Cloud judge via the opencode CLI — free, auth-free, no API key | Cloud-friendly |
> | `openai` | `cloud_model` | Any OpenAI-compatible API (DeepSeek, OpenAI, etc.) using `base_url` and `api_key` | Cloud-friendly |
> | `ollama` | `ollama_single_model` | Single local Ollama model | Offline |
> | `ensemble` | `ensemble_models` | Multiple local models; each judges independently and scores are averaged | Offline |
>
> **Important:** The judge provider is explicit — there is no automatic fallback.
> If the cloud is unreachable the run fails; if you want offline operation,
> flip `provider` to `"ollama"` or `"ensemble"`. This avoids silent score drift
> between runs using different judges.
>
> Example configs (see `config.example.yaml` for all options):
>
> ```yaml
> # Cloud (default for this machine):
> judge:
>   provider: "opencode"
>   cloud_model: "opencode/deepseek-v4-flash-free"
>
> # Local single model:
> judge:
>   provider: "ollama"
>   ollama_single_model: "llama3.1:8b"
>
> # Local ensemble (averages 3 judges):
> judge:
>   provider: "ensemble"
>   ensemble_models:
>     - qwen3:8b
>     - deepseek-r1:7b
>     - llama3.1:8b
>
> # Generic OpenAI-compatible API:
> judge:
>   provider: "openai"
>   cloud_model: "deepseek-chat"
>   base_url: "https://api.deepseek.com/v1"
>   api_key: "${DEEPSEEK_API_KEY}"
> ```
>
> `config.yaml` is gitignored so you can keep local settings private.

---

## Incremental runs — patching into an existing baseline

A full run across 15 models × 6 benchmarks can take many hours. If a model, benchmark, or an entire batch fails partway through, you don't need to repeat everything. The `--baseline` flag loads a prior results JSON and merges new results into it:

```bash
# You already have results/my_baseline.json with 14 models.
# You just pulled a new model. Add it without re-running everything:
python run_benchmark.py --baseline results/my_baseline.json --models nemotron-3-nano:30b

# A specific benchmark crashed for all models (e.g. Spider needed the
# SQLite databases). Fix the issue, then re-run only that benchmark:
python run_benchmark.py --baseline results/my_baseline.json --benchmarks sql

# One model failed halfway through (OOM, timeout). Redo that model:
python run_benchmark.py --baseline results/my_baseline.json --models gemma4:31b-mlx

# Cherry-pick a single cell: one model × one benchmark group
python run_benchmark.py --baseline results/my_baseline.json --models qwen3:32b --benchmarks coding

# Your run was killed before it could save (e.g. laptop closed, process
# SIGKILL'd). Recover by reconstructing a lightweight baseline from the
# log, then patch the missing pieces:
#
#   1. Parse the log to recreate per-sample records — you'll lose
#      per-sample timing but preserve accuracy scores.
#   2. Launch an incremental run for the models/benchmarks that never
#      completed.
```

### How it works

New results **replace** matching `(model, benchmark)` pairs in the baseline. Everything else is kept. The merged set is saved to a **new timestamped file** — the original baseline is never modified. The HTML dashboard is regenerated after every model (with auto-refresh) so you can watch progress.

A common workflow:

```bash
# 1. Initial full run — let it bake overnight
python run_benchmark.py

# 2. Next morning: review results, notice Spider all failed
#    Fix the issue (e.g. download SQLite databases), then patch:
python run_benchmark.py --baseline results/20260707_235042.json --benchmarks sql

# 3. Later: a new model lands. Patch again:
python run_benchmark.py --baseline results/20260708_091200.json --models llama4:latest
```

Each step produces a standalone `results/<timestamp>.json` that represents the complete picture up to that point.

---

## Reading the results

### Dashboard

After a run, open this file in your browser:

```bash
open results/report.html
```

`results/report.html` is generated automatically at the end of every run (and after each model during a run, with a 60-second auto-refresh so it stays current while the run is in progress). It has your real results baked in.

`scoring/benchmark_dashboard.html` is the reusable template — it opens with sample data and lets you drag-drop any `results/*.json` file to explore it. Don't use this as your main view.

To regenerate the report manually from any results file:

```bash
python scoring/generate_report.py results/<timestamp>.json
```

### Terminal output

Two Rich tables print at the end of each run:

**Accuracy table** — each cell is the mean score (0–100%) for that model on that benchmark. The OVERALL column is the mean across the **7 core benchmarks only** (MMLU, ARC, GSM8K, HumanEval, MBPP, Spider, Philosophical) — never the optional/harder ones a model may additionally carry (BFCL, EvalPlus, LiveCodeBench). Those score lower by design on the same underlying problems, so folding them in would rank the most-thoroughly-tested models worse; they stay visible as their own columns instead. A model that hasn't run the full core set yet shows its OVERALL averaged over fewer benchmarks, marked accordingly, and is not directly comparable to a full row.

**Speed table** — tokens/second and average latency per inference call. Relevant for deciding whether a model is fast enough for interactive or agentic use.

### JSON output

Every run saves a file to `results/<timestamp>.json`. Each record contains the prompt sent, the model's full response, pass/fail, extracted answer vs expected, latency, tokens/second, and — for philosophical runs — the judge's per-criterion scores and reasoning.

Load in pandas for deeper analysis:

```python
import pandas as pd, json

df = pd.DataFrame(json.load(open("results/20260706_194954.json")))

# Accuracy per model per benchmark
df.groupby(["model", "benchmark"])["score"].mean().unstack()

# Where did a coding model fail?
df[(df.benchmark == "humaneval") & (df.passed == False)][["model", "prompt", "exec_error"]]

# Speed vs accuracy tradeoff
df.groupby("model")[["score", "tok_per_sec"]].mean()
```

---

## Interpreting results

| Score pattern | What it means |
|---|---|
| High MMLU + low GSM8K | Broad knowledge but weak at chained reasoning |
| High HumanEval + low MBPP | Strong at well-specified problems, weaker with ambiguous specs |
| High SQL string match + low execution accuracy | Generates plausible-looking SQL that doesn't actually run |
| Low philosophical mean score | Shallow or one-sided responses; judge penalises lack of nuance |
| High tok/s + low accuracy | Fast but sloppy — problematic for agentic loops |
| Low tok/s + high accuracy | Slow but reliable — fine for batch tasks |

The practical output is a routing map: which models to assign to which task types. High-accuracy coding models for agent loops, strong reasoning models for complex analysis, fast small models for cheap classification or summarisation.

## Versioning

`v0.2` changed how several benchmarks are extracted, prompted and timed, so
**v0.1 results are not comparable to v0.2 results** — re-run rather than merge.
The behavioural changes, in brief:

- Multiple-choice answers are parsed from an explicit answer position rather
  than the first A/B/C/D character in the response.
- The Spider prompt includes the database schema, and scoring executes both the
  predicted and reference queries when the SQLite databases are present.
- Thinking mode is only requested from models that declare the capability.
- Read timeouts are sized to cover prefill, not just inter-token gaps.
- Speed comes from the server's own decode timings, so prompt processing is no
  longer charged to generation.
- Sampling is seeded and stratified rather than taking the first N items.

---

## Dependencies and credits

| Package | Purpose | Source |
|---|---|---|
| `openai` | HTTP client to Ollama's `/v1` endpoint | [github.com/openai/openai-python](https://github.com/openai/openai-python) |
| `datasets` | Loads all HuggingFace benchmark datasets | [github.com/huggingface/datasets](https://github.com/huggingface/datasets) |
| `huggingface_hub` | Dataset download and caching | [github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) |
| `rich` | Terminal tables and formatting | [github.com/Textualize/rich](https://github.com/Textualize/rich) |
| `pandas` | Result aggregation and analysis | [github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| `pyyaml` | Config parsing | [github.com/yaml/pyyaml](https://github.com/yaml/pyyaml) |
| `httpx` | Ollama model list endpoint | [github.com/encode/httpx](https://github.com/encode/httpx) |
| `gdown` | Optional Spider database download from Google Drive | [github.com/wkentaro/gdown](https://github.com/wkentaro/gdown) |
| [Ollama](https://ollama.com) | Local model serving | [github.com/ollama/ollama](https://github.com/ollama/ollama) |

### Datasets

| Dataset | License | Citation |
|---|---|---|
| MMLU | MIT | Hendrycks et al., 2020 |
| ARC | CC BY 4.0 | Clark et al., 2018 |
| GSM8K | MIT | Cobbe et al., 2021 |
| HumanEval | MIT | Chen et al., 2021 |
| MBPP | CC BY 4.0 | Austin et al., 2021 |
| Spider | CC BY 4.0 | Yu et al., 2018 |

The philosophical prompts and LLM-as-judge rubric are original to this repository. The LLM-as-judge evaluation methodology follows [Zheng et al., 2023](https://arxiv.org/abs/2306.05685).

---

## License

This harness is released under the [MIT License](LICENSE) — see `LICENSE` for
the full text.

That covers the code in this repository, including the philosophical prompts
and the LLM-as-judge rubric, which are original to it.

It does **not** cover the benchmark datasets. Those are third-party works under
their own terms (MIT and CC BY 4.0, listed in the table above) and are fetched
at runtime rather than redistributed here — `data/` is gitignored. If you
publish results, cite the datasets you used; the CC BY 4.0 ones require
attribution.

Model weights are likewise not covered: each model carries its own licence from
its publisher, and some place conditions on commercial use or on publishing
benchmark comparisons. Check the licence of any model before publishing scores
for it.

---

## Extending

Add a new benchmark category by:

1. Creating `benchmarks/yourname.py` with a class that extends `BaseBenchmark`
2. Implementing `load_samples()` and `score()`
3. Registering it in `BENCHMARK_REGISTRY` in `run_benchmark.py`
4. Adding a config block in `config.yaml`

The base class handles the run loop, result collection, latency tracking, and error handling.
