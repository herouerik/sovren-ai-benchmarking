<img width="1509" height="771" alt="image" src="https://github.com/user-attachments/assets/011f3b95-7d8c-4b19-8e30-8103d96257ca" />


# sovren-ai-benchmark

A self-contained benchmarking harness for local LLMs served via [Ollama](https://ollama.com). Runs a suite of standard and custom benchmarks against any model available at your local Ollama endpoint, scores them, and produces a comparative summary across models.

Built to work fully offline once datasets are cached.

---

## Structure

```
sovren-ai-benchmark/
├── run_benchmark.py        ← single entry point for everything
├── config.example.yaml     ← copy to config.yaml; what to run, against which models
├── benchmarks/             ← one file per category
├── harness/                ← shared infrastructure
├── scoring/                ← result display and analysis
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

### 4. HumanEval + MBPP — Python coding
Two standard coding benchmarks. The model generates a Python function; the harness writes it to a temp file and executes it in a subprocess against bundled unit tests. Pass or fail — no partial credit. Never uses `exec()`.

- **HumanEval** — 164 hand-written Python problems with test assertions. [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) — Chen et al., 2021. [Paper](https://arxiv.org/abs/2107.03374).
- **MBPP** — ~400 crowd-sourced Python problems, `sanitized` split. [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) — Austin et al., 2021. [Paper](https://arxiv.org/abs/2108.07732).

**Tells you:** does the code actually run and pass tests?

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
Ten curated open-ended philosophical questions — free will, justice, moral realism, suffering, epistemic power, and more. No ground truth exists. A judge model scores each response 1–5 on five rubric axes: depth of reasoning, coherence, acknowledgment of multiple perspectives, originality of insight, and clarity of expression. The mean judge score becomes the benchmark score.

**Tells you:** how well does the model reason through open-ended, ambiguous problems with no single correct answer?

**Method:** LLM-as-judge is a widely used evaluation pattern for open-ended generation. See [Zheng et al., 2023 — MT-Bench](https://arxiv.org/abs/2306.05685) for the canonical reference. The prompts and rubric in this repo are original.

The judge is configured via `judge.provider` in `config.yaml` — see the [Judge configuration](#judge-configuration) section for available backends.

---

### 7. BFCL — function calling / tool use

**`bfcl`** — the non-live, single-turn AST categories (`simple`, `multiple`,
`parallel`, `parallel_multiple`). The model gets a request plus a set of
function schemas and must emit the right call(s). Scored by BFCL's AST-match
rule — function name plus every ground-truth parameter drawn from an
acceptable-value set — reimplemented locally in `benchmarks/bfcl.py`.

**`bfcl_irrelevance`** — the inverse. 240 requests that *no* available function
can satisfy; a pass means the model called nothing. Without this, a model that
fires a tool at every prompt scores identically to one with judgment, because
nothing else in the suite penalises over-calling.

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
and ~28 tool calls, measured at ~450–500s.

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

## Two things the benchmark grid does not measure

Both ship as first-class parts of the harness — a tool, a committed data file, a
generator flag, and a dashboard panel each. Neither is merged into the benchmark grid,
because neither is the same kind of number.

### At-context throughput — `tools/measure_context_perf.py`

The `speeds` column is decode tok/s on benchmark prompts, which are short. For an agentic
coding loop that is the wrong operating point, and it does not merely understate — **it can
invert the ranking.**

Every model decodes more slowly as context grows, but by very different factors. On
Apple-Silicon MLX builds we have measured a dense ~27B model losing roughly an order of
magnitude between a near-empty context and a ~9k one, while a sparse-MoE model of
comparable total size lost only about half its rate. Rank on short-prompt decode and you
can choose the slowest model available while believing it is mid-pack.

Prefill matters as much. A harness that compacts rewrites the transcript and invalidates
the KV prefix, so **every compaction pays a full-window prefill**, not an incremental one.
Window size divided by prefill rate is that bill; when it approaches the interval between
compactions, the agent spends most of its wall clock re-reading its own context.

The model rows already carry `dense` vs `MoE` and active parameters, so the fact needed to
predict this is usually on screen — what was missing is a measurement at the operating
point.

> Figures for our own fleet are not published here. This repo is public; runs, summaries,
> dashboards and findings live in the private companion repo. Run the tool to get numbers
> for your own hardware.

```sh
python3 tools/measure_context_perf.py --config config.yaml \
    --context-tokens 9000 --host "<host label>" --out data/context_perf.json
```

`data/context_perf.json` is gitignored — a per-machine measurement, same category as
`config.yaml`. Picked up automatically from `data/context_perf.json`, or pass `--perf <path>`. Renders as
the **AT-CONTEXT THROUGHPUT** sidebar panel. Read the module docstring before changing
defaults — it records three traps (a resident model ignores a changed `num_ctx`; thinking
models spend the output budget on reasoning; a 40-token generation measures startup, not
decode) that each cost a wrong measurement to find.

### External reference figures — `data/external_reference.yaml`

Every score here is a "how does it behave on my machine" number: small-n, our scaffold, our
quantised builds, our hardware. That is the point, and it also means there is no way to
calibrate against results the wider field agrees on. This file carries a small set of
*published* figures for that calibration only.

They are rendered in a separate panel, **excluded from OVERALL, from ranking, and from every
model row** — a 500-sample leaderboard result and a 25-sample local run do not belong in one
column.

Currently pulled: SWE-bench Verified, LiveCodeBench, Terminal-Bench 2.0. Available upstream
without code changes by adding a `slug`: `osWorldVerified`, `browseComp`, `arcAgi2`, `hle`.

```sh
python3 tools/fetch_external_reference.py --dry-run   # show what would change
python3 tools/fetch_external_reference.py             # rewrite the scores blocks
```

Committed rather than fetched at report time on purpose: a benchmark run must not depend on
the network, and a published number that moves should move in a reviewable diff.

**The overlap is partial, and which part matters.** SWE-bench Verified evaluates frontier
hosted models only — read it as a reference *band*, not a comparison. LiveCodeBench carries
open-weight rows, so some of them name model families a local fleet can actually host; those
are genuinely comparable, and because the published figure is full-precision while a local
build is usually quantised, **the gap is an estimate of the quantisation penalty on your own
hardware.** The dashboard marks any such row "← runs here" by matching the external model
name against the models in your run.

Match on family, never on tag: an upstream `Qwen3.6-27B` and a local `qwen3.6:27b-mlx` are
the same weights at different precision on different hardware, and collapsing them into one
row would hide exactly the effect worth measuring.

## Usage

```bash
# Set up (first time only)
cd local-llm-benchmark
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

**Accuracy table** — each cell is the mean score (0–100%) for that model on that benchmark. The OVERALL column is the mean across all benchmarks run.

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
