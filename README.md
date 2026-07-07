<img width="1512" height="689" alt="image" src="https://github.com/user-attachments/assets/7278d200-21c1-4ffb-b374-204dc06e0dbc" />

# sovren-ai-benchmark

A self-contained benchmarking harness for local LLMs served via [Ollama](https://ollama.com). Runs a suite of standard and custom benchmarks against any model available at your local Ollama endpoint, scores them, and produces a comparative summary across models.

Built to work fully offline once datasets are cached.

---

## Structure

```
local-llm-benchmark/
├── run_benchmark.py        ← single entry point for everything
├── config.yaml             ← what to run and against which models
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

## How a run works

```
run_benchmark.py
  └─ loads config.yaml
  └─ for each model:
       └─ for each benchmark:
            └─ load_samples() — pulls dataset from HuggingFace (cached after first pull)
            └─ for each sample:
                 └─ client.complete() — POST to Ollama at localhost:11434/v1
                 └─ score() — exact match / code execution / LLM judge
            └─ print live pass rate
  └─ save results/<timestamp>.json
  └─ print Rich summary tables
```

Temperature is `0.0` for all deterministic benchmarks (MCQ, math, coding, SQL) so runs are reproducible. The philosophical judge also runs at `0.0` — subjectivity is in the rubric, not the sampling.

The harness calls Ollama via the [OpenAI Python SDK](https://github.com/openai/openai-python) pointed at Ollama's OpenAI-compatible `/v1` endpoint. No framework lock-in — it's plain HTTP under the hood.

---

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

## Extending

Add a new benchmark category by:

1. Creating `benchmarks/yourname.py` with a class that extends `BaseBenchmark`
2. Implementing `load_samples()` and `score()`
3. Registering it in `BENCHMARK_REGISTRY` in `run_benchmark.py`
4. Adding a config block in `config.yaml`

The base class handles the run loop, result collection, latency tracking, and error handling.
