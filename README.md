<img width="1509" height="771" alt="image" src="https://github.com/user-attachments/assets/011f3b95-7d8c-4b19-8e30-8103d96257ca" />


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
> separately. See `docs/ROADMAP.md` §E and its addendum.

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

### Written-up findings

Analyses of actual runs, rather than of the scoring machinery:

| Document | Question it answers |
|---|---|
| [`docs/FINDINGS-host-comparison.md`](docs/FINDINGS-host-comparison.md) | MacBook M4 vs the 6×P100 server — which host, model and use case wins, and which summaries are safe to read |
| [`docs/FINDINGS-qwen-factorial.md`](docs/FINDINGS-qwen-factorial.md) | Qwen 3.6 vs 3.8 × MLX vs GGUF, one factor at a time, with Fisher exact tests |
| [`docs/FINDINGS-qwen-evalplus.md`](docs/FINDINGS-qwen-evalplus.md) | The same factorial on the EvalPlus edge-case variants |

---

## Release notes

### v0.1 — initial release

The original harness: one entry point (`run_benchmark.py`), config-driven model
and benchmark selection, six benchmark categories (MMLU, ARC, GSM8K,
HumanEval + MBPP, Spider, philosophical LLM-as-judge), sandboxed code execution
for the coding benchmarks, an HTML dashboard generated from a results JSON, and
`--baseline` incremental patching so a new model could be added without
re-running everything. Inference went through Ollama's OpenAI-compatible `/v1`
endpoint. A wall-clock "memory guard" aborted models that looked like they were
swapping.

### v0.2 — measurement, scoring, and model-support overhaul

Everything below changes what the numbers mean. **v0.1 results are not
comparable to v0.2 results** — re-run rather than merge.

The scale of the correction, measured on the same models and hardware:

| what was wrong | before | after |
|---|---|---|
| MCQ answers extracted from the first `[ABCD]` in the text | `gemma4:26b-mlx` MMLU **0/5** | **5/5** |
| Spider prompt omitted the database schema | `llama3.2:3b` **0%** | **50%** |
| Spider, whole fleet (schema + representative sampling) | 0–20% | **35–80%** |
| `think` sent to models that do not support it | 4 models scored **1.00/5** on empty answers | **4.26–4.46/5** |
| Read timeout fired during prefill | `muse-glimmer:30b-mlx` **79.4** overall | **83.7** |
| Speed charged prompt processing to generation | `llama3.2:3b` **19.0** tok/s | **98.0** tok/s |

The v0.2 baseline is 20 model variants × 7 benchmarks × 2700 samples with
**1 errored sample** total. Under v0.1 measurement the same fleet produced
13 models with at least one aborted benchmark, nearly all of which turned out
to be measurement artefacts rather than real limits.

**Methodology**

- **Representative sampling.** Benchmarks took the *first* N samples, so every
  run measured a fixed head-slice of each dataset — and MMLU, configured with
  five subjects, drew all 20 samples from `abstract_algebra` alone because
  subjects were concatenated rather than interleaved. Sampling is now seeded
  (`sample_seed`), shuffled, and stratified: n=20 means 4 questions from each
  of 5 subjects, balanced at any n.
- **Decode speed separated from latency.** `tok_per_sec` divided completion
  tokens by *total* elapsed time, so prompt processing was charged to
  generation. Multiple-choice benchmarks emit a single token, making the figure
  meaningless. Results now carry `ttft`, `decode_tps`, `prompt_tokens`, and
  `completion_tokens`; `decode_tps` comes from the server's own timings and is
  the number comparable to published tok/s.
- **Realistic budgets.** `max_tokens` was 2048, which truncated reasoning
  models mid-answer — scoring zero on questions they were still working
  through. Now 4096, with `think_max_tokens` (16384) applied when thinking is
  on.
- **Sample counts raised** so a single question no longer moves a score 20
  points. Spider illustrates the risk: models scoring 100% at n=5 scored 55% at
  n=20.

**Bug fixes**

- **Multiple-choice answers were extracted with the *first* `[ABCD]` match in
  the response.** Any model that reasoned before answering had the letter taken
  from its restatement of the options, scoring correct answers as wrong. The
  extractor now prefers an explicit declaration (`the answer is C`, `**C**`,
  `\boxed{D}`) and otherwise takes the *last* standalone letter. Terse answers
  extract identically, so previously-scored terse responses are unaffected.
- **Fabricated timings polluted the dashboard.** Rows reconstructed by log
  recovery carried placeholder constants (`elapsed=5.0`, `tok_per_sec=10.0`)
  and were averaged into the speed column, understating fast models by 4–6×.
  They are now excluded, as are rows too short to measure throughput.
- **A run without `--baseline` silently overwrote the aggregated dashboard**
  with just that run's models. It now diverts to `report_<run_id>.html`.
- **`--output` was parsed and ignored.** It now works.
- **Results were only written at the very end**, so a run that died lost every
  completed model. It now checkpoints after each one.

**Model support**

- **Thinking is a first-class, recorded dimension.** Newer Ollama enables
  extended reasoning by default for models that declare the capability, which
  changes accuracy substantially — and `/v1` silently ignores the `think`
  parameter. Benchmark inference now uses Ollama's native `/api/chat`, where
  `think` is honoured, defaults per benchmark, can be overridden per model, and
  is stored on every result alongside the reasoning length.
- **Model variants.** The same model can be benchmarked under different
  settings and appear as separate dashboard rows (`model +think`) instead of
  overwriting each other — previously the second run silently replaced the
  first, since results merge on `(model, benchmark)`.
- **Speculative decoding is measured correctly** rather than aborted (see the
  guard notes below).

**Memory guard — rewritten**

- Aborts **mid-call** rather than classifying a sample after it has already
  cost the time.
- Detection is **timing-primary and burst-immune**: rates are averaged over
  wall-clock windows, not per-token gaps. Per-token gaps produced nonsense
  baselines (20000+ tok/s) whenever tokens arrived batched — which is *always*
  under speculative decoding, so the old approach systematically killed exactly
  the models it should have measured.
- **A long TTFT is no longer a fault.** Cold model load and extended thinking
  are legitimate; thinking tokens count as liveness.
- **OS pressure sensing**, portable across macOS (`vm_stat`) and Linux
  (`/proc/vmstat`), grades a trip as *hard* (model doesn't fit — skip it) or
  *soft* (one bad sample — continue).
- **"Too slow to benchmark" is now separate from "thrashing"**, reported as
  `too_slow` via a `min_decode_tps` floor.
- The absolute `max_baseline_seconds` / `spike_threshold` / `calibration_samples`
  settings are gone; the same config now works on GPU and CPU.

**Dashboard**

- **Architecture is shown**: dense vs MoE, expert topology (`MoE 128×8`), and
  active parameter counts. A sparse model reports only its *active* parameters,
  which is why a "larger" model can be several times faster and score
  differently — previously invisible.
- **Unreliable figures are left blank rather than guessed.** Parameter counts
  are cross-checked against file size; VRAM estimates are omitted when model
  metadata lacks the attention shape needed for the KV-cache term, instead of
  silently falling back to a weights-only number that understated usage by
  several GB.
- **The score heatmap actually separates scores.** It was three hard bands with
  only opacity varying inside each, so a whole band read as one colour (75% and
  100% were indistinguishable) while 74% vs 75% jumped a hue. It is now an
  11-step ramp across the same three sovren hues, with luminance rising
  monotonically — so the ordering survives greyscale and red-green colour
  blindness — spanning 30–100% rather than 0–100% because scores cluster near
  the top, and fanning out hardest over the last four steps.
- **Contrast fixed throughout.** Every step carries the ink that clears WCAG AA
  on it (worst pair 4.55:1), and the ramp steps across the luminance band where
  neither light nor dark ink would pass. Swap-marker text inherits its cell's
  ink rather than a fixed amber that measured **1.42:1** on the bright green
  steps. Column headers and rank-strip labels moved off `--muted`, which was
  2.6:1 on the near-black ground — below AA even for large text.
- Models are unloaded between runs, so two large models are never resident at
  once.

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
