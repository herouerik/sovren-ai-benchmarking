# Findings: MacBook M4 vs multi-GPU server

Cross-host comparison of the two hosts in this benchmark, drawn from
`results/merged.summary.json` (run `merged`, 2026-08-20 02:19, 8,470 samples,
34 model configs) and the n=100 runs that feed it (`exhaustive`,
`bfcl_refix`, `factorial_qwen`, `evalplus_qwen`).

Hosts:

- **MacBook M4**, 48 GB unified memory. Runs both MLX (nvfp4) and
  ollama/llama.cpp (GGUF) builds. `effective_ctx` capped at 32768 in these
  runs (16384 for devstral).
- **i9 GPU server**, 64 GB RAM, 6× Tesla P100 (6×16 GB = 96 GB VRAM). An
  RTX 2080 Ti (11 GB) is physically present but **excluded** from the pool —
  see `config-gpu-unified.yaml`; the mismatched smaller card was a repeated
  single-GPU OOM bottleneck at high context.

## Which numbers are usable

Only the 2026-08-19/20 summaries are valid for Spider and BFCL.

- Spider execution scoring was broken before `be63d11` — `gpu_sweep_final`
  and `report_20260818_*.html` show 0.05–0.15 where the corrected value is
  0.71–0.75.
- BFCL emitted invalid JSON Schema types before `27f2d37`; e.g. Qwen3.8-GGUF
  moved 0.55 → 0.75 and qwen3.6-128k 0.82 → 0.88 once re-run at n=100.

`merged.summary.json` previously advertised `total_samples: 10970` while its
own `sample_sizes` summed to 8,370 — the field was stale. Any earlier citation
of "10,970 samples" is wrong; the merge that added Needle 2 recomputed it to
8,470 and no score cell or `model_info` entry changed.

Throughput is **only comparable within a single summary**. `baseline_v0.2`
reports 2–3× higher tok/s than `merged` for the same models (llama3.2:3b:
97.1 vs 33.6) because of the metric change described in the README release
notes. Do not build cross-summary speed conclusions.

`results/report_final.html` is the template shell, not a rendered report —
it carries no embedded data and displays "SAMPLE DATA" until a
`results/*.json` file is loaded into its sidebar.

## The n=100 set

Seven configs have the full 600-item coding/tool suite at n=100 per
benchmark (HumanEval, HumanEval+, MBPP, MBPP+, Spider, BFCL). Everything
else in `merged` is n=10–25 and should be treated as directional only.

| config | host | build | composite /600 | tok/s |
|---|---|---|---|---|
| gemma4:31b | i9 GPU server | Q4_K_M | **80.3%** | 9.7 |
| qwen3.8:27b | MacBook M4 | Q4_K_M | 78.8% | 8.5 |
| qwen3.6-128k:latest | i9 GPU server | Q4_K_M | 78.2% | **39.0** |
| qwen3.8:27b-mlx | MacBook M4 | nvfp4 | 77.7% | 31.6 |
| qwen3.6:27b | MacBook M4 | Q4_K_M | 77.5% | 12.4 |
| Qwen3.8-27B-GGUF | i9 GPU server | Q4_K_M | 77.2% | 10.3 |
| qwen3.6:27b-mlx | MacBook M4 | nvfp4 | 76.7% | 13.7 |

The 95% CI on a 600-item composite is ~±3.4pp, so 76.7–80.3 is **one flat
band, not a ranking**. The pairwise Fisher tests in
`FINDINGS-qwen-factorial.md` and `FINDINGS-qwen-evalplus.md` agree (every
p > 0.5). The column that separates these configs by 4× is throughput.

## Same weights, three deployments

Qwen3.8-27B is the controlled experiment — identical weights, three
placements:

| placement | composite | tok/s |
|---|---|---|
| M4, MLX / nvfp4 | 77.7% | 31.6 |
| M4, GGUF / Q4_K_M | 78.8% | 8.5 |
| 6×P100, GGUF / Q4_K_M | 77.2% | 10.3 |

Quality is indistinguishable across all three. The 96 GB GPU pool delivers
**1.2× the M4's ollama speed and 0.33× the M4's MLX speed** on the same
model.

Two things follow:

1. **Runtime choice on the Mac is the single largest performance lever in
   this whole benchmark** — 3.5× on Qwen3.8 (MLX vs GGUF), at no measurable
   quality cost, for free. Any GGUF model on the M4 leaves most of the
   machine's throughput unused.
2. **P100 is the wrong generation.** Pascal has no FP16/INT8 tensor cores
   and no support for modern quant kernels, so six of them lose to one M4's
   unified-memory bandwidth. The excluded RTX 2080 Ti is the companion
   lesson: a heterogeneous pool bottlenecks on its smallest card.

## What each host is actually for

**MacBook M4 — interactive work.** `qwen3.8:27b-mlx` at 77.7% / 31.6 tok/s
is within noise of the server's best and faster than five of the seven
n=100 configs. Ceiling: 32k effective context and roughly 35B params.

**GPU server — context and model size, and nothing else.** That is not a
small win. `qwen3.6-128k` runs full 63/63 offload at **262,144** context
using only ~39 GiB of the pool; `llama4:scout` runs 131k at ~90 GiB; GLM-4.5-Air
(110.5B, Q3_K_M) does not fit on the Mac at all. 8× the context window is a
capability the M4 does not have at any speed.

**Strongest single config: `qwen3.6-128k:latest` on the GPU server.** 78.2%
composite, 39 tok/s (fastest in the n=100 set), best tool-calling in the
suite (BFCL 0.88), 262k context. `gemma4:31b` scores 2.1pp higher but at
9.7 tok/s — and that 2.1pp is inside the CI, so it trades real speed for a
difference the data cannot resolve.

## What the benchmark spread says about use cases

- **Tool calling is production-ready.** BFCL 0.75–0.88 after the schema fix.
  The `qwen3.6` variants lead (0.87–0.88) and beat `qwen3.8` (0.75–0.81) —
  the newer model is *worse* at tool use, one of the few deltas large enough
  to act on. Do not assume the newer release wins.
- **Text-to-SQL is the weak spot: 0.71–0.75 across every config.** One
  query in four is wrong, and the ceiling does not move with host, model
  size, or quantization. Local-model SQL needs review before it runs.
- **Edge cases are where local models break.** MBPP 0.70 → MBPP+ 0.59 for
  the server's best; HumanEval 0.95 → HumanEval+ 0.91 for gemma4:31b. These
  models write code that passes the happy path.
- **`qwen3-coder` is broken on Spider** — 0.35 on the M4 and 0.35 on the
  server, while scoring 1.00 on MMLU. Same defect on both hosts, so it is
  the model, not the setup. Its raw Spider outputs are worth a look.
- **Six of ten benchmarks are saturated.** MMLU, ARC, GSM8K and
  philosophical sit at 0.9–1.0 for every config above ~24B and carry no
  discriminating information. The plus-variants, Spider and BFCL are the
  only benchmarks currently separating models.

## Recommendations

1. **Run MLX on the Mac, always.** Free 3.5×, no measurable quality cost.
2. **Retire the P100 pool for interactive work.** It is slower than the
   laptop at the same quality, at far higher power draw (6×250 W spec TDP
   vs the M4's tens of watts — not measured here, but the direction is not
   in doubt). Keep it for two jobs only: >32k context, and >40B models.
3. **Route on context length, not on capability.** Quality is flat across
   hosts; context is not. M4 + MLX for the edit/test/review loop, server
   only when the task genuinely needs 100k+ tokens of repo or a 110B model.
4. **Split model choice by task.** `qwen3.6` family for agentic/tool-calling
   work; `qwen3.8-mlx` or `gemma4:31b` for code generation.
5. **Prune the suite.** Drop MMLU/ARC/GSM8K from routine runs; spend the
   compute lifting the n=20 configs to n=100 instead.
6. **Next run worth doing: `qwen3.6:35b-mlx` on the M4** at n=100. It shows
   72 tok/s — 2× the fastest server config — but only has n=20 behind it,
   making it the most promising unproven config in the set.

---

## Addendum: Needle 2 (45M) — the tiny-device end of the scale

`results/needle2.summary.json`, merged into `merged.summary.json`. Run with
`tools/needle_shim.py` + `config-needle.yaml` on the MacBook M4, **CPU-only**
(`cactus-needle` 2.0.7 exposes no Metal extra), sample_seed 42 so the item set
is identical to every other model's.

| | Needle 2 | qwen3.6-128k (best BFCL here) |
|---|---|---|
| params | 45M | 36.0B |
| on disk | **14 MB** | 22.5 GB |
| peak RAM | **~131 MB** | tens of GB |
| bfcl (v3 AST, n=100) | **41.0%** | 88.0% |
| decode | **1352 tok/s** | 39 tok/s |
| host | M4, CPU-only | 6×P100 |

41.0% of the tool calls, for 1/1600th of the disk and 35× the decode rate.
Whole run: 100 samples in 35 seconds.

### Only one cell is legitimate, and that is the point

Needle is purpose-built for tool calling, device control and structured
extraction. `config-needle.yaml` enables `bfcl` and nothing else on purpose —
running MMLU or HumanEval against a 45M model produces near-zero scores that
describe the benchmark's scope, not the model, and would give it the same
uninformative row as a crashed load.

**Do not compare this 41.0% to Cactus's published 42.6%.** Theirs is BFCL v4
overall (live + multi-turn + irrelevance); this harness runs
`BFCL_v3_{simple,multiple,parallel,parallel_multiple}` — non-live, single-turn
AST only. Different denominators; the near-match is coincidence.

### Where it breaks down

| BFCL category | score |
|---|---|
| simple | 68% |
| multiple | 52% |
| parallel | 24% |
| parallel_multiple | 20% |

| failure mode | n (of 59) |
|---|---|
| right function, wrong arguments | 26 |
| too few calls emitted | 16 |
| no call emitted | 11 |
| wrong function chosen | 5 |
| too many calls emitted | 1 |

The model is good at *choosing* a function (only 5 wrong picks in 100) and bad
at *filling it in* and at *counting*. Single-call routing at 68% is a usable
component behind a validation layer; anything needing parallel calls (20–24%)
is not. Argument extraction, not tool selection, is the ceiling.

### Two integration traps worth recording

1. **Needle carries state across `.complete()` calls.** One unrelated prior
   query flips a correct call into a refusal — reproduced as cold → `call`,
   after an unrelated prior → `respond` with no calls, after `.reset()` →
   `call` again. The shim resets before every request. Without that, samples
   contaminate each other and the score reads low for a reason that has
   nothing to do with the model.
2. **Grammar compile is per-tool-set (~2.6 s), not per-token.** BFCL hands a
   different tool set every sample. Charging that to `prompt_eval_duration`
   and multiplying by `prefill_tps` invented 7920 prompt tokens for a
   two-sentence request; it is reported on its own key instead.

Also fixed while wiring this up: `collect_model_info` rounded `size_gb` to one
decimal, so anything under ~50 MB became a flat `0.0`, which read as falsy and
made `_classify_params` drop the parameter count entirely. Needle 2 reported as
`size_gb: 0.0` with no `params`. Sub-GB models now keep four decimals.

### What this changes about the comparison

Adding a 45M model to a table of 27B models is where unfairness actually
enters — not in the CPU-vs-GPU axis. Two rules follow:

- **Band the dashboard by weight class.** A single ranked accuracy column
  invites reading 41.0% as "worse than" 88.0% when the honest reading is
  "different weight class, 1/800th the parameters". Needle's story only
  appears once the table can express *points per MB*.
- **Quality is hardware-independent; only speed is not.** The harness runs at
  temperature 0.0 and the 41.0% reproduced exactly across two runs. Needle
  would score 41.0% on an i7, a Pi 5, or an ESP32 — so there is no reason to
  re-run quality on other hardware, and running Needle on one machine against
  Qwen on another would confound model and host.
- **Any new host needs a bridge model.** Qwen3.8-27B is what makes M4↔server
  comparable. An i7 or Pi column is only interpretable if it also runs
  something already measured elsewhere (`llama3.2:3b` is the natural anchor).

---

## Addendum: `bfcl_irrelevance` — tool restraint is a separate capability

Added 2026-08-20 alongside the BFCL v3 → v4 move (see `docs/ROADMAP.md` §E
addendum). 240 non-live requests that no available function can satisfy; a
pass means the model called **nothing**.

> **This column is MacBook M4 only.** The GPU server was not reachable from
> the machine these runs were driven from (`192.168.68.115` times out; the M4
> has since moved network per ROADMAP.md:663). Blank server cells mean *not
> measured*, not zero. `gemma4:31b`, `qwen3.6-128k` and `Qwen3.8-27B-GGUF`
> need a run driven from the server before they have an irrelevance number.

| model | host | bfcl (v4 AST) | bfcl_irrelevance |
|---|---|---|---|
| needle2 (45M) | M4 | 41.0% | **74.0%** |
| qwen3.8:27b-mlx (27.8B) | M4 | 81% | **68.0%** |

Both at n=100, seed 42.

### The inversion is the finding

**A 45M model exercises better restraint than a 27.8B one** — 74% vs 68%,
while losing to it 41% vs 81% on tool *selection*. Tool-selection skill and
tool-restraint are separable capabilities, and the existing `bfcl` column was
only ever measuring the first. A model routed into an agent loop on the
strength of an 80%+ BFCL score can still over-call on a third of out-of-scope
requests.

### One shared failure mode: topical adjacency

Both models grab a tool that relates to the subject but cannot answer the
question. They fail `irrelevance_44` and `irrelevance_115` identically.

| asked | tool called | why it is wrong |
|---|---|---|
| how long to travel Boston → New York by car | `calculate_distance` | distance is not duration |
| casualty number of the Battle of Waterloo | `historical_event.get_date` | date is not casualties |
| top scorer for the Los Angeles Lakers | `get_sport_team_details` | team is not player stats |
| gene sequence for evolutionary changes in whales | `gene_sequencer` | the name matches, the capability does not |

This is the agent-loop failure that costs a turn and returns confidently wrong
data, and nothing in the suite could see it before this benchmark existed.
It also argues that the fix is not a better model but a validation layer: a
name-similar tool with the wrong capability is exactly what a schema check
cannot catch and a result check can.

### Consequence for the routing map

`bfcl` alone is not sufficient evidence for putting a model in an agent loop.
Read the two columns together — high selection *and* high restraint. On
current M4 evidence nothing tested clears both convincingly, which is an
argument for keeping a validation step between a local model and any tool it
can actually invoke.
