# What we learned running local LLMs for real work

Synthesis from ~16,000 benchmark samples across 37 model configurations on
three machines, plus getting `opencode` + a local 27B model working reliably as
a daily coding driver.

Written for someone deciding whether local models are worth it, which ones, and
what setup avoids the failure modes. Every number here is measured on this
fleet — where a claim is unproven or was wrong, it says so.

---

## 1. The short version

**Local coding models work.** A 27B model on a 48 GB laptop built the baseline
of a real codebase end to end. The gap to frontier is real but the floor is
already useful.

**The bottleneck is rarely the model.** Almost every failure this week was
configuration, contention or measurement error. Ranked by time lost:
co-resident models causing swap collapse, a sandbox running the wrong Python
interpreter, a second app holding a model, and a stale DHCP lease.

**Pick models on two axes, not one.** Tool *selection* and tool *restraint* are
separable and rank models differently. Optimising for one gives you an agent
that either can't act or won't stop acting.

**Runtime beats hardware.** The same model on the same laptop is ~3x faster on
MLX than on GGUF, and a 48 GB MacBook beats a 6x Tesla P100 server on the same
weights. Buy the right runtime before the right GPU.

---

## 2. Which models are actually worth running

Measured at n=100 per cell. "Selection" is BFCL (call the right function);
"restraint" is BFCL-irrelevance (call *nothing* when no function fits).

| model | params | selection | restraint | he+ | mbpp+ | verdict |
|---|---|---|---|---|---|---|
| **gemma4:31b** | 31.3B | 0.85 | **0.91** | 0.91 | 0.65 | best overall; needs the GPU server |
| **devstral-small-2** | 24.0B | 0.85 | **0.87** | 0.75 | 0.65 | best agentic pick that fits a laptop |
| **qwen3.6:27b(-mlx)** | 27.8B | **0.87** | 0.86 | 0.80 | 0.68 | best balance on the M4 |
| qwen3-coder:latest | 30.5B | 0.80 | 0.84 | 0.78 | 0.64 | solid, fast |
| qwen3.8:27b-mlx | 27.8B | 0.81 | **0.68** | **0.90** | 0.62 | best coder, worst restraint |
| gemma4-12b-agentic | 11.9B | 0.80 | **0.57** | 0.81 | 0.56 | name oversells it |
| qwen2.5-coder (7/14/32b) | – | **0.00** | 1.00* | – | – | never emits a tool call |

\* That 1.00 is not judgement. A model that never calls a tool passes every
irrelevance item and fails every selection item. It is the single best argument
for never reading the restraint column alone.

**If you run one model on a 48 GB laptop:** `qwen3.6:27b-mlx` for agentic work,
`qwen3.8:27b-mlx` if you mostly want code written and can tolerate over-calling.

### Thinking mode is a trade, not an upgrade

Two full think-on/off pairs, every other factor held constant:

| effect | off | on | delta | p |
|---|---|---|---|---|
| restraint | 142/200 | 163/200 | **+21** | **0.0185** |
| coding | 303/400 | 275/400 | **-28** | **0.0329** |

Both directions significant. The cost is uneven — the MoE lost 21/200 on coding
(p=0.0285) where the dense model lost 7/200 (p=0.49). So enable thinking on
dense models for agent loops, and turn it off when you want code.

Practical rule: **qwen3.8 in an agent loop needs thinking on.** Its default
restraint is 0.60-0.68 across both builds (pooled 128/200 vs qwen3.6's 172/200,
p = 5e-07); with thinking it reaches 0.82, statistically level with qwen3.6.

---

## 3. Setup that actually works (48 GB Apple Silicon)

Server-side, via a launchd agent — these must reach the `ollama serve` process,
which on macOS is the GUI app and does not read your shell profile:

```
OLLAMA_HOST=0.0.0.0
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_LOADED_MODELS=1     # load-bearing, not tuning
OLLAMA_NUM_PARALLEL=1          # each slot has its own KV cache
OLLAMA_CONTEXT_LENGTH=32768
```

**`MAX_LOADED_MODELS=1` is the whole ballgame on a 48 GB machine.** Measured:
macOS holds ~7.9 GB wired+compressed, leaving ~40 GB. One 27B model is
17-19 GB — comfortable. Two is 36+ GB before either has consumed any context,
and the machine goes to swap.

Swap collapse does **not** present as an out-of-memory error. Generation drops
to ~0.2 tok/s with request timeouts, the GPU sits at 100%, and the runner stops
responding to API-level unload. It looks exactly like a hung model.

**Keep-alive alone makes this worse.** A 30m keep-alive widens the window in
which a previous model is still resident from 5 minutes to 30. Enabling it
without the concurrency cap is an amplifier for the failure it appears to fix.

### Context length behaves differently per runtime

- **MLX allocates KV lazily.** `/api/ps` reported an identical 17.36 GB for
  `num_ctx` 8192 and 32768; it only grew once context was consumed (19.25 GB
  after ~30k tokens, so ~60 KB/token). Setting a large context costs nothing
  until you use it.
- **GGUF reserves up front.** `devstral-small-2` (14.1 GB on disk) estimates
  21.1 GB at 32k and 39.8 GB at 128k — the latter would swap this machine by
  itself.

So on MLX, dropping a client from 132k to 32k saves ~6 GB and is worth doing,
but it will not fix a hang caused by two resident models. Fix co-residency first.

### The client's context setting is not what you think

`/v1/chat/completions` has no `num_ctx` field. A context length configured in
your editor (e.g. `opencode.jsonc`) is only *that client's* token budget — it
cannot control what Ollama allocates. Pin `OLLAMA_CONTEXT_LENGTH` server-side or
the two agree only by coincidence of the default, and desync silently on
upgrade.

### Other things that cost real time

- **Another app holding a model.** An editor session pointed at the same Ollama
  kept a 17-22 GB model resident for hours. No client-side unload can override
  this — the model comes straight back. Check with
  `lsof -nP -iTCP:11434 | grep ESTABLISHED` before blaming the model.
- **DHCP.** This laptop had four addresses in a week, and one of the old ones
  was reassigned to a host that answers ping and refuses port 11434 — which
  looks exactly like broken Ollama rather than a moved address. Set a
  reservation.

---

## 4. Hardware: runtime matters more than silicon

Same model, same weights, three placements:

| placement | coding composite | tok/s |
|---|---|---|
| M4, MLX | 77.7% | **31.6** |
| M4, GGUF | 78.8% | 8.5 |
| 6x Tesla P100, GGUF | 77.2% | 10.3 |

Quality is indistinguishable. A 48 GB laptop on MLX decodes **3x faster than a
96 GB six-GPU server** on the same model. Pascal-era cards have no support for
modern quant kernels, so six of them lose to one M4's unified memory.

Caveat: the speedup is a **build** property, not a model property. The same two
models in GGUF go the other way (0.70x). MLX conversion quality varies per model.

**What the multi-GPU server is actually for:** context and capacity. It runs
262k context with full offload, and 110B-class models that do not fit the laptop
at all. Not speed.

**Quality is hardware-independent.** At temperature 0 the same model scores the
same number on any machine — verified by an exact cross-host reproduction
(84/84, 78/78, 64/65 across three benchmarks, different machines, a day apart).
So only speed needs re-measuring per host.

---

## 5. Benchmark design: what discriminates and what doesn't

Ranked by across-model standard deviation — how much a benchmark actually
separates models:

| benchmark | stdev | median | verdict |
|---|---|---|---|
| **bfcl** | **0.320** | 0.78 | most informative in the suite |
| **bfcl_irrelevance** | **0.223** | 0.81 | second, and orthogonal to the first |
| mmlu | 0.141 | 0.80 | |
| humaneval_plus | 0.117 | 0.82 | best coding discriminator |
| humaneval | 0.115 | 0.85 | |
| spider | 0.098 | 0.65 | hard ceiling: nothing beats 0.75 |
| arc / gsm8k | ~0.09 | 0.97-1.00 | **saturated — drop** |
| mbpp / mbpp_plus | 0.05-0.07 | 0.61-0.65 | ceiling, not a ranking |
| philosophical | 0.042 | 0.92 | **saturated — drop** |

The two tool benchmarks separate models 2-3x better than anything else — and
they were the last two added. Meanwhile four of the original benchmarks sit near
their ceiling and carry almost no information.

**Text-to-SQL is the standing weak spot**: 0.71-0.75 for everything, unmoved by
model size, host or quantisation. Don't ship unreviewed local-model SQL.

**Edge cases are where models separate, but the average drop is small.**
Plain-to-plus costs about -3pp on average across 27 models. The *outliers* are
the signal: codestral loses 18pp on HumanEval+, GLM-4.5-Air 13pp on MBPP+.

---

## 6. Measurement hygiene — the part that cost the most time

Most of a week's debugging was not about models. It was about the harness
producing confident wrong numbers. Every one of these shipped a plausible score:

| failure | how it looked | how it was caught |
|---|---|---|
| sandbox ran bare `python3`, not `sys.executable` | model scored **0.00** on both EvalPlus columns | 97/100 samples had `ModuleNotFoundError: numpy` while the model's code was correct |
| model deleted from Ollama | **0.00** across three benchmarks | `speed: 0.0` and `/api/show` returning MISSING |
| model has no `tools` capability | `bfcl` **0.00**, reads as "bad at tools" | capability check; it means *cannot*, not *fails* |
| reading `tok_per_sec` (includes prefill) instead of `decode_tps` | apparent **44%** benchmark-mix bias | recomputing on the right field: 0.4% |
| swap flags merged as a union | clean re-measurements still flagged 💀 | comparing flags against a fresh abort-free run |
| another process holding a model | 0.2 tok/s, "hung" | `lsof` on port 11434 |

Practices that earned their place:

1. **A zero is a claim, not a datum.** Distinguish *measured zero*, *cannot be
   measured*, and *not measured*. Use `null`, never `0.0`, for the last two —
   a 0.00 that means "unreachable" is indistinguishable from a real score.
2. **Diff every merge before pushing.** Compare every pre-existing
   (model, benchmark) cell before and after. Expect zero changes, or only ones
   you intended. This caught a stale sample count and confirmed corrections
   touched nothing else.
3. **Fix a seed and keep item identity stable.** Same seed + stable ordering
   gave byte-identical draws across machines, which turned "do these agree?"
   into a one-line check.
4. **Reproduce across hosts deliberately.** One model measured on two machines
   is the cheapest validation of a whole pipeline.
5. **Check what else is using the resource** before concluding the model is slow.
6. **Sanity-check a score against the model's own other scores.** 0.00 on
   HumanEval+ from something scoring 0.87 on plain HumanEval is a bug, not a
   result.
7. **Write down what you got wrong, next to the fix.** Several corrections here
   are second-order — a fix based on a wrong diagnosis that happened to work.

### Claims from this work that were wrong and later corrected

Kept deliberately, because a findings doc that only records successes teaches
the wrong lesson:

- "MBPP+ is the most discriminating column" — it has the *lowest* spread of any
  coding benchmark.
- "MBPP+ costs 10+ points vs MBPP" — actual mean is -2.8pp.
- "There's a 44% benchmark-mix bias in the speed column" — artifact of reading
  the wrong field; real spread 0.4%.
- "qwen3.6 is inherently better at tool restraint" — true at *default settings*;
  the gap closes when thinking is enabled.
- "The `/api/chat` endpoint reloads models it's asked to unload" — both endpoints
  behave identically; the real cause was another client holding it.

---

## 7. If you are starting from scratch

1. Apple Silicon with 48 GB+ unified memory, and **run MLX builds**.
2. Cap concurrency server-side before anything else
   (`OLLAMA_MAX_LOADED_MODELS=1`). It is the difference between "works" and
   "hangs forever".
3. One 24-31B model. `qwen3.6:27b-mlx` or `devstral-small-2` for agentic work.
4. 32k context is plenty; larger costs little on MLX and a lot on GGUF.
5. Benchmark tool *use* and tool *restraint* — not just coding — before trusting
   a model in an agent loop. Neither is predicted by the model's name.
6. Expect to spend more time on measurement correctness than on models.
