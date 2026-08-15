# GPU Server — Unified 6× P100 Pool Setup

**Target:** GPU Server (192.168.68.115)
**Purpose:** Reconfigure Ollama to use all 6× P100 (96GB VRAM) as a single pool on port 11434.
**After this:** Benchmark runs are triggered remotely from the benchmarking machine — no local benchmark code needed.

---

## Handover — GPU Server agent, 2026-08-15

Setup is **done and live**, but several details below turned out wrong once actually run against this box. Read this before trusting anything in sections 1–7 — they're left as-is for the historical record, not because they're accurate.

**What's actually running:** `ollama-unified.service`, active + enabled, `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6` (P100s only). Ollama itself is **0.32.9**, upgraded from the 0.20.5 this spec was written against (needed for `muse-glimmer` support — see below). `ollama.service`/`ollama-meta.service`/`ollama-b.service` (the real split-pool service names — see next point) are stopped + disabled.

**Corrections to the steps above:**
- **§1/§6 service names are fictional.** There is no `ollama-pool-a`/`ollama-pool-b` on this host. The real services are `ollama.service` (was Pool A, 4× P100, port 11434), `ollama-b.service` (was Pool B, 2× P100, port 11436), `ollama-meta.service` (GPU0/RTX2080Ti, port 11435). If you ever roll back, use these real names, not what §6 says.
- **§2.1 env file is wrong on two settings.** `OLLAMA_GPU_OVERHEAD=0` causes a full CPU fallback (0 layers offloaded) on this fleet's P100s — the value that actually works is `OLLAMA_GPU_OVERHEAD=268435456` (matches what `ollama.service`'s own override.conf already used, pre-existing fleet knowledge). Also add `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6` explicitly — without it Ollama also grabs GPU0 (RTX 2080 Ti, mismatched 11GB card), which was a repeated single-GPU OOM bottleneck for `gemma4:31b` at high context and got removed. Current real file is `/etc/ollama/unified.env` — read it directly rather than trusting this doc's §2.1 block.
- **§2.2 systemd unit**: dropped `Type=notify` (this Ollama build doesn't reliably send the readiness notification, risks a start-timeout hang) and `DeviceAllow` (unnecessary — the three real pre-existing services on this host don't use it either and work fine). Current real unit is `/etc/systemd/system/ollama-unified.service`.
- **§3 model list is stale.** `GLM-4.5-Air` isn't a real pullable name — the actual model is `MichelRosselli/GLM-4.5-Air:Q3_K_M`. `muse-glimmer` wasn't runnable at all on 0.20.5 (unknown architecture error) — that's why Ollama got upgraded to 0.32.9 this session.

**Context ceilings — use these, not blind defaults.** Each model's compute-graph buffer must fit on a *single* P100 (16GB) regardless of the pool's aggregate 96GB — this is architecture-dependent, not just model-size-dependent, and was wrong to assume as "131072 everywhere." Bisected and verified via `offloaded N/N layers to GPU` in the ollama-unified log (not just "loads without erroring" — partial offload silently degrades to CPU speed):

| Model | max ctx (full offload) |
|---|---|
| `deepseek-r1:70b` | 32768 |
| `MichelRosselli/GLM-4.5-Air:Q3_K_M` | 8192 |
| `gemma4:31b` | 98304 |
| `qwen3-coder-next:sovereign-128k` | 131072 |
| `llama4:scout` | 131072 (~90GB footprint — uses the pool best) |
| `qwen3.6-128k:latest` | 262144 (native max, only ~39GB used) |
| `muse-glimmer` (`hf.co/bartowski/Muse-Glimmer-30B-GGUF:Q4_K_M`) | 131072 (native max, only ~6GB/GPU — most efficient of all) |

These are already written into `config-gpu-unified.yaml` in this repo — that file is the current source of truth, not §3 of this doc.

**New capability on this pool:** `benchmarks/bfcl.py` (function-calling, single-turn) and `benchmarks/bfcl_multi_turn.py` (stateful multi-turn, Long-Context subcategory) landed this session — see `docs/ROADMAP.md` §E for the full design and status. Multi-turn is genuinely expensive (~450-500s/sample observed against qwen3.6 on this pool) and is disabled by default in `config.yaml`.

**Access note:** this agent did not have blanket sudo on this box — only specific pre-authorized `systemctl restart/stop/start` commands for the pre-existing services. Every privileged step (env/unit file writes, `daemon-reload`, the Ollama binary upgrade) was done by writing a script and asking the human operator to run it with `sudo`. Expect the same constraint if you're picking this up.

**Not pushed:** all changes are committed locally on `main` in `sovren-ai-benchmarking` (5 commits, 2026-08-15) but not pushed to `origin` — pending human review.

---

## 1. Stop Existing Pools

```bash
# Stop current ollama instances (adapt service names to your setup)
sudo systemctl stop ollama-pool-a
sudo systemctl stop ollama-pool-b
# Verify all GPU memory freed:
nvidia-smi
# Should show all 6 GPUs with minimal memory usage
```

---

## 2. Create Unified Ollama Service

### 2.1 Environment File: `/etc/ollama/unified.env`
```bash
# Unified 6× P100 pool (96GB VRAM)
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_GPU_OVERHEAD=0
OLLAMA_NUM_GPU=6
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE=24h
OLLAMA_FLASH_ATTENTION=1
```

### 2.2 systemd Service: `/etc/systemd/system/ollama-unified.service`
```ini
[Unit]
Description=Ollama Unified 6x P100 Pool (96GB)
After=network.target

[Service]
Type=notify
EnvironmentFile=/etc/ollama/unified.env
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=10

# GPU access
DeviceAllow=/dev/nvidia* rwm
DeviceAllow=/dev/dri/* rwm

[Install]
WantedBy=multi-user.target
```

### 2.3 Enable & Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama-unified
sudo systemctl start ollama-unified

# Verify all 6 GPUs visible to Ollama:
nvidia-smi -L
curl http://localhost:11434/api/ps
```

---

## 3. Load Target Models (One-Time)

Pull and warm up the three large models that need 64–96GB VRAM:

```bash
# Pull models (sequential to avoid OOM)
curl -X POST http://localhost:11434/api/pull -d '{"name": "deepseek-r1:70b"}'
curl -X POST http://localhost:11434/api/pull -d '{"name": "GLM-4.5-Air"}'
curl -X POST http://localhost:11434/api/pull -d '{"name": "qwen3-coder-next:sovereign-128k"}'

# Warm up each (keeps them resident with 24h keep_alive from env)
for m in deepseek-r1:70b GLM-4.5-Air qwen3-coder-next:sovereign-128k; do
  curl -X POST http://localhost:11434/api/generate -d "{\"model\": \"$m\", \"prompt\": \"warmup\", \"keep_alive\": \"24h\"}"
done
```

**VRAM notes:**
- Only ONE model should be resident during benchmark runs (the benchmark harness unloads between models).
- `OLLAMA_MAX_LOADED_MODELS=3` allows up to 3 resident, but we'll benchmark sequentially.
- If OOM occurs during benchmark, the remote runner will handle single-model sequencing.

---

## 4. Verify Pool Health

```bash
# Confirm all 6 GPUs show activity when a model loads:
nvidia-smi
# Run a quick test:
curl -X POST http://localhost:11434/api/generate -d '{"model": "deepseek-r1:70b", "prompt": "test", "stream": false}'
# Should complete without OOM, using multiple GPUs
```

---

## 5. What Happens Next (Remote)

After you confirm the unified pool is up and models are pulled:

1. **This machine** runs the full benchmark suite remotely via HTTP:
   ```bash
   # From sovereign-ai-benchmarking on the benchmarking machine
   .venv/bin/python run_benchmark.py --config config-gpu-unified.yaml --benchmarks mmlu gsm8k humaneval mbpp spider philosophical --n-samples 20
   ```

2. **This machine** merges results into the existing dashboard (M4 + GPU unified).

3. **No further action needed on this server** — just keep `ollama-unified` running.

---

## 6. Rollback — Restore Original 4+2 Pool Split

If the unified pool is unstable, or you need the original endpoints for dependent systems **tonight**, run this on the GPU server:

### 6.1 Stop Unified, Restore Original Services

```bash
# Stop unified pool
sudo systemctl stop ollama-unified
sudo systemctl disable ollama-unified

# Restore original pool-a (4x P100 on port 11434) and pool-b (2x P100 on port 11436)
# Adjust service names to match your actual setup:
sudo systemctl start ollama-pool-a   # 4x P100 → port 11434
sudo systemctl start ollama-pool-b   # 2x P100 → port 11436

# Verify both pools up and models pinned:
curl http://localhost:11434/api/ps   # should show qwen3-coder-next:sovereign-128k
curl http://localhost:11436/api/ps   # should show qwen3-coder:30b-sovereign
nvidia-smi                           # 4 GPUs active on pool-a, 2 on pool-b
```

### 6.2 Re-pin Original Models (If Evicted)

```bash
# Pool-a (port 11434): 79.7B sovereign with 128k context
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen3-coder-next:sovereign-128k", "prompt": "warmup", "keep_alive": "24h"}'

# Pool-b (port 11436): 30b sovereign coder
curl -X POST http://localhost:11436/api/generate \
  -d '{"model": "qwen3-coder:30b-sovereign", "prompt": "warmup", "keep_alive": "24h"}'

# Meta pool (port 11435, RTX 2080): lightweight sidecar
curl -X POST http://localhost:11435/api/generate \
  -d '{"model": "qwen2.5-coder:7b", "prompt": "warmup", "keep_alive": "24h"}'
```

### 6.3 Benchmarking Machine Config Rollback

On the benchmarking machine (192.168.68.106), restore the two-pool benchmark config:

```bash
cd /home/erik/git/pfc/sovren-ai-benchmarking
# Use the original config-gpu-server.yaml (points at pool-a 11434 for 79.7B)
# and config-gpu-server-poolb.yaml (points at pool-b 11436 for 30b/qwen3.6-128k)
# Both already committed in this repo.
```

### 6.4 Router Config Rollback (This Machine)

On the benchmarking machine, restore `config/inference.yaml` to the two-pool endpoints:

```yaml
endpoints:
  - name: gpu-server-a
    url: "http://192.168.68.115:11434/v1"
    driver: ollama
    priority: 80
    tags: [sovereign, gpu, pool-a]
    models:
      qwen3-coder-next:sovereign-128k: {ctx: 131072, temperature: 0.2}

  - name: gpu-server-b
    url: "http://192.168.68.115:11436/v1"
    driver: ollama
    priority: 75
    tags: [sovereign, gpu, pool-b]
    models:
      qwen3-coder:30b-sovereign: {ctx: 32768, temperature: 0.2}
      qwen3.6-128k:latest: {ctx: 32768, temperature: 0.2}

  - name: gpu-server-meta
    url: "http://192.168.68.115:11435/v1"
    driver: ollama
    priority: 70
    tags: [sovereign, gpu, meta]
    models:
      qwen2.5-coder:7b: {ctx: 8192, temperature: 0.2}
```

The original `inference.yaml` (before unified pool) is in git history — `git checkout HEAD~1 -- config/inference.yaml` restores it.

---

## 7. Dependent Systems Checklist

Before rolling back, verify these endpoints aren't hardcoded elsewhere:

| System | Expected Endpoint | Check |
|---|---|---|
| opencode agents | `gpu-server-a` (11434), `gpu-server-b` (11436) | `~/.config/opencode/opencode.jsonc` |
| verifier_agent fallbacks | Both pools in sovereign chain | `scripts/verifier_agent.py` |
| inference router | Both pools registered | `config/inference.yaml` |
| Any external scripts | Direct HTTP to 11434/11436 | grep for `192.168.68.115:1143` |

If any system calls the unified pool (11434) expecting the 79.7B but gets a different model, update its config to the restored pool-a endpoint.

---

## 7. Files in This Repo

| File | Purpose |
|---|---|
| `GPU_SERVER_UNIFIED_POOL_SPEC.md` | This document |
| `config-gpu-unified.yaml` | Benchmark config (used remotely from benchmarking machine) |

**No other files needed on this server.** The benchmark code, dashboard merging, and validation all run from the benchmarking machine (192.168.68.106) over HTTP.