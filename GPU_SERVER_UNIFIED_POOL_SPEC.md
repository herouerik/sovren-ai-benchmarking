# GPU Server — Unified 6× P100 Pool Setup

**Target:** GPU Server (192.168.68.115)
**Purpose:** Reconfigure Ollama to use all 6× P100 (96GB VRAM) as a single pool on port 11434.
**After this:** Benchmark runs are triggered remotely from the benchmarking machine — no local benchmark code needed.

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

## 6. Rollback (If Needed)

```bash
sudo systemctl stop ollama-unified
sudo systemctl start ollama-pool-a
sudo systemctl start ollama-pool-b
# Update benchmarking machine's config back to two pools
```

---

## 7. Files in This Repo

| File | Purpose |
|---|---|
| `GPU_SERVER_UNIFIED_POOL_SPEC.md` | This document |
| `config-gpu-unified.yaml` | Benchmark config (used remotely from benchmarking machine) |

**No other files needed on this server.** The benchmark code, dashboard merging, and validation all run from the benchmarking machine (192.168.68.106) over HTTP.