# MANUAL: GPU server steps — pending since 2026-08-22

> **Owner:** Erik (requires root/sudo — the reason these are still open).
> **Host:** i9 GPU server, `192.168.68.115`, 6× Tesla P100 (96 GB) + RTX 2080 Ti.
> Nothing here blocks overnight work; do them whenever convenient, in order.

Context: the 2026-08-20/21 benchmarking sessions established that the
production pool's GPU0 exclusion was never actually enforced (Vulkan bypass,
see ROADMAP §RTX 2080 Ti, bug 1), and that pool residency behaviour needs
pinning to stop model-load churn. The session that found all this had no
root, so `/etc/ollama/unified.env` and systemd units are untouched.

---

## 0. Pre-flight

```bash
pgrep -af run_benchmark || echo "clear"
curl -s localhost:11434/api/ps | python3 -m json.tool | grep model
```

If a sweep is mid-run, wait — restarting Ollama underneath it loses the run.
The restart unloads every resident model (one cold start afterwards).

## 1. Production pool env — the critical fix (root)

Edit `/etc/ollama/unified.env`, add:

```
OLLAMA_VULKAN=0
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_KEEP_ALIVE=8h
```

Why:

- `OLLAMA_VULKAN=0` — this Ollama build (0.32.9) ships Vulkan ON, and Vulkan
  enumerates GPUs independently of `CUDA_VISIBLE_DEVICES`. The production pool
  (`ollama-unified.service`, CUDA devices 1–6) has therefore been able to load
  models onto the RTX 2080 Ti (GPU0) all along. This is the likely mechanism
  behind the historical "leftover qwen2.5-coder:7b on GPU0" finding and part
  of the observed model-churn problem.
- `OLLAMA_MAX_LOADED_MODELS=2` — sovereign-128k (~43 GB) + 30b-sovereign
  (~19 GB) co-reside comfortably in 96 GB; a third big load forces an explicit
  eviction choice instead of silent thrash.
- `OLLAMA_KEEP_ALIVE=8h` — survives idle gaps so large models stop paying the
  ~123 s reload after every break. Deliberately not `-1`: a stray one-off
  model would squat a slot forever; 8 h reclaims it.

## 2. Restart and VERIFY (the part never possible without root)

```bash
sudo systemctl restart ollama-unified.service
sleep 5
journalctl -u ollama-unified.service -n 60 --no-pager | grep -iE "inference compute|vulkan|cuda"
```

Expected: compute listing shows only CUDA devices 1–6, **zero `Vulkan`
entries**, no mention of the RTX 2080 Ti. If Vulkan still appears, the env
file isn't being sourced — check how the unit loads it (`EnvironmentFile=`).

## 3. Make the `:11437` small-model pool durable

The RTX 2080 Ti instance is currently a plain background `ollama serve`
started by a benchmarking session — it dies silently on reboot/logout and has
no keep-alive policy. Promote it to a unit:

```bash
pkill -f "ollama serve" ; sleep 2   # only the :11437 process should exist first — check pgrep!

sudo tee /etc/systemd/system/ollama-gpu0.service > /dev/null <<'EOF'
[Unit]
Description=Ollama - RTX 2080 Ti small-model pool (port 11437)
After=network.target

[Service]
Environment=OLLAMA_HOST=0.0.0.0:11437
Environment=CUDA_VISIBLE_DEVICES=0
Environment=OLLAMA_VULKAN=0
Environment=OLLAMA_MODELS=/home/erik/.ollama-gpu0/models
Environment=OLLAMA_KEEP_ALIVE=-1
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama-gpu0.service
curl -s localhost:11437/api/tags >/dev/null && echo OK
```

Adjust `ExecStart` if `which ollama` differs. Verify with step 0-style probes:
"inference compute" must show exactly one CUDA device (CUDA0 = the 2080 Ti),
never Vulkan, and never more than one GPU.

`KEEP_ALIVE=-1` is correct *here* (unlike the main pool): small models are
5–9 GB on a dedicated 11 GB card, and this pool's job is instant trivial
responses.

## 4. One-liner while there

In the GPU server's `sovren-ai-benchmarking/config-gpu-unified.yaml`, ensure
`execution.host_label: "i9 GPU server"` is present. Without it the next sweep
reintroduces the stale host label and re-breaks the cell-level merge
(documented twice: commits `7db02af`, `b7a1721`).

## 5. Post-change verification (from any machine)

```bash
curl -s http://192.168.68.115:11434/api/tags | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin)['models']]"
curl -s http://192.168.68.115:11437/api/tags | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin)['models']]"
# warm the champion once, then confirm it stays resident:
curl -s http://192.168.68.115:11434/api/generate -d '{"model":"qwen3-coder-next:sovereign-128k","prompt":"ping","stream":false}' >/dev/null
curl -s http://192.168.68.115:11434/api/ps
```

Report back so residency + GPU0 exclusion can be confirmed cross-machine.

---

## Usage rules after this (from measured findings)

- **`llama4:scout` runs SOLO or not at all** — spans all 6 P100s at ~94 %
  VRAM; historically correlates with PCIe correctable errors on GPU5's riser.
- **Never request `gemma3:12b` on the 2080 Ti** — reproducible CUDA PDL crash
  during warmup that kills the whole `:11437` server process, not just the
  request. Documented dead end on Turing; revisit only on a newer
  llama.cpp/Ollama build.
- Working small-pool roster: `qwen3:8b` (32k ctx verified clean),
  `phi4:14b` (4096 ctx ceiling), `gemma3:4b` as the gemma-class replacement.
  Avoid `qwen2.5-coder:*` for anything agentic/tool-based — verified never
  emits tool calls despite advertised support.
- Main-pool routing per host-comparison findings: P100 array is for **>32k
  context and >40B models**; interactive ≤35B work belongs on the M4 (MLX).
- Don't drive `.109` (MacBook M4) while its fill sweep is running — two
  harnesses on one Ollama swap the machine (measured 0.2 tok/s vs 72).
