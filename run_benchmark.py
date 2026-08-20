#!/usr/bin/env python3
"""
Local LLM Benchmark Runner
Usage:
  python run_benchmark.py                          # run all benchmarks, all configured models
  python run_benchmark.py --models devstral-small-2 qwen3:32b
  python run_benchmark.py --benchmarks mmlu gsm8k
  python run_benchmark.py --n-samples 10          # quick test run
  python run_benchmark.py --list-models           # show available Ollama models
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

# Go fully offline if datasets have been pre-fetched; otherwise allow downloads.
if Path(".datasets_ready").exists():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
else:
    print("Tip: run `python prefetch_datasets.py` once to cache all datasets locally "
          "and silence HuggingFace network requests.")

from harness.client import OllamaClient, OpenCodeClient, build_client, resolve_env_vars
from benchmarks.base import BaseBenchmark, MemorySwapAbort
from benchmarks.reasoning import MMLUBenchmark, ARCBenchmark
from benchmarks.math import GSM8KBenchmark
from benchmarks.coding import HumanEvalBenchmark, MBPPBenchmark
from benchmarks.sql import SpiderBenchmark
from benchmarks.philosophical import PhilosophicalBenchmark
from benchmarks.speed import SpeedBenchmark
from benchmarks.bfcl import BFCLBenchmark, BFCLIrrelevanceBenchmark
from benchmarks.evalplus import HumanEvalPlusBenchmark, MBPPPlusBenchmark
from benchmarks.bfcl_multi_turn import BFCLMultiTurnLongContextBenchmark
from scoring.report import save_results, print_summary
from scoring.generate_report import aggregate, load_config_models, find_template

console = Console()

BENCHMARK_REGISTRY = {
    "mmlu":        MMLUBenchmark,
    "arc":         ARCBenchmark,
    "gsm8k":       GSM8KBenchmark,
    "humaneval":   HumanEvalBenchmark,
    "mbpp":        MBPPBenchmark,
    "spider":      SpiderBenchmark,
    "philosophical": PhilosophicalBenchmark,
    "speed":         SpeedBenchmark,
    "bfcl":          BFCLBenchmark,
    "bfcl_irrelevance": BFCLIrrelevanceBenchmark,
    "humaneval_plus": HumanEvalPlusBenchmark,
    "mbpp_plus":      MBPPPlusBenchmark,
    "bfcl_multi_turn_long_context": BFCLMultiTurnLongContextBenchmark,
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_ollama_models(base_url: str) -> list[str]:
    import httpx
    try:
        url = base_url.replace("/v1", "") + "/api/tags"
        resp = httpx.get(url, timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        console.print(f"[red]Could not reach Ollama: {e}[/red]")
        return []


def collect_run_metadata(cfg: dict) -> dict:
    """Collect git, hardware, Ollama, and config metadata for this run."""
    meta: dict = {}

    # Git info
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        meta["git_sha"] = sha
        meta["git_branch"] = branch
    except Exception:
        meta["git_sha"] = None
        meta["git_branch"] = None

    # Ollama version
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        meta["ollama_version"] = r.stdout.strip() or r.stderr.strip()
    except Exception:
        meta["ollama_version"] = None

    # Hardware
    meta["hardware"] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "python": platform.python_version(),
        # platform.node() is deliberately not recorded: it is the machine's
        # hostname, and these records are published. The fields above already
        # describe the hardware well enough to interpret a speed number.
    }

    # Config snapshot (sanitised — excludes full model list to keep records lean)
    exec_cfg = cfg.get("execution", {})
    meta["config_snapshot"] = {
        "n_models": len(cfg.get("models", [])),
        "benches_enabled": list(cfg.get("benchmarks", {}).keys()),
        "judge_provider": cfg.get("judge", {}).get("provider", "ollama"),
        "judge_model": cfg.get("judge", {}).get("cloud_model") or cfg.get("judge", {}).get("ollama_single_model", "ollama"),
        "parallel_models": exec_cfg.get("parallel_models", False),
        "max_workers": exec_cfg.get("max_workers", 1),
    }

    return meta


def resolve_model_entry(entry: str | dict, default_ctx: int | None = None) -> dict:
    """Resolve a model config entry (string or dict) to a run spec.

    Returns {model, ctx, think, label}. `label` is the identity used everywhere
    downstream — results records, merge keys, dashboard rows — so the same model
    can be benchmarked under different settings (notably thinking on vs off)
    and appear as separate, comparable rows instead of overwriting each other.

    The ctx is resolved: per-model > global default_ctx > 4096 hard fallback.
    `think` of None means "use the benchmark's own default".

    Example:
        "llama3.2:3b"
            → {model: llama3.2:3b, ctx: 32768, think: None, label: llama3.2:3b}
        {"model": "muse-glimmer:30b-mlx", "think": True}
            → label "muse-glimmer:30b-mlx +think"
    """
    if isinstance(entry, str):
        return {"model": entry, "ctx": default_ctx or 4096, "think": None, "label": entry}
    name = entry["model"]
    think = entry.get("think")
    label = entry.get("label") or (f"{name} +think" if think else name)
    return {
        "model": name,
        "ctx": entry.get("ctx", default_ctx) or 4096,
        "think": think,
        "label": label,
    }


def _estimate_vram(mi: dict, size_gb: float, ctx: int) -> float | None:
    """Estimate VRAM in GB at the given context length, or None if it can't be.

    Returns None when the model metadata lacks the attention shape needed for
    the KV cache term (some MLX conversions omit head_count entirely). It used
    to silently fall back to weights-only, which understated usage by several GB
    at 32k ctx while looking identical to a complete estimate — the worst
    outcome when the whole point of the number is judging whether a model fits.
    """
    arch = mi.get("general.architecture", "")
    prefix = f"{arch}."
    def g(key):
        return mi.get(prefix + key) or mi.get(key)

    n_layers = g("block_count")
    n_heads = g("attention.head_count")
    n_kv_heads = g("attention.head_count_kv") or n_heads
    emb_dim = g("embedding_length")
    if not all([n_layers, n_heads, emb_dim]):
        return None

    weights_gb = size_gb * 1.05
    per_head_dim = emb_dim // n_heads
    kv_bytes_per_token = 4 * n_kv_heads * per_head_dim * n_layers
    kv_gb = (ctx * kv_bytes_per_token) / (1024**3)
    return round(weights_gb + kv_gb, 1)


# Plausible bits-per-weight across the quantisations in use (nvfp4/q4 ~4 bits,
# q8 ~8, bf16 ~16). Outside this band the reported parameter count cannot be
# reconciled with the file on disk, so one of them is wrong.
_MIN_BITS_PER_PARAM, _MAX_BITS_PER_PARAM = 2.0, 20.0


def _classify_params(param_count, size_gb: float, reported: str | None) -> dict:
    """Reconcile the reported parameter count against the size on disk.

    Returns {params, params_active, sparse}. When the reported count is far
    smaller than the file can hold, it is the *active* parameter count of a
    sparse (MoE) model, not a wrong total — gemma4:26b-mlx reports 6.3B while
    occupying 15.5GB, and its 30 blocks / 2816 embedding confirm ~6B of dense
    compute inside a ~26B model. Reporting that as a plain "6.3B" hides the
    single most useful fact about the model's performance characteristics.
    """
    out: dict = {}
    if not reported or not param_count or not size_gb:
        return out
    bits_per_param = (size_gb * (1024**3) * 8) / param_count
    if bits_per_param <= _MAX_BITS_PER_PARAM:
        out["params"] = reported          # consistent: a dense total
    else:
        # Too many bytes for this many weights -> the count is active-only.
        out["params_active"] = reported
        out["sparse"] = True
    return out


def _model_name(entry: str | dict) -> str:
    """Resolve a model config entry to its model name string."""
    return entry if isinstance(entry, str) else entry["model"]


def _multi_model_models(raw: list, default_ctx: int | None = None) -> list[dict]:
    """Resolve a models list (strings and dicts) to run specs."""
    return [resolve_model_entry(m, default_ctx) for m in raw]


# Known inference hosts on this fleet, keyed by the IP each config's base_url
# actually points at — not by platform.node()/platform.machine(), which only
# describe whatever machine happens to be running run_benchmark.py itself.
# That distinction matters here specifically: this harness is routinely run
# FROM one machine to drive inference on ANOTHER (e.g. config-m4-remote.yaml
# runs on the GPU server but serves from the M4), so the script's own host is
# not the inference host and must never be used to label results.
# Deliberately generic: these labels are published in the dashboard and the
# summaries, which live in a public repo. They identify the *class* of machine
# (which is what makes a speed number interpretable) and nothing more — no
# hostnames, no owner, no serial. Keep any new entry to the same shape.
_KNOWN_HOSTS = {
    "192.168.68.115": "i9 GPU server",
    "192.168.68.106": "MacBook M4",  # stale — kept in case it reconnects here
    "192.168.68.110": "MacBook M4",  # current, as of 2026-08-21 (was .106,
    # then 172.30.185.105 per docs/ROADMAP.md:663, now this). Missing this
    # entry is what actually broke: for a *remote* config (base_url points at
    # another machine), execution.host_label's override never applies — only
    # this dict does. Verify with `curl http://<ip>:11434/api/tags` before
    # trusting an IP here again; it has moved three times now.
}


def _own_lan_ip() -> str | None:
    """This machine's primary LAN address, or None.

    Uses a UDP socket to a public address purely to ask the routing table which
    interface would be used — no packet is sent and nothing needs to be
    reachable.
    """
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None


# Set from config (execution.host_label) or the BENCH_HOST_LABEL env var. A
# machine declaring its own identity is far more robust than inferring it from
# an IP: this laptop moved from 192.168.68.106 to 172.30.185.105 simply by
# changing network, which silently turned every row it produced into an
# "unregistered host" and made the merge refuse to combine them with its own
# earlier rows.
_HOST_LABEL_OVERRIDE: str | None = None


def _infer_host_label(base_url: str) -> str:
    import os
    override = _HOST_LABEL_OVERRIDE or os.environ.get("BENCH_HOST_LABEL")
    # Only applies to a local endpoint: when base_url points at another
    # machine, that machine's identity is what matters, not this one's.
    if override:
        import re as _re
        m = _re.search(r"://([^:/]+)", base_url or "")
        h = m.group(1) if m else ""
        if h in ("localhost", "127.0.0.1", "::1") or h == _own_lan_ip():
            return override
    import re
    m = re.search(r"://([^:/]+)", base_url or "")
    host = m.group(1) if m else ""
    if host in _KNOWN_HOSTS:
        return _KNOWN_HOSTS[host]
    if host in ("localhost", "127.0.0.1", "::1"):
        # localhost means the script host and the inference host are the same
        # machine, so resolve to this machine's own LAN address and look it up:
        # a local run must produce the SAME label as a remote run against this
        # machine, or the dashboard shows one box as two different hosts.
        own = _own_lan_ip()
        if own and own in _KNOWN_HOSTS:
            return _KNOWN_HOSTS[own]
        # Never fall back to platform.node(): that hostname would be written
        # into model_info.host, which is published in the summaries and the
        # dashboard. Describe the machine class instead and leave it to be
        # registered in _KNOWN_HOSTS.
        return f"unregistered host ({platform.system()}/{platform.machine()})"
    return host or "unknown host"


def collect_model_info(model_entries: list[str | dict], default_ctx: int | None = None,
                       native_base_url: str | None = None) -> dict[str, dict]:
    """Query Ollama API for model details and disk size.

    Accepts both string model names and dict entries with 'model' and optional 'ctx'.

    Returns dict of model_name → {params, context_length, size_gb, quantization,
                                  vram_estimate, effective_ctx, official_name, host}.
    """
    import httpx
    # Was hardcoded to localhost, which is wrong for any config pointing
    # base_url at a remote host (e.g. config-m4-remote.yaml, run from a
    # different machine than the one actually serving the models) — every
    # /api/show call silently 404'd against the wrong machine's model store,
    # and every model in that run got empty attributes with no error surfaced.
    resolved_base_url = native_base_url or "http://localhost:11434"
    host_label = _infer_host_label(resolved_base_url)
    base_url = resolved_base_url.rstrip("/").removesuffix("/v1")

    # Disk size straight from the Ollama API rather than reconstructing its
    # on-disk manifest/blob path ourselves — that path assumed Ollama's models
    # live under this process's $HOME (~/.ollama/models/manifests/...), which
    # is only true for a per-user Ollama install. This box (and any systemd-
    # managed install) runs Ollama as a system service storing everything
    # under a different user's directory (e.g. /usr/share/ollama/.ollama),
    # so that lookup always missed, size_gb silently stayed 0, and the
    # resulting "vram_estimate" was just the KV-cache term with no weights
    # component at all (observed: ~1.5GB reported for an 80B model whose
    # real footprint is ~46GB). /api/tags reports the real size regardless
    # of where or as which user Ollama's model store actually lives.
    tags_by_name: dict[str, int] = {}
    try:
        tags_resp = httpx.get(f"{base_url}/api/tags", timeout=10)
        for m in tags_resp.json().get("models", []):
            tags_by_name[m["name"]] = m.get("size", 0)
    except Exception:
        pass

    configs = _multi_model_models(model_entries, default_ctx)
    info = {}
    for spec in configs:
        model_name, effective_ctx = spec["model"], spec["ctx"]
        entry = {"host": host_label}
        try:
            resp = httpx.post(f"{base_url}/api/show", json={"model": model_name}, timeout=10)
            data = resp.json()
            details = data.get("details", {})
            mi = data.get("model_info", {})

            entry["quantization"] = details.get("quantization_level", "?")

            # Ollama's /api/show does not expose the GGUF's own general.name
            # field (visible only in the server's load-time log, e.g.
            # "general.name = DeepSeek R1 Distill Llama 70B") — so the closest
            # real, non-fabricated official name comes from general.basename
            # (+ general.size_label) when present, falling back to
            # details.family (+ parameter_size) when it is not. Both are
            # actual GGUF/Ollama metadata, never guessed from the tag string.
            basename = mi.get("general.basename")
            size_label = mi.get("general.size_label")
            family = details.get("family")
            param_size = details.get("parameter_size")
            if basename:
                entry["official_name"] = f"{basename} {size_label}" if size_label else basename
            elif family:
                entry["official_name"] = f"{family} {param_size}" if param_size else family

            # Context length key varies by architecture
            ctx_key = next((k for k in mi if "context_length" in k), None)
            entry["context_length"] = mi.get(ctx_key) if ctx_key else None

            total = tags_by_name.get(model_name, 0)
            # One decimal is fine for multi-GB models but rounds anything
            # under ~50MB to a flat 0.0, which then reads as falsy and made
            # _classify_params drop the parameter count entirely (Needle 2:
            # a real 14MB / 45M-param model reported as size 0.0, no params).
            # Sub-GB models get the precision they need to stay non-zero.
            gb = total / (1024**3)
            entry["size_gb"] = round(gb, 1) if gb >= 1 else round(gb, 4)
            entry["effective_ctx"] = effective_ctx
            # Both of these are deliberately omitted rather than guessed when
            # the metadata does not support them — a blank tag is honest, a
            # wrong one silently misleads capacity decisions.
            entry.update(_classify_params(mi.get("general.parameter_count"),
                                          entry["size_gb"],
                                          details.get("parameter_size")))
            # Architecture and expert topology: dense vs sparse is a first-order
            # explanation for why a "larger" model can be faster and weaker.
            arch = mi.get("general.architecture")
            if arch:
                entry["architecture"] = arch
            experts = mi.get(f"{arch}.expert_count")
            used = mi.get(f"{arch}.expert_used_count")
            if experts:
                entry["experts"] = experts
                entry["experts_used"] = used
                entry["sparse"] = True
            elif arch and "moe" in arch.lower():
                # The architecture name is authoritative even when the expert
                # keys are absent: MLX conversions drop them, so qwen3_5_moe
                # reported a consistent total parameter count with no expert
                # metadata and was mislabelled "dense". Total params are still
                # shown — the active count simply is not recoverable here.
                entry["sparse"] = True
            vram = _estimate_vram(mi, entry["size_gb"], effective_ctx)
            if vram is not None:
                entry["vram_estimate"] = vram
            if spec["think"] is not None:
                entry["think"] = spec["think"]
        except Exception:
            pass
        # Keyed by label so each variant carries its own ctx/VRAM figures.
        info[spec["label"]] = entry
    return info


def main():
    parser = argparse.ArgumentParser(description="Local LLM Benchmark Runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", help="Override models from config")
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARK_REGISTRY.keys()), help="Benchmarks to run")
    parser.add_argument("--n-samples", type=int, help="Override sample count for all benchmarks")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--output", default=None, help="Output file path (default: results/<timestamp>.json)")
    parser.add_argument(
        "--baseline", metavar="PATH",
        help="Existing results JSON to patch. New results replace matching "
             "model+benchmark entries; everything else is kept. Saves a new "
             "timestamped file so the original is never modified.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    global _HOST_LABEL_OVERRIDE
    _HOST_LABEL_OVERRIDE = cfg.get("execution", {}).get("host_label")
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        api_key=cfg["ollama"].get("api_key", "ollama"),
        timeout=cfg["ollama"].get("timeout", 120),
    )

    # Build judge client and determine judge model(s)
    judge_cfg = cfg.get("judge", {})
    judge_provider = judge_cfg.get("provider", "ollama")

    use_ensemble = (judge_provider == "ensemble")
    ensemble_models = judge_cfg.get("ensemble_models", []) if use_ensemble else []

    if judge_provider == "openai":
        judge_client = build_client(judge_cfg)
        judge_model = resolve_env_vars(judge_cfg.get("cloud_model", "deepseek-chat"))
    elif judge_provider == "opencode":
        judge_client = OpenCodeClient(
            model=judge_cfg.get("cloud_model", "opencode/deepseek-v4-flash-free"),
            timeout=judge_cfg.get("timeout", 120),
        )
        judge_model = judge_cfg.get("cloud_model", "opencode/deepseek-v4-flash-free")
    else:
        # "ollama" or "ensemble" — both use local Ollama client
        judge_client = client
        judge_model = judge_cfg.get("ollama_single_model", "llama3.1:8b")

    if args.list_models:
        models = list_ollama_models(cfg["ollama"]["base_url"])
        console.print("\n[bold]Available Ollama models:[/bold]")
        for m in models:
            console.print(f"  {m}")
        return

    model_entries = args.models or cfg.get("models", [])
    if args.models:
        # Resolve CLI names against config entries so variants are selectable:
        # `--models "muse-glimmer:30b-mlx +think"` picks up that entry's think
        # setting instead of silently running it with the benchmark default.
        default_ctx_lookup = cfg.get("ollama", {}).get("default_ctx", 4096)
        by_key: dict[str, str | dict] = {}
        for raw in cfg.get("models", []):
            spec = resolve_model_entry(raw, default_ctx_lookup)
            by_key[spec["label"]] = raw
            by_key.setdefault(spec["model"], raw)
        model_entries = [by_key.get(name, name) for name in args.models]
    if not model_entries:
        console.print("[red]No models specified. Use --models or set models in config.yaml[/red]")
        sys.exit(1)
    # Resolve mixed string/dict entries into (name, ctx) pairs and a name list
    default_ctx = cfg.get("ollama", {}).get("default_ctx", 4096)
    model_configs = _multi_model_models(model_entries, default_ctx)
    models = [s["label"] for s in model_configs]
    spec_by_label = {s["label"]: s for s in model_configs}

    # Determine which benchmarks to run
    # config.yaml uses group names; expand them to individual registry keys
    BENCH_GROUPS = {
        "reasoning": ["mmlu", "arc"],
        "math":      ["gsm8k"],
        "coding":    ["humaneval", "mbpp"],
        "sql":       ["spider"],
    }
    bench_cfg = cfg.get("benchmarks", {})
    raw_selected = args.benchmarks or [k for k, v in bench_cfg.items() if v.get("enabled", True)]
    selected = []
    for name in raw_selected:
        selected.extend(BENCH_GROUPS.get(name, [name]))
    selected = [b for b in selected if b in BENCHMARK_REGISTRY]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect run metadata once
    run_meta = collect_run_metadata(cfg)
    run_start = datetime.now().isoformat(timespec="seconds")

    # Load baseline if given — new results will patch matching model+benchmark entries.
    baseline_results = []
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            console.print(f"[red]--baseline file not found: {baseline_path}[/red]")
            sys.exit(1)
        baseline_results = json.load(baseline_path.open())
        console.print(f"[dim]Baseline: {len(baseline_results)} results loaded from {baseline_path.name}[/dim]")

    # Accumulates only the results from THIS run; merged with baseline at the end.
    new_results = []

    config_models = [m if isinstance(m, str) else m["model"] for m in cfg.get("models", [])]
    report_path = Path(cfg["output"]["dir"]) / "report.html"
    # A partial run with no baseline would otherwise overwrite an aggregated
    # dashboard with just this run's models. Divert to a per-run file instead.
    if not args.baseline and report_path.exists():
        report_path = Path(cfg["output"]["dir"]) / f"report_{run_id}.html"
        console.print(f"[dim]No --baseline: writing {report_path.name} to preserve "
                      f"the aggregated report.html[/dim]")
    try:
        template_html = find_template().read_text()
    except FileNotFoundError:
        template_html = None

    def _merged() -> list:
        """Baseline minus any (model, benchmark) pairs being re-run, plus new results."""
        results_list = baseline_results.get("results", baseline_results) if isinstance(baseline_results, dict) else baseline_results
        if not results_list:
            return new_results
        patching = {(r["model"], r["benchmark"]) for r in new_results}
        kept = [r for r in results_list if (r["model"], r["benchmark"]) not in patching]
        return kept + new_results

    # Collect model info (params, context, size) once before the run starts
    model_info = collect_model_info(model_entries, default_ctx, cfg.get("ollama", {}).get("base_url"))

    # Merge baseline model_info for models not in current run (--models limits scope)
    if isinstance(baseline_results, dict) and "metadata" in baseline_results:
        for k, v in baseline_results["metadata"].get("model_info", {}).items():
            if k not in model_info:
                model_info[k] = v

    def _checkpoint(combined: list) -> None:
        """Write results so far to the run's output file (overwritten each time)."""
        out = Path(args.output) if args.output else Path(cfg["output"]["dir"]) / f"{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "metadata": {"run_id": run_id, "run_start": run_start, "partial": True,
                         "model_info": model_info, **run_meta},
            "results": combined,
        }, indent=2, default=str))

    def _refresh_report(is_live: bool) -> None:
        combined = _merged()
        if not template_html or not combined:
            return
        try:
            data = aggregate(combined, all_models=config_models or None,
                             sample_counts=sample_counts, model_info=model_info)
            html = template_html
            html = html.replace("// __INJECT_DATA__",
                                f"const BENCHMARK_DATA = {__import__('json').dumps(data, ensure_ascii=False)};")
            html = html.replace("<!-- __META_REFRESH__ -->",
                                '<meta http-equiv="refresh" content="60">' if is_live else '')
            report_path.write_text(html)
        except Exception as e:
            console.print(f"[dim]report update skipped: {e}[/dim]")

    # Determine parallel execution config
    exec_cfg = cfg.get("execution", {})
    parallel_models = exec_cfg.get("parallel_models", False)
    max_workers = exec_cfg.get("max_workers", 2)

    # Track sample counts per (model, benchmark) for dashboard/terminal display
    sample_counts: dict[tuple[str, str], int] = {}
    _sc_lock = __import__("threading").Lock()

    def _run_model(label: str) -> list[dict]:
        """Run all selected benchmarks for one model variant.

        `label` is the variant identity (may include a "+think" suffix);
        `model_name` is what Ollama is actually asked for.
        """
        spec = spec_by_label[label]
        model_name, model_ctx = spec["model"], spec["ctx"]
        model_results: list[dict] = []
        console.print(f"\n[bold cyan]═══ Model: {label} ═══[/bold cyan]")
        for bench_name in selected:
            cfg_key = next((g for g, members in BENCH_GROUPS.items() if bench_name in members), bench_name)
            bcfg = {**cfg["ollama"], **bench_cfg.get(cfg_key, {})}
            bcfg["judge_model"] = judge_model
            bcfg["judge_client"] = judge_client
            bcfg["use_ensemble"] = use_ensemble
            bcfg["ensemble_models"] = ensemble_models
            bcfg["memory_guard"] = exec_cfg.get("memory_guard", {"enabled": True})
            # A per-model think setting wins over the benchmark's own default,
            # which is how the same model runs as think-on and think-off rows.
            if spec["think"] is not None:
                bcfg["think"] = spec["think"]

            bench_class = BENCHMARK_REGISTRY[bench_name]
            bench = bench_class(client=client, config=bcfg)

            n_samples = args.n_samples or bcfg.get("n_samples", 20)
            console.print(f"  [yellow]Running {bench_name}[/yellow] ({n_samples} samples)...")

            def _on_sample(i, total, r):
                mark = "[green]✓[/green]" if r.get("passed") else "[red]✗[/red]"
                swap = "💀" if "swap" in (r.get("error") or "") else ""
                tps = f"  {r['tok_per_sec']:.1f} t/s" if r.get("tok_per_sec") else ""
                err = (r.get('exec_error') or r.get('error') or "")[:80]
                err_str = f"  {err}" if err else ""
                console.print(f"    {swap}{mark} {i}/{total}{tps}{err_str}", highlight=False)

            try:
                results = bench.run(model=model_name, n_samples=n_samples, on_sample=_on_sample, ctx=model_ctx)
                passed = sum(1 for r in results if r.get("passed"))
                score = sum(r.get("score", 0) for r in results) / max(len(results), 1)
                console.print(f"  [green]✓[/green] {bench_name}: {passed}/{len(results)} passed ({score:.1%})")
                # Add run metadata to every result record
                for r in results:
                    r["model"] = label      # variant identity, not the raw model
                    r["run_id"] = run_id
                    r["run_start"] = run_start
                    r["git_sha"] = run_meta["git_sha"]
                    r["git_branch"] = run_meta["git_branch"]
                    r["hardware"] = run_meta["hardware"]
                    r["ollama_version"] = run_meta["ollama_version"]
                with _sc_lock:
                    sample_counts[(label, bench_name)] = len(results)
                model_results.extend(results)
                # Checkpoint after every benchmark, not just every model — a
                # model's full benchmark set can run for hours, and a run
                # killed partway through one used to lose everything that
                # model had already finished (cost 300 samples twice on the
                # M4, both to an interrupted host). Skipped in the parallel
                # path: concurrent models writing to the same file would race.
                if not parallel_models:
                    try:
                        _checkpoint(new_results + model_results)
                    except Exception as e:
                        console.print(f"[dim]checkpoint skipped: {e}[/dim]")
            except MemorySwapAbort as e:
                swap_msg = str(e)
                console.print(f"  [red]💀 {bench_name} aborted — memory swap detected: {swap_msg[:120]}[/red]")
                swap_results = e.partial_results
                for r in swap_results:
                    r["model"] = label      # variant identity, not the raw model
                    r["run_id"] = run_id
                    r["run_start"] = run_start
                    r["git_sha"] = run_meta["git_sha"]
                    r["git_branch"] = run_meta["git_branch"]
                    r["hardware"] = run_meta["hardware"]
                    r["ollama_version"] = run_meta["ollama_version"]
                with _sc_lock:
                    sample_counts[(label, bench_name)] = len(swap_results)
                model_results.extend(swap_results)
                # Skip remaining benchmarks for this model — all will swap too
                console.print(f"  [red]💀 Skipping remaining benchmarks for {label} (memory swap)[/red]")
                break
            except Exception as e:
                console.print(f"  [red]✗ {bench_name} failed: {e}[/red]")
        return model_results

    if parallel_models:
        console.print(f"[bold]Parallel model execution ({max_workers} workers)[/bold]")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_model, model): model for model in models}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    new_results.extend(future.result())
                except Exception as e:
                    console.print(f"[red]Model {model} failed: {e}[/red]")
    else:
        for model in models:
            results = _run_model(model)
            # Evict before the next model loads. Ollama keeps a model warm for
            # ~5 minutes by default, so back-to-back large models would both be
            # resident and drive the machine into swap.
            if client.unload(model):
                console.print(f"  [dim]unloaded {model}[/dim]")
            new_results.extend(results)
            # Checkpoint after every model. A long run that dies partway used to
            # lose every completed model, because results were only written at
            # the very end.
            try:
                _checkpoint(_merged())
            except Exception as e:
                console.print(f"[dim]checkpoint skipped: {e}[/dim]")
            _refresh_report(is_live=model != models[-1])
            console.print(f"  [dim]report.html → step {models.index(model)+1}/{len(models)}[/dim]")

    if not new_results:
        console.print("[red]No results collected.[/red]")
        sys.exit(1)

    run_end = datetime.now().isoformat(timespec="seconds")

    all_results = _merged()
    if args.baseline:
        console.print(f"[dim]Patched {len(new_results)} results into baseline "
                      f"({len(all_results)} total after merge)[/dim]")

    # Wrap results with metadata
    wrapped = {
        "metadata": {
            "run_id": run_id,
            "run_start": run_start,
            "run_end": run_end,
            "duration_seconds": (datetime.fromisoformat(run_end) - datetime.fromisoformat(run_start)).total_seconds(),
            "model_info": model_info,
            **run_meta,
        },
        "results": all_results,
    }

    saved = save_results(wrapped, cfg["output"]["dir"], run_id, output_path=args.output)
    console.print(f"\n[dim]Results saved to {saved}[/dim]")

    _refresh_report(is_live=False)
    console.print(f"[dim]Dashboard → file://{report_path.resolve()}[/dim]")

    print_summary(all_results, sample_counts)


if __name__ == "__main__":
    main()
