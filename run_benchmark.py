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
        "node": platform.node(),
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


def resolve_model_entry(entry: str | dict, default_ctx: int | None = None) -> tuple[str, int]:
    """Given a model config entry (string or dict), return (model_name, ctx).

    The ctx is resolved: per-model > global default_ctx > 4096 hard fallback.
    This ensures every model has a deterministic ctx used for API calls, VRAM
    estimates, and dashboard display — no mismatch between what's shown and used.

    Example:
        "llama3.2:3b" → ("llama3.2:3b", 32768)  (using global default)
        {"model": "qwen3-coder:latest", "ctx": 16384} → ("qwen3-coder:latest", 16384)
    """
    if isinstance(entry, dict):
        ctx = entry.get("ctx", default_ctx)
    else:
        ctx = default_ctx
    return entry if isinstance(entry, str) else entry["model"], ctx or 4096


def _estimate_vram(mi: dict, size_gb: float, ctx: int) -> float:
    """Estimate VRAM in GB at the given context length."""
    arch = mi.get("general.architecture", "")
    prefix = f"{arch}."
    def g(key):
        return mi.get(prefix + key) or mi.get(key)

    weights_gb = size_gb * 1.05
    n_layers = g("block_count")
    n_heads = g("attention.head_count")
    n_kv_heads = g("attention.head_count_kv") or n_heads
    emb_dim = g("embedding_length")
    if not all([n_layers, n_heads, emb_dim]):
        return round(weights_gb, 1)

    per_head_dim = emb_dim // n_heads
    kv_bytes_per_token = 4 * n_kv_heads * per_head_dim * n_layers
    kv_gb = (ctx * kv_bytes_per_token) / (1024**3)
    return round(weights_gb + kv_gb, 1)


def _model_name(entry: str | dict) -> str:
    """Resolve a model config entry to its model name string."""
    return entry if isinstance(entry, str) else entry["model"]


def _multi_model_models(raw: list, default_ctx: int | None = None) -> list[tuple[str, int]]:
    """Resolve models list (strings and dicts) to (name, ctx) pairs.

    Every entry gets a resolved ctx: per-model > global default > 4096.
    """
    return [resolve_model_entry(m, default_ctx) for m in raw]


def collect_model_info(model_entries: list[str | dict], default_ctx: int | None = None) -> dict[str, dict]:
    """Query Ollama API for model details and disk size.

    Accepts both string model names and dict entries with 'model' and optional 'ctx'.

    Returns dict of model_name → {params, context_length, size_gb, quantization,
                                  vram_estimate, effective_ctx}.
    """
    import httpx
    base_url = "http://localhost:11434"
    MANIFEST_DIR = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    BLOBS_DIR = Path.home() / ".ollama" / "models" / "blobs"

    configs = _multi_model_models(model_entries, default_ctx)
    info = {}
    for model_name, effective_ctx in configs:
        entry = {}
        try:
            resp = httpx.post(f"{base_url}/api/show", json={"model": model_name}, timeout=10)
            data = resp.json()
            details = data.get("details", {})
            mi = data.get("model_info", {})

            entry["params"] = details.get("parameter_size", "?")
            entry["quantization"] = details.get("quantization_level", "?")

            # Context length key varies by architecture
            ctx_key = next((k for k in mi if "context_length" in k), None)
            entry["context_length"] = mi.get(ctx_key) if ctx_key else None

            # Disk size from blob files
            name_part, tag_part = (model_name.rsplit(":", 1) + ["latest"])[:2]
            manifest_path = MANIFEST_DIR / name_part / tag_part
            total = 0
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                for layer in manifest.get("layers", []):
                    digest = layer["digest"]
                    blob_name = digest.replace(":", "-")
                    blob_path = BLOBS_DIR / blob_name
                    if blob_path.exists():
                        total += blob_path.stat().st_size
                    else:
                        total += layer.get("size", 0)
            entry["size_gb"] = round(total / (1024**3), 1)
            entry["effective_ctx"] = effective_ctx
            entry["vram_estimate"] = _estimate_vram(mi, entry["size_gb"], effective_ctx)
        except Exception:
            pass
        info[model_name] = entry
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
    if not model_entries:
        console.print("[red]No models specified. Use --models or set models in config.yaml[/red]")
        sys.exit(1)
    # Resolve mixed string/dict entries into (name, ctx) pairs and a name list
    default_ctx = cfg.get("ollama", {}).get("default_ctx", 4096)
    model_configs = _multi_model_models(model_entries, default_ctx)
    models = [name for name, _ in model_configs]
    model_ctx_map = dict(model_configs)

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

    config_models = cfg.get("models", [])
    report_path = Path(cfg["output"]["dir"]) / "report.html"
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
    model_info = collect_model_info(model_entries, default_ctx)

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

    def _run_model(model_name: str) -> list[dict]:
        """Run all selected benchmarks for one model. Used both sequentially and in parallel."""
        model_ctx = model_ctx_map.get(model_name)
        model_results: list[dict] = []
        console.print(f"\n[bold cyan]═══ Model: {model_name} ═══[/bold cyan]")
        for bench_name in selected:
            cfg_key = next((g for g, members in BENCH_GROUPS.items() if bench_name in members), bench_name)
            bcfg = {**cfg["ollama"], **bench_cfg.get(cfg_key, {})}
            bcfg["judge_model"] = judge_model
            bcfg["judge_client"] = judge_client
            bcfg["use_ensemble"] = use_ensemble
            bcfg["ensemble_models"] = ensemble_models
            bcfg["memory_guard"] = exec_cfg.get("memory_guard", {"enabled": True})

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
                    r["run_id"] = run_id
                    r["run_start"] = run_start
                    r["git_sha"] = run_meta["git_sha"]
                    r["git_branch"] = run_meta["git_branch"]
                    r["hardware"] = run_meta["hardware"]
                    r["ollama_version"] = run_meta["ollama_version"]
                with _sc_lock:
                    sample_counts[(model_name, bench_name)] = len(results)
                model_results.extend(results)
            except MemorySwapAbort as e:
                swap_msg = str(e)
                console.print(f"  [red]💀 {bench_name} aborted — memory swap detected: {swap_msg[:120]}[/red]")
                swap_results = e.partial_results
                for r in swap_results:
                    r["run_id"] = run_id
                    r["run_start"] = run_start
                    r["git_sha"] = run_meta["git_sha"]
                    r["git_branch"] = run_meta["git_branch"]
                    r["hardware"] = run_meta["hardware"]
                    r["ollama_version"] = run_meta["ollama_version"]
                with _sc_lock:
                    sample_counts[(model_name, bench_name)] = len(swap_results)
                model_results.extend(swap_results)
                # Skip remaining benchmarks for this model — all will swap too
                console.print(f"  [red]💀 Skipping remaining benchmarks for {model_name} (memory swap)[/red]")
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
            new_results.extend(results)
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

    saved = save_results(wrapped, cfg["output"]["dir"], run_id)
    console.print(f"\n[dim]Results saved to {saved}[/dim]")

    _refresh_report(is_live=False)
    console.print(f"[dim]Dashboard → file://{report_path.resolve()}[/dim]")

    print_summary(all_results, sample_counts)


if __name__ == "__main__":
    main()
