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
import sys
import yaml
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

    # Build separate judge client if a cloud provider is configured
    judge_cfg = cfg.get("judge", {})
    judge_provider = judge_cfg.get("provider", "ollama")
    if judge_provider == "openai":
        judge_client = build_client(judge_cfg)
        judge_model = resolve_env_vars(judge_cfg.get("model", "deepseek-chat"))
    elif judge_provider == "opencode":
        # OpenCode CLI — free, auth-free, key-free (wraps opencode binary)
        judge_client = OpenCodeClient(
            model=judge_cfg.get("model", "opencode/deepseek-v4-flash-free"),
            timeout=judge_cfg.get("timeout", 120),
        )
        judge_model = judge_cfg.get("model", "opencode/deepseek-v4-flash-free")
    else:
        judge_client = client
        judge_model = cfg.get("judge_model", "llama3.1:8b")

    if args.list_models:
        models = list_ollama_models(cfg["ollama"]["base_url"])
        console.print("\n[bold]Available Ollama models:[/bold]")
        for m in models:
            console.print(f"  {m}")
        return

    models = args.models or cfg.get("models", [])
    if not models:
        console.print("[red]No models specified. Use --models or set models in config.yaml[/red]")
        sys.exit(1)

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
        if not baseline_results:
            return new_results
        patching = {(r["model"], r["benchmark"]) for r in new_results}
        kept = [r for r in baseline_results if (r["model"], r["benchmark"]) not in patching]
        return kept + new_results

    def _refresh_report(is_live: bool) -> None:
        combined = _merged()
        if not template_html or not combined:
            return
        try:
            data = aggregate(combined, all_models=config_models or None)
            html = template_html
            html = html.replace("// __INJECT_DATA__",
                                f"const BENCHMARK_DATA = {__import__('json').dumps(data, ensure_ascii=False)};")
            html = html.replace("<!-- __META_REFRESH__ -->",
                                '<meta http-equiv="refresh" content="60">' if is_live else '')
            report_path.write_text(html)
        except Exception as e:
            console.print(f"[dim]report update skipped: {e}[/dim]")

    total_steps = len(models) * len(selected)
    step = 0

    for model in models:
        console.print(f"\n[bold cyan]═══ Model: {model} ═══[/bold cyan]")
        is_last_model = (model == models[-1])

        for bench_name in selected:
            step += 1
            is_last_bench = (bench_name == selected[-1])
            is_last_step  = is_last_model and is_last_bench

            # resolve group name for config lookup (e.g. mmlu → reasoning)
            cfg_key = next((g for g, members in BENCH_GROUPS.items() if bench_name in members), bench_name)
            bcfg = {**cfg["ollama"], **bench_cfg.get(cfg_key, {})}
            bcfg["judge_model"] = judge_model
            bcfg["judge_client"] = judge_client

            bench_class = BENCHMARK_REGISTRY[bench_name]
            bench = bench_class(client=client, config=bcfg)

            n_samples = args.n_samples or bcfg.get("n_samples", 20)
            console.print(f"  [yellow]Running {bench_name}[/yellow] ({n_samples} samples)...")

            def _on_sample(i, total, r):
                mark = "[green]✓[/green]" if r.get("passed") else "[red]✗[/red]"
                tps = f"  {r['tok_per_sec']:.1f} t/s" if r.get("tok_per_sec") else ""
                err = f"  {r['exec_error'][:80]}" if r.get("exec_error") else ""
                console.print(f"    {mark} {i}/{total}{tps}{err}", highlight=False)

            try:
                results = bench.run(model=model, n_samples=n_samples, on_sample=_on_sample)
                passed = sum(1 for r in results if r.get("passed"))
                score = sum(r.get("score", 0) for r in results) / max(len(results), 1)
                console.print(f"  [green]✓[/green] {bench_name}: {passed}/{len(results)} passed ({score:.1%})")
                new_results.extend(results)
            except Exception as e:
                console.print(f"  [red]✗ {bench_name} failed: {e}[/red]")

            _refresh_report(is_live=not is_last_step)
            console.print(f"  [dim]report.html → step {step}/{total_steps}[/dim]")

    if not new_results:
        console.print("[red]No results collected.[/red]")
        sys.exit(1)

    all_results = _merged()
    if args.baseline:
        console.print(f"[dim]Patched {len(new_results)} results into baseline "
                      f"({len(all_results)} total after merge)[/dim]")

    saved = save_results(all_results, cfg["output"]["dir"], run_id)
    console.print(f"\n[dim]Results saved to {saved}[/dim]")

    _refresh_report(is_live=False)
    console.print(f"[dim]Dashboard → file://{report_path.resolve()}[/dim]")

    print_summary(all_results)


if __name__ == "__main__":
    main()
