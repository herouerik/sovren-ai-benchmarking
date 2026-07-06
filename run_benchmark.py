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
import sys
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from harness.client import OllamaClient
from benchmarks.reasoning import MMLUBenchmark, ARCBenchmark
from benchmarks.math import GSM8KBenchmark
from benchmarks.coding import HumanEvalBenchmark, MBPPBenchmark
from benchmarks.sql import SpiderBenchmark
from benchmarks.philosophical import PhilosophicalBenchmark
from scoring.report import save_results, print_summary

console = Console()

BENCHMARK_REGISTRY = {
    "mmlu":        MMLUBenchmark,
    "arc":         ARCBenchmark,
    "gsm8k":       GSM8KBenchmark,
    "humaneval":   HumanEvalBenchmark,
    "mbpp":        MBPPBenchmark,
    "spider":      SpiderBenchmark,
    "philosophical": PhilosophicalBenchmark,
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        api_key=cfg["ollama"].get("api_key", "ollama"),
        timeout=cfg["ollama"].get("timeout", 120),
    )

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
    bench_cfg = cfg.get("benchmarks", {})
    selected = args.benchmarks or [k for k, v in bench_cfg.items() if v.get("enabled", True)]
    selected = [b for b in selected if b in BENCHMARK_REGISTRY]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for model in models:
        console.print(f"\n[bold cyan]═══ Model: {model} ═══[/bold cyan]")

        for bench_name in selected:
            bcfg = {**cfg["ollama"], **bench_cfg.get(bench_name, {})}
            bcfg["judge_model"] = cfg.get("judge_model", "qwen3:32b")

            bench_class = BENCHMARK_REGISTRY[bench_name]
            bench = bench_class(client=client, config=bcfg)

            n_samples = args.n_samples or bcfg.get("n_samples", 20)
            console.print(f"  [yellow]Running {bench_name}[/yellow] ({n_samples} samples)...")

            try:
                results = bench.run(model=model, n_samples=n_samples)
                passed = sum(1 for r in results if r.get("passed"))
                score = sum(r.get("score", 0) for r in results) / max(len(results), 1)
                console.print(f"  [green]✓[/green] {bench_name}: {passed}/{len(results)} passed ({score:.1%})")
                all_results.extend(results)
            except Exception as e:
                console.print(f"  [red]✗ {bench_name} failed: {e}[/red]")

    if not all_results:
        console.print("[red]No results collected.[/red]")
        sys.exit(1)

    output_path = args.output or str(Path(cfg["output"]["dir"]) / f"{run_id}.json")
    saved = save_results(all_results, cfg["output"]["dir"], run_id)
    console.print(f"\n[dim]Results saved to {saved}[/dim]")

    print_summary(all_results)


if __name__ == "__main__":
    main()
