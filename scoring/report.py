import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def save_results(results, output_dir: str, run_id: str):
    """Save results to a JSON file. Accepts a list of records or a wrapped {metadata, results} dict."""
    path = Path(output_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def print_summary(results: list[dict], sample_counts: dict[tuple[str, str], int] | None = None):
    df = pd.DataFrame(results)
    if df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    console.print(Panel.fit("[bold]Benchmark Results Summary[/bold]", box=box.DOUBLE))

    # Detect swap-affected (model, benchmark) pairs
    swap_models: set[str] = set()
    swap_benches: dict[tuple[str, str], bool] = {}
    if "error" in df.columns:
        swap_rows = df[df["error"].notna() & df["error"].str.contains("swap_abort", na=False)]
        for _, r in swap_rows.iterrows():
            key = (r["model"], r["benchmark"])
            swap_benches[key] = True
            swap_models.add(r["model"])

    # Per-model per-benchmark accuracy
    pivot = df.groupby(["model", "benchmark"])["score"].mean().unstack(fill_value=0.0)

    table = Table(box=box.SIMPLE_HEAD, show_footer=True)
    table.add_column("Model", style="bold cyan", footer="Mean")

    benchmarks = list(pivot.columns)
    for b in benchmarks:
        table.add_column(b.upper(), justify="right", footer=f"{pivot[b].mean():.2%}")

    overall_style = "bold green"
    if swap_models:
        overall_style = "bold yellow"
    table.add_column("OVERALL", justify="right", style=overall_style, footer=f"{df['score'].mean():.2%}")

    for model in pivot.index:
        row_scores = []
        for b in benchmarks:
            score_val = pivot.loc[model, b]
            n = sample_counts.get((model, b)) if sample_counts else None
            is_swap = swap_benches.get((model, b), False)
            if is_swap:
                score_str = f"💀 {score_val:.2%} n={n}" if n else f"💀 {score_val:.2%}"
            else:
                score_str = f"{score_val:.2%}" + (f" n={n}" if n else "")
            row_scores.append(score_str)
        overall = f"{pivot.loc[model].mean():.2%}"
        table.add_row(model, *row_scores, overall)

    console.print(table)

    # Swap warning
    if swap_models:
        console.print(f"\n[red]💀 Models with insufficient memory:[/red]")
        for m in sorted(swap_models):
            affected = [b for (mod, b) in swap_benches if mod == m]
            console.print(f"  [red]💀 {m}[/red] — aborted {len(affected)} benchmark(s): "
                          f"{', '.join(affected)}")
            console.print(f"     [dim]Overall score is not comparable — includes "
                          f"failed samples from swap-aborted benchmarks[/dim]")

    # Speed table
    speed_table = Table(title="Inference Speed (tok/s)", box=box.SIMPLE_HEAD)
    speed_table.add_column("Model", style="cyan")
    speed_table.add_column("Benchmark")
    speed_table.add_column("Avg tok/s", justify="right")
    speed_table.add_column("Avg latency (s)", justify="right")

    speed = df[df["tok_per_sec"] > 0].groupby(["model", "benchmark"])[["tok_per_sec", "elapsed"]].mean()
    for (model, bench), row in speed.iterrows():
        speed_table.add_row(model, bench, f"{row['tok_per_sec']:.1f}", f"{row['elapsed']:.1f}s")

    console.print(speed_table)

    # Error summary
    errors = df[df["error"].notna() & (df["error"] != "")]
    if not errors.empty:
        console.print(f"\n[red]Errors: {len(errors)} failed calls[/red]")
        for _, e in errors.head(5).iterrows():
            console.print(f"  {e['model']} / {e['benchmark']}: {e['error']}")
