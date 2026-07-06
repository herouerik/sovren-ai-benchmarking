import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def save_results(results: list[dict], output_dir: str, run_id: str):
    path = Path(output_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def print_summary(results: list[dict]):
    df = pd.DataFrame(results)
    if df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    console.print(Panel.fit("[bold]Benchmark Results Summary[/bold]", box=box.DOUBLE))

    # Per-model per-benchmark accuracy
    pivot = df.groupby(["model", "benchmark"])["score"].mean().unstack(fill_value=0.0)

    table = Table(box=box.SIMPLE_HEAD, show_footer=True)
    table.add_column("Model", style="bold cyan", footer="Mean")

    benchmarks = list(pivot.columns)
    for b in benchmarks:
        table.add_column(b.upper(), justify="right", footer=f"{pivot[b].mean():.2%}")

    table.add_column("OVERALL", justify="right", style="bold green", footer=f"{df['score'].mean():.2%}")

    for model in pivot.index:
        row_scores = [f"{pivot.loc[model, b]:.2%}" for b in benchmarks]
        overall = f"{pivot.loc[model].mean():.2%}"
        table.add_row(model, *row_scores, overall)

    console.print(table)

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
