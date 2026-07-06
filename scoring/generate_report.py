#!/usr/bin/env python3
"""Generate an HTML benchmark report from a results JSON file.

Usage:
    python scoring/generate_report.py results/20260702_143022.json
    python scoring/generate_report.py results/20260702_143022.json --output results/report.html
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def aggregate(raw: list[dict]) -> dict:
    """Aggregate per-sample records into a dashboard-ready payload."""
    models = sorted(set(r["model"] for r in raw))
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in raw))  # preserve order

    scores: dict[str, dict[str, float]] = {}
    speeds: dict[str, float] = {}

    for model in models:
        mrows = [r for r in raw if r["model"] == model]
        scores[model] = {}
        for bench in benchmarks:
            brows = [r for r in mrows if r["benchmark"] == bench]
            if brows:
                scores[model][bench] = sum(r["score"] for r in brows) / len(brows)
        toks = [r["tok_per_sec"] for r in mrows if r.get("tok_per_sec", 0) > 0]
        speeds[model] = sum(toks) / len(toks) if toks else 0.0

    run_id = raw[0].get("run_id", "UNKNOWN") if raw else "UNKNOWN"

    return {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_samples": len(raw),
        "models": models,
        "benchmarks": benchmarks,
        "scores": scores,
        "speeds": speeds,
    }


def find_template() -> Path:
    here = Path(__file__).parent
    candidates = [
        here / "benchmark_dashboard.html",
        here.parent / "benchmark_dashboard.html",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "benchmark_dashboard.html not found. Expected it alongside this script in scoring/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("input", help="Path to results JSON file")
    parser.add_argument("--output", default=None, help="Output HTML path (default: same dir as input, report.html)")
    parser.add_argument("--template", default=None, help="Path to dashboard HTML template")
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Results JSON must be a list of per-sample records.")

    data = aggregate(raw)

    template_path = Path(args.template) if args.template else find_template()
    with open(template_path) as f:
        html = f.read()

    injection = f"const BENCHMARK_DATA = {json.dumps(data, indent=2, ensure_ascii=False)};"
    html = html.replace("// __INJECT_DATA__", injection)

    output_path = Path(args.output) if args.output else input_path.parent / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report written → {output_path}")
    print(f"Models: {len(data['models'])}  Benchmarks: {len(data['benchmarks'])}  Samples: {data['total_samples']}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
