#!/usr/bin/env python3
"""Merge multiple benchmark summary aggregates into one dashboard payload.

Each *.summary.json produced by run_benchmark.py --summary is an *aggregate*
(a small, diffable form of a run: scores/speeds/sample_sizes/model_info, no
per-sample records). This script merges several of those aggregates (e.g. the
M4 baseline plus a GPU-server run) into a single payload, then regenerates the
HTML dashboard so models from both backends render side by side.

Usage:
    .venv/bin/python merge_summaries.py \
        results/baseline_v0.2.summary.json \
        /tmp/gpu-val.summary.json \
        --output results/report_merged.html \
        --summary results/merged.summary.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _union(*lists):
    seen = set()
    out = []
    for lst in lists:
        for item in lst or []:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _merge_scores(scores_list):
    """scores[model][bench] -> float. Later file wins for the same (model, bench)."""
    merged = {}
    for scores in scores_list:
        for model, bench_scores in (scores or {}).items():
            bucket = merged.setdefault(model, {})
            for bench, score in bench_scores.items():
                bucket[bench] = score
    return merged


def _merge_sample_sizes(sizes_list):
    """sample_sizes[model][bench] -> int. Later file wins."""
    merged = {}
    for sizes in sizes_list:
        for model, bench_sizes in (sizes or {}).items():
            bucket = merged.setdefault(model, {})
            for bench, n in bench_sizes.items():
                bucket[bench] = n
    return merged


def _merge_speeds(speeds_list):
    """speeds[model] -> tok/s. Later file wins."""
    merged = {}
    for speeds in speeds_list:
        for model, tok_s in (speeds or {}).items():
            merged[model] = tok_s
    return merged


def _merge_timestamps(ts_list):
    """model_timestamps[model] -> iso ts. Max wins (latest completed)."""
    merged = {}
    for ts_map in ts_list:
        for model, ts in (ts_map or {}).items():
            if model not in merged or ts > merged[model]:
                merged[model] = ts
    return merged


def _merge_swap_benches(sw_list):
    """swap_benches[model] -> [bench,...]. Union per model."""
    merged = {}
    for sw in sw_list:
        for model, benches in (sw or {}).items():
            bucket = merged.setdefault(model, [])
            for b in benches or []:
                if b not in bucket:
                    bucket.append(b)
    return merged


def _merge_model_info(info_list):
    """model_info[model] -> dict. Later file wins per model."""
    merged = {}
    for info in info_list:
        for model, meta in (info or {}).items():
            merged[model] = meta
    return merged


def merge_summaries(summaries: list[dict]) -> dict:
    """Merge aggregate payloads. Later entries in the list take precedence."""
    if not summaries:
        raise ValueError("No summaries to merge")

    merged = {
        "run_id": "+".join(s.get("run_id", "?") for s in summaries),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "models": _union(*(s.get("models", []) for s in summaries)),
        "benchmarks": _union(*(s.get("benchmarks", []) for s in summaries)),
        "scores": _merge_scores([s.get("scores", {}) for s in summaries]),
        "speeds": _merge_speeds([s.get("speeds", {}) for s in summaries]),
        "sample_sizes": _merge_sample_sizes([s.get("sample_sizes", {}) for s in summaries]),
        "model_timestamps": _merge_timestamps([s.get("model_timestamps", {}) for s in summaries]),
        "swap_benches": _merge_swap_benches([s.get("swap_benches", {}) for s in summaries]),
        "model_info": _merge_model_info([s.get("model_info", {}) for s in summaries]),
    }

    # total_samples = sum over per-model per-bench sample sizes (excludes speed)
    total = 0
    for model, benches in merged["sample_sizes"].items():
        total += sum(n for b, n in benches.items() if b != "speed")
    merged["total_samples"] = total

    all_models = _union(*(s.get("all_models", []) for s in summaries))
    if all_models:
        merged["all_models"] = all_models
    return merged


def render_dashboard(payload: dict, template_path: Path | str, output_path: Path | str) -> None:
    """Inject the merged payload into the dashboard template and write the HTML."""
    template_path = Path(template_path)
    output_path = Path(output_path)
    html = template_path.read_text()
    injection = f"const BENCHMARK_DATA = {json.dumps(payload, indent=2, ensure_ascii=False)};"
    if "// __INJECT_DATA__" not in html:
        raise ValueError(f"template {template_path} has no // __INJECT_DATA__ marker")
    html = html.replace("// __INJECT_DATA__", injection)
    html = html.replace("<!-- __META_REFRESH__ -->", "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="Paths to *.summary.json aggregates (later = higher precedence)")
    parser.add_argument("--output", default="results/report_merged.html", help="Output HTML dashboard path")
    parser.add_argument("--summary", default=None, help="Also write the merged payload as JSON")
    parser.add_argument("--template", default=None, help="Dashboard template path (default: scoring/benchmark_dashboard.html)")
    args = parser.parse_args()

    summaries = []
    for p in args.summaries:
        path = Path(p)
        if not path.exists():
            sys.exit(f"summary file not found: {path}")
        with open(path) as f:
            summaries.append(json.load(f))

    merged = merge_summaries(summaries)
    template = Path(args.template) if args.template else Path(__file__).parent / "scoring" / "benchmark_dashboard.html"
    render_dashboard(merged, template, args.output)

    print(f"Merged {len(args.summaries)} summaries -> {len(merged['models'])} models, "
          f"{len(merged['benchmarks'])} benchmarks, {merged['total_samples']} total samples")
    print(f"Dashboard: {Path(args.output).resolve()}")

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"Summary:   {Path(args.summary).resolve()}")


if __name__ == "__main__":
    main()
