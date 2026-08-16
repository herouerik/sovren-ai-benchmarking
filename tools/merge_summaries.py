#!/usr/bin/env python3
"""Merge dashboard summaries from several machines into one.

Models are benchmarked on more than one host — the M4 MacBook, the GPU server,
and (soon) the u7 laptop — and no single machine holds every per-sample record.
The dashboard is therefore built from a *merge* of the small aggregate
summaries each machine produces, not from any one results file. That merge was
being done by hand; this makes it repeatable.

A summary is what `generate_report.py --summary` writes: scores, speeds,
sample_sizes, model_info, model_timestamps and swap_benches, keyed by model
label. ~16KB per run against ~5MB of raw records, so it is cheap to copy
between machines and to commit.

Merging is per model label. When the same label appears in more than one input,
the LAST input listed wins — so put the newest or most authoritative file last.
That is deliberate rather than merging cell-by-cell: a model's scores should
come from one coherent run on one machine, otherwise a row mixes hosts and its
speed figures become meaningless.

Usage:
    python3 tools/merge_summaries.py a.summary.json b.summary.json \
        --out results/merged.summary.json
    python3 scoring/generate_report.py results/merged.summary.json \
        --from-summary --output results/report_final.html
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Per-model maps that are merged by taking whole entries from the winning input.
_BY_MODEL = ("scores", "speeds", "sample_sizes", "model_info",
             "model_timestamps", "swap_benches")


def merge(summaries: list[dict]) -> dict:
    merged: dict = {k: {} for k in _BY_MODEL}
    models: list[str] = []
    benchmarks: list[str] = []
    all_models: list[str] = []
    total = 0
    provenance: dict[str, str] = {}

    for idx, s in enumerate(summaries):
        for m in s.get("models", []):
            if m not in models:
                models.append(m)
        for b in s.get("benchmarks", []):
            if b not in benchmarks:
                benchmarks.append(b)
        for m in s.get("all_models", []):
            if m not in all_models:
                all_models.append(m)
        for key in _BY_MODEL:
            for model, value in (s.get(key) or {}).items():
                merged[key][model] = value
        for m in s.get("models", []):
            provenance[m] = s.get("run_id", f"input{idx}")
        total += s.get("total_samples", 0) or 0

    # Order models by overall score so the dashboard opens ranked, matching what
    # a single-machine report does.
    def overall(m: str) -> float:
        cells = merged["scores"].get(m) or {}
        return sum(cells.values()) / len(cells) if cells else -1.0

    merged["models"] = sorted(models, key=overall, reverse=True)
    merged["benchmarks"] = benchmarks
    if all_models:
        merged["all_models"] = all_models
    merged["total_samples"] = total
    merged["run_id"] = "merged"
    merged["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    merged["merged_from"] = provenance
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="summary JSON files; later files win on conflict")
    ap.add_argument("--out", required=True, help="path for the merged summary")
    args = ap.parse_args()

    summaries = []
    for p in args.inputs:
        data = json.loads(Path(p).read_text())
        if "scores" not in data:
            raise SystemExit(f"{p} is not a summary (no 'scores' key) — "
                             f"generate one with generate_report.py --summary")
        summaries.append(data)
        print(f"  {p}: {len(data.get('models', []))} models, "
              f"{len(data.get('benchmarks', []))} benchmarks")

    merged = merge(summaries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    hosts: dict[str, int] = {}
    for mi in merged["model_info"].values():
        hosts[mi.get("host", "unknown")] = hosts.get(mi.get("host", "unknown"), 0) + 1
    print(f"\n  merged → {out} ({out.stat().st_size // 1024}KB)")
    print(f"  {len(merged['models'])} models, {len(merged['benchmarks'])} benchmarks, "
          f"{merged['total_samples']} samples")
    for h, n in sorted(hosts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d} models  {h}")


if __name__ == "__main__":
    main()
