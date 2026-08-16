#!/usr/bin/env python3
"""Generate an HTML benchmark report from a results JSON file.

Usage:
    python scoring/generate_report.py results/20260702_143022.json
    python scoring/generate_report.py results/20260702_143022.json --output results/report.html
    python scoring/generate_report.py results/20260702_143022.json --live   # adds 60s auto-refresh
"""

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def load_config_models(config_path: Path) -> list[str]:
    """Extract ordered model names from config.yaml. Handles string and dict entries."""
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        raw = cfg.get("models", [])
        # Must match the label used in results records, or a think-on variant
        # shows as a separate pending model forever.
        out = []
        for m in raw:
            if isinstance(m, str):
                out.append(m)
            else:
                out.append(m.get("label") or
                           (f"{m['model']} +think" if m.get("think") else m["model"]))
        return out
    except Exception:
        pass
    # Fallback manual parser for string-only entries
    models = []
    in_models = False
    try:
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "models:":
                in_models = True
                continue
            if in_models:
                if stripped.startswith("- "):
                    m = stripped[2:].split("#")[0].strip()
                    if m:
                        models.append(m)
                elif stripped and not stripped.startswith("#"):
                    in_models = False
    except Exception:
        pass
    return models


def _is_recovered_timing(r: dict) -> bool:
    """True for rows whose timing was backfilled by log recovery, not measured.

    The log-recovery path preserves accuracy but has no per-sample timing, so it
    wrote constants (elapsed=5.0, tok_per_sec=10.0). Averaging those into the
    speed column understated the fast models by 4-6x, so they are excluded.
    """
    return (r.get("elapsed") == 5.0 and r.get("tok_per_sec") == 10.0)


# Below this many generated tokens a row says nothing about sustained throughput:
# MMLU/ARC answer with a single letter, so decode_elapsed collapses toward zero
# and decode_tps explodes. Such rows are excluded from the speed average.
MIN_TOKENS_FOR_SPEED = 8


def _speed_of(r: dict) -> float:
    """Generation rate for a row, or 0.0 if the row can't support one.

    Prefers the true decode rate recorded by streaming; falls back to the
    end-to-end rate (prefill included, so it understates decode) for older
    results captured before streaming was enabled.
    """
    if _is_recovered_timing(r):
        return 0.0
    n_tok = r.get("completion_tokens")
    decode = r.get("decode_tps")
    if decode and n_tok is not None:
        return decode if n_tok >= MIN_TOKENS_FOR_SPEED else 0.0
    return r.get("tok_per_sec") or 0.0


def aggregate(raw: list[dict], all_models: list[str] | None = None,
              sample_counts: dict[tuple[str, str], int] | None = None,
              model_info: dict[str, dict] | None = None) -> dict:
    """Aggregate per-sample records into a dashboard-ready payload."""
    models = list(dict.fromkeys(r["model"] for r in raw))  # run order
    benchmarks = [b for b in dict.fromkeys(r["benchmark"] for r in raw) if b != "speed"]

    scores: dict[str, dict[str, float]] = {}
    speeds: dict[str, float] = {}
    sample_sizes: dict[str, dict[str, int]] = {}
    swap_benches: dict[str, list[str]] = {}

    model_timestamps: dict[str, str] = {}
    for model in models:
        mrows = [r for r in raw if r["model"] == model]
        scores[model] = {}
        sample_sizes[model] = {}
        swap_benches[model] = []
        for bench in benchmarks:
            brows = [r for r in mrows if r["benchmark"] == bench]
            if brows:
                scores[model][bench] = sum(r["score"] for r in brows) / len(brows)
                # Detect swap-aborted benchmarks
                if any("swap_abort" in (r.get("error") or "") for r in brows):
                    swap_benches[model].append(bench)
                n = sample_counts.get((model, bench), len(brows)) if sample_counts else len(brows)
                sample_sizes[model][bench] = n
        toks = [s for s in (_speed_of(r) for r in mrows) if s > 0]
        speeds[model] = sum(toks) / len(toks) if toks else 0.0
        # Latest sample timestamp = when this model's evaluation completed
        ts_vals = [r["ts"] for r in mrows if r.get("ts")]
        if ts_vals:
            model_timestamps[model] = max(ts_vals)

    run_id = raw[0].get("run_id", "UNKNOWN") if raw else "UNKNOWN"

    payload: dict = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_samples": len([r for r in raw if r.get("benchmark") != "speed"]),
        "models": models,
        "benchmarks": benchmarks,
        "scores": scores,
        "speeds": speeds,
        "sample_sizes": sample_sizes,
        "model_timestamps": model_timestamps,
        "swap_benches": swap_benches,
    }
    if all_models:
        # Ensure all entries are strings (not raw config dicts)
        payload["all_models"] = [m if isinstance(m, str) else m["model"] for m in all_models]
    if model_info:
        payload["model_info"] = model_info
    return payload


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
    parser.add_argument("input", help="Path to results JSON file (or a summary, with --from-summary)")
    parser.add_argument(
        "--from-summary", action="store_true",
        help="Treat the input as an already-aggregated summary rather than "
             "per-sample records. Needed to rebuild the dashboard from a merge "
             "of several machines, where no single machine holds every row.",
    )
    parser.add_argument("--output", default=None, help="Output HTML path (default: same dir as input, report.html)")
    parser.add_argument("--template", default=None, help="Path to dashboard HTML template")
    parser.add_argument("--config", default=None, help="Path to config.yaml (injects all_models for status markers)")
    parser.add_argument("--live", action="store_true", help="Inject 60s meta-refresh (use during an ongoing run)")
    parser.add_argument(
        "--summary", metavar="PATH",
        help="Also write the dashboard's aggregate payload as JSON. This is the "
             "small, diffable form of a run (~20KB vs ~5MB of per-sample "
             "records) and is enough to rebuild the dashboard, so a committed "
             "report stays reproducible without tracking raw model output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path) as f:
        raw = json.load(f)

    if args.from_summary:
        # Already aggregated — typically a merge across machines, where no
        # single machine holds every per-sample record.
        if not (isinstance(raw, dict) and "scores" in raw):
            raise ValueError("--from-summary expects an aggregated summary "
                             "(with a 'scores' key), not per-sample records.")
        data = raw
    else:
        # Handle both wrapped {metadata, results} and flat list formats
        if isinstance(raw, dict) and "results" in raw:
            records = raw["results"]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError("Results JSON must be a list of per-sample records or {metadata, results}.")

        # Auto-detect config.yaml if not specified
        config_path = Path(args.config) if args.config else (input_path.parent.parent / "config.yaml")
        all_models = load_config_models(config_path) if config_path.exists() else []

        # Extract model_info from metadata if available
        model_info = raw.get("metadata", {}).get("model_info") if isinstance(raw, dict) else None
        data = aggregate(records, all_models=all_models or None, model_info=model_info)

    template_path = Path(args.template) if args.template else find_template()
    with open(template_path) as f:
        html = f.read()

    injection = f"const BENCHMARK_DATA = {json.dumps(data, indent=2, ensure_ascii=False)};"
    html = html.replace("// __INJECT_DATA__", injection)

    meta_refresh = '<meta http-equiv="refresh" content="60">' if args.live else ''
    html = html.replace("<!-- __META_REFRESH__ -->", meta_refresh)

    output_path = Path(args.output) if args.output else input_path.parent / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Summary written → {summary_path} "
              f"({summary_path.stat().st_size // 1024}KB)")

    print(f"Report written → {output_path}")
    print(f"Models: {len(data['models'])}  Benchmarks: {len(data['benchmarks'])}  Samples: {data['total_samples']}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
