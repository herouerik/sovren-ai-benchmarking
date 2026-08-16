#!/usr/bin/env python3
"""Export a small, clean model scoreboard from results/merged.summary.json.

2026-08-15: built for pfclabs-codingswarm's Phase 2 "complexity-matched
execution routing" (#87) — it needs to rank the local/sovereign tier's
models by fitness for a task (tool-calling reliability, code quality,
speed) without depending on this repo's own HTML report or its embedded
JS data blob, which is a fragile interface for another project to parse.
This reads the same merged summary the HTML report itself renders from,
so it stays in sync with whatever the dashboard shows without needing to
scrape rendered markup.

Erik, 2026-08-15: "possibly we can make that service deliver a far better
interface... but it is a start to test the idea." This is that start —
a real, versioned, regenerable export, not a one-off scrape. A proper
HTTP endpoint or a report_final.html <link rel> to a JSON sibling file
would be the next step if this proves useful across more than one
consuming project.

Usage:
    python3 tools/export_model_scoreboard.py [--out PATH]

Output shape (see the `note` field for what's NOT covered): local/
sovereign-tier models only (GPU server + MacBook, Ollama-hosted) — cloud
models (Gemini/Claude/NIM) are never locally benchmarked here and won't
appear.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "results" / "merged.summary.json"
DEFAULT_OUT = REPO_ROOT / "results" / "model_scoreboard.summary.json"

_NOTE = (
    "Local/sovereign-tier only (GPU server + MacBook, Ollama-hosted). Cloud "
    "models (Gemini/Claude/NIM) are not covered by this benchmark and are "
    "not in this file. Consumers that also route cloud candidates (e.g. "
    "pfclabs-codingswarm's atomic_pipeline.py _STRONG_PROVIDERS) need a "
    "separate mechanism for that half of the decision. Regenerate this file "
    "by re-running tools/export_model_scoreboard.py whenever a new "
    "benchmark run completes."
)


def export(source: Path, out: Path) -> dict:
    data = json.loads(source.read_text())

    models = data.get("all_models", data.get("models", []))
    info = data.get("model_info", {})
    scores = data.get("scores", {})
    speeds = data.get("speeds", {})

    out_models: dict[str, dict] = {}
    for name in models:
        inf = info.get(name, {})
        sc = scores.get(name, {})
        overall = round(sum(sc.values()) / len(sc), 4) if sc else None
        host = inf.get("host", "")
        host_short = (
            "gpu_server" if "P100" in host
            else "macbook" if "MacBook" in host
            else host
        )
        out_models[name] = {
            "host": host_short,
            "bfcl": sc.get("bfcl"),
            "humaneval": sc.get("humaneval"),
            "overall": overall,
            "speed_tok_s": round(speeds[name], 1) if speeds.get(name) is not None else None,
            "effective_ctx": inf.get("effective_ctx", inf.get("context_length")),
            "params": inf.get("params"),
        }

    result = {
        "source": str(source.relative_to(REPO_ROOT)),
        "run_id": data.get("run_id"),
        "generated_from_run_timestamp": data.get("timestamp"),
        "note": _NOTE,
        "models": out_models,
    }
    out.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = export(args.source, args.out)
    print(f"Wrote {len(result['models'])} models to {args.out}")


if __name__ == "__main__":
    main()
