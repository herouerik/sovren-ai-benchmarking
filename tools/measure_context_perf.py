#!/usr/bin/env python3
"""Measure prefill and decode throughput AT A REALISTIC CONTEXT SIZE.

Why this exists
---------------
The harness's headline `speeds` figure is decode tok/s measured on benchmark prompts,
which are short. For an agentic coding loop that is the wrong operating point, and it
does not merely understate — **it can invert the ranking.**

Every model decodes more slowly as context grows, but not by the same factor — the
collapse is architecture-dependent, and a dense model can fall much further than a
sparse-MoE one of comparable total size. So a short-prompt ranking need not survive at
the operating point, and can invert.

Model rows already carry the architecture (`dense` vs `MoE`, active parameters), so the
information needed to predict this is usually present — what is missing is a measurement
at the operating point. That is what this tool adds.

(No measured figures are published in this repo. Run the tool for your own hardware.)

Prefill matters as much as decode, because an agent harness that compacts rewrites the
transcript and invalidates the KV prefix — so every compaction pays a *full-window*
prefill, not an incremental one. Divide the window size by the prefill rate to get that
bill. When it approaches the interval between compactions, the session spends most of its
wall clock re-reading its own context and appears to make no progress.

DECODE SAMPLE SIZE
------------------
`--predict` must be large enough that per-request overhead is amortised: over a few dozen
tokens `eval_duration` is mostly startup, and the same model can vary by ~2x between runs.
Prefill is far more reproducible than decode, so treat prefill as the reliable figure and
give decode a real sample (200+ tokens, the default).

Three traps, all learned the hard way
-------------------------------------
1. **A resident model does not re-honour a changed `num_ctx`.** Ollama keeps the model
   loaded with the context it was loaded with, so a second measurement at a different
   size silently reports the first size. This tool unloads (`keep_alive: 0`) before
   every measurement.
2. **Thinking models spend the output budget on reasoning tokens.** `think: false`,
   or `decode_tps` measures the wrong thing (or returns an empty completion).
3. **A short generation does not measure decode.** See DECODE SAMPLE SIZE above.

Usage
-----
    python3 tools/measure_context_perf.py --models <model-a> <model-b> \
        --context-tokens 9000 --host "<host label>" --out data/context_perf.json

    # or take the model list from a benchmark config
    python3 tools/measure_context_perf.py --config config.yaml --out data/context_perf.json

Feed the result to the dashboard:

    python3 scoring/generate_report.py results/merged.summary.json --from-summary \
        --perf data/context_perf.json --output results/report_final.html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# Measured empirically on code-shaped filler: 4.40 bytes per token. Used only to size
# the filler prompt; the *reported* token count always comes from the server's own
# prompt_eval_count, never from this estimate.
BYTES_PER_TOKEN = 4.40
FILLER = "def process_record(record, config, logger):  # one row from the upstream feed\n"


def _post(base_url: str, path: str, payload: dict, timeout: int):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload(base_url: str, model: str) -> None:
    """Trap 1: a resident model keeps its original num_ctx. Always unload first."""
    try:
        _post(base_url, "/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass
    time.sleep(4)


def measure(base_url: str, model: str, context_tokens: int, num_ctx: int,
            predict: int, timeout: int) -> dict | None:
    reps = max(1, int(context_tokens * BYTES_PER_TOKEN / len(FILLER)))
    prompt = FILLER * reps + "\nReply: ok"
    unload(base_url, model)
    try:
        d = _post(base_url, "/api/generate", {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,            # trap 2
            "options": {"num_ctx": num_ctx, "num_predict": predict},
        }, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # A timeout is itself a result: at these rates a large prompt can legitimately
        # exceed any reasonable deadline. Record it rather than dropping the model.
        return {"error": f"{type(e).__name__}: {e}", "timed_out_after_s": timeout}

    pe, ped = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 0) / 1e9
    ec, ed = d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9
    if not pe or ped <= 0:
        return {"error": "server returned no prompt_eval timing"}
    out = {
        "prompt_tokens": pe,
        "prefill_tps": round(pe / ped, 1),
        "num_ctx": num_ctx,
    }
    if ec and ed > 0:
        out["decode_tps"] = round(ec / ed, 1)
        out["decode_tokens"] = ec
    return out


def models_from_config(path: Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        sys.exit("--config needs pyyaml (pip install -r requirements.txt)")
    cfg = yaml.safe_load(path.read_text()) or {}
    out = []
    for m in (cfg.get("models") or []):
        out.append(m if isinstance(m, str) else (m.get("name") or m.get("id")))
    return [m for m in out if m]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--config", help="read the model list from a benchmark config.yaml")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--context-tokens", type=int, default=9000,
                    help="approximate prompt size to measure at (default 9000)")
    ap.add_argument("--num-ctx", type=int, default=32768,
                    help="window to load the model with (default 32768)")
    ap.add_argument("--predict", type=int, default=200,
                    help="tokens to generate for the decode figure (default 200). Do not "
                         "lower this much: at 40 tokens per-request startup dominates and "
                         "the same model can vary 2x between runs. See DECODE SAMPLE SIZE.")
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument("--host", default="", help="host label, matching the summary's model_info")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = list(args.models)
    if args.config:
        models += [m for m in models_from_config(Path(args.config)) if m not in models]
    if not models:
        sys.exit("no models: pass --models and/or --config")

    results: dict[str, dict] = {}
    for m in models:
        print(f"  measuring {m} at ~{args.context_tokens} tokens ...", flush=True)
        r = measure(args.base_url, m, args.context_tokens, args.num_ctx,
                    args.predict, args.timeout)
        results[m] = r or {"error": "no result"}
        if "error" in results[m]:
            print(f"    {results[m]['error']}")
        else:
            print(f"    prefill {r['prefill_tps']} tok/s   "
                  f"decode {r.get('decode_tps','—')} tok/s   (n={r['prompt_tokens']} tokens)")

    doc = {
        "schema": 1,
        "measured": date.today().isoformat(),
        "host": args.host,
        "base_url": args.base_url,
        "target_context_tokens": args.context_tokens,
        "num_ctx": args.num_ctx,
        "note": ("Prefill and decode at a realistic context size. Not comparable to the "
                 "summary's `speeds` field, which is measured on short benchmark prompts."),
        "models": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"\nWritten → {out}  ({len(results)} models)")


if __name__ == "__main__":
    main()
