#!/usr/bin/env python3
"""Decompose a model x build x host comparison into isolated effects.

The question "is Qwen 3.8 better than 3.6" was unanswerable from the main
dashboard because every candidate comparison moved several things at once:
model, quantisation/engine (nvfp4+MLX vs Q4_K_M+llama.cpp), and host
(M4 vs the P100 pool). This holds two factors fixed at a time and reports the
third, with a significance test, so a difference is only claimed when the data
supports it.

One factor cannot be separated on this hardware: MLX runs only on Apple
Silicon, so "MLX vs GGUF" on the M4 is really "MLX+nvfp4 vs llama.cpp+Q4_K_M".
That is the actionable comparison anyway — you choose a build, never a
quantisation in isolation — but it is not a quantisation result and is not
reported as one.

Usage:
    python3 tools/factorial_effects.py results/<run>.json [--out docs/FINDINGS.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact test on a 2x2 of pass/fail counts."""
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) < 0 or n == 0:
        return 1.0
    obs = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    p = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j = a + c - i
        if j < 0 or j > c + d:
            continue
        pr = comb(a + b, i) * comb(c + d, j) / comb(n, a + c)
        if pr <= obs + 1e-12:
            p += pr
    return min(p, 1.0)


def load(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text())
    records = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    passes: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # model -> bench -> [pass, total]
    speed: dict = defaultdict(list)
    for r in records:
        m, b = r["model"], r["benchmark"]
        cell = passes[m][b]
        cell[0] += 1 if r.get("passed") else 0
        cell[1] += 1
        if r.get("decode_tps"):
            speed[m].append(r["decode_tps"])
    return passes, speed


def totals(passes: dict, model: str, benches: list[str]) -> tuple[int, int]:
    p = t = 0
    for b in benches:
        if b in passes.get(model, {}):
            p += passes[model][b][0]
            t += passes[model][b][1]
    return p, t


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def compare(passes, speed, a: str, b: str, benches: list[str]) -> dict:
    pa, ta = totals(passes, a, benches)
    pb, tb = totals(passes, b, benches)
    p = fisher_two_sided(pa, ta - pa, pb, tb - pb)
    return {
        "a": a, "b": b,
        "a_pass": pa, "a_total": ta, "b_pass": pb, "b_total": tb,
        "a_pct": 100 * pa / ta if ta else 0, "b_pct": 100 * pb / tb if tb else 0,
        "p": p, "significant": p < 0.05,
        "a_tps": median(speed.get(a, [])), "b_tps": median(speed.get(b, [])),
        "per_bench": {
            bm: (100 * passes[a][bm][0] / passes[a][bm][1] if bm in passes.get(a, {}) else None,
                 100 * passes[b][bm][0] / passes[b][bm][1] if bm in passes.get(b, {}) else None)
            for bm in benches
        },
    }


def render(cmps: list[tuple[str, dict]], benches: list[str]) -> str:
    out = ["# Factorial findings: model x build\n",
           "Each row holds every other factor fixed. `p` is a two-sided Fisher exact",
           "test on pooled pass/fail items; a difference is only real when p < 0.05.\n"]
    for title, c in cmps:
        verdict = ("**significant**" if c["significant"]
                   else "not distinguishable")
        out.append(f"## {title}\n")
        out.append(f"- {c['a']}: **{c['a_pct']:.1f}%** ({c['a_pass']}/{c['a_total']}), "
                   f"{c['a_tps']:.1f} tok/s")
        out.append(f"- {c['b']}: **{c['b_pct']:.1f}%** ({c['b_pass']}/{c['b_total']}), "
                   f"{c['b_tps']:.1f} tok/s")
        delta = c["a_pct"] - c["b_pct"]
        out.append(f"- difference: {delta:+.1f} points, p = {c['p']:.3f} -> {verdict}")
        if c["b_tps"]:
            out.append(f"- speed ratio: {c['a_tps'] / c['b_tps']:.2f}x")
        out.append("\n| benchmark | " + " | ".join([c["a"], c["b"], "delta"]) + " |")
        out.append("|---|---|---|---|")
        for bm in benches:
            x, y = c["per_bench"].get(bm, (None, None))
            if x is None or y is None:
                continue
            out.append(f"| {bm} | {x:.0f}% | {y:.0f}% | {x - y:+.0f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()

    passes, speed = load(Path(args.results))
    models = list(passes)
    benches = args.benchmarks or sorted({b for m in models for b in passes[m]})

    def find(*needles: str) -> str | None:
        for m in models:
            if all(n in m for n in needles):
                return m
        return None

    # Exact names for the GGUF arms so "qwen3.6:27b" never matches "…-mlx".
    m36g = next((m for m in models if m == "qwen3.6:27b"), None)
    m38g = next((m for m in models if m == "qwen3.8:27b"), None)
    m36m = find("qwen3.6:27b-mlx")
    m38m = find("qwen3.8:27b-mlx")

    cmps = []
    if m38m and m36m:
        cmps.append(("Model effect, MLX build held fixed (3.8 vs 3.6)",
                     compare(passes, speed, m38m, m36m, benches)))
    if m38g and m36g:
        cmps.append(("Model effect, GGUF build held fixed (3.8 vs 3.6)",
                     compare(passes, speed, m38g, m36g, benches)))
    if m38m and m38g:
        cmps.append(("Build effect on Qwen 3.8 (MLX/nvfp4 vs GGUF/Q4_K_M)",
                     compare(passes, speed, m38m, m38g, benches)))
    if m36m and m36g:
        cmps.append(("Build effect on Qwen 3.6 (MLX/nvfp4 vs GGUF/Q4_K_M)",
                     compare(passes, speed, m36m, m36g, benches)))

    text = render(cmps, benches)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
