#!/usr/bin/env python3
"""Refresh data/external_reference.yaml from its upstream sources.

The external figures are *committed*, not fetched at report time — a benchmark run must
not depend on the network, and a published number that moves should move in a diff
somebody can review. This script is how that diff gets produced.

Each benchmark section in the YAML declares its own `source_url`, so adding a new
external benchmark needs no code change here: give it a `source_url`, a `label`, a
`scale`, and a `pattern` describing how to read its rows out of the fetched text.

    python3 tools/fetch_external_reference.py --dry-run    # show what would change
    python3 tools/fetch_external_reference.py              # rewrite the scores blocks
    python3 tools/fetch_external_reference.py --only swe_bench_verified

Adding a benchmark needs no code change: give it a `slug` (the upstream
`<!-- AUTO:START slug=... -->` marker), a `label`, a `scale` and a `source_url`.
Available upstream slugs as of 2026-08-31: sweVerified, liveCodeBench, terminalBench2,
osWorldVerified, browseComp, arcAgi2, hle.

Scores are written as fractions (0.0–1.0) to match the harness's internal scale, so a
published "96.0%" is stored as 0.960.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_model_cards import harvest as harvest_cards  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml — pip install -r requirements.txt")

_DATA = Path(__file__).resolve().parent.parent / "data"
REF = _DATA / "external_reference.yaml"       # hand-authored definitions; never written
SCORES = _DATA / "external_scores.yaml"       # machine-generated; rewritten wholesale

# Upstream rows are markdown tables delimited by `<!-- AUTO:START slug=... -->`, which is a
# far more stable anchor than a heading: the slug is machine-generated and survives
# rewording, whereas headings get edited. A section declares `slug` and this reads its block.
_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*\[(?P<model>[^\]]+)\]\([^)]*\)\s*\|"
    r"\s*(?P<provider>[^|]*?)\s*\|\s*(?P<license>[^|]*?)\s*\|"
    r"\s*(?P<score>[\d.]+)\s*%?\s*\|", re.M)


def load_aliases(ref: Path | None = None) -> dict[str, list[str]]:
    """{benchmark_key: [card label variants]} from the definitions file.

    The alias list is not cosmetic: hf_model_cards decides a table's ORIENTATION by
    testing which axis these labels match, so a benchmark with no aliases is invisible to
    the card harvester even if the card reports it.
    """
    doc = yaml.safe_load((ref or REF).read_text()) or {}
    out = {}
    for key, b in (doc.get("benchmarks") or {}).items():
        al = b.get("aliases") or ([b["label"]] if b.get("label") else [])
        if al:
            out[key] = al
    return out


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "sovren-ai-benchmark"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse(text: str, slug: str) -> dict[str, dict]:
    """Read one slug's table. Returns {model: {score, provider, license}}.

    `license` is kept because it is the only field that says whether a reference point is
    even theoretically runnable locally — an Open row is a model a local fleet could host,
    a Closed row is a permanent external yardstick.
    """
    i = text.find(f"AUTO:START slug={slug}")
    if i < 0:
        raise LookupError(f"slug {slug!r} not present upstream")
    j = text.find("AUTO:END", i)
    block = text[i: j if j > 0 else len(text)]
    out: dict[str, dict] = {}
    for m in _ROW.finditer(block):
        name = m.group("model").strip().strip("*_`")
        out[name] = {
            "score": round(float(m.group("score")) / 100.0, 4),
            "provider": m.group("provider").strip() or None,
            "license": m.group("license").strip() or None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="refresh a single benchmark key")
    ap.add_argument("--ref", default=str(REF), help="definitions file (read-only)")
    ap.add_argument("--scores", default=str(SCORES), help="scores file (rewritten)")
    args = ap.parse_args()

    path = Path(args.ref)
    doc = yaml.safe_load(path.read_text())
    benches = doc.get("benchmarks") or {}
    scores_path = Path(args.scores)
    try:
        scores_doc = yaml.safe_load(scores_path.read_text()) or {}
    except FileNotFoundError:
        scores_doc = {}
    scores_doc.setdefault("_meta", {})
    out_benches = scores_doc.setdefault("benchmarks", {})
    changed, skipped = [], []

    for key, b in benches.items():
        if args.only and key != args.only:
            continue
        url = b.get("source_url")
        if not url:
            skipped.append(f"{key}: no source_url"); continue
        slug = b.get("slug")
        if not slug:
            skipped.append(f"{key}: no slug"); continue
        try:
            rows = parse(fetch(url), slug)
        except Exception as e:
            skipped.append(f"{key}: fetch/parse failed — {type(e).__name__}: {e}"); continue
        if not rows:
            skipped.append(f"{key}: slug {slug!r} matched no rows (upstream format changed?)"); continue

        scores = {k: v["score"] for k, v in rows.items()}
        old = (out_benches.get(key) or {}).get("scores") or {}
        added = {k: v for k, v in scores.items() if k not in old}
        removed = {k: v for k, v in old.items() if k not in scores}
        moved = {k: (old[k], v) for k, v in scores.items() if k in old and old[k] != v}
        if not (added or removed or moved):
            print(f"  {key}: unchanged ({len(scores)} rows)")
            continue

        changed.append(key)
        print(f"  {key}: {len(added)} added, {len(removed)} removed, {len(moved)} changed")
        for k, v in added.items():   print(f"     + {k}: {v}")
        for k, v in removed.items(): print(f"     - {k}: {v}")
        for k, (a, v) in moved.items(): print(f"     ~ {k}: {a} -> {v}")
        if not args.dry_run:
            out_benches[key] = {
                "retrieved": date.today().isoformat(),
                "scores": scores,
                "providers": {k: v["provider"] for k, v in rows.items() if v["provider"]},
                "licenses": {k: v["license"] for k, v in rows.items() if v["license"]},
            }

    # ---- Hugging Face model cards -------------------------------------------------
    # Leaderboard mirrors publish a top-N view, which for coding benchmarks is entirely
    # frontier hosted models. Cards are where open-weight numbers live, so this is what
    # makes the external panel populate for models a laptop can run.
    cards = ((doc.get("sources") or {}).get("hf_cards") or [])
    if cards and not args.only:
        print(f"  harvesting {len(cards)} model card(s)...")
        harvested = harvest_cards(cards, load_aliases(path), verbose=True)
        for key, models in harvested.items():
            if key not in benches:
                continue
            tgt = out_benches.setdefault(key, {})
            sc = tgt.setdefault("scores", {})
            src = tgt.setdefault("sources", {})
            selfrep = tgt.setdefault("self_reported", {})
            added = 0
            for model, rec in models.items():
                # The leaderboard mirror is an aggregator; a vendor card is a primary
                # claim about its own model. Neither is verification. Keep whichever is
                # already there for a model the mirror listed, and add the rest.
                if model in sc and model not in src:
                    continue
                if model not in sc:
                    added += 1
                sc[model] = rec["score"]
                src[model] = rec["source"]
                selfrep[model] = rec["self_reported"]
            if added:
                changed.append(f"{key}(+{added} from cards)")
                tgt["retrieved"] = date.today().isoformat()
            print(f"    {key}: {added} new model row(s)")

    for s in skipped:
        print(f"  SKIP {s}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if changed:
        # Only the scores file is ever rewritten. The definitions file carries hand-written
        # prose and comments that a wholesale yaml re-dump would destroy, which would make
        # this script unsafe to run unattended — the whole point of the split.
        scores_doc["_meta"] = {
            "generated_by": "tools/fetch_external_reference.py",
            "warning": "DO NOT EDIT BY HAND — regenerated wholesale on every refresh.",
            "definitions_live_in": REF.name,
            "last_refresh": date.today().isoformat(),
        }
        scores_path.write_text(
            "# Machine-generated. Regenerated wholesale by tools/fetch_external_reference.py.\n"
            "# Benchmark DEFINITIONS (labels, notes, column flags) live in "
            "external_reference.yaml\n"
            "# and are never touched by the refresher — that split is what makes a code-only\n"
            "# refresh safe to run without losing hand-written prose.\n"
            + yaml.safe_dump(scores_doc, sort_keys=False, allow_unicode=True, width=100))
        print(f"\nUpdated {scores_path} ({', '.join(changed)}). "
              f"{REF.name} untouched. Review the diff before committing.")
    else:
        print("\nNothing to update.")


if __name__ == "__main__":
    main()
