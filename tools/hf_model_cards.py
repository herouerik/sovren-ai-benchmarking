#!/usr/bin/env python3
"""Read published benchmark scores out of Hugging Face model cards.

Why bother, when there are leaderboards
---------------------------------------
Public leaderboards publish a top-N view, and for coding benchmarks that view is
frontier hosted models. A locally-runnable open-weight build almost never appears, so a
leaderboard mirror can calibrate a local fleet only in the abstract — "here is the band
you are not in".

Model cards are the opposite: a vendor publishing an open-weight model reports its own
benchmark numbers, usually in a table that also names several competitors. One card can
therefore yield SWE-bench Verified for three or four models a laptop can actually run.
That is the difference between an external panel that is interesting and one that is
useful.

The two hard parts, and how they are handled
--------------------------------------------
**1. Cards have no common format.** Observed in the wild, all on cards for models in one
small fleet:

    HTML table, benchmarks as rows      most Qwen cards
    HTML table, capability+name nested  Qwen3.8-27B (two <div>s in the label cell)
    Markdown table, models as rows      Devstral (transposed, and scores carry '%')
    Markdown table, no benchmarks       Gemma (architecture tables only)
    No table at all                     several smaller cards

So parsing is **shape-based, not markup-based**: pull every table, strip each cell to
text, then decide orientation by testing which axis matches a known benchmark alias. An
earlier attempt keyed on a CSS class (`benchmark-name`) and worked on exactly one card.
Orientation detection against the alias list is what makes one parser cover all of them.

**2. Vendors grade their competitors.** A Qwen card reports Gemma's score; a Devstral
card reports GLM's. Those numbers are second-hand and are not neutral — a vendor picks
the configuration, the scaffold and the baseline it compares against. Every score is
therefore tagged:

    self_reported   the card belongs to this model — the vendor's own claim
    third_party     the card belongs to someone else — a competitor's measurement

and a self-reported score always wins over a third-party one for the same model and
benchmark. Both are published claims rather than independent verification; the tag is
there so a reader knows which kind they are looking at.

Usage as a library (see tools/fetch_external_reference.py):

    from hf_model_cards import harvest
    rows = harvest(["Qwen/Qwen3.6-27B", "mistralai/Devstral-Small-2-24B-Instruct-2512"],
                   aliases={"swe_bench_verified": ["swe-bench verified", "swe bench verified"]})

Standalone, to see what a card offers before adding it:

    python3 tools/hf_model_cards.py Qwen/Qwen3.6-27B
    python3 tools/hf_model_cards.py Qwen/Qwen3.6-27B --all      # every benchmark found
"""
from __future__ import annotations

import html
import re
import sys
import urllib.error
import urllib.request

CARD_URL = "https://huggingface.co/{repo}/raw/main/README.md"
_NUM = re.compile(r"^-?\d+(?:\.\d+)?%?$")
_TAG = re.compile(r"<[^>]+>")


def fetch_card(repo: str, timeout: int = 30) -> str | None:
    req = urllib.request.Request(CARD_URL.format(repo=repo),
                                 headers={"User-Agent": "sovren-ai-benchmark"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def _clean(cell: str) -> str:
    """Cell text. Where a label nests <div>s (capability + benchmark name), the LAST one
    is the specific label — 'Agentic coding' + 'SWE-bench Pro' must not become one word."""
    divs = re.findall(r"<div[^>]*>(.*?)</div>", cell, re.S)
    txt = divs[-1] if divs else cell
    txt = html.unescape(_TAG.sub("", txt))
    return re.sub(r"\s+", " ", txt).replace("*", "").strip()


def _num(v: str) -> float | None:
    """Scores appear as 77.2, 68.0%, or '--'. Normalise to a 0–1 fraction."""
    v = v.strip().rstrip("%")
    if not v or not _NUM.match(v + ("%" if v != v.rstrip("%") else "")):
        if not re.match(r"^-?\d+(?:\.\d+)?$", v):
            return None
    try:
        f = float(v)
    except ValueError:
        return None
    if f < 0:
        return None
    return f / 100.0 if f > 1.0 else f


def _tables(md: str) -> list[list[list[str]]]:
    """Every table in the card, as rows of cleaned cells — HTML and markdown alike."""
    out = []
    for tb in re.findall(r"<table.*?</table>", md, re.S):
        rows = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S):
            rows.append([_clean(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)])
        if rows:
            out.append(rows)

    block: list[list[str]] = []
    for line in md.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
                continue                      # markdown separator row
            block.append(cells)
        else:
            if len(block) > 1:
                out.append(block)
            block = []
    if len(block) > 1:
        out.append(block)
    return out


def _canon(label: str, aliases: dict[str, list[str]]) -> str | None:
    l = label.lower().strip()
    for key, alts in aliases.items():
        if l in (a.lower() for a in alts):
            return key
    return None


def _rows_from_table(rows, aliases):
    """Yield (benchmark_key, model_label, fraction) from one table, either orientation."""
    if len(rows) < 2:
        return
    header = rows[0]
    if len(header) < 3:
        return

    # Orientation: does the HEADER name benchmarks (models-as-rows), or does the first
    # COLUMN name them (benchmarks-as-rows)? Decided by which axis the alias list
    # recognises more of — never by markup, which varies per vendor.
    hdr_hits = sum(1 for c in header[1:] if _canon(c, aliases))
    col_hits = sum(1 for r in rows[1:] if len(r) == len(header) and _canon(r[0], aliases))
    if hdr_hits == 0 and col_hits == 0:
        return

    if hdr_hits >= col_hits:                                  # models as rows
        bench_of = {i: _canon(c, aliases) for i, c in enumerate(header[1:], start=1)}
        for r in rows[1:]:
            if len(r) != len(header) or not r[0]:
                continue
            for i, key in bench_of.items():
                if not key:
                    continue
                v = _num(r[i])
                if v is not None:
                    yield key, r[0], v
    else:                                                     # benchmarks as rows
        models = header[1:]
        for r in rows[1:]:
            if len(r) != len(header):
                continue
            key = _canon(r[0], aliases)
            if not key:
                continue
            for m, cell in zip(models, r[1:]):
                v = _num(cell)
                if v is not None and m:
                    yield key, m, v


def _owns(repo: str, model_label: str) -> bool:
    """Is this row the card's own model? Compare on family, ignoring org and quant tags."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    name = norm(repo.split("/")[-1])
    for suffix in ("fp8", "nvfp4", "gguf", "instruct", "it", "bf16", "awq", "int4", "mlx"):
        name = name.replace(suffix, "")
    m = norm(model_label)
    return bool(m) and (m in name or name in m) and min(len(m), len(name)) > 4


def _family(label: str) -> str:
    """Collapse label spellings to one key. Cards disagree on punctuation for the same
    weights — "Qwen3.6-35B-A3B" on one card and "Qwen3.6-35BA3B" on its own — and without
    this the same model appears twice in a benchmark with different provenance."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def harvest(repos, aliases, verbose=False):
    """{benchmark_key: {model_label: {score, source, self_reported}}} across cards.

    One row per model FAMILY per benchmark. A self-reported score beats a third-party one;
    between two of the same kind the first card listed wins, so order `repos` most-trusted
    first. Deduping matters because a model is usually named on several cards: its own
    vendor's, and every competitor that benchmarked against it.
    """
    out: dict[str, dict[str, dict]] = {}
    seen: dict[str, dict[str, str]] = {}     # bench -> family -> winning label
    for repo in repos:
        md = fetch_card(repo)
        if md is None:
            if verbose:
                print(f"    {repo}: card not fetchable")
            continue
        n, sup = 0, 0
        for table in _tables(md):
            for key, model, val in _rows_from_table(table, aliases):
                self_rep = _owns(repo, model)
                fam = _family(model)
                bench_seen = seen.setdefault(key, {})
                prev_label = bench_seen.get(fam)
                if prev_label is not None:
                    prev = out[key][prev_label]
                    if prev["self_reported"] or not self_rep:
                        sup += 1
                        continue
                    del out[key][prev_label]      # self-reported supersedes third-party
                out.setdefault(key, {})[model] = {
                    "score": round(val, 4), "source": repo, "self_reported": self_rep}
                bench_seen[fam] = model
                n += 1
        if verbose:
            extra = f", {sup} superseded/skipped" if sup else ""
            print(f"    {repo}: {n} score(s){extra}")
    return out


def _main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repos = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    md = fetch_card(repos[0])
    if md is None:
        sys.exit(f"could not fetch card for {repos[0]}")
    if show_all:
        # No alias filter: report every table row that looks like a benchmark, to help
        # decide which aliases are worth declaring.
        seen = set()
        for table in _tables(md):
            if len(table) < 2 or len(table[0]) < 3:
                continue
            for r in table[1:]:
                if len(r) == len(table[0]) and r[0] and sum(
                        1 for c in r[1:] if _num(c) is not None) >= 2:
                    if r[0] not in seen:
                        seen.add(r[0])
                        print(f"  {r[0]}")
        print(f"\n{len(seen)} candidate benchmark labels. Header: {table[0][1:]}")
        return
    from fetch_external_reference import load_aliases   # noqa: E402
    rows = harvest(repos, load_aliases(), verbose=True)
    for key, models in rows.items():
        print(f"\n{key}")
        for m, v in sorted(models.items(), key=lambda kv: -kv[1]["score"]):
            tag = "self" if v["self_reported"] else "3rd-party"
            print(f"   {m:<28} {v['score']*100:5.1f}  [{tag}, {v['source']}]")


if __name__ == "__main__":
    _main()
