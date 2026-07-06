#!/usr/bin/env python3
"""Download and cache all benchmark datasets from HuggingFace.

Run this once after cloning the repo. Subsequent benchmark runs will
use the local cache with no network requests (HF_DATASETS_OFFLINE=1).

Usage:
    python prefetch_datasets.py
"""
import os
import sys
from datasets import load_dataset
from pathlib import Path

DOWNLOADS = [
    ("cais/mmlu",                        dict(name="abstract_algebra",  split="test")),
    ("cais/mmlu",                        dict(name="logical_fallacies",  split="test")),
    ("cais/mmlu",                        dict(name="formal_logic",       split="test")),
    ("cais/mmlu",                        dict(name="high_school_mathematics", split="test")),
    ("cais/mmlu",                        dict(name="philosophy",         split="test")),
    ("allenai/ai2_arc",                  dict(name="ARC-Challenge",      split="test")),
    ("openai/gsm8k",                     dict(name="main",               split="test")),
    ("openai/openai_humaneval",          dict(split="test")),
    ("google-research-datasets/mbpp",    dict(name="sanitized",          split="test")),
    ("xlangai/spider",                   dict(split="validation")),
]

def main():
    if not os.environ.get("HF_TOKEN"):
        print("Note: no HF_TOKEN set. Downloads will work fine — unauthenticated")
        print("      HF requests are not throttled for dataset downloads of this size.")
        print("      Set HF_TOKEN env var only if you hit rate limits.\n")
    print("Prefetching benchmark datasets into local HuggingFace cache...\n")
    failed = []
    for i, (path, kwargs) in enumerate(DOWNLOADS, 1):
        name = f"{path} ({kwargs.get('name', kwargs.get('split', ''))})"
        print(f"[{i}/{len(DOWNLOADS)}] {name} ...", end=" ", flush=True)
        try:
            load_dataset(path, **kwargs)
            print("cached")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"WARNING: {len(failed)} dataset(s) failed to download:")
        for f in failed:
            print(f"  - {f}")
        print("\nBenchmarks using failed datasets will be skipped during runs.")
        sys.exit(1)
    else:
        Path(".datasets_ready").write_text("ok")
        print("All datasets cached. Future runs will use local cache only (no HF requests).")

if __name__ == "__main__":
    main()
