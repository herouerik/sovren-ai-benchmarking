#!/usr/bin/env python3
"""Download and cache all benchmark datasets from HuggingFace, plus Spider databases.

Run this once after cloning the repo. Subsequent benchmark runs will
use the local cache with no network requests (HF_DATASETS_OFFLINE=1).

Usage:
    python prefetch_datasets.py
"""
import os
import sys
import zipfile
import shutil
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

# Spider database Google Drive file ID (full dataset zip ~100MB)
SPIDER_GDRIVE_ID = "1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"
SPIDER_DB_DIR = Path("data/spider/database")


def download_spider_databases(local_zip: str | None = None) -> bool:
    """Download Spider SQLite databases for execution-based SQL scoring.

    Returns True if databases are available (downloaded or already present).
    """
    if SPIDER_DB_DIR.exists() and any(SPIDER_DB_DIR.rglob("*.sqlite")):
        n = len(list(SPIDER_DB_DIR.rglob("*.sqlite")))
        print(f"  Spider databases already present ({n} .sqlite files). Skipping download.")
        return True

    # If caller supplied a pre-downloaded zip, use it directly.
    if local_zip:
        zip_path = Path(local_zip)
        if not zip_path.exists():
            print(f"  [!] --spider-zip file not found: {zip_path}")
            return False
        print(f"  Using local zip: {zip_path}")
        return _extract_spider_zip(zip_path, remove_after=False)

    print("  Attempting to download Spider database files (~100 MB) via Google Drive...")
    print("  These are needed for execution-based SQL scoring (string match is a fallback).")

    try:
        import gdown
    except ImportError:
        print("  [!] gdown not installed. Run: pip install gdown")
        _print_manual_spider_instructions()
        return False

    zip_path = Path("data/spider_tmp.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
        if not result or not zip_path.exists():
            raise RuntimeError("gdown returned no file")
    except Exception as e:
        print(f"  [!] Automatic download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        _print_manual_spider_instructions()
        return False

    return _extract_spider_zip(zip_path, remove_after=True)


def _extract_spider_zip(zip_path: Path, remove_after: bool = True) -> bool:
    print("  Extracting database/ directory...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            db_entries = [f for f in zf.namelist() if f.endswith(".sqlite") and "database" in f]
            if not db_entries:
                raise RuntimeError("No .sqlite files found in zip")

            SPIDER_DB_DIR.mkdir(parents=True, exist_ok=True)
            for entry in db_entries:
                parts = entry.split("/")
                db_idx = next((i for i, p in enumerate(parts) if p == "database"), None)
                if db_idx is None or db_idx + 1 >= len(parts):
                    continue
                db_name = parts[db_idx + 1]
                fname = parts[-1]
                dest = SPIDER_DB_DIR / db_name / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(entry))

        n = len(list(SPIDER_DB_DIR.rglob("*.sqlite")))
        print(f"  Extracted {n} database files to {SPIDER_DB_DIR}/")
        if remove_after:
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"  [!] Extraction failed: {e}")
        if remove_after and zip_path.exists():
            zip_path.unlink()
        _print_manual_spider_instructions()
        return False


def _print_manual_spider_instructions():
    print()
    print("  --- Manual Spider database download ---")
    print("  1. Download the Spider dataset from: https://yale-lily.github.io/spider")
    print("     (click 'Download Spider v1.0' — it's the Google Drive zip)")
    print(f"  2. Extract the zip and copy the database/ folder to: {SPIDER_DB_DIR}/")
    print("     Expected structure: data/spider/database/<db_name>/<db_name>.sqlite")
    print("  Without this, Spider scoring falls back to normalised string matching.")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prefetch benchmark datasets")
    parser.add_argument(
        "--spider-zip",
        metavar="PATH",
        help="Path to a pre-downloaded Spider zip (skips Google Drive download)",
    )
    cli = parser.parse_args()

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

    print(f"\n[{len(DOWNLOADS) + 1}/{len(DOWNLOADS) + 1}] Spider SQLite databases ...")
    spider_ok = download_spider_databases(local_zip=cli.spider_zip)

    print()
    if failed:
        print(f"WARNING: {len(failed)} dataset(s) failed to download:")
        for f in failed:
            print(f"  - {f}")
        print("\nBenchmarks using failed datasets will be skipped during runs.")
        sys.exit(1)
    else:
        Path(".datasets_ready").write_text("ok")
        if spider_ok:
            print("All datasets cached. Future runs will use local cache + Spider execution scoring.")
        else:
            print("HF datasets cached (Spider string-match only — see instructions above).")
        print("Run: python run_benchmark.py")


if __name__ == "__main__":
    main()
