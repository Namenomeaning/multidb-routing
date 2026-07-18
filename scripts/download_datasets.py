#!/usr/bin/env python3
"""Fetch the raw source datasets for the multi-DB routing benchmark into ``data/_raw/``.

This is the FROM-SCRATCH entry point. It is provided for provenance and transparency.
The PRIMARY way to reproduce the paper numbers is the prebuilt artifacts already shipped
under ``data/{multidb,spider,bird}/`` (see ../README.md) — you do NOT need to run this to
re-run the evaluation.

Covers every source listed in the thesis CLAUDE.md §10:
  - Spider (SQL)              https://yale-lily.github.io/spider   (Google-Drive zip; manual)
  - BIRD (SQL)               https://bird-bench.github.io/        (JSON-only range fetch; automated)
  - DocSpider (Mongo tier)   Cambridge NLP article                (manual)
  - MongoDB-EAI native       HF: mongodb-eai/natural-language-to-mongosh   (automated if `datasets`)
  - CypherBench (Neo4j)      HF: megagonlabs/cypherbench                   (automated if `datasets`)
  - Text2Cypher 2024 (Neo4j) HF: neo4j/text2cypher-2024v1                  (automated if `datasets`)

Idempotent: a source whose target dir already has content is skipped. Prints provenance +
on-disk size per source at the end.

⚠️ UNVERIFIED END-TO-END. External endpoints, formats, and access terms change; some sources
gate behind a manual download or license click. This script was NOT run as part of producing
the shipped artifacts (those predate it). Spot-check at least one source before relying on it,
and confirm each dataset's redistribution terms before publishing anything derived from it.

Usage:
    python scripts/download_datasets.py            # fetch all automatable sources
    python scripts/download_datasets.py --only bird mongodb_eai
    python scripts/download_datasets.py --list     # print the source manifest and exit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # multidb-routing/scripts
ROOT = HERE.parent                              # multidb-routing/
RAW = ROOT / "data" / "_raw"

# BIRD's official zip; the bundled fetcher pulls only the JSON members (~tens of MB, not 8.9 GB).
BIRD_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"


def _dir_size(p: Path) -> str:
    if not p.exists():
        return "absent"
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    for unit in ("B", "K", "M", "G"):
        if total < 1024 or unit == "G":
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.0f}G"


def _has_content(p: Path) -> bool:
    return p.exists() and any(p.rglob("*"))


def fetch_bird(target: Path) -> str:
    """Reuse the bundled range-GET fetcher (JSON members only). Automated."""
    target.mkdir(parents=True, exist_ok=True)
    fetcher = HERE / "build" / "fetch_bird_json_remote.py"
    cmd = [
        sys.executable, str(fetcher),
        "--url", BIRD_ZIP_URL,
        "--out", str(target / "train"),
        "--want", "train/train.json", "train/train_tables.json",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return "ok"


def fetch_hf(repo: str, target: Path) -> str:
    """Save a HuggingFace dataset to disk. Automated if `datasets` is installed."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        return ("SKIP: `datasets` not installed (pip install datasets), or fetch manually from "
                f"https://huggingface.co/datasets/{repo}")
    target.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo)
    ds.save_to_disk(str(target))
    return "ok"


def manual(url: str, target: Path) -> str:
    """Source that cannot be reliably auto-fetched (Drive/article paywall/license click)."""
    target.mkdir(parents=True, exist_ok=True)
    return f"MANUAL: download from {url} and unzip into {target}"


# name -> (target subdir under data/_raw, fetch callable, provenance note)
SOURCES = {
    "spider": (
        "spider",
        lambda t: manual("https://yale-lily.github.io/spider", t),
        "Spider SQL (Yu et al. 2018) — Google-Drive zip, manual download + license.",
    ),
    "bird": (
        "sql_bird",
        fetch_bird,
        "BIRD SQL (Li et al. 2024) — JSON members range-fetched from the official train.zip.",
    ),
    "docspider": (
        "docspider",
        lambda t: manual(
            "https://www.cambridge.org/core/journals/natural-language-processing/article/"
            "docspider-a-dataset-of-crossdomain-natural-language-querying-for-mongodb/"
            "1E35B1DBF843B9E0F444B595B975695A", t),
        "DocSpider (MongoDB tier) — Cambridge NLP article supplement, manual.",
    ),
    "mongodb_eai": (
        "mongodb_native",
        lambda t: fetch_hf("mongodb-eai/natural-language-to-mongosh", t),
        "MongoDB-EAI NL→mongosh native — HuggingFace.",
    ),
    "cypherbench": (
        "neo4j",
        lambda t: fetch_hf("megagonlabs/cypherbench", t),
        "CypherBench (Neo4j) — HuggingFace.",
    ),
    "text2cypher": (
        "neo4j_native",
        lambda t: fetch_hf("neo4j/text2cypher-2024v1", t),
        "Text2Cypher 2024 (Neo4j native) — HuggingFace.",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="fetch only these sources")
    ap.add_argument("--list", action="store_true", help="print the source manifest and exit")
    ap.add_argument("--force", action="store_true", help="re-fetch even if the target already has content")
    args = ap.parse_args()

    if args.list:
        print("Sources (CLAUDE.md §10) -> data/_raw/<dir>:\n")
        for name, (sub, _, note) in SOURCES.items():
            print(f"  {name:14s} -> data/_raw/{sub:16s}  {note}")
        return

    print("⚠️  FROM-SCRATCH downloader — UNVERIFIED end-to-end. The shipped artifacts under")
    print("    data/{multidb,spider,bird}/ are the primary reproduction path; see README.md.\n")

    names = args.only or list(SOURCES)
    results: dict[str, str] = {}
    for name in names:
        sub, fetch, note = SOURCES[name]
        target = RAW / sub
        print(f"[{name}] {note}")
        if _has_content(target) and not args.force:
            results[name] = f"skip (exists, {_dir_size(target)}); --force to refetch"
            print(f"  -> {results[name]}\n")
            continue
        try:
            results[name] = fetch(target)
        except Exception as exc:  # noqa: BLE001
            results[name] = f"ERROR: {type(exc).__name__}: {exc}"
        print(f"  -> {results[name]}  [{_dir_size(target)}]\n")

    print("=== summary ===")
    for name in names:
        print(f"  {name:14s} {results[name]}")
    print("\nNext: run scripts/*.py (see scripts/README.md) to rebuild data/{multidb,spider,bird}.")
    print("Confirm each dataset's redistribution terms before publishing derived artifacts.")


if __name__ == "__main__":
    main()
