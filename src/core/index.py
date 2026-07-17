"""Build retrieval index vectors for a v2 benchmark set: two arms per DB.

  raw  view = schema_text (Sudarshan baseline — embed raw DDL)
  card view = B4 render: card(description+glossary) + A(entities/fields/declared rel)

Both arms use the SAME embedder (OpenRouter, project policy) so the only variable is
the indexed TEXT (raw DDL vs semantic card) — the clean E1 / head-to-head contrast.
Saves raw.npy, card.npy (L2-normalized; cosine = dot) + manifest.json (id order + model).

Usage:
  python build_index.py --bench <route_dir> --semantic <semantic_dir> --out <index_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.core.openrouter import embed, embedding_model  # noqa: E402
from src.core.semantic_card import render_entities, render_relations  # noqa: E402

BATCH = 96


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def card_text(card: dict, inv: dict) -> str:
    gloss = "; ".join(f"{k}: {v}" for k, v in (card.get("term_glossary") or {}).items())
    return (f"{card.get('domain_description','')}\n"
            f"Glossary: {gloss}\n"
            f"Entities:\n{render_entities(inv)}\n"
            f"Relationships:\n{render_relations(inv)}")


def embed_all(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        vecs.extend(embed(chunk))
        print(f"    embedded {min(i+BATCH, len(texts))}/{len(texts)}", flush=True)
    return np.asarray(vecs, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench, sem, out = Path(args.bench), Path(args.semantic), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dbs = {d["instance_id"]: d for d in load(bench / "databases.jsonl")}
    cards = {c["instance_id"]: c for c in load(sem / "cards.jsonl")}
    invs = {i["instance_id"]: i for i in load(sem / "inventory.jsonl")}

    ids = [iid for iid in dbs if iid in cards and iid in invs]
    missing = [iid for iid in dbs if iid not in ids]
    print(f"{bench.name}: {len(ids)} DBs to index ({len(missing)} missing semantic)")

    print("  raw arm...")
    raw = embed_all([dbs[i]["schema_text"] for i in ids])
    print("  card arm...")
    card = embed_all([card_text(cards[i], invs[i]) for i in ids])

    np.save(out / "raw.npy", raw)
    np.save(out / "card.npy", card)
    (out / "manifest.json").write_text(json.dumps({
        "instance_ids": ids,
        "missing_semantic": missing,
        "embedding_model": embedding_model(),
        "dim": int(raw.shape[1]),
        "arms": ["raw", "card"],
    }, indent=2))
    print(f"  saved raw{raw.shape} card{card.shape} -> {out}")


if __name__ == "__main__":
    main()
