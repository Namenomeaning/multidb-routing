"""MACHINERY smoke test (NOT a scientific result): embed 2 arms for a small DB set,
run test queries, report GT rank under raw-DDL vs card index.

A 10-DB pool makes top-5 trivial — this only proves the embed->retrieve path runs
end-to-end and that the card text is well-formed. Real recall conclusions need scale
(see memory eval-slice-bias). Caps queries for speed.

Usage:
  python smoke_retrieval.py --bench <route_dir> --semantic <semantic_dir> [--max-q 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.clients.openrouter import embed  # noqa: E402
from build_semantic import build_inventory, render_entities, render_relations  # noqa: E402


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def card_text(card: dict, inv: dict) -> str:
    """B4 index text: card (description + glossary) + inventory A (entities/fields/declared rel)."""
    gloss = "; ".join(f"{k}: {v}" for k, v in (card.get("term_glossary") or {}).items())
    return (f"{card.get('domain_description','')}\n"
            f"Glossary: {gloss}\n"
            f"Entities:\n{render_entities(inv)}\n"
            f"Relationships:\n{render_relations(inv)}")


def cosine_topk(qvec: list[float], index: list[tuple[str, list[float]]], k: int):
    scored = sorted(((sum(a * b for a, b in zip(qvec, v)), iid) for iid, v in index), reverse=True)
    return scored[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="route dir with databases.jsonl + splits/test.jsonl")
    ap.add_argument("--semantic", required=True, help="dir with cards.jsonl + inventory.jsonl")
    ap.add_argument("--max-q", type=int, default=60)
    args = ap.parse_args()

    bench, sem = Path(args.bench), Path(args.semantic)
    dbs = {d["instance_id"]: d for d in load(bench / "databases.jsonl")}
    cards = {c["instance_id"]: c for c in load(sem / "cards.jsonl")}
    invs = {i["instance_id"]: i for i in load(sem / "inventory.jsonl")}
    built = sorted(cards)                      # only DBs we built semantic for
    print(f"DBs in semantic set: {len(built)}")

    # build two indexes
    raw_texts = [dbs[iid]["schema_text"] for iid in built]
    card_texts = [card_text(cards[iid], invs[iid]) for iid in built]
    raw_vecs = embed(raw_texts)
    card_vecs = embed(card_texts)
    raw_index = list(zip(built, raw_vecs))
    card_index = list(zip(built, card_vecs))

    # test queries restricted to the built DBs
    builtset = set(built)
    qs = [q for q in load(bench / "splits" / "test.jsonl") if q["instance_id"] in builtset][:args.max_q]
    qvecs = embed([q["question"] for q in qs])

    def rank_of(topk, gt):
        for r, (_, iid) in enumerate(topk, 1):
            if iid == gt:
                return r
        return None

    raw_hits5 = card_hits5 = raw_r1 = card_r1 = 0
    for q, qv in zip(qs, qvecs):
        gt = q["instance_id"]
        rr = rank_of(cosine_topk(qv, raw_index, 5), gt)
        cr = rank_of(cosine_topk(qv, card_index, 5), gt)
        raw_hits5 += rr is not None
        card_hits5 += cr is not None
        raw_r1 += rr == 1
        card_r1 += cr == 1
    n = len(qs)
    print(f"\nMACHINERY SMOKE (n={n} queries over {len(built)} DBs — top-5 trivial at this scale):")
    print(f"  raw-DDL  arm: GT-in-top5 {raw_hits5}/{n}  R@1 {raw_r1}/{n}")
    print(f"  card     arm: GT-in-top5 {card_hits5}/{n}  R@1 {card_r1}/{n}")
    print("  (interpret as: pipeline runs end-to-end; NOT a recall conclusion)")


if __name__ == "__main__":
    main()
