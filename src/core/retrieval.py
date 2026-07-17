"""Scale retrieval eval: raw-DDL arm (Sudarshan baseline) vs card arm (ours), same embedder.

Two-layer metrics (CLAUDE.md §3): retrieval recall (GT-in-top-k) reported separately.
Sudarshan-comparable: R@1 / R@3 / R@5 / mAP (micro per-query). Headline: DB-macro (per-DB
equal weight, memory benchmark-v3). Paired McNemar raw-vs-card on R@5 + R@1 hits.

Stratified slice (memory eval-slice-bias): --per-db-cap caps queries/GT-DB so no DB dominates;
every GT-DB with >=1 test query is represented. Default cap=5.

Usage:
  python retrieval_eval.py --bench <route_dir> --index <index_dir> [--per-db-cap 5] [--arms raw,card]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from src.core.openrouter import embed  # noqa: E402

BATCH = 96


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def stratified(queries: list[dict], cap: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_db = defaultdict(list)
    for q in queries:
        by_db[q["instance_id"]].append(q)
    out = []
    for db in sorted(by_db):
        items = by_db[db]
        idx = rng.permutation(len(items))[:cap]
        out.extend(items[i] for i in idx)
    return out


def embed_all(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH):
        vecs.extend(embed(texts[i:i + BATCH]))
        print(f"    q-embed {min(i+BATCH,len(texts))}/{len(texts)}", flush=True)
    return np.asarray(vecs, dtype=np.float32)


def ranks_for(qv: np.ndarray, index: np.ndarray, ids: list[str], gts: list[str]) -> list[int | None]:
    sims = qv @ index.T                      # (nq, ndb) cosine (both L2-normalized)
    order = np.argsort(-sims, axis=1)
    pos = {iid: j for j, iid in enumerate(ids)}
    out = []
    for i, gt in enumerate(gts):
        g = pos.get(gt)
        if g is None:
            out.append(None); continue
        rank = int(np.where(order[i] == g)[0][0]) + 1
        out.append(rank)
    return out


def metrics(ranks: list[int | None], gts: list[str]) -> dict:
    n = len(ranks)
    def hit(k): return sum(1 for r in ranks if r and r <= k)
    micro = {
        "R@1": hit(1) / n, "R@3": hit(3) / n, "R@5": hit(5) / n,
        "mAP": sum((1.0 / r) for r in ranks if r) / n,  # single-GT -> MRR == mAP
        "GT_unfound": sum(1 for r in ranks if r is None),
    }
    # DB-macro: average per-DB R@1/R@5
    by_db = defaultdict(list)
    for r, g in zip(ranks, gts):
        by_db[g].append(r)
    macro_r1 = np.mean([np.mean([1.0 if (r and r == 1) else 0.0 for r in rs]) for rs in by_db.values()])
    macro_r5 = np.mean([np.mean([1.0 if (r and r <= 5) else 0.0 for r in rs]) for rs in by_db.values()])
    micro["DBmacro_R@1"] = float(macro_r1)
    micro["DBmacro_R@5"] = float(macro_r5)
    return micro


def mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided binomial McNemar p over the discordant pairs (no continuity fudge)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def mcnemar(a: list[int | None], b: list[int | None], k: int) -> dict:
    """paired: a-hit & not b-hit  vs  b-hit & not a-hit at top-k, with exact two-sided p."""
    a_only = sum(1 for ra, rb in zip(a, b) if (ra and ra <= k) and not (rb and rb <= k))
    b_only = sum(1 for ra, rb in zip(a, b) if (rb and rb <= k) and not (ra and ra <= k))
    return {"raw_only": a_only, "card_only": b_only, "discordant": a_only + b_only,
            "p": mcnemar_p(a_only, b_only)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-db-cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    args = ap.parse_args()

    bench, idxd = Path(args.bench), Path(args.index)
    man = json.loads((idxd / "manifest.json").read_text())
    ids = man["instance_ids"]
    raw = np.load(idxd / "raw.npy"); card = np.load(idxd / "card.npy")

    qs = stratified(load(bench / "splits" / "test.jsonl"), args.per_db_cap, args.seed)
    gts = [q["instance_id"] for q in qs]
    print(f"{bench.name}: {len(ids)} DBs, {len(qs)} test queries (cap {args.per_db_cap}/DB), embed={man['embedding_model']}")
    qv = embed_all([q["question"] for q in qs])

    r_raw = ranks_for(qv, raw, ids, gts)
    r_card = ranks_for(qv, card, ids, gts)
    m_raw, m_card = metrics(r_raw, gts), metrics(r_card, gts)

    print(f"\n{'metric':14s} {'raw-DDL':>10s} {'card':>10s}  (Sudarshan baseline vs ours)")
    for k in ["R@1", "R@3", "R@5", "mAP", "DBmacro_R@1", "DBmacro_R@5", "GT_unfound"]:
        print(f"  {k:12s} {m_raw[k]:>10.4f} {m_card[k]:>10.4f}")
    print(f"\nMcNemar @5: {mcnemar(r_raw, r_card, 5)}")
    print(f"McNemar @1: {mcnemar(r_raw, r_card, 1)}")


if __name__ == "__main__":
    main()
