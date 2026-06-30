"""Baseline 1 — embedding-similarity routing (no LLM), factorial representation {card, raw}.

For each query: cosine(query, every DB vector) -> argmax = routed instance (final-routing top-1),
plus recall@k. Pure retrieval-as-routing floor. Two prebuilt indices reused (index/card.npy,
index/raw.npy); no embedding rebuild, no LLM.

Two-layer reporting (CLAUDE.md §3): R@1(all) = top-1 routing accuracy; recall@5/@10 reported
separately. Per-GT-engine split (the multi-engine claim is per engine). Bootstrap CI on R@1.

Usage: python baseline_embed_routing.py --set <route_dir> [--per-db-cap 5] [--seed 260611]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all, stratified  # noqa: E402


def boot_ci(hits: list[int], iters: int = 1000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(hits, dtype=float)
    if len(a) == 0:
        return (0.0, 0.0)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(iters)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--per-db-cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--dump", default="", help="write per-qid top-1 hits jsonl (for paired McNemar)")
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    cards = {c["instance_id"] for c in load(root / "semantic" / "cards.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    pos = {i: j for j, i in enumerate(ids)}
    idx = {"card": np.load(root / "index" / "card.npy"),
           "raw": np.load(root / "index" / "raw.npy")}

    import hashlib
    qs = [q for q in load(root / "splits" / "test.jsonl") if q["instance_id"] in cards]
    qs = stratified(qs, args.per_db_cap, args.seed)
    gts = [q["instance_id"] for q in qs]
    qids = [q.get("query_id") or hashlib.md5((q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
            for q in qs]
    dump_rows = []
    print(f"{root.name}: {len(ids)} DBs, {len(qs)} q (cap{args.per_db_cap} seed{args.seed})")

    qv = embed_all([q["question"] for q in qs])
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)

    for rep in ("raw", "card"):
        M = idx[rep]
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sims = qv @ M.T                                  # [Nq, Ndb]
        order = np.argsort(-sims, axis=1)                # ranked db indices per query
        r1, r5, r10 = [], [], []
        per_eng = defaultdict(lambda: [0, 0])
        for row, gt, qid in zip(order, gts, qids):
            gj = pos[gt]
            rank = int(np.where(row == gj)[0][0]) + 1
            r1.append(int(rank == 1)); r5.append(int(rank <= 5)); r10.append(int(rank <= 10))
            ge = dbs[gt]["engine"]
            per_eng[ge][0] += int(rank == 1); per_eng[ge][1] += 1
            if args.dump:
                dump_rows.append({"qid": qid, "rep": rep, "gt": gt, "engine": ge,
                                  "correct": int(rank == 1)})
        lo, hi = boot_ci(r1)
        print(f"\n== EMBED-top1 [{rep}] ==  R@1={np.mean(r1):.3f} (95%CI {lo:.3f}-{hi:.3f})  "
              f"R@5={np.mean(r5):.3f}  R@10={np.mean(r10):.3f}  N={len(gts)}")
        for ge in sorted(per_eng):
            ok, n = per_eng[ge]
            print(f"   └ GT-engine {ge:11s} N={n:4d}  R@1={ok/max(n,1):.3f}")

    if args.dump:
        Path(args.dump).write_text("\n".join(json.dumps(r) for r in dump_rows))
        print(f"\ndumped {len(dump_rows)} per-qid rows -> {args.dump}")


if __name__ == "__main__":
    main()
