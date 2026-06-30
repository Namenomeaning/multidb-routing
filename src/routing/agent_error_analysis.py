"""Error analysis — WHY does the agent pick wrong when GT is in the pool?

Reuses the cached extractions/holistic/hybrid from agent_rerank.py (no LLM calls).
For every in-pool query (GT in top-K) that the chosen arm got WRONG, dumps the structured
evidence and aggregates failure patterns:

  - cross-engine vs intra-engine error (picked a DB of different/same engine as GT)
  - GT gated out by hybrid (connectivity=0 → dropped before the agent ever saw it)
  - GT vs pick deterministic score (GT below / tied / above the pick)
  - GT extraction fidelity (how many query phrases the LLM marked unmatched for the GT card)
  - same-domain sibling (pick & GT share top domain words) — the Sudarshan intra-domain trap

Usage: python agent_error_analysis.py --set <route_dir> [--kmax 8] [--arm hybrid|holistic|decomp]
       (must match the slice args used in the agent_rerank run that built the cache)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all  # noqa: E402
from agent_rerank import (  # noqa: E402
    score_candidate, connectivity, coverage, field2ent, top_pool, KMAX_HOLISTIC,
)

STOP = set("the a an of to in for and or by with on at from is are how many what "
           "which who list show give count number each all that this".split())


def domain_words(card: dict, k: int = 8) -> set[str]:
    txt = (card.get("domain_description", "") or "").lower()
    toks = [w.strip(".,;:()[]'\"") for w in txt.split()]
    toks = [w for w in toks if len(w) > 3 and w not in STOP]
    return set([w for w, _ in Counter(toks).most_common(k)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-dbs", type=int, default=30)
    ap.add_argument("--per-db-cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--arm", default="hybrid", choices=["hybrid", "holistic", "decomp"])
    ap.add_argument("--n", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--show", type=int, default=40, help="max wrong cases to print")
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    # rebuild the SAME slice (must match agent_rerank args)
    rng = np.random.default_rng(args.seed)
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        if d in cards:
            by_engine[dbs[d]["engine"]].append(d)
    pick_dbs, ei = [], 0
    engines = sorted(by_engine)
    while len(pick_dbs) < args.max_dbs and any(by_engine.values()):
        eng = engines[ei % len(engines)]
        if by_engine[eng]:
            pick_dbs.append(by_engine[eng].pop(0))
        ei += 1
    qs = []
    for d in pick_dbs:
        items = by_db[d]
        idx = rng.permutation(len(items))[:args.per_db_cap]
        qs.extend(items[i] for i in idx)
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    gts = [q["instance_id"] for q in qs]

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, args.kmax)

    cache_dir = root / "agent_cache"
    extr_cache = {(r["qid"], r["cand"]): r["out"]
                  for r in load(cache_dir / "extractions.jsonl")}
    holi_cache = {r["qid"]: r["out"] for r in load(cache_dir / "holistic.jsonl")}
    hyb_cache = {r["qid"]: r["pick"] for r in load(cache_dir / "hybrid.jsonl")}

    def sc(qid, c):
        return score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], args.n)

    def survivors(qid, pool):
        topK = pool[:KMAX_HOLISTIC]
        scored = sorted(((sc(qid, c), c) for c in topK), key=lambda x: -x[0])
        return [c for s, c in scored if s > 0] or topK

    def pick_of(qid, pool):
        poolK = pool[:args.K]
        if args.arm == "hybrid":
            return hyb_cache.get(qid, pool[0])
        if args.arm == "holistic":
            ch = holi_cache.get(qid, {}).get("choice")
            return pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]
        scored = [(sc(qid, c), c) for c in poolK]
        return scored[max(range(len(scored)), key=lambda i: (scored[i][0], -i))][1]

    cats = Counter()
    wrong = []
    for q, pool in zip(qs, pools):
        qid = q["_qid"]; gt = q["instance_id"]; poolK = pool[:args.K]
        if gt not in poolK:
            continue
        pick = pick_of(qid, pool)
        if pick == gt:
            continue
        gt_eng, pk_eng = dbs[gt]["engine"], dbs[pick]["engine"]
        gt_s, pk_s = sc(qid, gt), sc(qid, pick)
        surv = survivors(qid, pool)
        gt_gated = (args.arm == "hybrid" and gt not in surv)
        gt_ex = extr_cache.get((qid, gt), {})
        pk_ex = extr_cache.get((qid, pick), {})
        gt_unm = len(gt_ex.get("unmatched", []) or [])
        gt_mat = len(gt_ex.get("matched", []) or [])
        dom_overlap = len(domain_words(cards.get(gt, {})) & domain_words(cards.get(pick, {})))

        # categorize (priority order)
        if gt_gated:
            cat = "GT gated out (conn=0)"
        elif pk_eng != gt_eng:
            cat = "cross-engine pick"
        elif gt_s < pk_s:
            cat = "GT score < pick (det. wrong)"
        elif gt_s == pk_s:
            cat = "GT tied with pick (saturation)"
        else:
            cat = "GT score > pick (agent overrode score)"
        cats[cat] += 1
        if dom_overlap >= 2 and pk_eng == gt_eng:
            cats["  ...of which same-domain sibling"] += 1
        wrong.append((cat, q["question"], gt, gt_eng, gt_s, gt_mat, gt_unm,
                      pick, pk_eng, pk_s, dom_overlap, gt_gated))

    n_in = sum(1 for q, pool in zip(qs, pools) if q["instance_id"] in pool[:args.K])
    print(f"{root.name}  arm={args.arm}  K={args.K} n={args.n}  in-pool={n_in}  wrong={len(wrong)}  "
          f"in-pool acc={1 - len(wrong)/max(n_in,1):.3f}")
    print("\n== failure pattern counts ==")
    for c, n in cats.most_common():
        print(f"  {n:3d}  {c}")

    print(f"\n== wrong cases (up to {args.show}) ==")
    for (cat, ques, gt, ge, gs, gm, gu, pk, pe, ps, dov, gated) in wrong[:args.show]:
        print(f"\n[{cat}]")
        print(f"  Q: {ques[:140]}")
        print(f"  GT  {gt[:42]:42s} eng={ge:8s} score={gs:.3f} matched={gm} unmatched={gu}{'  [GATED]' if gated else ''}")
        print(f"  PICK{pk[:42]:42s} eng={pe:8s} score={ps:.3f}  domain-overlap={dov}")
        print(f"    GT card : {(cards.get(gt,{}).get('domain_description','') or '')[:160]}")
        print(f"    PICKcard: {(cards.get(pk,{}).get('domain_description','') or '')[:160]}")


if __name__ == "__main__":
    main()
