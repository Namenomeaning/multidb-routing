"""P1 mục 4-5 — cache-only isolation analysis (NO LLM calls).

Mục 4 (RQ2 isolation): does the domain-triage step SIGNIFICANTLY drop GT vs the unfiltered
top-10 card pool? McNemar on (GT-in-top10) vs (GT-in-triaged-pick). Discordant cells:
b = GT in top10 but triage dropped it; c = GT in pick but not top10 (=0, pick subset of top10).
If b is non-significant -> triage shrinks the pool at no real recall cost (the defensible RQ2 claim,
separated from the card+top10 recall gain).

Mục 5 (RQ3 coupling): characterize the score ties on the FILTERED pool. Tie-rate overall, and
split by pool composition (single-engine pool vs multi-engine pool) — if ties concentrate in
single-engine/same-domain pools, that confirms the triage gathers near-duplicate domains and the
ties are benchmark near-dup ambiguity, not a generic scoring defect. (Raw-top5 tie-rate would need
fresh maps on raw candidates = LLM -> deferred to the confirmed benchmark run.)

Usage:  python src/routing/p1_isolation_analysis.py --set <route_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load  # noqa: E402
from agent_rerank import top_pool  # noqa: E402
from decomp_card_eval import score_fixed  # noqa: E402

N = 2.0
MARGIN = 0.2  # near-tie band used for the headline V2 tie-break


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)


def near(sc: list[tuple[float, str]]) -> list[str]:
    top = sc[0][0]
    return [c for s, c in sc if top - s <= MARGIN and (top == 0 or s > 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    args = ap.parse_args()

    root = Path(args.set)
    name = root.name
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    card_idx = np.load(root / "index" / "card.npy")
    pool_ids = json.loads((root / "index" / "manifest.json").read_text())["instance_ids"]
    # GT keyed by BOTH query_id (ours_multidb) and md5(instance_id|question)[:12] (spider/bird),
    # matching the qid schemes used across the eval caches.
    gt_all = {}
    for q in load(root / "splits" / "test.jsonl"):
        iid = q["instance_id"]
        if q.get("query_id"):
            gt_all[q["query_id"]] = iid
        h = hashlib.md5((iid + "|" + q["question"]).encode()).hexdigest()[:12]
        gt_all[h] = iid

    tdir = root / "triage_cache"
    fdir = root / "agent_flow_cache"

    # ---------------- Mục 4: triage recall isolation (top10 vs picked) ----------------
    qv_p = tdir / f"qv_strat_cap{args.cap}_s{args.seed}.npy"
    ids_p = tdir / f"qv_strat_cap{args.cap}_s{args.seed}.ids.json"
    tri_p = tdir / f"triage_full_desc_cap{args.cap}_s{args.seed}.jsonl"
    print(f"\n========== {name} ==========")
    if qv_p.exists() and tri_p.exists():
        qids = json.loads(ids_p.read_text())
        qv = np.load(qv_p)
        picks = {r["qid"]: r["picked"] for r in load(tri_p)}
        pools10 = {qid: pool for qid, pool in zip(qids, top_pool(qv, card_idx, pool_ids, 10))}
        nq = b = c = both = 0
        in10 = inpick = 0
        for qid in qids:
            g = gt_all.get(qid)
            if g is None or qid not in picks:
                continue
            nq += 1
            h10 = g in pools10.get(qid, [])
            hpk = g in picks[qid]
            in10 += h10
            inpick += hpk
            if h10 and hpk:
                both += 1
            elif h10 and not hpk:
                b += 1
            elif hpk and not h10:
                c += 1
        p = mcnemar_exact(b, c)
        print(f"\n[Mục 4] triage recall isolation  (n={nq})")
        print(f"  recall  top10 (unfiltered) = {in10}/{nq} = {in10/nq:.3f}")
        print(f"  recall  picked (triaged)   = {inpick}/{nq} = {inpick/nq:.3f}")
        print(f"  drop = {in10-inpick} queries ({(in10-inpick)/nq*100:.1f}pp of all; "
              f"{b}/{in10} = {b/in10*100:.1f}% of in-pool GT dropped)")
        print(f"  McNemar top10-vs-picked: b(dropped)={b}, c(gained)={c}, p={p:.4g}")
        print(f"  -> triage cost {'SIGNIFICANT (hurts recall)' if p < 0.05 else 'NOT significant (recall-neutral shrink)'}")
    else:
        print(f"[Mục 4] SKIP — missing {qv_p.name} or {tri_p.name}")

    # ---------------- Mục 5: tie coupling on filtered pool ----------------
    parse = {r["qid"]: r["out"] for r in load(fdir / "parse_v2.jsonl")}
    mp = {(r["qid"], r["cand"]): r["out"] for r in load(fdir / "map_cal_v2.jsonl")}
    pool_by_q = defaultdict(list)
    for (qid, cand) in mp:
        pool_by_q[qid].append(cand)

    n_eval = 0
    n_tie = 0
    tie_single_eng = tie_multi_eng = 0
    n_single = n_multi = 0
    tie_all_same_eng = 0  # of tied queries, how many have the tied set all-same-engine
    for qid, pool in pool_by_q.items():
        ph = (parse.get(qid) or {}).get("phrases") or []
        if not ph or len(pool) < 2:
            continue
        n_eval += 1
        sc = [(score_fixed(ph, mp.get((qid, c), {}), adjs.get(c, {}), invs[c], N), c) for c in pool]
        sc.sort(key=lambda x: -x[0])
        pool_eng = {invs[c]["engine"] for c in pool}
        single = len(pool_eng) == 1
        n_single += single
        n_multi += (not single)
        nr = near(sc)
        is_tie = len(nr) >= 2
        if is_tie:
            n_tie += 1
            tie_eng = {invs[c]["engine"] for c in nr}
            if len(tie_eng) == 1:
                tie_all_same_eng += 1
            if single:
                tie_single_eng += 1
            else:
                tie_multi_eng += 1
    print(f"\n[Mục 5] tie coupling on filtered pool  (n={n_eval})")
    print(f"  overall tie-rate (δ={MARGIN}) = {n_tie}/{n_eval} = {n_tie/n_eval*100:.1f}%")
    print(f"  of tied queries, tied set all-same-engine = {tie_all_same_eng}/{n_tie} = "
          f"{tie_all_same_eng/n_tie*100:.1f}%")
    if n_single:
        print(f"  tie-rate | single-engine pool = {tie_single_eng}/{n_single} = {tie_single_eng/n_single*100:.1f}%")
    if n_multi:
        print(f"  tie-rate | multi-engine  pool = {tie_multi_eng}/{n_multi} = {tie_multi_eng/n_multi*100:.1f}%")
    print(f"  (raw-top5 tie-rate needs fresh maps = LLM -> deferred to confirmed run)")


if __name__ == "__main__":
    main()
