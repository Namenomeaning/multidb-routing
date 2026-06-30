"""Query-side robustness: card vs raw retrieval under Spider-Syn / Spider-Realistic.

Same DB index (raw.npy + card.npy of spider_route, 206 DBs) — only the QUERY changes:
  Spider-Syn  [Gan et al. ACL 2021]   : schema words in the question replaced by synonyms.
                                         Has BOTH original (SpiderQuestion) + synonym (SpiderSynQuestion)
                                         → paired clean-vs-perturbed drop per arm.
  Spider-Realistic [Deng et al. NAACL 2021]: explicit column mentions removed from the question.

Complements E3c (schema-side masking). Tests whether card's NL domain prose is more robust than raw
DDL to query↔schema lexical mismatch. Retrieval layer only. Pool = full 206 (route among all).

Usage: python robustness_query_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all  # noqa: E402

SPIDER = HERE.parent.parent / "data" / "spider"
SYN = Path("/tmp/spidersyn/syn_dev.json")
REAL = Path("/tmp/spider-realistic.json")


def ranks_for(qv, index, ids, gts):
    order = np.argsort(-(qv @ index.T), axis=1)
    pos = {iid: j for j, iid in enumerate(ids)}
    out = []
    for i, gt in enumerate(gts):
        g = pos.get(gt)
        out.append(int(np.where(order[i] == g)[0][0]) + 1 if g is not None else 10**9)
    return out


def report(tag, ranks, gts):
    n = len(ranks)
    by = defaultdict(list)
    for r, g in zip(ranks, gts):
        by[g].append(r)
    def mi(k): return sum(1 for r in ranks if r <= k) / n
    def ma(k): return float(np.mean([np.mean([r <= k for r in rs]) for rs in by.values()]))
    return {"R@1": mi(1), "R@5": mi(5), "R@10": mi(10), "mR@1": ma(1), "mR@5": ma(5)}


def mcnemar(a, b, k):
    a1 = sum(1 for x, y in zip(a, b) if x <= k and y > k)
    b1 = sum(1 for x, y in zip(a, b) if y <= k and x > k)
    from math import comb
    nn = a1 + b1
    p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(min(a1, b1) + 1)) / 2 ** nn)
    return a1, b1, p


def main():
    db2iid = {d["db_id"]: d["instance_id"] for d in load(SPIDER / "databases.jsonl")}
    man = json.loads((SPIDER / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    raw = np.load(SPIDER / "index" / "raw.npy")
    card = np.load(SPIDER / "index" / "card.npy")

    syn = json.load(open(SYN))
    real = json.load(open(REAL))
    gts_syn = [db2iid[r["db_id"]] for r in syn]
    gts_real = [db2iid[r["db_id"]] for r in real]

    print(f"index 206 DBs | SYN {len(syn)}q/{len(set(gts_syn))}DB | REAL {len(real)}q/{len(set(gts_real))}DB")
    print("embedding queries (clean+syn+real)...")
    qv_clean = embed_all([r["SpiderQuestion"] for r in syn])
    qv_syn = embed_all([r["SpiderSynQuestion"] for r in syn])
    qv_real = embed_all([r["question"] for r in real])

    res = {
        ("SYN", "clean", "raw"): ranks_for(qv_clean, raw, ids, gts_syn),
        ("SYN", "clean", "card"): ranks_for(qv_clean, card, ids, gts_syn),
        ("SYN", "syn", "raw"): ranks_for(qv_syn, raw, ids, gts_syn),
        ("SYN", "syn", "card"): ranks_for(qv_syn, card, ids, gts_syn),
        ("REAL", "real", "raw"): ranks_for(qv_real, raw, ids, gts_real),
        ("REAL", "real", "card"): ranks_for(qv_real, card, ids, gts_real),
    }

    print(f"\n{'set/cond/arm':22s} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'mR@1':>7} {'mR@5':>7}")
    for key, ranks in res.items():
        g = gts_syn if key[0] == "SYN" else gts_real
        m = report("/".join(key), ranks, g)
        print(f"{'/'.join(key):22s} {m['R@1']:>7.3f} {m['R@5']:>7.3f} {m['R@10']:>7.3f} {m['mR@1']:>7.3f} {m['mR@5']:>7.3f}")

    print("\n-- Spider-Syn paired drop (clean -> syn), same 1034 q --")
    for arm in ("raw", "card"):
        c = report("", res[("SYN", "clean", arm)], gts_syn)
        s = report("", res[("SYN", "syn", arm)], gts_syn)
        print(f"  {arm:4s} R@1 {c['R@1']:.3f}->{s['R@1']:.3f} ({s['R@1']-c['R@1']:+.3f})  "
              f"R@5 {c['R@5']:.3f}->{s['R@5']:.3f} ({s['R@5']-c['R@5']:+.3f})")
    for k in (1, 5):
        a1, b1, p = mcnemar(res[("SYN", "syn", "raw")], res[("SYN", "syn", "card")], k)
        print(f"  McNemar SYN@{k} card vs raw: raw_only={a1} card_only={b1} p={p:.4g}")
    print("\n-- Spider-Realistic, raw vs card --")
    for k in (1, 5):
        a1, b1, p = mcnemar(res[("REAL", "real", "raw")], res[("REAL", "real", "card")], k)
        print(f"  McNemar REAL@{k} card vs raw: raw_only={a1} card_only={b1} p={p:.4g}")


if __name__ == "__main__":
    main()
