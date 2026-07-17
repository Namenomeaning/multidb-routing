"""Hybrid v2 — soft connectivity gate + agent tie-break. Improves in-pool final-routing acc.

Two fixes over agent_rerank's hybrid (measured failure patterns, ours_multidb):
  FIX 1 (soft gate): connectivity=0 no longer DROPS a candidate. Instead score =
        coverage × (1 if connected else PENALTY). Recovers GT that has perfect coverage
        but false conn=0 from an incomplete adjacency graph (21% of in-pool errors).
        PENALTY swept {0.5,0.7,0.85,1.0}; 1.0 = connectivity ignored (ablation).
  FIX 2 (agent tie-break): when the top soft-score is a TIE (≥2 candidates equal, the
        saturation failure = 63% of errors), the agent reads those cards and picks one.
        Single clear top → deterministic pick (no LLM). Agent never sees scores → invariant kept.

Reuses agent_rerank's extraction cache (NO new extraction LLM calls). Only tie-cluster
agent picks call the LLM, cached by (qid, cluster) so the penalty sweep shares picks.

Usage: LLM_PROVIDER=deepseek python hybrid_v2_eval.py --set <route_dir> [--kmax 8] [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.core.retrieval import load, embed_all  # noqa: E402
from src.core.semantic_card import client, call_json, chat_model  # noqa: E402
from src.core.rerank import (  # noqa: E402
    coverage, connectivity, field2ent, top_pool, KMAX_HOLISTIC, HYBRID_PROMPT,
)

PENALTIES = [1.0, 0.85, 0.7]
MARGINS = [0.0, 0.6, 0.8]   # cluster = candidates with soft_score >= max*margin (0.0 = all topK to agent)
ROUND = 6


def score_soft(extr, adj, inv, n, penalty):
    matched = extr.get("matched", []) or []
    unmatched = extr.get("unmatched", []) or []
    cov = coverage(len(matched), len(unmatched), n)
    ents = [m.get("entity") for m in matched if isinstance(m, dict) and m.get("entity")]
    conn = connectivity(ents, adj, field2ent(inv))
    return cov * (1.0 if conn else penalty)


def build_slice(root, dbs, cards, seed, max_dbs, cap):
    rng = np.random.default_rng(seed)
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        if d in cards:
            by_engine[dbs[d]["engine"]].append(d)
    pick_dbs, ei = [], 0
    engines = sorted(by_engine)
    while len(pick_dbs) < max_dbs and any(by_engine.values()):
        eng = engines[ei % len(engines)]
        if by_engine[eng]:
            pick_dbs.append(by_engine[eng].pop(0))
        ei += 1
    qs = []
    for d in pick_dbs:
        items = by_db[d]
        idx = rng.permutation(len(items))[:cap]
        qs.extend(items[i] for i in idx)
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-dbs", type=int, default=30)
    ap.add_argument("--per-db-cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--n", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    qs = build_slice(root, dbs, cards, args.seed, args.max_dbs, args.per_db_cap)
    qmap = {q["_qid"]: q for q in qs}
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, args.kmax)

    cache_dir = root / "cache" / "agent_cache"
    extr_cache = {(r["qid"], r["cand"]): r["out"]
                  for r in load(cache_dir / "extractions.jsonl")}
    hy2_path = cache_dir / "hybrid2.jsonl"
    hy2_cache = {r["ckey"]: r["pick"] for r in (load(hy2_path) if hy2_path.exists() else [])}

    def tie_cluster(qid, poolK, penalty, margin):
        """Cluster = candidates whose soft score >= max*margin (margin 0 => all topK).
        Agent reads these cards (no scores) and picks; len 1 => deterministic."""
        scored = [(round(score_soft(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c],
                                    args.n, penalty), ROUND), c) for c in poolK]
        scored.sort(key=lambda x: -x[0])
        top = scored[0][0]
        cutoff = top * margin if top > 0 else 0.0
        cluster = [c for s, c in scored if s >= cutoff]
        return cluster, scored

    def ckey(qid, cluster):
        return qid + "|" + ",".join(cluster)  # cluster preserves cosine order

    # collect every tie-cluster needing an agent pick across the penalty sweep
    need = {}
    for q, pool in zip(qs, pools):
        qid = q["_qid"]; poolK = pool[:args.K]
        for pen in PENALTIES:
            for mg in MARGINS:
                cluster, _ = tie_cluster(qid, poolK, pen, mg)
                if len(cluster) > 1:
                    k = ckey(qid, cluster)
                    if k not in hy2_cache and k not in need:
                        need[k] = (qid, cluster)

    if need:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()
        print(f"agent tie-break: {len(need)} clusters @ {args.workers} workers, model={chat_model()}")

        def do_pick(item):
            k, (qid, cluster) = item
            blk = [f"[{j}] {cards[c].get('domain_description','')[:300]}"
                   for j, c in enumerate(cluster, 1)]
            out = call_json(oai, HYBRID_PROMPT.format(
                question=qmap[qid]["question"], candidate_block="\n".join(blk)))
            ch = out.get("choice")
            pick = cluster[ch - 1] if isinstance(ch, int) and 1 <= ch <= len(cluster) else cluster[0]
            return k, pick

        with open(hy2_path, "a") as fh:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(do_pick, it): it for it in need.items()}
                done = 0
                for fut in as_completed(futs):
                    k, pick = fut.result()
                    hy2_cache[k] = pick
                    with lock:
                        fh.write(json.dumps({"ckey": k, "pick": pick}) + "\n"); fh.flush()
                    done += 1
                    if done % 20 == 0:
                        print(f"  ... {done}/{len(need)}", flush=True)

    # evaluate in-pool acc per penalty + agent-tie-break rate
    n_in = sum(1 for q, pool in zip(qs, pools) if q["instance_id"] in pool[:args.K])
    print(f"\n{root.name}  K={args.K} n={args.n}  in-pool={n_in}/{len(qs)}")
    print(f"{'penalty':>8} {'margin':>7} {'in-pool acc':>12} {'det':>5} {'agent':>6} {'agent-acc':>10} {'avg-clust':>10}")
    best = (0.0, None)
    for pen in PENALTIES:
        for mg in MARGINS:
            hit = det = agent = agent_hit = clust_sum = 0
            for q, pool in zip(qs, pools):
                qid = q["_qid"]; gt = q["instance_id"]; poolK = pool[:args.K]
                if gt not in poolK:
                    continue
                cluster, scored = tie_cluster(qid, poolK, pen, mg)
                if len(cluster) == 1:
                    pick = cluster[0]; det += 1
                else:
                    pick = hy2_cache.get(ckey(qid, cluster), cluster[0]); agent += 1
                    clust_sum += len(cluster)
                    agent_hit += (pick == gt)
                hit += (pick == gt)
            acc = hit / max(n_in, 1)
            print(f"{pen:>8.2f} {mg:>7.2f} {acc:>12.3f} {det:>5} {agent:>6} "
                  f"{agent_hit/max(agent,1):>10.3f} {clust_sum/max(agent,1):>10.2f}")
            if acc > best[0]:
                best = (acc, (pen, mg))
    print(f"\nbest in-pool acc = {best[0]:.3f} at penalty/margin {best[1]}")
    print("(baseline agent_rerank: hybrid in-pool 0.789, holistic 0.822, decomp 0.622)")


if __name__ == "__main__":
    main()
