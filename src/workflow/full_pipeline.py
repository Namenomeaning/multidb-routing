"""Full-set final-routing eval: v1 (specific-domain prompt) vs v0 (base), with McNemar.

Runs the WHOLE test split (not the 30-DB slice). For each query: retrieve top-kmax over the
FULL card index, extract over top-5 (deterministic Coverage×Connectivity score), then the agent
picks among top-5 cards+scores under two prompts (v0 base, v1 specific-domain). Reuses every
existing cache (extractions.jsonl, pv_v0_base.jsonl, pv_v1_specific.jsonl) and only adds the
queries not yet cached.

Reports two layers (CLAUDE.md §3): pool recall (GT-in-top-K) separate from final R@1; final R@1
both overall (comparable to Sudarshan 78.65% on Spider) and conditioned on GT-in-pool. Plus
engine-acc and DB-macro, and McNemar v1-vs-v0.

Usage: LLM_PROVIDER=deepseek python full_eval_v1.py --set <route_dir> [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(HERE))
from src.core.retrieval import load, embed_all, stratified  # noqa: E402
from src.core.semantic_card import client, call_json, chat_model, render_entities  # noqa: E402
from src.core.rerank import score_candidate, top_pool, KMAX_HOLISTIC, EXTRACT_PROMPT  # noqa: E402
from src.workflow.prompt_variants import VARIANTS  # noqa: E402

ARMS = {"v0_base": VARIANTS["v0_base"], "v1_specific": VARIANTS["v1_specific"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-db-cap", type=int, default=5, help="stratified cap queries/GT-DB (matches retrieval slice)")
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--engine", default="", help="restrict pool+queries to one engine (single-type routing); '' = all")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0=all, for smoke)")
    args = ap.parse_args()
    tag = args.engine or "all"  # pick-cache suffix so engine subsets don't clobber the mixed run

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    if args.engine:  # single-type routing: keep only this engine's DBs in the pool + index
        keep = [j for j, i in enumerate(ids) if dbs[i]["engine"] == args.engine]
        ids = [ids[j] for j in keep]
        card_idx = card_idx[keep]

    qs = [q for q in load(root / "splits" / "test.jsonl")
          if q["instance_id"] in cards and (not args.engine or dbs[q["instance_id"]]["engine"] == args.engine)]
    qs = stratified(qs, args.per_db_cap, args.seed)
    if args.limit:
        qs = qs[:args.limit]
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    qmap = {q["_qid"]: q for q in qs}
    print(f"{root.name}: {len(ids)} DBs, {len(qs)} test queries, kmax={args.kmax}, model will be set per-call")

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, args.kmax)
    pool_by = {q["_qid"]: pool for q, pool in zip(qs, pools)}

    cache_dir = root / "agent_cache"
    cache_dir.mkdir(exist_ok=True)
    ex_path = cache_dir / "extractions.jsonl"
    extr_cache = {(r["qid"], r["cand"]): r["out"] for r in (load(ex_path) if ex_path.exists() else [])}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    oai = client(); lock = threading.Lock()
    print(f"model={chat_model()}")

    def run(tasks, fn, fh, label):
        if not tasks:
            return
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, t): t for t in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                done += 1
                if done % 50 == 0:
                    print(f"  ... {label} {done}/{len(tasks)}", flush=True)

    # Phase 1: extraction over top-5 (the holistic candidate set)
    ex_tasks = [(q["_qid"], c) for q in qs for c in pool_by[q["_qid"]][:KMAX_HOLISTIC]
                if (q["_qid"], c) not in extr_cache]
    def do_ex(t):
        qid, cand = t
        out = call_json(oai, EXTRACT_PROMPT.format(
            question=qmap[qid]["question"], engine=invs[cand]["engine"],
            entity_block=render_entities(invs[cand])))
        extr_cache[(qid, cand)] = out
        return {"qid": qid, "cand": cand, "out": out}
    with open(ex_path, "a") as fh:
        print(f"extraction: {len(ex_tasks)} calls @ {args.workers}", flush=True)
        run(ex_tasks, do_ex, fh, "extract")

    def sc(qid, c):
        return score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], args.n)

    def block(qid, top5):
        return "\n".join(f"[{j}] (score {sc(qid, c):.3f}) {cards[c].get('domain_description','')[:300]}"
                         for j, c in enumerate(top5, 1))

    # Phase 2: each arm's holistic pick
    arm_cache = {}
    for vid, tmpl in ARMS.items():
        path = cache_dir / f"pv_{vid}_{tag}.jsonl"
        cache = {r["qid"]: r["out"] for r in (load(path) if path.exists() else [])}
        tasks = [q["_qid"] for q in qs if q["_qid"] not in cache]
        def do_pick(qid, tmpl=tmpl):
            top5 = list(pool_by[qid][:KMAX_HOLISTIC])
            out = call_json(oai, tmpl.format(question=qmap[qid]["question"],
                                             candidate_block=block(qid, top5)))
            return {"qid": qid, "out": out}
        with open(path, "a") as fh:
            print(f"{vid}: {len(tasks)} picks @ {args.workers}", flush=True)
            run(tasks, do_pick, fh, vid)
        # reload (run appended)
        arm_cache[vid] = {r["qid"]: r["out"] for r in load(path)}

    # ---- eval ----
    def pick(qid, pool, cache):
        ch = cache.get(qid, {}).get("choice")
        return pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]

    nq = len(qs)
    in_pool = sum(1 for q in qs if q["instance_id"] in pool_by[q["_qid"]][:args.K])
    retr_r1 = sum(1 for q in qs if pool_by[q["_qid"]][0] == q["instance_id"])
    print(f"\n== {root.name} [{tag}]  K={args.K} n={args.n}  N={nq}  pool-recall@{args.K}={in_pool/nq:.3f}  "
          f"retrieval R@1={retr_r1/nq:.3f} ==")
    print(f"{'arm':14s} {'R@1(all)':>9} {'R@1|inpool':>11} {'engine(all)':>12} {'DBmacro R@1':>12}")
    hits_by = {}
    for vid, cache in arm_cache.items():
        inst = eng = inst_in = 0
        per_db = defaultdict(lambda: [0, 0])
        # per-GT-engine slice of the SAME cross-engine run: [inst_ok, n, inpool_ok, n_inpool]
        per_eng = defaultdict(lambda: [0, 0, 0, 0])
        hitlist = []
        for q in qs:
            gt = q["instance_id"]; pool = pool_by[q["_qid"]]; poolK = pool[:args.K]
            p = pick(q["_qid"], pool, cache)
            ok = (p == gt)
            inst += ok; eng += (dbs[p]["engine"] == dbs[gt]["engine"])
            per_db[gt][0] += ok; per_db[gt][1] += 1
            ge = dbs[gt]["engine"]
            per_eng[ge][0] += ok; per_eng[ge][1] += 1
            if gt in poolK:
                inst_in += ok
                per_eng[ge][2] += ok; per_eng[ge][3] += 1
            hitlist.append((q["_qid"], ok, gt in poolK))
        hits_by[vid] = hitlist
        dbmacro = float(np.mean([c / t for c, t in per_db.values()]))
        print(f"{vid:14s} {inst/nq:>9.3f} {inst_in/max(in_pool,1):>11.3f} {eng/nq:>12.3f} {dbmacro:>12.3f}")
        for ge in sorted(per_eng):
            ok, n, oin, nin = per_eng[ge]
            print(f"   └ GT-engine {ge:11s} N={n:4d}  R@1(all)={ok/max(n,1):.3f}  "
                  f"R@1|inpool={oin/max(nin,1):.3f} (in-pool {nin}/{n})")

    # McNemar v1 vs v0 (overall + in-pool), paired by qid
    def mcnemar(a, b):
        a_only = sum(1 for x, y in zip(a, b) if x and not y)
        b_only = sum(1 for x, y in zip(a, b) if y and not x)
        nn = a_only + b_only
        p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(min(a_only, b_only) + 1)) / 2 ** nn)
        return a_only, b_only, p
    v0, v1 = hits_by["v0_base"], hits_by["v1_specific"]
    a_all = [h[1] for h in v1]; b_all = [h[1] for h in v0]
    a_in = [h[1] for h in v1 if h[2]]; b_in = [h[1] for h in v0 if h[2]]
    for tag, a, b in [("overall", a_all, b_all), ("in-pool", a_in, b_in)]:
        ao, bo, p = mcnemar(a, b)
        print(f"McNemar v1 vs v0 ({tag}): v1_only={ao} v0_only={bo} p={p:.4f}")


if __name__ == "__main__":
    main()
