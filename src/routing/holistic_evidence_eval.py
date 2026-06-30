"""Holistic + rich evidence, and engine-correct vs instance-correct split.

Two questions:
  (A) Does giving the agent the EXTRACTION REASONS (matched/unmatched phrases) on top of the
      score help it disambiguate? New arm `holi_ev`: per candidate the agent sees
      card + deterministic score + matched phrases + unmatched phrases, then picks.
      Compare to `holi` (card + score only, from agent_rerank cache) and retrieval top-1.
  (B) Split every arm into engine-correct (picked the right ENGINE, regardless of instance)
      vs instance-correct (the GT instance). Separates genuine benchmark ambiguity
      (same domain replicated across engines -> arbitrary GT label) from real routing error.

Reuses agent_rerank extraction + holistic caches (no new extraction calls). Only holi_ev
picks call the LLM (cached to holistic_ev.jsonl).

Usage: LLM_PROVIDER=deepseek python holistic_evidence_eval.py --set <route_dir> [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all  # noqa: E402
from build_semantic import client, call_json, chat_model  # noqa: E402
from agent_rerank import score_candidate, top_pool, KMAX_HOLISTIC  # noqa: E402
from hybrid_v2_eval import build_slice  # noqa: E402

HOLI_EV_PROMPT = """Pick the single database that can best answer the question. For each candidate
you are shown its domain, plus which parts of the question its schema can and cannot account for.

## Question
{question}

## Candidate databases
{candidate_block}

Return ONLY this JSON object: {{"choice": <number>}}
"""


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

    cache_dir = root / "agent_cache"
    extr_cache = {(r["qid"], r["cand"]): r["out"] for r in load(cache_dir / "extractions.jsonl")}
    holi_cache = {r["qid"]: r["out"] for r in load(cache_dir / "holistic.jsonl")}
    ev_path = cache_dir / "holistic_ev.jsonl"
    ev_cache = {r["qid"]: r["out"] for r in (load(ev_path) if ev_path.exists() else [])}

    def sc(qid, c):
        return score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], args.n)

    def ev_block(qid, top5):
        lines = []
        for j, c in enumerate(top5, 1):
            ex = extr_cache.get((qid, c), {})
            mat = ", ".join(m.get("phrase", "") for m in (ex.get("matched", []) or []) if isinstance(m, dict))[:200]
            unm = ", ".join(ex.get("unmatched", []) or [])[:200]
            lines.append(f"[{j}] (score {sc(qid, c):.3f}) {cards[c].get('domain_description','')[:280]}\n"
                         f"    matched: {mat or '(none)'}\n    unmatched: {unm or '(none)'}")
        return "\n".join(lines)

    # generate holi_ev picks (uncached only)
    todo = [q["_qid"] for q in qs if q["_qid"] not in ev_cache]
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client(); lock = threading.Lock()
        print(f"holi_ev: {len(todo)} calls @ {args.workers} workers, model={chat_model()}")
        pool_by = {q["_qid"]: pool for q, pool in zip(qs, pools)}

        def do(qid):
            top5 = list(pool_by[qid][:KMAX_HOLISTIC])
            out = call_json(oai, HOLI_EV_PROMPT.format(
                question=qmap[qid]["question"], candidate_block=ev_block(qid, top5)))
            return qid, out

        with open(ev_path, "a") as fh:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(do, qid): qid for qid in todo}
                done = 0
                for fut in as_completed(futs):
                    qid, out = fut.result()
                    ev_cache[qid] = out
                    with lock:
                        fh.write(json.dumps({"qid": qid, "out": out}) + "\n"); fh.flush()
                    done += 1
                    if done % 20 == 0:
                        print(f"  ... {done}/{len(todo)}", flush=True)

    def pick_holi(qid, pool, cache):
        ch = cache.get(qid, {}).get("choice")
        return pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]

    # eval: instance-correct + engine-correct, in-pool only
    arms = {"retrieval": lambda qid, pool: pool[0],
            "holi(score)": lambda qid, pool: pick_holi(qid, pool, holi_cache),
            "holi+evidence": lambda qid, pool: pick_holi(qid, pool, ev_cache)}
    n_in = sum(1 for q, pool in zip(qs, pools) if q["instance_id"] in pool[:args.K])
    print(f"\n{root.name}  K={args.K} n={args.n}  in-pool={n_in}/{len(qs)}")
    print(f"{'arm':16s} {'inst-acc':>9} {'engine-acc':>11} {'eng-right/inst-wrong':>21}")
    for name, fn in arms.items():
        inst = eng = eng_not_inst = 0
        for q, pool in zip(qs, pools):
            gt = q["instance_id"]; poolK = pool[:args.K]
            if gt not in poolK:
                continue
            pick = fn(q["_qid"], pool)
            ok_inst = (pick == gt)
            ok_eng = (dbs[pick]["engine"] == dbs[gt]["engine"])
            inst += ok_inst; eng += ok_eng
            eng_not_inst += (ok_eng and not ok_inst)
        print(f"{name:16s} {inst/max(n_in,1):>9.3f} {eng/max(n_in,1):>11.3f} {eng_not_inst:>21d}")

    # McNemar holi+evidence vs holi(score), in-pool instance-correct
    from math import comb
    def hits(cache):
        out = []
        for q, pool in zip(qs, pools):
            gt = q["instance_id"]
            if gt not in pool[:args.K]:
                continue
            out.append(pick_holi(q["_qid"], pool, cache) == gt)
        return out
    a, b = hits(ev_cache), hits(holi_cache)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    nn = a_only + b_only
    p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(min(a_only, b_only) + 1)) / 2 ** nn)
    print(f"\nMcNemar holi+evidence vs holi(score): ev_only={a_only} score_only={b_only} p={p:.4f}")


if __name__ == "__main__":
    main()
