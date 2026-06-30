#!/usr/bin/env python3
"""Feasibility micro-probe: AGENT (LLM-controlled inspect/commit loop) vs WORKFLOW
(fixed deterministic cascade: parse-once -> map-per-candidate -> Coverage x Connectivity -> argmax).

Both arms start from the SAME retrieved top-K card pool (retrieval held constant) so the
comparison isolates the DECISION mechanism, not retrieval. Small slice, dev-style probe.

NOTE: the AGENT arm lets the LLM self-decide when to stop & which DB to commit -- this is the
'full agent' style that the locked no-self-confidence invariant warns against. This script only
tests FEASIBILITY (is an agent competitive with the workflow?), not the final thesis design.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all                      # noqa: E402
from agent_rerank import top_pool                                # noqa: E402
from build_semantic import client, call_json, chat_model, render_entities  # noqa: E402
from decomp_card_eval import PARSE_PROMPT, MAP_PROMPT, score_fixed  # noqa: E402

KPOOL = 5  # candidate pool both arms share (top-5)


# ---------------------------------------------------------------- AGENT arm

AGENT_PROMPT = """You are a database-routing agent. A natural-language question must be answered by
EXACTLY ONE database. You see candidate databases (id, engine, short description). You may INSPECT a
candidate's full schema (entities + glossary) before deciding -- inspect only when the descriptions
are insufficient to choose. When confident, COMMIT one database id.

Question: {q}

Candidates:
{cands}

Observations so far (schemas you already inspected):
{obs}

Decide the next action. Inspect at most when needed; do not inspect a db you already inspected.
Return ONLY this JSON: {{"action": "inspect" | "commit", "db_id": "<one candidate id>", "reason": "<short>"}}"""


def cand_block(pool, dbs, cards):
    out = []
    for c in pool:
        desc = (cards[c].get("domain_description") or "")[:240]
        out.append(f'- id={c} | engine={dbs[c]["engine"]} | {desc}')
    return "\n".join(out)


def schema_obs(c, invs, cards):
    ents = render_entities(invs[c])
    gl = cards[c].get("term_glossary") or {}
    gtxt = "; ".join(f"{k}: {v}" for k, v in list(gl.items())[:12])
    return f"[{c}] entities: {ents}\n   glossary: {gtxt}"


def run_agent(oai, q, pool, dbs, cards, invs, max_steps=4):
    """LLM-controlled loop. Returns (pick, n_calls, n_inspect, trace)."""
    obs, inspected, calls, insp = {}, set(), 0, 0
    last_db = pool[0]
    for _ in range(max_steps):
        obs_txt = "\n".join(obs.values()) if obs else "(none yet)"
        out = call_json(oai, AGENT_PROMPT.format(
            q=q, cands=cand_block(pool, dbs, cards), obs=obs_txt))
        calls += 1
        act = (out.get("action") or "").lower()
        dbid = out.get("db_id")
        if dbid in pool:
            last_db = dbid
        if act == "commit" and dbid in pool:
            return dbid, calls, insp, "commit"
        if act == "inspect" and dbid in pool and dbid not in inspected:
            inspected.add(dbid)
            obs[dbid] = schema_obs(dbid, invs, cards)
            insp += 1
        else:
            # invalid / repeat inspect / commit unknown -> force commit best guess
            return last_db, calls, insp, "forced"
    return last_db, calls, insp, "maxsteps"


# ---------------------------------------------------------------- WORKFLOW arm

def run_workflow(oai, q, pool, cards, invs, adjs, n=2.0):
    """Fixed cascade: parse once -> map each candidate -> score_fixed -> argmax.
    Returns (pick, n_calls)."""
    from build_index import card_text
    parse = call_json(oai, PARSE_PROMPT.format(question=q))
    phrases = parse.get("phrases") or []
    calls = 1
    sc = []
    for c in pool:
        mp = call_json(oai, MAP_PROMPT.format(
            engine=invs[c]["engine"],
            schema_block=card_text(cards[c], invs[c]),
            phrases=json.dumps(phrases, ensure_ascii=False)))
        calls += 1
        s = score_fixed(phrases, mp, adjs.get(c, {}), invs[c], n)
        sc.append((s, c))
    # argmax; ties -> highest retrieval rank (pool order)
    best = max(s for s, _ in sc)
    pick = next(c for s, c in sc if s == best)
    return pick, calls


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--max-dbs", type=int, default=9)
    ap.add_argument("--per-db-cap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=260621)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    rng = np.random.default_rng(args.seed)
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        if q["instance_id"] in cards:
            by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        by_engine[dbs[d]["engine"]].append(d)
    engines = sorted(by_engine)
    pick_dbs, ei = [], 0
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
    gts = {id(q): q["instance_id"] for q in qs}

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, KPOOL)

    print(f"{root.name}: {len(qs)} q from {len(pick_dbs)} GT-DBs (cap {args.per_db_cap}/DB), "
          f"pool top-{KPOOL}, model={chat_model()}, max_steps={args.max_steps}")
    in_pool = sum(1 for q, p in zip(qs, pools) if q["instance_id"] in p)
    print(f"retrieval: GT-in-pool {in_pool}/{len(qs)} = {in_pool/len(qs):.3f}\n")

    oai = client()

    def work_one(i):
        q, pool = qs[i], pools[i]
        gt = q["instance_id"]
        wf_pick, wf_calls = run_workflow(oai, q["question"], pool, cards, invs, adjs)
        ag_pick, ag_calls, ag_insp, ag_stop = run_agent(
            oai, q["question"], pool, dbs, cards, invs, args.max_steps)
        return dict(gt=gt, inpool=gt in pool,
                    wf=wf_pick, wf_ok=wf_pick == gt, wf_calls=wf_calls,
                    ag=ag_pick, ag_ok=ag_pick == gt, ag_calls=ag_calls,
                    ag_insp=ag_insp, ag_stop=ag_stop)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(work_one, range(len(qs))))

    N = len(res)
    inp = [r for r in res if r["inpool"]]
    def acc(key, rs): return sum(r[key] for r in rs) / max(len(rs), 1)
    print("="*64)
    print(f"slice N={N}  (GT-in-pool n={len(inp)})")
    print(f"{'arm':<12}{'R@1 all':>10}{'R@1|in-pool':>14}{'avg LLM calls':>16}")
    print(f"{'WORKFLOW':<12}{acc('wf_ok',res):>10.3f}{acc('wf_ok',inp):>14.3f}"
          f"{sum(r['wf_calls'] for r in res)/N:>16.2f}")
    print(f"{'AGENT':<12}{acc('ag_ok',res):>10.3f}{acc('ag_ok',inp):>14.3f}"
          f"{sum(r['ag_calls'] for r in res)/N:>16.2f}")
    avg_insp = sum(r['ag_insp'] for r in res)/N
    stops = defaultdict(int)
    for r in res:
        stops[r['ag_stop']] += 1
    print(f"\nagent: avg schema inspections/query = {avg_insp:.2f}  stop-reasons {dict(stops)}")
    # disagreements
    dis = [r for r in res if r['wf_ok'] != r['ag_ok']]
    print(f"disagreements (one right, one wrong): {len(dis)}")
    for r in dis:
        win = "AGENT" if r['ag_ok'] else "WORKFLOW"
        print(f"  GT={r['gt']:<28} wf={r['wf']:<24} ag={r['ag']:<24} -> {win} wins (insp={r['ag_insp']})")


if __name__ == "__main__":
    main()
