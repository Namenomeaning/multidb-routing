"""Test Sudarshan-faithful prompts (EXTRACT phrase->Entity.field multi-map +
ADJ Priority2/3 'use cautiously', recall-bias removed).

For N cases x top-5 pool: rebuild adjacency PER candidate with the NEW ADJ_PROMPT
(declared-connected DBs still skip LLM, same as build), run NEW faithful EXTRACT,
compute Coverage x Connectivity. Compare OLD adjacency (adjacency.jsonl, my
recall-biased prompt) vs NEW adjacency: density, fully-connected rate, conn outcome,
decomp pick + correctness. Shows whether faithful prompt revives connectivity.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all
from build_semantic import (client, call_json, chat_model, render_entities,
                            render_indexed_entities, render_confirmed_edges,
                            declared_graph_connected, merge_adjacency, ADJ_PROMPT)
from agent_rerank import (EXTRACT_PROMPT, score_candidate, connectivity, field2ent,
                          resolve_nodes, top_pool)

N = 2.0
NCASE = 8


def graph_connected(adj: dict) -> int:
    names = adj.get("entity_index", [])
    n = len(names)
    if n <= 1:
        return 1
    g = defaultdict(set)
    for e in adj.get("edges", []):
        g[e["from"]].add(e["to"]); g[e["to"]].add(e["from"])
    seen, dq = set(), deque([0])
    while dq:
        x = dq.popleft()
        if x in seen: continue
        seen.add(x); dq.extend(g[x] - seen)
    return 1 if len(seen) == n else 0


def main():
    root = HERE.parent.parent / "data" / "multidb"
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    old_adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        if d in invs:
            by_engine[dbs[d]["engine"]].append(d)
    engines = sorted(by_engine)
    rng = np.random.default_rng(11)
    qs, ei = [], 0
    while len(qs) < NCASE:
        eng = engines[ei % len(engines)]
        if by_engine[eng]:
            d = by_engine[eng].pop(int(rng.integers(len(by_engine[eng]))))
            qs.append(by_db[d][int(rng.integers(len(by_db[d])))])
        ei += 1

    print(f"model={chat_model()}  set=ours_multidb  cases={len(qs)}\n")
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, 5)
    oai = client()

    # rebuild adjacency for every candidate in any pool, NEW prompt
    cand_ids = {c for pool in pools for c in pool}
    new_adjs = {}
    print(f"rebuilding adjacency for {len(cand_ids)} candidate DBs (new ADJ_PROMPT)...")
    for c in sorted(cand_ids):
        inv = invs[c]
        if declared_graph_connected(inv):
            adj = merge_adjacency(inv, {"implicit_edges": []})
            adj["llm_skipped"] = True
        else:
            raw = call_json(oai, ADJ_PROMPT.format(
                engine=inv["engine"], indexed_entity_block=render_indexed_entities(inv),
                confirmed_edges_block=render_confirmed_edges(inv),
                surface_block=render_entities(inv)))
            adj = merge_adjacency(inv, raw)
            adj["llm_skipped"] = False
        new_adjs[c] = adj

    # density comparison aggregate
    old_fc = sum(graph_connected(old_adjs[c]) for c in cand_ids if len(old_adjs[c].get("entity_index",[]))>1)
    new_fc = sum(graph_connected(new_adjs[c]) for c in cand_ids if len(new_adjs[c].get("entity_index",[]))>1)
    multi = sum(1 for c in cand_ids if len(new_adjs[c].get("entity_index",[]))>1)
    print(f"\nfully-connected (multi-entity cands, n={multi}):  OLD={old_fc}/{multi}  NEW={new_fc}/{multi}")
    oi = sum(old_adjs[c].get("n_inferred",0) for c in cand_ids)
    ni = sum(new_adjs[c].get("n_inferred",0) for c in cand_ids)
    print(f"total inferred edges over cands:  OLD={oi}  NEW={ni}\n")

    old_hit = new_hit = 0
    for qi, (q, pool) in enumerate(zip(qs, pools), 1):
        gt = q["instance_id"]
        in_pool = gt in pool
        print(f"=== CASE {qi} === GT={gt} ({dbs[gt]['engine']})  in_pool={in_pool}")
        print(f"Q: {q['question']}")
        old_sc, new_sc = {}, {}
        for c in pool:
            extr = call_json(oai, EXTRACT_PROMPT.format(
                question=q["question"], engine=invs[c]["engine"],
                entity_block=render_entities(invs[c])))
            os_ = score_candidate(extr, old_adjs.get(c, {}), invs[c], N)
            ns_ = score_candidate(extr, new_adjs[c], invs[c], N)
            old_sc[c], new_sc[c] = os_, ns_
            # diag conn under each adj
            maps = extr.get("mappings", []) or []
            tgts = [t for m in maps if isinstance(m,dict) for t in (m.get("targets") or []) if isinstance(t,str)]
            oc = connectivity(tgts, old_adjs.get(c,{}), field2ent(invs[c]))
            nc = connectivity(tgts, new_adjs[c], field2ent(invs[c]))
            nn = len(resolve_nodes(tgts, new_adjs[c].get("entity_index",[]), field2ent(invs[c])))
            tag = " <-GT" if c == gt else ""
            print(f"  {c[:32]:32s}{tag}  nodes={nn}  conn OLD={oc} NEW={nc}  "
                  f"score OLD={os_:.3f} NEW={ns_:.3f}")
        op = max(old_sc, key=lambda k:(old_sc[k], -pool.index(k)))
        np_ = max(new_sc, key=lambda k:(new_sc[k], -pool.index(k)))
        oh, nh = op==gt, np_==gt
        old_hit += oh; new_hit += nh
        print(f"  -> OLD-adj pick={op[:32]} {'OK' if oh else 'WRONG'}   "
              f"NEW-adj pick={np_[:32]} {'OK' if nh else 'WRONG'}\n")
    print(f"SUMMARY ({len(qs)} case, faithful extract)  decomp hit: OLD-adj={old_hit}  NEW-adj={new_hit}")


if __name__ == "__main__":
    main()
