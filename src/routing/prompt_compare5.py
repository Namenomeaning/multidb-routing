"""5-case A/B: OLD extract prompt (phrase->1 entity) vs SUD-faithful prompt
(phrase->Table.Column, multi-mapping, include/exclude lists, N/A sparingly).

For each query: build top-5 card-cosine pool, run BOTH prompts per candidate,
compute Coverage x Connectivity per candidate (same deterministic code), argmax.
Report per-case: GT, pool, decomp pick + correctness under each prompt, and the
connectivity/node stats that drive the difference.

Format-only deviation from Sudarshan: JSON output (robust parse) instead of plain
'phrase - Table.Column' lines. Semantics (column-level, multi-map, exclusions, N/A
sparingly) follow phrase_mapping_prompt.txt verbatim.
"""
from __future__ import annotations
import json, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all
from build_semantic import client, call_json, chat_model, render_entities
from agent_rerank import (EXTRACT_PROMPT, coverage, connectivity, field2ent,
                          score_candidate, top_pool)

N = 2.0  # coverage penalty (same as main runs)

# ---- SUD-faithful prompt (column-level, multi-map, exclusions) ----
SUD_PROMPT = """Role: expert DBA specializing in NLP and schema mapping.
Objective: map meaningful words/phrases in the QUERY to the most relevant COLUMNS in the
schema. Identify the semantic connection between query terms and the data structure.

## QUERY
{question}

## TABLE INFORMATION (engine: {engine})  -- format: Entity(col type PK, col type, ...)
{entity_block}

FOLLOW STRICTLY:
1. Identify mappable terms: nouns (entities/attributes), verbs (actions), adjectives/adverbs
   ONLY if they represent a state/category/quantifiable attribute (e.g. "oldest"->Age,
   "completed"->Status), and specific values ('Smith', 123, 'Completed').
2. Column mapping: direct & semantic match; value->most likely Table.Column by type;
   verb->column reflecting result/status/entity of the action; entity-type reference
   ("students")->the identifying column (e.g. Student.StuID).
3. Multiple mappings allowed: ONE phrase may map to SEVERAL columns -- list all valid ones.
4. N/A: if no match exists, map the phrase to N/A (use sparingly).
5. DO NOT map: SQL keywords (select/from/where/order by), functions (count/sum/avg),
   NL phrases ("number of","how many","list all"), general modifiers
   (different/unique/various/top/lowest), stop words (what/who/the/a/in).
   Special rule: "number of students" -> map only "students" to its identifying column;
   "student id" -> the actual id column.

Return ONLY this JSON:
{{"mappings": [{{"phrase": "<span>", "targets": ["Entity.column", "Entity2.column2"]}},
              {{"phrase": "<unmatched span>", "targets": []}}]}}
Targets must use entity/column names exactly as written above. [] targets = N/A.
Extraction only -- no score, ranking, or confidence.
"""


def sud_score(out: dict, adj: dict, inv: dict, n: float) -> tuple[float, int, int, int]:
    """Coverage x Connectivity from SUD multi-map output. Returns (score, conn, nodes, na)."""
    maps = out.get("mappings", []) or []
    matched = [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])]
    na = [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])]
    cov = coverage(len(matched), len(na), n)
    targets: list[str] = []
    for m in matched:
        targets.extend(t for t in m["targets"] if isinstance(t, str))
    f2e = field2ent(inv)
    conn = connectivity(targets, adj, f2e)
    # node count for diagnostics
    from agent_rerank import resolve_nodes
    nodes = resolve_nodes(targets, adj.get("entity_index", []), f2e)
    return cov * conn, conn, len(nodes), len(na)


def old_diag(out: dict, adj: dict, inv: dict) -> tuple[int, int]:
    ents = [m.get("entity") for m in (out.get("matched") or []) if isinstance(m, dict) and m.get("entity")]
    from agent_rerank import resolve_nodes
    nodes = resolve_nodes(ents, adj.get("entity_index", []), field2ent(inv))
    return connectivity(ents, adj, field2ent(inv)), len(nodes)


def main():
    root = HERE.parent.parent / "data" / "multidb"
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    # pick 5 queries: 1 per engine round-robin from cards-present GT-DBs, deterministic
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        if d in invs:
            by_engine[dbs[d]["engine"]].append(d)
    engines = sorted(by_engine)
    rng = np.random.default_rng(7)
    qs = []
    ei = 0
    while len(qs) < 5:
        eng = engines[ei % len(engines)]
        if by_engine[eng]:
            d = by_engine[eng].pop(rng.integers(len(by_engine[eng])))
            qs.append(by_db[d][int(rng.integers(len(by_db[d])))])
        ei += 1

    print(f"model={chat_model()}  set=ours_multidb  N_pool={len(ids)}\n")
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, 5)
    oai = client()

    old_hit = sud_hit = 0
    for qi, (q, pool) in enumerate(zip(qs, pools), 1):
        gt = q["instance_id"]
        print(f"=== CASE {qi} === GT={gt} ({dbs[gt]['engine']})")
        print(f"Q: {q['question']}")
        in_pool = gt in pool
        print(f"pool(top5)={pool}  GT_in_pool={in_pool}")
        old_scores, sud_scores = {}, {}
        for c in pool:
            eb = render_entities(invs[c])
            old_out = call_json(oai, EXTRACT_PROMPT.format(
                question=q["question"], engine=invs[c]["engine"], entity_block=eb))
            sud_out = call_json(oai, SUD_PROMPT.format(
                question=q["question"], engine=invs[c]["engine"], entity_block=eb))
            os_ = score_candidate(old_out, adjs.get(c, {}), invs[c], N)
            oc, on = old_diag(old_out, adjs.get(c, {}), invs[c])
            ss_, sc, sn, sna = sud_score(sud_out, adjs.get(c, {}), invs[c], N)
            old_scores[c] = os_; sud_scores[c] = ss_
            tag = " <-GT" if c == gt else ""
            print(f"  {c[:34]:34s}{tag}")
            print(f"     OLD score={os_:.3f} conn={oc} nodes={on}")
            print(f"     SUD score={ss_:.3f} conn={sc} nodes={sn} na={sna}")
        old_pick = max(old_scores, key=lambda k: (old_scores[k], -pool.index(k)))
        sud_pick = max(sud_scores, key=lambda k: (sud_scores[k], -pool.index(k)))
        oh, sh = old_pick == gt, sud_pick == gt
        old_hit += oh; sud_hit += sh
        print(f"  -> OLD pick={old_pick[:34]} {'OK' if oh else 'WRONG'}")
        print(f"  -> SUD pick={sud_pick[:34]} {'OK' if sh else 'WRONG'}\n")
    print(f"SUMMARY (5 case)  OLD decomp hit={old_hit}/5   SUD decomp hit={sud_hit}/5")


if __name__ == "__main__":
    main()
