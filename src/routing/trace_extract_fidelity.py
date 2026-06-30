"""Trace LLM-extract fidelity on the in-pool FAILURES (cases 1,4,7 from faithful_adj_test).
For GT vs the winning-wrong DB: print full schema (entities+fields) + the faithful extract
mappings (phrase->targets, N/A). Goal: see whether coverage saturation is (a) faithful
extract correctly finding the query maps onto a sibling DB too (benchmark ambiguity), or
(b) sloppy over-matching onto the wrong DB / under-matching GT (extract-fidelity bug).
Ground-truth check, Sudarshan column-level mapping.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load
from build_semantic import client, call_json, chat_model, render_entities
from agent_rerank import EXTRACT_PROMPT, coverage

N = 2.0

# (question, GT id, winning-wrong id) from the 8-case run
CASES = [
    ("Can you find out which patients have undergone the most genetic tests and list the top ten?",
     "mongo_schema_genomic_analysis_healthcare_db", "pgbird_024_thrombosis_prediction"),
    ("For each company id, what are the companies and how many gas stations does each one operate?",
     "mongoscale_doc_038_gas_company", "mongoscale_doc_014_company_office"),
    ("Give me a list of id and status of orders which belong to the customer named 'Jeramie'.",
     "mongoscale_doc_066_tracking_orders", "mongoscale_doc_022_customers_and_addresses"),
]


def dump(oai, q, cid, invs, role):
    inv = invs.get(cid)
    if inv is None:
        print(f"  [{role}] {cid}  -- NOT FOUND in inventory"); return
    eb = render_entities(inv)
    extr = call_json(oai, EXTRACT_PROMPT.format(question=q, engine=inv["engine"], entity_block=eb))
    maps = extr.get("mappings", []) or []
    matched = [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])]
    na = [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])]
    cov = coverage(len(matched), len(na), N)
    print(f"  [{role}] {cid}  ({inv['engine']})  coverage={cov:.3f}  matched={len(matched)} na={len(na)}")
    print(f"      SCHEMA: {eb.strip()[:400]}")
    for m in matched:
        print(f"        + '{m['phrase']}' -> {m['targets']}")
    for m in na:
        print(f"        x '{m['phrase']}' -> N/A")


def main():
    root = HERE.parent.parent / "data" / "multidb"
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    oai = client()
    print(f"model={chat_model()}\n")
    for i, (q, gt, wrong) in enumerate(CASES, 1):
        print(f"==== CASE {i}: {q}")
        dump(oai, q, gt, invs, "GT   ")
        dump(oai, q, wrong, invs, "WRONG")
        print()


if __name__ == "__main__":
    main()
