"""Smoke test: verify DeepSeek KV prefix-cache fires with the reordered EXTRACT_PROMPT.
Same DB (static instructions+schema prefix), 4 different queries (variable tail). Call 1 builds
cache (all miss); calls 2-4 should hit on the instructions+schema prefix.
Run: LLM_PROVIDER=deepseek python src/routing/smoke_cache_test.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load
import build_semantic as bs
from build_semantic import client, call_json, chat_model, render_entities, CACHE_STATS
from agent_rerank import EXTRACT_PROMPT

DB = "pgbird_005_mondial_geo"   # 34 entities -> large cacheable prefix
QUERIES = [
    "Which countries border France and what are their populations?",
    "List the rivers that flow through Germany.",
    "What is the GDP of the country whose capital is Tokyo?",
    "Find the mountains located in Switzerland above 4000 meters.",
]


def main():
    invs = {i["instance_id"]: i for i in load(
        HERE.parent.parent / "data" / "multidb" / "semantic" / "inventory.jsonl")}
    inv = invs[DB]
    eb = render_entities(inv)
    oai = client()
    print(f"provider={bs._provider()}  model={chat_model()}  DB={DB} entities={len(inv['entities'])}\n")
    prev = dict(CACHE_STATS)
    for i, q in enumerate(QUERIES, 1):
        call_json(oai, EXTRACT_PROMPT.format(question=q, engine=inv["engine"], entity_block=eb))
        dh = CACHE_STATS["hit"] - prev["hit"]
        dm = CACHE_STATS["miss"] - prev["miss"]
        prev = dict(CACHE_STATS)
        tot = dh + dm
        print(f"call {i}: prompt_tokens={tot}  cache_HIT={dh}  cache_MISS={dm}  "
              f"hit_ratio={dh/tot*100:.0f}%" if tot else f"call {i}: no usage")
        time.sleep(3)   # DeepSeek cache build is best-effort, few-sec delay
    print(f"\nTOTAL  calls={CACHE_STATS['calls']}  hit={CACHE_STATS['hit']}  miss={CACHE_STATS['miss']}"
          f"  overall_hit_ratio={CACHE_STATS['hit']/(CACHE_STATS['hit']+CACHE_STATS['miss'])*100:.0f}%")
    print("PASS: cache firing" if CACHE_STATS["hit"] > 0 else "WARN: no cache hits (cold or unsupported)")


if __name__ == "__main__":
    main()
