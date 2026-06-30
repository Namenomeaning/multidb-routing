"""Rebuild ours_multidb adjacency.jsonl with the faithful + cache-optimized ADJ_PROMPT.
DeepSeek-direct. Cache pattern: 1 warmup call builds the static instruction-block prefix, short
delay lets DeepSeek persist it, then the rest run in parallel (each hits the cached instruction
prefix; per-DB schema tail is unique -> miss). DBs whose declared graph is already connected
skip the LLM (deterministic).

Run: LLM_PROVIDER=deepseek python src/routing/rebuild_adjacency_parallel.py
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load
import build_semantic as bs
from build_semantic import (client, call_json, chat_model, render_entities,
                            render_indexed_entities, render_confirmed_edges,
                            declared_graph_connected, merge_adjacency, ADJ_PROMPT, CACHE_STATS)

DEFAULT_ROOT = str(HERE.parent.parent / "data" / "multidb" / "semantic")


def build_one(oai, inv):
    raw = call_json(oai, ADJ_PROMPT.format(
        engine=inv["engine"], indexed_entity_block=render_indexed_entities(inv),
        confirmed_edges_block=render_confirmed_edges(inv), surface_block=render_entities(inv)))
    return merge_adjacency(inv, raw)


def fully_connected(adj):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--warmup-sleep", type=float, default=3.0)
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUT = ROOT / "adjacency.jsonl"
    invs = [i for i in load(ROOT / "inventory.jsonl") if i.get("entities")]
    oai = client()
    print(f"provider={bs._provider()} model={chat_model()} DBs={len(invs)} workers={args.workers}\n", flush=True)

    skip, need = [], []
    for inv in invs:
        (skip if declared_graph_connected(inv) else need).append(inv)
    print(f"skip LLM (declared connected) = {len(skip)}   need LLM = {len(need)}\n", flush=True)

    results = {}
    for inv in skip:
        adj = merge_adjacency(inv, {"implicit_edges": []})
        adj["llm_skipped"] = "declared graph already connected"
        results[inv["instance_id"]] = adj

    if need:
        # warmup: one call builds the static instruction-block cache prefix
        print(f"warmup on {need[0]['instance_id']} ...", flush=True)
        results[need[0]["instance_id"]] = build_one(oai, need[0])
        print(f"  warmup done; sleep {args.warmup_sleep}s for cache persist", flush=True)
        time.sleep(args.warmup_sleep)

        rest = need[1:]
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_one, oai, inv): inv for inv in rest}
            for fut in as_completed(futs):
                inv = futs[fut]
                results[inv["instance_id"]] = fut.result()
                done += 1
                if done % 20 == 0:
                    print(f"  ... {done}/{len(rest)}", flush=True)

    # write in original inventory order
    with open(OUT, "w") as f:
        for inv in invs:
            f.write(json.dumps(results[inv["instance_id"]]) + "\n")

    # summary
    multi = [a for a in results.values() if len(a.get("entity_index", [])) > 1]
    fc = sum(fully_connected(a) for a in multi)
    inf = sum(a.get("n_inferred", 0) for a in results.values())
    drp = sum(a.get("n_dropped", 0) for a in results.values())
    flag = [a["instance_id"] for a in results.values() if a.get("density_flag")]
    dens = [a.get("density", 0) for a in multi]
    print(f"\n=== REBUILD DONE -> {OUT} ===")
    print(f"DBs written={len(results)}  inferred_edges={inf}  dropped={drp}")
    print(f"fully-connected (n>1): {fc}/{len(multi)} = {fc/len(multi)*100:.1f}%  median_density={st.median(dens):.3f}")
    print(f"density_flag DBs={len(flag)}: {flag[:10]}")
    h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
    print(f"CACHE: calls={CACHE_STATS['calls']} hit={h} miss={m} hit_ratio={h/(h+m)*100:.0f}%" if h+m else "CACHE: n/a")


if __name__ == "__main__":
    main()
