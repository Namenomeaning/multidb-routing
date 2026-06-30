"""Run Sudarshan's VERBATIM adjacency_list_prompt.txt (full from-scratch, output idx:targets)
on our 8-case candidate DBs. Compare resulting graph density + connectivity vs our fill-in
adjacency. Question: does THEIR own prompt produce a sparser graph (connectivity alive) on
our schemas, or does it also over-connect?

Faithful: prompt text loaded verbatim from the cached repo file; only [Schema Information] is
filled. Their prompt self-indexes tables 0..N-1 in presentation order, so we present entities
in inventory order and map output indices back by position. Output is plain-text idx:targets
(their format), parsed by code.
"""
from __future__ import annotations
import json, sys, re
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import statistics as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all
from build_semantic import client, chat_model, chat_model as _cm
from build_semantic import entity_index
from agent_rerank import field2ent, resolve_nodes, top_pool
from openai import OpenAI

ROOT = HERE.parent.parent / "data" / "multidb"
PROMPT_FILE = HERE.parent.parent / "refs" / "sudarshan" / "DB-Routing-prompts" / "adjacency_list_prompt.txt"
CACHE = ROOT / "agent_cache" / "sud_adj_verbatim.jsonl"

PROMPT_TMPL = PROMPT_FILE.read_text()


def render_schema(inv: dict) -> str:
    """DDL-ish: indexed tables with fields+types+PK, then declared FK/relationships."""
    lines = []
    for i, e in enumerate(inv["entities"]):
        parts = [f["name"] + (f" {f['type']}" if f["type"] else "") + (" PK" if f.get("pk") else "")
                 for f in e["fields"]]
        lines.append(f"Table {i}: {e['name']}({', '.join(parts)})")
    rels = inv.get("declared_relations", [])
    if rels:
        lines.append("Foreign keys / relationships:")
        for r in rels:
            if r.get("kind") == "fk":
                lines.append(f"  {r['from_entity']}.{r['via']} -> {r['to_entity']}.{r.get('to_field','')}")
            else:
                lines.append(f"  {r['from_entity']} -[{r['via']}]-> {r['to_entity']}")
    return "\n".join(lines)


def call_text(oai: OpenAI, prompt: str, retries: int = 3) -> str:
    import time
    last = None
    for a in range(retries + 1):
        try:
            r = oai.chat.completions.create(model=chat_model(), temperature=0,
                                            messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception as exc:  # noqa
            last = exc; time.sleep(8 * (2 ** a))
    raise last


def parse_adj(text: str, n: int) -> list[tuple[int, int]]:
    """Parse 'idx:targets' lines -> undirected edge set (within range, no self-loop)."""
    edges = set()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)\s*:\s*([\d,\s]*)$", line)
        if not m:
            continue
        src = int(m.group(1))
        if not (0 <= src < n):
            continue
        for tgt in re.findall(r"\d+", m.group(2)):
            t = int(tgt)
            if 0 <= t < n and t != src:
                edges.add((min(src, t), max(src, t)))
    return sorted(edges)


def connected_full(nodes, edges):
    if len(nodes) <= 1:
        return 1
    g = defaultdict(set)
    for a, b in edges:
        g[a].add(b); g[b].add(a)
    seen, dq = set(), deque([next(iter(nodes))])
    while dq:
        x = dq.popleft()
        if x in seen: continue
        seen.add(x); dq.extend(g[x] - seen)
    return 1 if set(nodes) <= seen else 0


def main():
    invs = {i["instance_id"]: i for i in load(ROOT / "semantic" / "inventory.jsonl")}
    dbs = {d["instance_id"]: d for d in load(ROOT / "databases.jsonl")}
    man = json.loads((ROOT / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(ROOT / "index" / "card.npy")
    our_adj = {r["instance_id"]: r for r in load(ROOT / "agent_cache" / "abl_adj_new.jsonl")}

    # same 8-case subset (seed 11) + pools
    by_db = defaultdict(list)
    for q in load(ROOT / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_eng = defaultdict(list)
    for d in sorted(by_db):
        if d in invs: by_eng[dbs[d]["engine"]].append(d)
    engines = sorted(by_eng)
    rng = np.random.default_rng(11)
    qs, ei = [], 0
    while len(qs) < 8:
        eng = engines[ei % len(engines)]
        if by_eng[eng]:
            d = by_eng[eng].pop(int(rng.integers(len(by_eng[eng]))))
            qs.append(by_db[d][int(rng.integers(len(by_db[d])))])
        ei += 1
    import hashlib
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5((q["instance_id"]+"|"+q["question"]).encode()).hexdigest()[:12]
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, 5)
    cand_ids = sorted({c for pool in pools for c in pool})

    # extract cache (sample 0) for connectivity-on-query
    ex = defaultdict(dict)
    for r in load(ROOT / "agent_cache" / "abl_extracts_k3.jsonl"):
        ex[(r["qid"], r["cand"])][r["s"]] = (r["out"].get("mappings") or [])

    # run verbatim adjacency, cached
    done = {r["instance_id"]: r for r in (load(CACHE) if CACHE.exists() else [])}
    oai = client()
    print(f"model={chat_model()}  candidates={len(cand_ids)}\n")
    with open(CACHE, "a") as f:
        for c in cand_ids:
            if c in done: continue
            inv = invs[c]; n = len(inv["entities"])
            prompt = PROMPT_TMPL.replace("[Schema Information]", render_schema(inv))
            txt = call_text(oai, prompt)
            edges = parse_adj(txt, n)
            rec = {"instance_id": c, "n": n, "edges": [list(e) for e in edges], "raw": txt[:500]}
            f.write(json.dumps(rec) + "\n"); f.flush()
            done[c] = rec
            print(f"  {c[:36]:36s} n={n:2d} sud_edges={len(edges):3d}", flush=True)

    # ---- compare density + fully-connected ----
    print("\n=== GRAPH DENSITY / FULLY-CONNECTED (36 cand) ===")
    def stats(get_edges, get_n):
        fc, dens = [], []
        for c in cand_ids:
            n = get_n(c)
            es = get_edges(c)
            if n <= 1: continue
            nodes = set(range(n))
            fc.append(connected_full(nodes, es))
            dens.append(len(set(map(tuple, es))) / (n*(n-1)/2))
        return sum(fc), len(fc), st.median(dens)
    sf, st_n, sd = stats(lambda c: [tuple(e) for e in done[c]["edges"]], lambda c: done[c]["n"])
    of, ot_n, od = stats(
        lambda c: [(e["from"], e["to"]) for e in our_adj[c]["edges"]],
        lambda c: len(our_adj[c]["entity_index"]))
    print(f"  SUDARSHAN verbatim:  fully-connected {sf}/{st_n}  median_density={sd:.3f}")
    print(f"  OURS fill-in:        fully-connected {of}/{ot_n}  median_density={od:.3f}")

    # ---- connectivity on the 8 queries under Sudarshan graph ----
    print("\n=== CONNECTIVITY on query-matched nodes (8 cases x 5 cand) ===")
    sud1 = our1 = tot = 0
    for q, pool in zip(qs, pools):
        for c in pool:
            maps = ex[(q["_qid"], c)].get(0, [])
            tgts = [t for m in maps if isinstance(m, dict) for t in (m.get("targets") or []) if isinstance(t, str)]
            f2e = field2ent(invs[c]); names = entity_index(invs[c])
            nodes = resolve_nodes(tgts, names, f2e)
            sud_e = [tuple(e) for e in done[c]["edges"]]
            our_e = [(e["from"], e["to"]) for e in our_adj[c]["edges"]]
            sud1 += connected_full(nodes, sud_e)
            our1 += connected_full(nodes, our_e)
            tot += 1
    print(f"  connectivity=1 under SUDARSHAN graph: {sud1}/{tot} = {sud1/tot*100:.0f}%")
    print(f"  connectivity=1 under OURS graph:      {our1}/{tot} = {our1/tot*100:.0f}%")


if __name__ == "__main__":
    main()
