"""Small-scale agent rerank experiment (final-routing layer, NOT retrieval).

Three arms over the SAME top-K card-retrieval pool — isolates the user-observed failure
("LLM ignores the score and picks whatever looks most plausible"):

  RETRIEVAL  : top-1 by card cosine (no rerank)                       — baseline
  HOLISTIC   : one LLM call sees all K cards + their computed scores, picks one
               — REPRODUCES the failure mode (LLM-as-ranker)
  DECOMP     : per-candidate LLM extraction (schema-aware, extraction-only) →
               deterministic Coverage×Connectivity → argmax            — ours (Sudarshan 2601.19825)

DECOMP structurally removes the LLM-picks-from-score step: the LLM only maps query→schema
entities per candidate (in isolation), code computes the score, argmax decides. This is the
locked invariant (m3-design-decisions: LLM extracts, never evaluates) and the Sudarshan
decomposed rerank that beat holistic LLM rerank by +8-11 R@1.

Coverage = exp(-n·x), x = unmatched/(matched+unmatched)        [n = thesis hyperparam, swept]
Connectivity = 1 if matched entities form one connected component over the candidate's
               adjacency graph (declared+inferred edges) via BFS, else 0  [≤1 node ⇒ 1]
score = Coverage × Connectivity  (multiplicative; no retrieval term; engine-neutral)

Two-layer metrics (CLAUDE.md §3): pool recall (GT-in-top-K) reported SEPARATELY from final
R@1; rerank arms also reported as R@1 | GT-in-pool so retrieval misses don't mask rerank skill.

Extractions + holistic picks are cached to JSONL → n/K sweep and reruns cost no LLM calls.

Usage:
  python agent_rerank.py --set <route_dir> --kmax 8 --max-dbs 12 --per-db-cap 3 [--sleep 1.5]
  python agent_rerank.py --set <route_dir> --score-only      # re-score from cache, no LLM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
from src.prompts import load as load_prompt  # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.core.retrieval import load, embed_all  # noqa: E402  (also wires repo root onto path)
from src.core.semantic_card import client, call_json, chat_model, render_entities  # noqa: E402
from src.core.index import card_text  # noqa: E402

KMAX_HOLISTIC = 5  # holistic arm shows top-5 (Sudarshan rerank scope)


# ---------------------------------------------------------------- prompts (extraction-only, engine-neutral)

# Adapted from Sudarshan et al. (arXiv:2601.19825) phrase_mapping_prompt.txt — verbatim
# instruction structure (column-level mapping, multiple mappings per phrase, the
# include/exclude lists, the identifier-vs-counting special rule, N/A sparingly).
# Generalized from SQL TableName.ColumnName to Entity.field so it applies to Mongo
# collections + Neo4j node labels/properties. JSON output (robust parse) instead of plain
# 'phrase - Table.Column' lines. Verbatim source:
# refs/sudarshan/DB-Routing-prompts/phrase_mapping_prompt.txt
#
# SECTION ORDER (KV-cache optimization): instructions + output spec are STATIC (identical
# every call) and the SCHEMA block is identical for every query hitting the same DB, so both
# form a long cacheable prefix; the variable QUERY is appended LAST. DeepSeek context caching
# (full-prefix match) then charges only the short query tail on repeat calls over the same DB
# — prompt_cache_hit_tokens covers instructions+schema. Content faithful to Sudarshan; only
# the placement of the query (moved to the end) differs from their layout.
EXTRACT_PROMPT = load_prompt("extract")

HOLISTIC_PROMPT = load_prompt("holistic")

# HYBRID: deterministic gate (drop connectivity=0 = structurally invalid) already applied;
# agent selects among the structurally-valid survivors using their domain cards. Same prompt
# shape as holistic but the candidate set is pre-filtered → agent only breaks remaining ties.
HYBRID_PROMPT = load_prompt("hybrid")


# ---------------------------------------------------------------- deterministic scoring

def coverage(matched: int, unmatched: int, n: float) -> float:
    total = matched + unmatched
    if total == 0:
        return 0.0
    return math.exp(-n * (unmatched / total))


def field2ent(inv: dict) -> dict[str, int]:
    """field-name (lower) -> owning entity index; entity order == adjacency entity_index order."""
    m: dict[str, int] = {}
    for i, e in enumerate(inv["entities"]):
        for f in e["fields"]:
            m.setdefault(f["name"].lower(), i)
    return m


def resolve_nodes(tokens: list[str], names: list[str], f2e: dict[str, int]) -> set[int]:
    """Map LLM 'entity' strings (bare entity, 'Entity.field', or bare field) to entity indices."""
    pos = {nm.lower(): i for i, nm in enumerate(names)}
    nodes: set[int] = set()
    for t in tokens:
        low = t.strip().lower()
        if not low:
            continue
        if low in pos:
            nodes.add(pos[low]); continue
        if "." in low:                                   # 'Entity.field' -> Entity
            pre, suf = low.split(".", 1)
            if pre in pos:
                nodes.add(pos[pre]); continue
            if suf in f2e:
                nodes.add(f2e[suf]); continue
        if low in f2e:                                   # bare field -> owning entity
            nodes.add(f2e[low])
    return nodes


def connectivity(matched_entities: list[str], adj: dict, f2e: dict[str, int]) -> int:
    """1 if resolved matched entities form a single connected component over the adjacency graph."""
    names = adj.get("entity_index", [])
    nodes = resolve_nodes(matched_entities, names, f2e)
    if len(nodes) <= 1:
        return 1
    g = defaultdict(set)
    for e in adj.get("edges", []):
        g[e["from"]].add(e["to"]); g[e["to"]].add(e["from"])
    seen, dq = set(), deque([next(iter(nodes))])
    while dq:
        x = dq.popleft()
        if x in seen:
            continue
        seen.add(x)
        dq.extend(g[x] - seen)
    return 1 if nodes <= seen else 0


def score_candidate(extr: dict, adj: dict, inv: dict, n: float) -> float:
    """Sudarshan-faithful: phrase->multiple Entity.field mappings (column-level).
    Coverage over phrases (matched = has >=1 target, N/A = empty targets); Connectivity
    BFS over ALL mapped target entities. Multiplicative."""
    maps = extr.get("mappings", []) or []
    matched = [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])]
    na = [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])]
    cov = coverage(len(matched), len(na), n)
    targets: list[str] = []
    for m in matched:
        targets.extend(t for t in m["targets"] if isinstance(t, str))
    conn = connectivity(targets, adj, field2ent(inv))
    return cov * conn


# ---------------------------------------------------------------- pool

def top_pool(qv: np.ndarray, index: np.ndarray, ids: list[str], k: int) -> list[list[str]]:
    sims = qv @ index.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return [[ids[j] for j in row] for row in order]


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="route dir (has databases/semantic/index/splits)")
    ap.add_argument("--kmax", type=int, default=8, help="pool size to extract over (cache); sweep K<=kmax")
    ap.add_argument("--max-dbs", type=int, default=12, help="GT DBs to sample queries from")
    ap.add_argument("--per-db-cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--sleep", type=float, default=1.5)  # retained (unused: concurrency replaces it)
    ap.add_argument("--workers", type=int, default=8, help="concurrent OpenRouter requests")
    ap.add_argument("--score-only", action="store_true", help="re-score from cache, no LLM calls")
    args = ap.parse_args()

    root = Path(args.set)
    name = root.name
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    # sample queries: stratified, --max-dbs GT-DBs balanced across engines (round-robin), cap/DB
    rng = np.random.default_rng(args.seed)
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        if d in cards:
            by_engine[dbs[d]["engine"]].append(d)
    pick_dbs, ei = [], 0
    engines = sorted(by_engine)
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
    for q in qs:  # stable cache key (spider has no query_id)
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    gts = [q["instance_id"] for q in qs]
    print(f"{name}: {len(ids)} DBs in pool, sampling {len(qs)} q from {len(pick_dbs)} GT-DBs "
          f"(cap {args.per_db_cap}/DB), kmax={args.kmax}, model={chat_model()}")

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, args.kmax)

    cache_dir = root / "cache" / "agent_cache"
    cache_dir.mkdir(exist_ok=True)
    ex_path, ho_path, hy_path = (cache_dir / "extractions.jsonl",
                                 cache_dir / "holistic.jsonl", cache_dir / "hybrid.jsonl")
    extr_cache = {(r["qid"], r["cand"]): r["out"] for r in (load(ex_path) if ex_path.exists() else [])}
    holi_cache = {r["qid"]: r["out"] for r in (load(ho_path) if ho_path.exists() else [])}
    hyb_cache = {r["qid"]: r["pick"] for r in (load(hy_path) if hy_path.exists() else [])}

    def survivors(qid: str, pool: list[str], n: float = 2.0) -> list[str]:
        """Gate: keep candidates with score>0 (connectivity=1 & some coverage); sort by score desc.
        Drops structurally-invalid (conn=0) candidates before the agent sees them."""
        topK = pool[:KMAX_HOLISTIC]
        scored = [(score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in topK]
        surv = [c for s, c in sorted(scored, key=lambda x: -x[0]) if s > 0]
        return surv or topK

    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()
        qmap = {q["_qid"]: q for q in qs}

        def run_pool(tasks, fn, fh, label):
            if not tasks:
                return
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fn, t): t for t in tasks}
                for fut in as_completed(futs):
                    key, out = fut.result()
                    with lock:
                        fh.write(json.dumps(out) + "\n"); fh.flush()
                    done += 1
                    if done % 20 == 0:
                        print(f"  ... {label} {done}/{len(tasks)}", flush=True)

        # Phase 1: DECOMP extraction (all uncached (qid,cand), fully parallel)
        ex_tasks = [(q["_qid"], c) for q, pool in zip(qs, pools) for c in pool
                    if (q["_qid"], c) not in extr_cache]
        def do_extract(t):
            qid, cand = t
            out = call_json(oai, EXTRACT_PROMPT.format(
                question=qmap[qid]["question"], engine=invs[cand]["engine"],
                entity_block=render_entities(invs[cand])))
            extr_cache[(qid, cand)] = out
            return t, {"qid": qid, "cand": cand, "out": out}
        with open(ex_path, "a") as f_ex:
            print(f"  extraction: {len(ex_tasks)} calls @ {args.workers} workers", flush=True)
            run_pool(ex_tasks, do_extract, f_ex, "extract")

        # Phase 2: HOLISTIC pick (needs extraction scores; parallel over queries)
        ho_tasks = [(q["_qid"], tuple(pool[:KMAX_HOLISTIC])) for q, pool in zip(qs, pools)
                    if q["_qid"] not in holi_cache]
        def do_holistic(t):
            qid, top5 = t
            blk = [f"[{j}] (match score "
                   f"{score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], 2.0):.3f}) "
                   f"{cards[c].get('domain_description','')[:300]}"
                   for j, c in enumerate(top5, 1)]
            out = call_json(oai, HOLISTIC_PROMPT.format(
                question=qmap[qid]["question"], candidate_block="\n".join(blk)))
            holi_cache[qid] = out
            return t, {"qid": qid, "out": out}
        with open(ho_path, "a") as f_ho:
            print(f"  holistic: {len(ho_tasks)} calls @ {args.workers} workers", flush=True)
            run_pool(ho_tasks, do_holistic, f_ho, "holistic")

        # Phase 3: HYBRID — deterministic gate (conn=0 dropped) then agent picks among survivors
        pool_by_qid = {q["_qid"]: pool for q, pool in zip(qs, pools)}
        hy_tasks = [q["_qid"] for q in qs if q["_qid"] not in hyb_cache]
        def do_hybrid(qid):
            surv = survivors(qid, pool_by_qid[qid])
            if len(surv) == 1:
                pick = surv[0]
            else:
                blk = [f"[{j}] {cards[c].get('domain_description','')[:300]}" for j, c in enumerate(surv, 1)]
                out = call_json(oai, HYBRID_PROMPT.format(
                    question=qmap[qid]["question"], candidate_block="\n".join(blk)))
                ch = out.get("choice")
                pick = surv[ch - 1] if isinstance(ch, int) and 1 <= ch <= len(surv) else surv[0]
            hyb_cache[qid] = pick
            return qid, {"qid": qid, "pick": pick}
        with open(hy_path, "a") as f_hy:
            print(f"  hybrid: {len(hy_tasks)} queries (gate+agent) @ {args.workers} workers", flush=True)
            run_pool(hy_tasks, do_hybrid, f_hy, "hybrid")

    # ----- scoring / eval sweep (pure code) -----
    print("\n== final-routing eval (sweep n, K) ==")
    print(f"{'K':>3} {'n':>4} {'poolRec':>8} {'retr':>6} {'holi':>6} {'decomp':>7} {'hybrid':>7} "
          f"{'dec|in':>7} {'holi|in':>8} {'hyb|in':>7}")
    diag = None
    for K in sorted({5, args.kmax}):
        for n in (1.0, 2.0, 3.0):
            in_pool = retr_hit = holi_hit = dec_hit = hyb_hit = 0
            dec_hit_in = holi_hit_in = hyb_hit_in = n_in = 0
            gt_top_unique = gt_tied = gt_below = tie_sizes = 0
            for qi, (q, pool) in enumerate(zip(qs, pools)):
                qid = q["_qid"]
                gt = gts[qi]
                poolK = pool[:K]
                gt_in = gt in poolK
                in_pool += gt_in
                retr_hit += (pool[0] == gt)
                scored = [(score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
                best = max(range(len(scored)), key=lambda i: (scored[i][0], -i))  # tie-break = cosine order
                dec_pick = scored[best][1]
                dec_hit += (dec_pick == gt)
                ho = holi_cache.get(qid, {})
                ch = ho.get("choice")
                holi_pick = pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]
                holi_hit += (holi_pick == gt)
                hyb_pick = hyb_cache.get(qid, pool[0])
                hyb_hit += (hyb_pick == gt)
                if gt_in:
                    n_in += 1
                    dec_hit_in += (dec_pick == gt)
                    holi_hit_in += (holi_pick == gt)
                    hyb_hit_in += (hyb_pick == gt)
                    top = max(s for s, _ in scored)
                    gt_score = next(s for s, c in scored if c == gt)
                    at_top = sum(1 for s, _ in scored if s == top)
                    if gt_score < top:
                        gt_below += 1
                    elif at_top == 1:
                        gt_top_unique += 1
                    else:
                        gt_tied += 1; tie_sizes += at_top
            nq = len(qs)
            print(f"{K:>3} {n:>4.0f} {in_pool/nq:>8.3f} {retr_hit/nq:>6.3f} {holi_hit/nq:>6.3f} "
                  f"{dec_hit/nq:>7.3f} {hyb_hit/nq:>7.3f} {dec_hit_in/max(n_in,1):>7.3f} "
                  f"{holi_hit_in/max(n_in,1):>8.3f} {hyb_hit_in/max(n_in,1):>7.3f}")
            if K == args.kmax and n == 2.0:
                diag = (n_in, gt_top_unique, gt_tied, gt_below, tie_sizes)
    nq = len(qs)
    if diag:
        n_in, uniq, tied, below, tsz = diag
        print(f"\n-- decomp diagnostic (K={args.kmax}, n=2, in-pool only, n_in={n_in}) --")
        print(f"  GT alone at top score   : {uniq}/{n_in}  (decomp gets these right)")
        print(f"  GT tied at top score    : {tied}/{n_in}  (avg tie set {tsz/max(tied,1):.1f}; tie-break decides)")
        print(f"  GT below top score      : {below}/{n_in}  (decomp wrong: a sibling scores higher)")
    print(f"(retr/holi_R@1 invariant to n. GT-in-pool@K={args.kmax} = {in_pool}/{nq})")

    # ----- paired McNemar (exact) on in-pool decisions, K=5, n=2 -----
    from math import comb
    def hits(arm: str, K: int = 5, n: float = 2.0):
        out = []
        for qi, (q, pool) in enumerate(zip(qs, pools)):
            qid = q["_qid"]; gt = gts[qi]; poolK = pool[:K]
            if gt not in poolK:
                continue
            if arm == "decomp":
                scored = [(score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
                pick = scored[max(range(len(scored)), key=lambda i: (scored[i][0], -i))][1]
            elif arm == "holistic":
                ch = holi_cache.get(qid, {}).get("choice")
                pick = pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]
            else:  # hybrid
                pick = hyb_cache.get(qid, pool[0])
            out.append(pick == gt)
        return out
    def mcnemar_exact(a, b):
        b_only = sum(1 for x, y in zip(a, b) if y and not x)   # b wins
        a_only = sum(1 for x, y in zip(a, b) if x and not y)   # a wins
        nn = a_only + b_only
        if nn == 0:
            return a_only, b_only, 1.0
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / (2 ** nn))
        return a_only, b_only, p
    hd, hh, hy = hits("decomp"), hits("holistic"), hits("hybrid")
    print(f"\n-- McNemar exact, in-pool, K=5 (n_in={len(hy)}) --")
    for name, a, b in [("hybrid vs decomp", hy, hd), ("hybrid vs holistic", hy, hh),
                       ("holistic vs decomp", hh, hd)]:
        a_only, b_only, p = mcnemar_exact(a, b)
        print(f"  {name:20s}: first_only={a_only} second_only={b_only} p={p:.4f}")


if __name__ == "__main__":
    main()
