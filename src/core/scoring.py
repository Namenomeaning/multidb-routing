"""Clean ablation: does the decomp (Coverage×Connectivity) rerank improve when the LLM mapping
sees the CARD representation (what we embed) instead of the bare entity inventory, and when the
query is parsed ONCE into a fixed phrase set (equal coverage denominator across candidates)?

The old `agent_rerank.py` decomp arm maps against `render_entities(inv)` (bare ≈ raw DDL) and
extracts per-candidate (variable denominator). This confounds representation (schema vs card) with
mechanism (per-candidate parse vs fixed parse). Arms here isolate each factor.

  A  decomp-bare    : per-candidate extract, target = render_entities(inv)         (= old decomp)
  B  decomp-card    : per-candidate extract, target = card_text(card,inv)          (+ card repr)
  C  decomp-cardfix : parse ONCE (schema-blind, fixed phrases) → map fixed phrases
                      against card_text(card,inv); coverage denom = |fixed phrases| (+ fixed parse)
  Ctie              : arm C ranking + LLM tie-break among score-ties (S5)          (the reported pipeline)

This module also writes the caches the SQL head-to-head (workflow/sudarshan.py) replays for its
OURS column: parse.jsonl, map_card.jsonl (arm-C map), tiebreak.jsonl (S5 pick).

INVARIANT (m3-design-decisions): LLM extracts/maps only — no score/confidence. Coverage(e^-n·x) ×
Connectivity(BFS over the candidate's adjacency, pure code) decides. Adjacency graph is NEVER shown
to the LLM (it would be the answer + violate the invariant); the MAP block carries only what
build_index embeds (domain_description + glossary + entities + DECLARED relations), matching
retrieval representation. Connectivity (code) uses the full adjacency (declared ∪ inferred).

Two-layer metrics (CLAUDE.md §3): pool recall@K reported separately; rerank judged R@1 | GT-in-pool.
Caches under <set>/cache/decomp_card_cache/ (own dir → agent_rerank caches untouched). --score-only reruns
the deterministic scoring with no LLM calls.

Usage:
  LLM_PROVIDER=deepseek python decomp_card_eval.py --set <route_dir> --max-dbs 12 --per-db-cap 3
  python decomp_card_eval.py --set <route_dir> --score-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from math import comb
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
from src.core.retrieval import load, stratified, embed_all  # noqa: E402
from src.core.rerank import (coverage, connectivity, field2ent, resolve_nodes, top_pool,  # noqa: E402
                          score_candidate, EXTRACT_PROMPT, KMAX_HOLISTIC)
from src.core.semantic_card import client, call_json, chat_model, render_entities, CACHE_STATS  # noqa: E402
from src.core.index import card_text  # noqa: E402

K = 5  # Sudarshan rerank scope (top-5)


# ---------------------------------------------------------------- prompts

# Schema-blind query decomposition. Inclusion/exclusion criteria + identifier-vs-counting special
# rule are VERBATIM from Sudarshan phrase_mapping_prompt.txt (cache refs/sudarshan/),
# with the schema + mapping stripped (THESIS-VARIANT: Sudarshan does parse+map in ONE schema-aware
# call; we split parse off so the phrase set — hence the coverage denominator — is fixed and equal
# across all candidates). Output = flat phrase list (Sudarshan flat style, no entity grouping).
PARSE_PROMPT = load_prompt("parse")

# Map a FIXED phrase set against ONE candidate's representation. Mapping rules (direct/semantic,
# value-by-type, verb-to-concept, entity→identifying field, multiple targets, N/A sparingly) are
# faithful to Sudarshan phrase_mapping; the SCHEMA block is the card representation (= what we
# embed) instead of raw DDL. Flat output (Sudarshan style). KV-cache: instructions + schema first
# (static per DB), the variable PHRASES list last.
MAP_PROMPT = load_prompt("map")

# CALIBRATED map (THESIS-VARIANT). Fixes BOTH measured failure modes:
#  - OVER-map (loose): keeps tight's VALUE-REF/proper-noun abstention so values aren't forced onto
#    generic fields (coverage stops saturating at ~1).
#  - UNDER-map (tight): tight abstained even on genuine attribute fields (e.g. "most reviewed" vs a
#    review-count field) → coverage collapsed + evidence went non-discriminative. Fix: DERIVED /
#    QUANTIFIABLE attribute phrases are SCHEMA-REF and MUST map to the field that supports that
#    aggregate. This mirrors the PARSE inclusion class "quantifiable attribute" (parse↔map aligned,
#    not invented). Multi-target re-emphasised (Sudarshan original phrase_mapping: "list ALL valid").
# Calibration principle: map iff a genuinely relevant field exists; abstain ONLY when none does.
MAP_PROMPT_CAL = load_prompt("map_cal")

# TIGHT map (ablation arm of the deployed pipeline, used by workflow/final_routing.py --map tight):
# abstain-heavy — maps derived/attribute phrases but withholds on values/proper-nouns. Kept as an
# ablation option, NOT the locked config (which is map_cal). No per-mapping confidence threshold
# (would make the LLM self-score, violating the no-self-confidence invariant).
MAP_PROMPT_TIGHT = load_prompt("map_tight")

# Tie-break prompt: among score-tied candidates the LLM reads their domain cards and picks one
# (the "LLM tie-breaking" step, S5). The LLM only SELECTS — it emits no confidence number, preserving
# the no-self-confidence invariant. Feeds tiebreak.jsonl, replayed by the SQL baseline comparison.
TIEBREAK_PROMPT = load_prompt("tiebreak")


# ---------------------------------------------------------------- scoring (arm C: fixed denominator)

def _components(adj: dict) -> list[set[int]]:
    """Connected components of the candidate adjacency graph (node ids = entity indices)."""
    g: dict[int, set[int]] = defaultdict(set)
    for e in adj.get("edges", []):
        g[e["from"]].add(e["to"]); g[e["to"]].add(e["from"])
    nodes = set(g) | set(range(len(adj.get("entity_index", []))))
    seen: set[int] = set(); comps: list[set[int]] = []
    for s in nodes:
        if s in seen:
            continue
        c: set[int] = set(); dq = deque([s])
        while dq:
            x = dq.popleft()
            if x in seen:
                continue
            seen.add(x); c.add(x); dq.extend(g[x] - seen)
        comps.append(c)
    return comps


def connectivity_faithful(by_phrase: list[list[str]], adj: dict, f2e: dict) -> int:
    """Sudarshan-faithful connectivity (arXiv:2601.19825, Step 3, VERIFIED from paper):
    each phrase may admit MULTIPLE candidate mappings; connectivity = 1 iff there EXISTS one
    combination (one entity per phrase) forming a connected subgraph. Equivalently: 1 iff some
    connected component covers every phrase (each phrase has >=1 candidate entity in it).

    This replaces the earlier THESIS-VARIANT that flattened ALL candidates of ALL phrases into one
    set and required the whole set connected — strictly harsher than the paper, which zeroed the
    correct DB whenever a phrase's alternative senses spanned disconnected tables (measured: it
    killed 31/50 GT-conn0 cases on ours_multidb)."""
    names = adj.get("entity_index", [])
    pn = [resolve_nodes(ts, names, f2e) for ts in by_phrase if ts]
    pn = [s for s in pn if s]                       # phrases with >=1 resolvable entity
    if len(pn) <= 1:
        return 1
    for comp in _components(adj):
        if all(s & comp for s in pn):
            return 1
    return 0


def score_fixed(phrases: list[str], mapping: dict, adj: dict, inv: dict, n: float) -> float:
    """Coverage over the FIXED phrase set (denom = len(phrases), equal across candidates) ×
    Connectivity (Sudarshan-faithful: exists a one-entity-per-phrase selection that is connected)."""
    by = {m["phrase"]: (m.get("targets") or [])
          for m in (mapping.get("mappings") or []) if isinstance(m, dict)}
    matched = [p for p in phrases if by.get(p)]
    na = len(phrases) - len(matched)
    cov = coverage(len(matched), na, n)
    by_phrase = [[t for t in by[p] if isinstance(t, str)] for p in matched]
    conn = connectivity_faithful(by_phrase, adj, field2ent(inv))
    return cov * conn


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="route dir (databases/semantic/index/splits)")
    ap.add_argument("--max-dbs", type=int, default=12)
    ap.add_argument("--per-db-cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=260613)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
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

    # sample: engine-balanced GT-DBs (round-robin), stratified cap/DB
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
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    gts = [q["instance_id"] for q in qs]
    qmap = {q["_qid"]: q for q in qs}
    print(f"{name}: {len(ids)} DBs in pool, {len(qs)} q from {len(pick_dbs)} GT-DBs "
          f"(cap {args.per_db_cap}/DB), K={K}, model={chat_model()}")

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, K)
    pool_by = {q["_qid"]: pool for q, pool in zip(qs, pools)}

    cdir = root / "cache" / "decomp_card_cache"
    cdir.mkdir(exist_ok=True)
    bare_p, card_p, parse_p, map_p, tie_p = (
        cdir / "extract_bare.jsonl", cdir / "extract_card.jsonl", cdir / "parse.jsonl",
        cdir / "map_card.jsonl", cdir / "tiebreak.jsonl")
    bare = {(r["qid"], r["cand"]): r["out"] for r in (load(bare_p) if bare_p.exists() else [])}
    cardx = {(r["qid"], r["cand"]): r["out"] for r in (load(card_p) if card_p.exists() else [])}
    parse = {r["qid"]: r["out"] for r in (load(parse_p) if parse_p.exists() else [])}
    mapc = {(r["qid"], r["cand"]): r["out"] for r in (load(map_p) if map_p.exists() else [])}
    tie = {r["qid"]: r["pick"] for r in (load(tie_p) if tie_p.exists() else [])}

    def tieset(qid: str, pool: list[str], mp: dict, n: float = 2.0) -> list[str]:
        poolK = pool[:K]
        ph = parse.get(qid, {}).get("phrases") or []
        sc = [(score_fixed(ph, mp.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
        top = max(s for s, _ in sc)
        return [c for s, c in sc if s == top]

    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()

        def run(tasks, fn, fh, label):
            if not tasks:
                return
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fn, t): t for t in tasks}
                for fut in as_completed(futs):
                    out = fut.result()
                    with lock:
                        fh.write(json.dumps(out) + "\n"); fh.flush()
                    done += 1
                    if done % 25 == 0:
                        print(f"  ... {label} {done}/{len(tasks)}", flush=True)

        # arm A: bare per-candidate extract (EXTRACT_PROMPT + render_entities)
        ta = [(q["_qid"], c) for q, pool in zip(qs, pools) for c in pool if (q["_qid"], c) not in bare]
        def do_bare(t):
            qid, c = t
            out = call_json(oai, EXTRACT_PROMPT.format(
                question=qmap[qid]["question"], engine=invs[c]["engine"],
                entity_block=render_entities(invs[c])))
            bare[t] = out
            return {"qid": qid, "cand": c, "out": out}
        with open(bare_p, "a") as f:
            print(f"  [A bare] {len(ta)} extract calls @ {args.workers}w", flush=True)
            run(ta, do_bare, f, "A")

        # arm B: card per-candidate extract (EXTRACT_PROMPT + card_text)
        tb = [(q["_qid"], c) for q, pool in zip(qs, pools) for c in pool if (q["_qid"], c) not in cardx]
        def do_card(t):
            qid, c = t
            out = call_json(oai, EXTRACT_PROMPT.format(
                question=qmap[qid]["question"], engine=invs[c]["engine"],
                entity_block=card_text(cards[c], invs[c])))
            cardx[t] = out
            return {"qid": qid, "cand": c, "out": out}
        with open(card_p, "a") as f:
            print(f"  [B card] {len(tb)} extract calls @ {args.workers}w", flush=True)
            run(tb, do_card, f, "B")

        # arm C step 1: parse once per query (schema-blind, fixed phrases)
        tp = [q["_qid"] for q in qs if q["_qid"] not in parse]
        def do_parse(qid):
            out = call_json(oai, PARSE_PROMPT.format(question=qmap[qid]["question"]))
            parse[qid] = out
            return {"qid": qid, "out": out}
        with open(parse_p, "a") as f:
            print(f"  [C parse] {len(tp)} parse calls @ {args.workers}w", flush=True)
            run(tp, do_parse, f, "Cparse")

        # arm C step 2: map fixed phrases against card_text per candidate
        tm = [(q["_qid"], c) for q, pool in zip(qs, pools) for c in pool if (q["_qid"], c) not in mapc]
        def do_map(t):
            qid, c = t
            phrases = parse.get(qid, {}).get("phrases") or []
            out = call_json(oai, MAP_PROMPT.format(
                engine=invs[c]["engine"], schema_block=card_text(cards[c], invs[c]),
                phrases=json.dumps(phrases, ensure_ascii=False)))
            mapc[t] = out
            return {"qid": qid, "cand": c, "out": out}
        with open(map_p, "a") as f:
            print(f"  [C map] {len(tm)} map calls @ {args.workers}w", flush=True)
            run(tm, do_map, f, "Cmap")

        # tie-break (ours): among top-score ties, the LLM reads cards and picks one (S5).
        def make_tiebreak(mp, cache, label, fh):
            tt = [q["_qid"] for q in qs
                  if q["_qid"] not in cache and len(tieset(q["_qid"], pool_by[q["_qid"]], mp)) > 1]
            def do_tie(qid):
                surv = tieset(qid, pool_by[qid], mp)
                blk = "\n".join(f"[{j}] {cards[c].get('domain_description','')[:300]}"
                                for j, c in enumerate(surv, 1))
                out = call_json(oai, TIEBREAK_PROMPT.format(question=qmap[qid]["question"], candidate_block=blk))
                ch = out.get("choice")
                pk = surv[ch - 1] if isinstance(ch, int) and 1 <= ch <= len(surv) else surv[0]
                cache[qid] = pk
                return {"qid": qid, "pick": pk}
            print(f"  [{label}] {len(tt)} tied queries @ {args.workers}w", flush=True)
            run(tt, do_tie, fh, label)
        with open(tie_p, "a") as f:
            make_tiebreak(mapc, tie, "C tie-break", f)

        h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
        if h + m:
            print(f"  CACHE: calls={CACHE_STATS['calls']} hit_ratio={h/(h+m)*100:.0f}%")

    # ---------------- deterministic scoring + metrics ----------------
    def pick(arm: str, qid: str, pool: list[str], n: float) -> str:
        poolK = pool[:K]
        if arm == "Ctie":  # card map (S3) + deterministic score (S4) + LLM tie-break (S5) — the reported pipeline
            ts = tieset(qid, pool, mapc, n)
            return ts[0] if len(ts) == 1 else tie.get(qid, ts[0])
        if arm == "A":
            sc = [(score_candidate(bare.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
        elif arm == "B":
            sc = [(score_candidate(cardx.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
        else:  # C
            ph = parse.get(qid, {}).get("phrases") or []
            sc = [(score_fixed(ph, mapc.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in poolK]
        b = max(range(len(sc)), key=lambda i: (sc[i][0], -i))  # tie-break = cosine order
        return sc[b][1]

    arms = ["A", "Ctie"]
    in_pool = sum(1 for q in qs if q["instance_id"] in pool_by[q["_qid"]][:K])
    retr_r1 = sum(1 for q in qs if pool_by[q["_qid"]][0] == q["instance_id"])
    nq = len(qs)
    print(f"\n== two-layer eval (K={K}) ==  pool recall@{K}={in_pool}/{nq}={in_pool/nq:.3f}  "
          f"retrieval R@1={retr_r1/nq:.3f}")
    print(f"{'arm':10} {'n':>3} {'R@1(all)':>9} {'R@1|inpool':>11}")
    hits_n2 = {}
    for arm in arms:
        for n in (1.0, 2.0, 3.0):
            inst = inst_in = 0
            hl = []
            for q in qs:
                gt = q["instance_id"]; pool = pool_by[q["_qid"]]
                p = pick(arm, q["_qid"], pool, n)
                ok = (p == gt); inst += ok
                if gt in pool[:K]:
                    inst_in += ok; hl.append(ok)
            print(f"{arm:10} {n:>3.0f} {inst/nq:>9.3f} {inst_in/max(in_pool,1):>11.3f}")
            if n == 2.0:
                hits_n2[arm] = hl

    # McNemar exact (in-pool, n=2): the comparisons that answer the question
    def mcnemar(a, b):
        a_only = sum(1 for x, y in zip(a, b) if x and not y)
        b_only = sum(1 for x, y in zip(a, b) if y and not x)
        nn = a_only + b_only
        if nn == 0:
            return a_only, b_only, 1.0
        k = min(a_only, b_only)
        return a_only, b_only, min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / (2 ** nn))
    print(f"\n-- McNemar exact, in-pool, n=2 (n_in={len(hits_n2['A'])}) --")
    for nm, x, y in [("Ctie vs A (ours vs Sudarshan)", "Ctie", "A")]:
        ao, bo, p = mcnemar(hits_n2[x], hits_n2[y])
        print(f"  {nm:36}: {x}_only={ao} {y}_only={bo} p={p:.4f}")

    # ---- RANKING-step comparison (tie-break-independent): does the score put GT in the top
    # cluster, and how discriminative is the cluster? This isolates the ranking mechanism
    # (parse→map→coverage→connectivity) of Sudarshan (A) vs ours (B/C), separate from any tie-break.
    def arm_scores(arm, qid, poolK, n=2.0):
        ph = parse.get(qid, {}).get("phrases") or []
        if arm == "A":
            return [score_candidate(bare.get((qid, c), {}), adjs.get(c, {}), invs[c], n) for c in poolK]
        if arm == "B":
            return [score_candidate(cardx.get((qid, c), {}), adjs.get(c, {}), invs[c], n) for c in poolK]
        return [score_fixed(ph, mapc.get((qid, c), {}), adjs.get(c, {}), invs[c], n) for c in poolK]  # C

    print(f"\n-- RANKING step (tie-break-independent, in-pool, n=2): Sudarshan(A) / ours(C) --")
    print(f"{'arm':6} {'GT_in_top_cluster':>17} {'GT_alone':>9} {'GT_below':>9} {'avg_tie':>8}")
    for arm in ("A", "C"):
        uniq = tied = below = tsz = nin = 0
        for q in qs:
            gt = q["instance_id"]; poolK = pool_by[q["_qid"]][:K]
            if gt not in poolK:
                continue
            nin += 1
            sc = arm_scores(arm, q["_qid"], poolK)
            top = max(sc); at = sum(1 for s in sc if s == top)
            if sc[poolK.index(gt)] < top:
                below += 1
            elif at == 1:
                uniq += 1
            else:
                tied += 1; tsz += at
        intop = uniq + tied
        print(f"{arm:6} {f'{intop}/{nin}={intop/nin:.3f}':>17} {f'{uniq}':>9} {f'{below}':>9} "
              f"{tsz/max(tied,1):>8.1f}")


if __name__ == "__main__":
    main()
