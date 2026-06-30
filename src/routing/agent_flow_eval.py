"""End-to-end agent-flow probe (behavior study, ~100 queries): how should the FINAL routing decision
be made over the triaged candidate set, and does an agent tie-break beat plain coverage argmax?

Pipeline (retrieval layers reused from cache, not recomputed):
  query → card embed top-10 → domain triage (desc) → picked ~3      [reused: triage_cache]
        → parse query once (schema-blind, fixed phrases)            [Sudarshan-faithful, THESIS-VARIANT split]
        → map fixed phrases → each picked candidate (loose card map) [LLM extraction only]
        → Coverage(e^-n·x) × Connectivity(BFS over per-engine adjacency)  [deterministic — invariant]
        → FINAL DECISION variants compared:
            V1 cov-argmax     : pick top coverage×conn (ties broken by pool order = arbitrary)
            V2 cov+tiebreak   : if top score is a TIE, an agent reads the tied candidates' FULL card
                                context + why-they-tied (matched entities, cov, conn) and picks one;
                                no tie → V1's pick.   (hybrid gate+agent — keeps invariant: agent makes
                                a routing CHOICE, never a self-confidence number fed to a formula)

Coverage/Connectivity are engine-agnostic at scoring time; the per-engine adaptation (PG FK / Mongo
doc-path co-residence / Neo4j graph rels) lives in adjacency.jsonl, built per engine. So one coverage
flow serves all three DB types.

Two-layer metric: R@1 reported BOTH overall and | GT-in-triaged-pool (routing layer isolated from
retrieval/triage misses). Caches under <set>/agent_flow_cache/. --score-only skips all LLM.

Usage:
  LLM_PROVIDER=deepseek python agent_flow_eval.py --set <route_dir> --n-queries 100
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
from retrieval_eval import load, stratified  # noqa: E402
from build_semantic import client, call_json, CACHE_STATS  # noqa: E402
from build_index import card_text  # noqa: E402
from agent_rerank import field2ent, connectivity, coverage  # noqa: E402
from decomp_card_eval import (PARSE_PROMPT, MAP_PROMPT, MAP_PROMPT_TIGHT,  # noqa: E402
                              MAP_PROMPT_CAL, score_fixed)

N = 2.0  # coverage penalty (Sudarshan hyperparameter, used throughout the thesis runs)


# ---------------------------------------------------------------- final-decision prompts

# Agent tie-break: invoked ONLY when the deterministic score ties. Full context (each tied card +
# why they tied) + explicit ordered criteria. Output is a CHOICE among given options (the final
# routing decision), not a confidence number — invariant preserved.
TIEBREAK_HEAD = """You are making the FINAL database-routing decision. Several candidate databases
scored NEARLY EQUAL on the automatic structural check (coverage of the query's terms × connectivity
of the matched schema entities) — they are too close to separate automatically, so the decision falls
to you. Each candidate below lists its score and the actual phrase→field mapping evidence: which query
phrases it COVERED (and onto which fields) and which it could NOT match (N/A). Use this mapping
evidence — a candidate that covers a phrase by forcing it onto a generic/thematic field is weaker than
one with a genuine field for it. Choose the SINGLE database that can actually answer the question.

Apply these criteria IN ORDER:
1. DOMAIN: does the database's subject actually match what the question is about? Reject a candidate
   whose domain is only adjacent (e.g. a question about sculptures vs a database that records only
   exhibitions/themes).
2. ENTITIES: does it contain the specific entities/attributes the question needs — not just loosely
   related fields? A value forced onto a generic field does NOT count as having that entity.
3. RELATIONSHIPS: can those entities be joined/traversed to answer the question (the links exist)?
Pick the candidate that best satisfies 1→2→3. Use ONLY what each card states; invent nothing.
Return ONLY this JSON object: {{"choice": <number>}}
"""
TIEBREAK_PROMPT = TIEBREAK_HEAD + """
## Tied candidates (all equal on the automatic check)
{tied_block}

## Question
{question}
"""

def matched_entities(phrases, mapping):
    by = {m["phrase"]: (m.get("targets") or [])
          for m in (mapping.get("mappings") or []) if isinstance(m, dict)}
    ents = []
    for p in phrases:
        for t in by.get(p, []):
            if isinstance(t, str):
                ents.append(t)
    return sorted(set(ents))


def map_evidence(phrases, mapping):
    """Per-candidate mapping evidence for the tie-break context: which phrases were covered (and onto
    which fields) and which were N/A. Shows the agent WHY coverage scored as it did."""
    by = {m["phrase"]: [t for t in (m.get("targets") or []) if isinstance(t, str)]
          for m in (mapping.get("mappings") or []) if isinstance(m, dict)}
    covered, na = [], []
    for p in phrases:
        tg = by.get(p) or []
        (covered if tg else na).append(f"{p} → {', '.join(tg)}" if tg else p)
    cov_str = "; ".join(covered) or "(none)"
    na_str = "; ".join(na) or "(none)"
    return cov_str, na_str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tie-margin", type=float, default=0.05,
                    help="near-tie band: candidates with (top_score - score) <= margin go to the "
                         "agent tie-break. 0.0 = only exact ties (old V2 behavior).")
    ap.add_argument("--map", choices=["loose", "tight", "cal"], default="loose",
                    help="mapping prompt: loose = Sudarshan-faithful (over-maps → coverage saturates); "
                         "tight = abstain-heavy (under-maps real attributes); "
                         "cal = calibrated (map derived attrs, abstain only values/proper-nouns).")
    ap.add_argument("--ptag", default="v1",
                    help="parse/map cache version. v1 = original caches; bump (e.g. v2) when PARSE or "
                         "MAP prompt changes so phrases/mappings are recomputed, not reused stale.")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--dump", default="", help="write per-qid V2 correctness jsonl (paired McNemar)")
    args = ap.parse_args()

    root = Path(args.set)
    name = root.name
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}

    # reuse triage picks (qid -> picked ~3) from the pipeline benchmark cache
    tri_p = root / "triage_cache" / f"triage_full_desc_cap{args.cap}_s{args.seed}.jsonl"
    if not tri_p.exists():
        print(f"MISSING {tri_p} — run retrieval_pipeline_benchmark.py first"); return
    picks = {r["qid"]: r["picked"] for r in load(tri_p)}

    qs = stratified(load(root / "splits" / "test.jsonl"), args.cap, args.seed)
    qs = [q for q in qs if q["instance_id"] in cards]
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    qs = [q for q in qs if q["_qid"] in picks]

    # engine-balanced sample of n-queries (behavior study across the 3 DB types)
    rng = np.random.default_rng(args.seed)
    by_eng = defaultdict(list)
    for q in qs:
        by_eng[dbs[q["instance_id"]]["engine"]].append(q)
    for e in by_eng:
        idx = rng.permutation(len(by_eng[e]))
        by_eng[e] = [by_eng[e][i] for i in idx]
    engines = sorted(by_eng)
    sample, ei = [], 0
    while len(sample) < args.n_queries and any(by_eng.values()):
        e = engines[ei % len(engines)]
        if by_eng[e]:
            sample.append(by_eng[e].pop(0))
        ei += 1
    qmap = {q["_qid"]: q for q in sample}
    print(f"{name}: {len(sample)} queries (engine-balanced) | triaged pool reused")

    cdir = root / "agent_flow_cache"
    cdir.mkdir(exist_ok=True)
    map_prompt = {"tight": MAP_PROMPT_TIGHT, "cal": MAP_PROMPT_CAL}.get(args.map, MAP_PROMPT)
    # separate caches per (parse version, map prompt, margin) so configs never reuse stale results.
    # ptag=v1 keeps the original un-suffixed filenames; bumping ptag forces fresh parse + map.
    psuf = "" if args.ptag == "v1" else f"_{args.ptag}"
    parse_p = cdir / f"parse{psuf}.jsonl"
    map_p = cdir / f"map_{args.map}{psuf}.jsonl"
    tie_p = cdir / f"tiebreak_{args.map}{psuf}_m{args.tie_margin}.jsonl"
    parse = {r["qid"]: r["out"] for r in (load(parse_p) if parse_p.exists() else [])}
    mp = {(r["qid"], r["cand"]): r["out"] for r in (load(map_p) if map_p.exists() else [])}
    tie = {r["qid"]: r["choice"] for r in (load(tie_p) if tie_p.exists() else [])}

    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()

        def run(fn, todo, path, desc):
            if not todo:
                return
            with open(path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(fn, *a) for a in todo]
                done = 0
                for fut in as_completed(futs):
                    rec = fut.result()
                    with lock:
                        f.write(json.dumps(rec) + "\n"); f.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"  {desc} {done}/{len(todo)}", flush=True)

        def do_parse(qid):
            out = call_json(oai, PARSE_PROMPT.format(question=qmap[qid]["question"]))
            parse[qid] = out
            return {"qid": qid, "out": out}
        run(do_parse, [(q,) for q in qmap if q not in parse], parse_p, "parse")

        def do_map(qid, c):
            ph = parse.get(qid, {}).get("phrases") or []
            out = call_json(oai, map_prompt.format(
                engine=invs[c]["engine"], schema_block=card_text(cards[c], invs[c]),
                phrases=json.dumps(ph)))
            mp[(qid, c)] = out
            return {"qid": qid, "cand": c, "out": out}
        todo_map = [(q, c) for q in qmap for c in picks[q] if (q, c) not in mp]
        run(do_map, todo_map, map_p, "map")

    # ---- score + decide
    margin = args.tie_margin

    def scored(qid):
        ph = parse.get(qid, {}).get("phrases") or []
        pool = picks[qid]
        sc = [(score_fixed(ph, mp.get((qid, c), {}), adjs.get(c, {}), invs[c], N), c) for c in pool]
        sc.sort(key=lambda x: -x[0])
        return ph, sc

    def near(sc):
        """Candidates within `margin` of the top score → near-tie set fed to the agent.
        1 element => clear winner (take it, no LLM). >=2 => agent tie-breaks among the close ones.
        Zero-score candidates are excluded when a positive top exists (they are clearly worse)."""
        if not sc:
            return []
        top = sc[0][0]
        return [c for s, c in sc if top - s <= margin and (top == 0 or s > 0)]

    # tie-break agent fires only on ties (computed now, cached)
    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client(); lock = threading.Lock()

        def do_tie(qid):
            ph, sc = scored(qid)
            tied = near(sc)
            score_of = {c: s for s, c in sc}
            blocks = []
            for i, c in enumerate(tied, 1):
                cov_str, na_str = map_evidence(ph, mp.get((qid, c), {}))
                blocks.append(
                    f"[{i}] engine={cards[c]['engine']}  (auto-score={score_of.get(c, 0):.3f})\n"
                    f"{card_text(cards[c], invs[c])}\n"
                    f"COVERED phrases: {cov_str}\n"
                    f"N/A (no field matched): {na_str}")
            out = call_json(oai, TIEBREAK_PROMPT.format(
                question=qmap[qid]["question"], tied_block="\n\n".join(blocks)), thinking=True)
            ch = out.get("choice")
            pick = tied[ch - 1] if isinstance(ch, int) and 1 <= ch <= len(tied) else tied[0]
            tie[qid] = pick
            return {"qid": qid, "choice": pick}

        from concurrent.futures import ThreadPoolExecutor as TPE
        tie_todo = []
        for qid in qmap:
            _, sc = scored(qid)
            if qid not in tie and len(near(sc)) >= 2:
                tie_todo.append((qid,))

        def _run(fn, todo, path, desc):
            if not todo:
                return
            with open(path, "a") as f, TPE(max_workers=args.workers) as ex:
                futs = [ex.submit(fn, *a) for a in todo]
                for k, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    with lock:
                        f.write(json.dumps(rec) + "\n"); f.flush()
                    if k % 25 == 0:
                        print(f"  {desc} {k}/{len(todo)}", flush=True)
        _run(do_tie, tie_todo, tie_p, "tiebreak")

    # ---- metrics (two-layer)
    nq = len(qmap)
    in_pool = [q for q in qmap if qmap[q]["instance_id"] in picks[q]]
    n_in = len(in_pool)
    n_ties = 0
    tie_sizes = []
    v1 = v2 = 0
    v1_in = v2_in = 0
    # per-GT-engine slice of V2 (the decision rule): [v2_ok, n, v2_in_ok, n_in]
    per_eng = defaultdict(lambda: [0, 0, 0, 0])
    dump_rows = []
    for qid in qmap:
        gt = qmap[qid]["instance_id"]
        ph, sc = scored(qid)
        if not sc:                       # empty triaged pool → automatic miss for every variant
            dump_rows.append({"qid": qid, "correct": 0, "in_pool": int(gt in picks.get(qid, [])),
                              "engine": dbs[gt]["engine"]})
            continue
        tied = near(sc)
        if len(tied) >= 2:
            n_ties += 1; tie_sizes.append(len(tied))
        p1 = sc[0][1]
        p2 = tie.get(qid, p1) if len(tied) >= 2 else p1
        ig = gt in picks[qid]
        v1 += p1 == gt; v2 += p2 == gt
        ge = dbs[gt]["engine"]
        per_eng[ge][0] += p2 == gt; per_eng[ge][1] += 1
        dump_rows.append({"qid": qid, "correct": int(p2 == gt), "in_pool": int(ig), "engine": ge})
        if ig:
            v1_in += p1 == gt; v2_in += p2 == gt
            per_eng[ge][2] += p2 == gt; per_eng[ge][3] += 1

    print(f"\n=== {name}: {nq} queries, {n_in} with GT-in-triaged-pool ===")
    print(f"near-tie margin δ={margin}: {n_ties}/{nq} queries ({n_ties/nq*100:.0f}%) have >=2 "
          f"candidates within δ → LLM tie-break; avg near-tie size "
          f"{np.mean(tie_sizes) if tie_sizes else 0:.2f}")
    print(f"{'variant':28s} {'R@1(all)':>9s} {'R@1|in-pool':>12s}")
    print(f"{'V1 cov-argmax':28s} {v1/nq:>9.3f} {v1_in/n_in:>12.3f}")
    print(f"{'V2 cov+LLM-tiebreak':28s} {v2/nq:>9.3f} {v2_in/n_in:>12.3f}")
    for ge in sorted(per_eng):
        ok, n, oin, nin = per_eng[ge]
        print(f"   └ GT-engine {ge:11s} N={n:4d}  V2 R@1(all)={ok/max(n,1):.3f}  "
              f"R@1|in-pool={oin/max(nin,1):.3f} (in-pool {nin}/{n})")
    h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
    if h + m:
        print(f"CACHE: calls={CACHE_STATS['calls']} hit_ratio={h/(h+m)*100:.0f}%")
    if args.dump:
        from pathlib import Path as _P
        _P(args.dump).write_text("\n".join(json.dumps(r) for r in dump_rows))
        print(f"dumped {len(dump_rows)} per-qid rows -> {args.dump}")


if __name__ == "__main__":
    main()
