"""Provable decomp ablation on 8-case subset. Sudarshan-faithful, ground-truth, reproducible.

PROOF METHOD:
  1. Freeze all LLM outputs to JSONL caches (extract sampled k=3, adjacency rebuilt with the
     faithful ADJ_PROMPT). Scoring is pure deterministic code over the frozen caches → anyone
     re-runs `--score-only` and gets identical numbers WITHOUT spending credit. That is the
     reproducibility proof.
  2. Quantify extract NON-DETERMINISM: per (query, candidate) report coverage over the 3
     samples (mean / min / max) → numerically proves the instability claim.
  3. Cumulative ablation C0->C3, each adds ONE Sudarshan-faithful fix; report decomp hit/8 +
     per-case pick. Shows which fix (if any) recovers routing.

CONFIGS (cumulative):
  C0  faithful extract (sample 0), NEW adjacency, tie-break = pool order      [current decomp]
  C1  + stability: score = MEAN coverage*conn over the 3 extract samples
  C2  + exclusion: drop NA-phrases that are pure operation/modifier/stopword (Sudarshan
        exclusion list, enforced in code) from the coverage denominator
  C3  + Sudarshan tie-break: on score ties, pick max avg cosine(query-phrase, matched-entity)
        embeddings (their step 4), replacing pool-order

Usage:
  python decomp_ablation.py            # build caches (LLM) then score
  python decomp_ablation.py --score-only   # re-score from caches, no LLM (the proof)
"""
from __future__ import annotations
import argparse, json, sys, re
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all
from build_semantic import (client, call_json, chat_model, render_entities,
                            render_indexed_entities, render_confirmed_edges,
                            declared_graph_connected, merge_adjacency, ADJ_PROMPT)
from agent_rerank import (EXTRACT_PROMPT, coverage, connectivity, field2ent,
                          resolve_nodes, top_pool)

N = 2.0
NCASE = 8
KSAMP = 3
ROOT = HERE.parent.parent / "data" / "multidb"
CACHE = ROOT / "agent_cache"
EXC_PATH = CACHE / "abl_extracts_k3.jsonl"
ADJ_PATH = CACHE / "abl_adj_new.jsonl"

# Sudarshan exclusion list (operations / aggregations / modifiers / stopwords) — enforced in
# code. A NA-phrase made ONLY of these tokens is dropped from the coverage denominator
# (Sudarshan: these are never mappable phrases). Field-like nouns (name,id,status,date...) are
# NOT here — only clear non-content tokens.
EXCLUDE = {
    # aggregation / counting
    "number","count","how","many","total","sum","average","avg","amount","quantity",
    # ranking / superlative / comparative
    "most","least","top","bottom","highest","lowest","largest","smallest","biggest",
    "oldest","newest","earliest","latest","more","less","fewer","greatest","maximum",
    "minimum","max","min","ten","five","three","first","last","up",
    # set operations / quantifiers
    "different","unique","distinct","various","all","each","every","any","same","both",
    # generic request verbs / connectors / stopwords
    "list","give","find","show","get","return","tell","provide","display","belong","belongs",
    "named","call","called","is","are","was","were","do","does","did","has","have","had",
    "the","a","an","of","in","to","on","at","for","with","by","which","what","who","whom",
    "whose","where","when","and","or","that","this","these","those","me","please","there",
    "their","its","each","out","can","you","undergone","operate","one",
}


def _tok(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def clean_na(maps: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Return (matched, na_kept, n_dropped). NA phrase dropped if all tokens in EXCLUDE."""
    matched = [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])]
    na_all = [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])]
    na_kept, dropped = [], 0
    for m in na_all:
        toks = _tok(m.get("phrase", ""))
        if toks and all(t in EXCLUDE for t in toks):
            dropped += 1
        else:
            na_kept.append(m)
    return matched, na_kept, dropped


def score_from_maps(maps, adj, inv, n, *, exclusion: bool):
    matched, na, _ = clean_na(maps) if exclusion else (
        [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])],
        [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])], 0)
    cov = coverage(len(matched), len(na), n)
    targets = [t for m in matched for t in m["targets"] if isinstance(t, str)]
    conn = connectivity(targets, adj, field2ent(inv))
    return cov * conn, cov, conn, matched


def build_caches(qs, pools, invs):
    oai = client()
    # adjacency (new prompt), per candidate, cached
    cand_ids = sorted({c for pool in pools for c in pool})
    done_adj = {r["instance_id"]: r for r in (load(ADJ_PATH) if ADJ_PATH.exists() else [])}
    with open(ADJ_PATH, "a") as f:
        for c in cand_ids:
            if c in done_adj:
                continue
            inv = invs[c]
            if declared_graph_connected(inv):
                adj = merge_adjacency(inv, {"implicit_edges": []}); adj["llm_skipped"] = True
            else:
                raw = call_json(oai, ADJ_PROMPT.format(
                    engine=inv["engine"], indexed_entity_block=render_indexed_entities(inv),
                    confirmed_edges_block=render_confirmed_edges(inv),
                    surface_block=render_entities(inv)))
                adj = merge_adjacency(inv, raw); adj["llm_skipped"] = False
            f.write(json.dumps(adj) + "\n"); f.flush()
            done_adj[c] = adj
            print(f"  adj {c[:34]} skip={adj['llm_skipped']} inferred={adj['n_inferred']}", flush=True)
    # extract k=3 samples per (qid, cand)
    done_ex = {(r["qid"], r["cand"], r["s"]) for r in (load(EXC_PATH) if EXC_PATH.exists() else [])}
    with open(EXC_PATH, "a") as f:
        tot = 0
        for q, pool in zip(qs, pools):
            for c in pool:
                for s in range(KSAMP):
                    if (q["_qid"], c, s) in done_ex:
                        continue
                    out = call_json(oai, EXTRACT_PROMPT.format(
                        question=q["question"], engine=invs[c]["engine"],
                        entity_block=render_entities(invs[c])))
                    f.write(json.dumps({"qid": q["_qid"], "cand": c, "s": s, "out": out}) + "\n")
                    f.flush(); tot += 1
                    if tot % 20 == 0:
                        print(f"  extract {tot} ...", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    dbs = {d["instance_id"]: d for d in load(ROOT / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(ROOT / "semantic" / "inventory.jsonl")}
    man = json.loads((ROOT / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(ROOT / "index" / "card.npy")

    by_db = defaultdict(list)
    for q in load(ROOT / "splits" / "test.jsonl"):
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
    import hashlib
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]

    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, 5)
    CACHE.mkdir(exist_ok=True)

    if not args.score_only:
        print("building caches (LLM)...")
        build_caches(qs, pools, invs)

    adjs = {r["instance_id"]: r for r in load(ADJ_PATH)}
    ex = defaultdict(dict)  # (qid,cand) -> {s: maps}
    for r in load(EXC_PATH):
        ex[(r["qid"], r["cand"])][r["s"]] = (r["out"].get("mappings") or [])

    # ---- PROOF 1: non-determinism (coverage across 3 samples, with exclusion off) ----
    print("\n=== NON-DETERMINISM (coverage over 3 samples, raw) ===")
    spreads = []
    for q in qs:
        for c in pools[qs.index(q)]:
            covs = []
            for s in range(KSAMP):
                maps = ex[(q["_qid"], c)].get(s, [])
                _, cov, _, _ = score_from_maps(maps, adjs[c], invs[c], N, exclusion=False)
                covs.append(cov)
            if max(covs) - min(covs) > 1e-6:
                spreads.append(max(covs) - min(covs))
    print(f"  (q,cand) pairs with coverage spread>0: {len(spreads)}  "
          f"mean spread={np.mean(spreads):.3f}  max={max(spreads):.3f}" if spreads
          else "  all deterministic")

    # ---- embeddings for C3 tie-break ----
    all_phrases, all_entities = set(), set()
    for q in qs:
        for c in pools[qs.index(q)]:
            for s in range(KSAMP):
                for m in ex[(q["_qid"], c)].get(s, []):
                    if m.get("targets"):
                        all_phrases.add(m["phrase"])
                        for t in m["targets"]:
                            all_entities.add(t.split(".")[0] if "." in t else t)
    vocab = sorted(all_phrases | all_entities)
    vmat = embed_all(vocab) if vocab else np.zeros((0, 1))
    vidx = {w: i for i, w in enumerate(vocab)}

    def tiebreak(maps) -> float:
        sims = []
        for m in maps:
            if not m.get("targets"):
                continue
            pv = vmat[vidx[m["phrase"]]]
            for t in m["targets"]:
                e = t.split(".")[0] if "." in t else t
                if e in vidx:
                    ev = vmat[vidx[e]]
                    sims.append(float(pv @ ev / (np.linalg.norm(pv) * np.linalg.norm(ev) + 1e-9)))
        return float(np.mean(sims)) if sims else 0.0

    # ---- ablation configs ----
    def run(cfg) -> tuple[int, list]:
        hit, detail = 0, []
        for q, pool in zip(qs, pools):
            gt = q["instance_id"]
            sc, tb = {}, {}
            for c in pool:
                samp = ex[(q["_qid"], c)]
                if cfg in ("C0",):
                    s0 = samp.get(0, [])
                    val, *_ = score_from_maps(s0, adjs[c], invs[c], N, exclusion=False)
                    tb[c] = tiebreak(s0)
                else:
                    excl = cfg in ("C2", "C3")
                    vals = [score_from_maps(samp.get(s, []), adjs[c], invs[c], N, exclusion=excl)[0]
                            for s in range(KSAMP)]
                    val = float(np.mean(vals))
                    tb[c] = tiebreak(samp.get(0, []))
                sc[c] = val
            if cfg == "C3":
                key = lambda k: (round(sc[k], 6), tb[k])          # ties -> semantic cosine
            else:
                key = lambda k: (sc[k], -pool.index(k))           # ties -> pool order
            pick = max(sc, key=key)
            ok = pick == gt
            hit += ok
            detail.append((gt in pool, gt, pick, ok, round(sc.get(gt, 0), 3), round(max(sc.values()), 3)))
        return hit, detail

    print("\n=== ABLATION (decomp hit / 8) ===")
    for cfg in ("C0", "C1", "C2", "C3"):
        hit, detail = run(cfg)
        inpool = sum(1 for d in detail if d[0])
        print(f"\n{cfg}: hit={hit}/8  (in-pool {inpool}/8, ceiling)")
        for ip, gt, pick, ok, gts, tops in detail:
            print(f"   {'OK ' if ok else 'XX '} GT={gt[:30]:30s} gtScore={gts} topScore={tops}"
                  f" pick={pick[:30]}{'' if ip else '  [GT not in pool]'}")


if __name__ == "__main__":
    main()
