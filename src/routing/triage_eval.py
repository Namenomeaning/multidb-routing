"""Pipeline-B triage ablation: does a wider retrieval (top-10) + an agent domain-triage that PICKS
a relevant subset keep the GT in the candidate list better than a plain top-5, and what candidate
INFO does the agent need to pick well?

Flow tested:
  query --embed--> top-10 cross-type pool
        --agent triage (reads candidate info, SELECTS relevant subset)--> picked list
  (downstream coverage/rerank would run only on `picked`; not scored here — this measures the GATE)

We compare against the plain top-5 retrieval pool (no triage). The headline is GATE-RECALL: does
GT survive into `picked`. A triage that drops GT caps everything downstream (lesson:
m3-design-decisions removed a complete+0-error gate that lost GT and made M3 < M1). So the gate is
phrased RECALL-PROTECTIVE and DOMAIN-RELATEDNESS-ONLY ("keep any topically-related domain; drop only
clearly-unrelated topics") — it does NOT judge answerability (fields / can-it-answer); that is the
downstream coverage step's job. We report gate-recall first.

Info-level ablation — what to show the agent for each top-10 candidate:
  L1 desc   : domain_description only            ("is description enough?")
  L2 +gloss : domain_description + glossary
  L3 full   : full card_text (desc + glossary + entities + relations)

INVARIANT: the agent SELECTS a subset (triage/extraction of relevance), it does NOT output a final
winner or a confidence number — final routing stays downstream/deterministic. This is Pipeline-B
step-2 triage (m3-design-decisions; Sudarshan-recommended), not a scorer.

Two-layer metrics: pure retrieval recall@5 / recall@10 reported separately from gate-recall.
Caches under <set>/triage_cache/. --score-only recomputes metrics from cache (no LLM).

Usage:
  LLM_PROVIDER=deepseek python triage_eval.py --set <route_dir> --max-dbs 12 --per-db-cap 3
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
from retrieval_eval import load, embed_all  # noqa: E402
from agent_rerank import top_pool  # noqa: E402
from build_semantic import client, call_json, chat_model, render_entities, render_relations, CACHE_STATS  # noqa: E402

KTOP = 10  # wide retrieval pool the triage reads
K5 = 5     # plain-retrieval comparison


# ---------------------------------------------------------------- prompts

# RECALL-FIRST first-stage domain filter (cascade / multi-stage retrieval: the first stage maximizes
# recall, precision is deferred to the later re-ranking stage — Wang et al. 2011 "A cascade ranking
# model for efficient ranked retrieval", SIGIR; standard retrieve-then-rerank IR). Grounded in that
# general principle, NOT in any observed test query (THESIS-VARIANT: Sudarshan has no domain-filter
# prompt — its retrieval is pure dense top-k; this LLM filter is thesis-original, Pipeline-B step-2).
# The agent judges SUBJECT-DOMAIN relatedness only; it does NOT judge answerability / fields / best
# (the next stage does). It selects a subset — no rank, no winner, no confidence number (invariant).
#
# CACHE NOTE: TRIAGE_HEAD is byte-identical on every call → DeepSeek KV-cache prefix; the per-query
# candidates + question come AFTER it (only the prefix is cacheable — candidates differ per query).
TRIAGE_HEAD = """Role: You are the FIRST STAGE of a two-stage database router. Your job is
recall-oriented filtering by subject domain; a later precise stage decides which single database
actually answers the question.
Objective: from the candidate databases, return every one whose subject domain could be related to the
question's topic, so the correct database is never lost before the precise stage runs.

Instructions:
1. Judge SUBJECT-DOMAIN (topic) relatedness only — what each database is about vs what the question is
   about. Do NOT judge whether a database has the right fields, can answer the question, or is the best
   one; that is the next stage's job.
2. Favor recall: keep a database whenever its domain could plausibly be related to the question. Drop a
   database ONLY when its subject domain is clearly and unambiguously unrelated to the question's topic.
3. When in doubt, KEEP. Losing the correct database here cannot be recovered later; an extra unrelated
   candidate is cheap because the next stage removes it.
4. Do not rank the databases and do not pick a single winner — just return the related subset.
Return ONLY this JSON object: {{"relevant": [<numbers of the databases to keep>]}}
"""

# static prefix first (cacheable), then variable candidates, then the question last
TRIAGE_PROMPT = TRIAGE_HEAD + """
## Candidate databases
{candidate_block}

## Question
{question}
"""


def info_block(level: str, card: dict, inv: dict) -> str:
    desc = card.get("domain_description", "")
    if level == "desc":
        return desc
    gloss = "; ".join(f"{k}: {v}" for k, v in (card.get("term_glossary") or {}).items())
    if level == "gloss":
        return f"{desc}\nGlossary: {gloss}"
    # full
    return (f"{desc}\nGlossary: {gloss}\nEntities:\n{render_entities(inv)}\n"
            f"Relationships:\n{render_relations(inv)}")


def candidate_block(level: str, pool: list[str], cards: dict, invs: dict) -> str:
    out = []
    for i, c in enumerate(pool, 1):
        out.append(f"[{i}] (engine={cards[c]['engine']})\n{info_block(level, cards[c], invs[c])}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--max-dbs", type=int, default=12)
    ap.add_argument("--per-db-cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--levels", default="desc,gloss,full")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.set)
    name = root.name
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    ids = [c for c in cards if c in invs and c in dbs]
    card_idx = np.load(root / "index" / "card.npy")
    pool_ids = json.loads((root / "index" / "manifest.json").read_text())["instance_ids"]

    # sample: engine-balanced GT-DBs (round-robin), stratified cap/DB  (same as decomp_card_eval)
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
        sel = rng.permutation(len(items))[:args.per_db_cap]
        qs.extend(items[i] for i in sel)
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    gt = {q["_qid"]: q["instance_id"] for q in qs}

    qv = embed_all([q["question"] for q in qs])
    pools10 = {q["_qid"]: pool for q, pool in zip(qs, top_pool(qv, card_idx, pool_ids, KTOP))}

    # ---- retrieval recall (two-layer, reported separately)
    in5 = sum(gt[qid] in pools10[qid][:K5] for qid in gt)
    in10 = sum(gt[qid] in pools10[qid] for qid in gt)
    nq = len(gt)
    print(f"\n{name}: {nq} queries from {len(pick_dbs)} GT-DBs, pool={len(pool_ids)} DBs")
    print(f"== retrieval (pure) ==  recall@5={in5}/{nq}={in5/nq:.3f}  recall@10={in10}/{nq}={in10/nq:.3f}")

    levels = args.levels.split(",")
    cdir = root / "triage_cache"
    cdir.mkdir(exist_ok=True)
    cache = {}  # (level,qid) -> picked ids
    for lv in levels:
        p = cdir / f"triage_{lv}.jsonl"
        if p.exists():
            for r in load(p):
                cache[(lv, r["qid"])] = r["picked"]

    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()

        def do_one(lv, qid):
            pool = pools10[qid]
            q = next(x for x in qs if x["_qid"] == qid)
            cb = candidate_block(lv, pool, cards, invs)
            out = call_json(oai, TRIAGE_PROMPT.format(question=q["question"], candidate_block=cb))
            rel = out.get("relevant") or []
            picked = [pool[i - 1] for i in rel if isinstance(i, int) and 1 <= i <= len(pool)]
            return lv, qid, picked

        for lv in levels:
            todo = [qid for qid in gt if (lv, qid) not in cache]
            if not todo:
                continue
            p = cdir / f"triage_{lv}.jsonl"
            with open(p, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(do_one, lv, qid) for qid in todo]
                done = 0
                for fut in as_completed(futs):
                    lv2, qid, picked = fut.result()
                    cache[(lv2, qid)] = picked
                    with lock:
                        f.write(json.dumps({"qid": qid, "picked": picked}) + "\n"); f.flush()
                    done += 1
                    if done % 20 == 0:
                        print(f"  [{lv}] {done}/{len(todo)}", flush=True)

    # ---- gate metrics
    print(f"\n== triage gate (from top-{KTOP}) ==")
    print(f"{'level':6} {'gateR(all)':>11} {'gateR|in10':>11} {'avg_pick':>9} {'med_pick':>8} {'kept%':>6}")
    for lv in levels:
        keep_all = keep_in10 = 0
        sizes = []
        n_in10 = 0
        for qid in gt:
            picked = cache.get((lv, qid))
            if picked is None:
                continue
            sizes.append(len(picked))
            hit = gt[qid] in picked
            keep_all += hit
            if gt[qid] in pools10[qid]:
                n_in10 += 1
                keep_in10 += hit
        avg = np.mean(sizes) if sizes else 0
        med = int(np.median(sizes)) if sizes else 0
        gr_all = keep_all / nq
        gr_in10 = keep_in10 / n_in10 if n_in10 else 0
        print(f"{lv:6} {gr_all:>11.3f} {gr_in10:>11.3f} {avg:>9.2f} {med:>8} {avg/KTOP*100:>5.0f}%")

    print(f"\nbaseline ref: plain top-5 recall = {in5/nq:.3f} (no triage, fixed 5 candidates downstream)")
    h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
    if h + m:
        print(f"CACHE: calls={CACHE_STATS['calls']} hit_ratio={h/(h+m)*100:.0f}%")


if __name__ == "__main__":
    main()
