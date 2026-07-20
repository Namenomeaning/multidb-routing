"""Build the domain-triage cache that the deterministic §3.3 workflow (final_routing.py) consumes.

Pipeline steps S1→S2 (retrieval → triage):
  question → embed → card top-K pool  [S1, deterministic]
          → LLM domain triage keeps the subject-domain-related subset  [S2, LLM SELECTS only]

The triage LLM only returns a subset of candidate numbers — no rank, no winner, no confidence
number (no-self-confidence invariant). Deterministic Coverage×Connectivity scoring and the LLM
tie-break run downstream in final_routing.py, which reads this cache.

Writes  <set>/cache/triage_cache/triage_full_desc_cap{cap}_s{seed}.jsonl : one {qid, picked} per line,
on the SAME stratified(cap, seed) slice final_routing.py reconstructs, so the two align by qid.
"picked" is the list of instance_ids kept (candidate numbers resolved against the top-K pool order).

Append-cached: re-running only fills qids not already in the file. --score-only makes NO API calls
(reads the cache and reports triage recall / avg picked size for an offline integrity check).

Run:
  LLM_PROVIDER=deepseek python -m src.workflow.triage --set data/multidb --cap 5 --seed 260714
  python -m src.workflow.triage --set data/multidb --cap 5 --seed 260611 --score-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_RR = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in sys.path:
    sys.path.insert(0, str(_RR))

from src.core.retrieval import load, stratified, embed_all  # noqa: E402
from src.core.rerank import top_pool  # noqa: E402
from src.core.semantic_card import client, call_json, CACHE_STATS  # noqa: E402
from src.prompts import load as load_prompt  # noqa: E402

# static prefix first (KV-cache prefix), variable candidates + question appended by .format()
TRIAGE_PROMPT = load_prompt("triage")


def qid_of(q: dict) -> str:
    return q.get("query_id") or hashlib.md5(
        (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]


def candidate_block(pool: list[str], cards: dict) -> str:
    """Domain-level (desc) candidate list — engine + domain_description only (the 'desc' view)."""
    out = []
    for i, c in enumerate(pool, 1):
        out.append(f"[{i}] (engine={cards[c]['engine']})\n{cards[c].get('domain_description', '')}")
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--cap", type=int, default=5, help="queries per GT-DB in the stratified slice")
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--ktop", type=int, default=10, help="card top-K pool the triage reads (S1)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--score-only", action="store_true", help="no API: read cache, report recall")
    args = ap.parse_args()

    root = Path(args.set)
    ids = json.loads((root / "index" / "manifest.json").read_text())["instance_ids"]
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    card_idx = np.load(root / "index" / "card.npy")

    # SAME slice construction as final_routing.py (stratified → card filter → qid) → qids align exactly
    qs = stratified(load(root / "splits" / "test.jsonl"), args.cap, args.seed)
    qs = [q for q in qs if q["instance_id"] in cards]
    for q in qs:
        q["_qid"] = qid_of(q)
    qmap = {q["_qid"]: q for q in qs}
    gts = {q["_qid"]: q["instance_id"] for q in qs}

    cdir = root / "cache" / "triage_cache"
    cdir.mkdir(parents=True, exist_ok=True)
    tri_p = cdir / f"triage_full_desc_cap{args.cap}_s{args.seed}.jsonl"

    picked: dict[str, list[str]] = {}
    if tri_p.exists():
        picked = {r["qid"]: r["picked"] for r in load(tri_p)}

    # ---- retrieve card top-K pool per query (needed to report recall + to build any missing triage)
    todo = [qid for qid in qmap if qid not in picked]
    if not args.score_only and todo:
        # query-vector cache keyed on the exact slice, so re-runs skip re-embedding
        qv_p = cdir / f"qv_cap{args.cap}_s{args.seed}.npy"
        ids_p = cdir / f"qv_cap{args.cap}_s{args.seed}.ids.json"
        order = [q["_qid"] for q in qs]
        if qv_p.exists() and json.loads(ids_p.read_text()) == order:
            qv = np.load(qv_p)
        else:
            qv = embed_all([qmap[qid]["question"] for qid in order])
            np.save(qv_p, qv)
            ids_p.write_text(json.dumps(order))
        pools = {qid: pool for qid, pool in zip(order, top_pool(qv, card_idx, ids, args.ktop))}

        oai = client()

        def do_one(qid: str) -> tuple[str, list[str]]:
            pool = pools[qid]
            out = call_json(oai, TRIAGE_PROMPT.format(
                candidate_block=candidate_block(pool, cards),
                question=qmap[qid]["question"]))
            rel = out.get("relevant") or []
            keep = [pool[i - 1] for i in rel if isinstance(i, int) and 1 <= i <= len(pool)]
            return qid, keep

        with open(tri_p, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(do_one, qid) for qid in todo]
            for n, fut in enumerate(futs, 1):
                qid, keep = fut.result()
                picked[qid] = keep
                f.write(json.dumps({"qid": qid, "picked": keep}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"  triage {n}/{len(todo)}  (cache hit/miss={CACHE_STATS['hit']}/{CACHE_STATS['miss']})")
    elif not args.score_only:
        print("all qids already cached — nothing to build")

    # ---- report (offline): triage gate-recall + avg picked size on the cached slice
    have = [qid for qid in qmap if qid in picked]
    if not have:
        print(f"no triage picks on disk for {tri_p.name} — run without --score-only to build")
        return
    in_pick = sum(gts[qid] in picked[qid] for qid in have)
    avg_sz = sum(len(picked[qid]) for qid in have) / len(have)
    print(f"\n=== {root.name}  triage_full_desc_cap{args.cap}_s{args.seed} ===")
    print(f"slice: {len(qs)} queries | cached: {len(have)}")
    print(f"triage gate-recall (GT in picked): {in_pick}/{len(have)} = {in_pick/len(have):.3f}")
    print(f"avg picked pool size: {avg_sz:.2f}  (from card top-{args.ktop})")


if __name__ == "__main__":
    main()
