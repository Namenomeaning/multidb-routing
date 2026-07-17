"""Prompt-variant sweep for the holistic (card + score) final pick.

Holds the candidate block IDENTICAL to the 0.822 holi(score) arm (top-5 cards + deterministic
score). Only the INSTRUCTION text changes. Tests whether a sharper, engine-NEUTRAL instruction
lifts in-pool instance/engine accuracy, or whether 0.822 is a data ceiling (prompt-insensitive).

Engine-neutral guard: no prompt may name or prefer an engine (m3-design-decisions anti-pattern).

Reuses agent_rerank extraction cache (no extraction calls). Each variant's picks cached to
agent_cache/pv_<id>.jsonl. Usage: LLM_PROVIDER=deepseek python prompt_variant_eval.py --set <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.core.retrieval import load, embed_all  # noqa: E402
from src.core.semantic_card import client, call_json, chat_model  # noqa: E402
from src.core.rerank import score_candidate, top_pool, KMAX_HOLISTIC  # noqa: E402
from src.workflow.hybrid import build_slice  # noqa: E402

# every variant ends with the same candidate block + JSON contract; only the framing differs.
HEAD = "## Question\n{question}\n\n## Candidate databases (score = how much of the question this schema covers and connects)\n{candidate_block}\n\n"
TAIL = 'Return ONLY this JSON object: {{"choice": <number>}}\n'

VARIANTS = {
    "v0_base": "Pick the single database that can best answer the question.\n\n" + HEAD + TAIL,
    "v1_specific":
        "Pick the database whose domain matches the question MOST SPECIFICALLY. "
        "Several candidates may overlap on generic terms; choose the one whose core subject "
        "actually is what the question is about, not one that merely mentions similar words.\n\n"
        + HEAD + TAIL,
    "v2_evidence":
        "Choose the single best database. The score reflects how completely each schema covers "
        "and structurally connects the things the question asks for — treat it as evidence, but "
        "also judge whether the database's domain genuinely fits the question's intent.\n\n"
        + HEAD + TAIL,
    "v3_eliminate":
        "Decide which single database answers the question. First rule out candidates whose "
        "domain is only superficially related; among the rest pick the one that most precisely "
        "owns the entities and the specific aspect the question targets.\n\n"
        + HEAD + TAIL,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-dbs", type=int, default=30)
    ap.add_argument("--per-db-cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--n", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")

    qs = build_slice(root, dbs, cards, args.seed, args.max_dbs, args.per_db_cap)
    qmap = {q["_qid"]: q for q in qs}
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, args.kmax)

    cache_dir = root / "cache" / "agent_cache"
    extr_cache = {(r["qid"], r["cand"]): r["out"] for r in load(cache_dir / "extractions.jsonl")}

    def sc(qid, c):
        return score_candidate(extr_cache.get((qid, c), {}), adjs.get(c, {}), invs[c], args.n)

    def block(qid, top5):
        return "\n".join(f"[{j}] (score {sc(qid, c):.3f}) {cards[c].get('domain_description','')[:300]}"
                         for j, c in enumerate(top5, 1))

    pool_by = {q["_qid"]: pool for q, pool in zip(qs, pools)}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    oai = client(); lock = threading.Lock()

    caches = {}
    for vid, tmpl in VARIANTS.items():
        path = cache_dir / f"pv_{vid}.jsonl"
        cache = {r["qid"]: r["out"] for r in (load(path) if path.exists() else [])}
        todo = [q["_qid"] for q in qs if q["_qid"] not in cache]
        if todo:
            print(f"{vid}: {len(todo)} calls @ {args.workers}, model={chat_model()}")

            def do(qid, tmpl=tmpl):
                top5 = list(pool_by[qid][:KMAX_HOLISTIC])
                out = call_json(oai, tmpl.format(question=qmap[qid]["question"],
                                                 candidate_block=block(qid, top5)))
                return qid, out
            with open(path, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(do, qid): qid for qid in todo}
                for fut in as_completed(futs):
                    qid, out = fut.result()
                    cache[qid] = out
                    with lock:
                        fh.write(json.dumps({"qid": qid, "out": out}) + "\n"); fh.flush()
        caches[vid] = cache

    def pick(qid, pool, cache):
        ch = cache.get(qid, {}).get("choice")
        return pool[ch - 1] if isinstance(ch, int) and 1 <= ch <= min(KMAX_HOLISTIC, len(pool)) else pool[0]

    n_in = sum(1 for q, pool in zip(qs, pools) if q["instance_id"] in pool[:args.K])
    print(f"\n{root.name}  K={args.K} n={args.n}  in-pool={n_in}/{len(qs)}  (baseline holi 0.822/eng0.878)")
    print(f"{'variant':14s} {'inst-acc':>9} {'engine-acc':>11}")
    for vid, cache in caches.items():
        inst = eng = 0
        for q, pool in zip(qs, pools):
            gt = q["instance_id"]
            if gt not in pool[:args.K]:
                continue
            p = pick(q["_qid"], pool, cache)
            inst += (p == gt); eng += (dbs[p]["engine"] == dbs[gt]["engine"])
        print(f"{vid:14s} {inst/max(n_in,1):>9.3f} {eng/max(n_in,1):>11.3f}")


if __name__ == "__main__":
    main()
