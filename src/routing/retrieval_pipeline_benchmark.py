"""Full retrieval-pipeline benchmark: OURS (card embed top-10 -> domain triage -> small subset)
vs SUDARSHAN (raw-DDL embed, plain top-K), on the SAME stratified slice the card-vs-raw head-to-head
used (cap 5 queries / GT-DB, all GT-DBs covered -> DB-macro honest, per eval-slice-bias rule).

The card-vs-raw RECALL head-to-head already exists (retrieval_eval.py / report 260611). This adds the
TRIAGE layer to show the full pipeline: ours hands FEWER candidates to the downstream reranker while
keeping recall >= Sudarshan's top-5. Two-layer reporting (CLAUDE.md §3): retrieval recall only; the
agent rerank is a separate layer, not scored here.

Columns reported per set:
  Sudarshan raw@5   : raw-DDL embed, top-5            (baseline rerank pool, 5 candidates)
  Sudarshan raw@10  : raw-DDL embed, top-10
  card@5 / card@10  : card embed (representation effect, = existing head-to-head)
  OURS              : card embed top-10 -> triage(desc) -> picked subset (avg size reported)

Headline = recall (queries with GT in the pool) vs candidates-handed-downstream. Ours wins if recall
>= raw@5 at a SMALLER average pool. McNemar pairs ours-hit vs Sudarshan-raw@5-hit.

Usage:
  LLM_PROVIDER=deepseek python retrieval_pipeline_benchmark.py --set <route_dir> [--cap 5]
  python retrieval_pipeline_benchmark.py --set <route_dir> --score-only   # no LLM, from cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, stratified, embed_all, ranks_for  # noqa: E402
from agent_rerank import top_pool  # noqa: E402
from build_semantic import client, call_json, CACHE_STATS  # noqa: E402
from triage_eval import TRIAGE_PROMPT, candidate_block  # noqa: E402


def split_slice(queries: list[dict], split: str, ktest: int, kdev: int) -> list[dict]:
    """Query-level, test-first stratified benchmark split (deterministic, no RNG).
    Per GT-DB: order its queries by a stable hash, assign the first ktest to TEST (so EVERY GT-DB is
    covered in test → DB-macro honest), the next kdev to DEV (disjoint, prompt-iteration only).
    cap ktest bounds DB-popularity skew on micro recall (eval-slice-bias rule). 'all' = test ∪ dev."""
    by = defaultdict(list)
    for q in queries:
        by[q["instance_id"]].append(q)
    out = []
    for db in sorted(by):
        items = sorted(by[db], key=lambda q: hashlib.md5(
            (q.get("query_id") or q["question"]).encode()).hexdigest())
        test, dev = items[:ktest], items[ktest:ktest + kdev]
        out += {"test": test, "dev": dev, "all": test + dev}[split]
    return out


def recall_at(ranks, k):
    """micro recall = hits / total queries."""
    n = len(ranks)
    return sum(1 for r in ranks if r and r <= k) / n


def macro_recall_at(ranks, gts, k):
    """DB-macro recall = mean over GT-DBs of that DB's hit-rate (project headline metric)."""
    by = defaultdict(list)
    for r, g in zip(ranks, gts):
        by[g].append(1.0 if (r and r <= k) else 0.0)
    return float(np.mean([np.mean(v) for v in by.values()]))


def macro_bool(hits, gts):
    """DB-macro over a per-query boolean hit list (for the OURS pipeline)."""
    by = defaultdict(list)
    for h, g in zip(hits, gts):
        by[g].append(1.0 if h else 0.0)
    return float(np.mean([np.mean(v) for v in by.values()]))


def mcnemar_p(a_hit, b_hit):
    """exact two-sided McNemar over paired booleans (a_only vs b_only)."""
    a_only = sum(1 for a, b in zip(a_hit, b_hit) if a and not b)
    b_only = sum(1 for a, b in zip(a_hit, b_hit) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return a_only, b_only, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ktest", type=int, default=10, help="test queries per GT-DB (test-first, all DBs)")
    ap.add_argument("--kdev", type=int, default=3, help="dev queries per GT-DB (overflow, disjoint)")
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--ktop", type=int, default=10)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="test",
                    help="test = report (all GT-DBs, cap ktest); dev = prompt iteration (disjoint)")
    ap.add_argument("--ptag", default="recall1", help="prompt-version tag in triage cache filename")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.set)
    name = root.name
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    raw = np.load(root / "index" / "raw.npy")
    card = np.load(root / "index" / "card.npy")

    all_q = [q for q in load(root / "splits" / "test.jsonl") if q["instance_id"] in cards]
    qs = split_slice(all_q, args.split, args.ktest, args.kdev)
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    gts = [q["instance_id"] for q in qs]
    nq = len(qs)

    cdir = root / "triage_cache"
    cdir.mkdir(exist_ok=True)
    tag = f"{args.split}_kt{args.ktest}_kd{args.kdev}"
    qv_p = cdir / f"qv_{tag}.npy"
    qids_p = cdir / f"qv_{tag}.ids.json"
    if qv_p.exists() and json.loads(qids_p.read_text()) == [q["_qid"] for q in qs]:
        qv = np.load(qv_p)
    else:
        qv = embed_all([q["question"] for q in qs])
        np.save(qv_p, qv)
        qids_p.write_text(json.dumps([q["_qid"] for q in qs]))

    r_raw = ranks_for(qv, raw, ids, gts)
    r_card = ranks_for(qv, card, ids, gts)
    pools10 = {q["_qid"]: pool for q, pool in zip(qs, top_pool(qv, card, ids, args.ktop))}

    # ---- triage(desc) on card top-10
    tri_p = cdir / f"triage_{args.ptag}_desc_{tag}.jsonl"
    picked = {}
    if tri_p.exists():
        for r in load(tri_p):
            picked[r["qid"]] = r["picked"]
    if not args.score_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        oai = client()
        lock = threading.Lock()
        qmap = {q["_qid"]: q for q in qs}

        def do_one(qid):
            pool = pools10[qid]
            cb = candidate_block("desc", pool, cards, invs)
            out = call_json(oai, TRIAGE_PROMPT.format(question=qmap[qid]["question"], candidate_block=cb))
            rel = out.get("relevant") or []
            return qid, [pool[i - 1] for i in rel if isinstance(i, int) and 1 <= i <= len(pool)]

        todo = [q["_qid"] for q in qs if q["_qid"] not in picked]
        if todo:
            with open(tri_p, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(do_one, qid) for qid in todo]
                done = 0
                for fut in as_completed(futs):
                    qid, pk = fut.result()
                    picked[qid] = pk
                    with lock:
                        f.write(json.dumps({"qid": qid, "picked": pk}) + "\n"); f.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"  triage {done}/{len(todo)}", flush=True)

    # ---- pipeline hits + pool sizes
    ours_hit, ours_sizes = [], []
    for q in qs:
        pk = picked.get(q["_qid"], [])
        ours_sizes.append(len(pk))
        ours_hit.append(q["instance_id"] in pk)
    raw5_hit = [bool(r and r <= 5) for r in r_raw]
    card10_in = [q["instance_id"] in pools10[q["_qid"]] for q in qs]

    print(f"\n=== {name} [{args.split}]: {len(ids)} DBs, {nq} q (ktest{args.ktest}/kdev{args.kdev}, all GT-DBs in test) ===")
    print(f"{'method':28s} {'avg pool':>9s} {'micro_R':>8s} {'DBmacro_R':>10s}")
    print(f"{'Sudarshan raw-DDL top-5':28s} {5.0:>9.2f} {recall_at(r_raw,5):>8.3f} {macro_recall_at(r_raw,gts,5):>10.3f}")
    print(f"{'Sudarshan raw-DDL top-10':28s} {10.0:>9.1f} {recall_at(r_raw,10):>8.3f} {macro_recall_at(r_raw,gts,10):>10.3f}")
    print(f"{'card top-5':28s} {5.0:>9.2f} {recall_at(r_card,5):>8.3f} {macro_recall_at(r_card,gts,5):>10.3f}")
    print(f"{'card top-10':28s} {10.0:>9.1f} {recall_at(r_card,10):>8.3f} {macro_recall_at(r_card,gts,10):>10.3f}")
    print(f"{'OURS card10->triage(desc)':28s} {np.mean(ours_sizes):>9.2f} {sum(ours_hit)/nq:>8.3f} {macro_bool(ours_hit,gts):>10.3f}")

    a, b, p = mcnemar_p(ours_hit, raw5_hit)
    print(f"\nMcNemar OURS vs Sudarshan-raw@5: ours_only={a} sud_only={b} p={p:.4f}")
    gate_in10 = sum(h for h, c in zip(ours_hit, card10_in) if c)
    print(f"triage gate-recall | GT-in-card-top10: {gate_in10}/{sum(card10_in)} = "
          f"{gate_in10/max(1,sum(card10_in)):.3f}  (1.0 = triage drops no GT)")
    h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
    if h + m:
        print(f"CACHE: calls={CACHE_STATS['calls']} hit_ratio={h/(h+m)*100:.0f}%")


if __name__ == "__main__":
    main()
