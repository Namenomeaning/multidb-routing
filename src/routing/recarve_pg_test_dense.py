"""Re-carve a DENSER PostgreSQL test split for MultiDB-Route — 0 LLM, leakage-safe.

Problem: PG test was median 3 q/DB (45% of PG DBs <3 q) -> per-DB recall un-estimable.
The queries exist but were parked in support-balanced.jsonl. The card consumed only the
FIRST MAX_SUPPORT_QUESTIONS (=5) per DB (build_semantic.load_support_questions: v[:5]); every
support query at index >=5 was NEVER seen by the card -> safe to promote into test (no leakage).

This script promotes those card-unseen PG support spares into test until each PG DB reaches a
target of >=10 test q (capped by availability). Mongo/Neo4j untouched (median 9 / 31 already ok).

Writes splits/test-dense.jsonl. Asserts test-dense ∩ card-seen-first-5 = 0.

Usage: python src/routing/recarve_pg_test_dense.py --set <route_dir> [--target 10]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MAX_SUPPORT = 5  # must match build_semantic.MAX_SUPPORT_QUESTIONS (card-seen prefix)


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--target", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.set)

    db_eng = {d["instance_id"]: d["engine"] for d in load(root / "databases.jsonl")}
    test = load(root / "splits" / "test.jsonl")
    support = load(root / "splits" / "support-balanced.jsonl")

    # support grouped in FILE ORDER (same order load_support_questions sees)
    sup_by_db = defaultdict(list)
    for q in support:
        sup_by_db[q["instance_id"]].append(q)
    card_seen = set()  # (iid, question) the card actually consumed -> must never enter test
    spare = defaultdict(list)  # card-unseen support rows, promotable
    for iid, rows in sup_by_db.items():
        for r in rows[:MAX_SUPPORT]:
            card_seen.add((iid, r["question"]))
        spare[iid] = rows[MAX_SUPPORT:]

    test_count = defaultdict(int)
    for q in test:
        test_count[q["instance_id"]] += 1

    existing = {(q["instance_id"], q["question"]) for q in test}
    seen = set(existing)
    promoted = []
    for iid, eng in db_eng.items():
        need = max(0, args.target - test_count[iid])
        for r in spare[iid]:
            if need <= 0:
                break
            key = (iid, r["question"])
            if key in seen or key in card_seen:  # skip dups + card-seen
                continue
            seen.add(key)
            r = dict(r)
            r["_promoted_from"] = "support_card_unseen"
            promoted.append(r)
            need -= 1

    # ---- dedupe (test.jsonl has 7 pre-existing internal dups; clean them here)
    dense, seen2, n_dropped = [], set(), 0
    for q in test + promoted:
        key = (q["instance_id"], q["question"])
        if key in seen2:
            n_dropped += 1
            continue
        seen2.add(key)
        dense.append(q)

    # ---- leakage guard: no test row may be a card-seen support question
    leak = [q for q in dense if (q["instance_id"], q["question"]) in card_seen]
    assert not leak, f"LEAKAGE: {len(leak)} test rows were seen by the card!"

    out = root / "splits" / "test-dense.jsonl"
    with open(out, "w") as f:
        for q in dense:
            f.write(json.dumps(q) + "\n")

    # report
    import statistics
    pg = [d for d in db_eng if db_eng[d] == "postgresql"]
    new_c = defaultdict(int)
    for q in dense:
        new_c[q["instance_id"]] += 1
    pgv = sorted(new_c[d] for d in pg)
    print(f"wrote {out}  total {len(dense)} q (was {len(test)}; +{len(promoted)} promoted PG; "
          f"−{n_dropped} pre-existing dups cleaned)")
    print(f"PG test q/DB: med {int(statistics.median(pgv))} min {pgv[0]} max {pgv[-1]} "
          f"<3q={sum(1 for v in pgv if v<3)} <10q={sum(1 for v in pgv if v<10)}  total {sum(pgv)}")
    print(f"leakage check PASS (0 card-seen rows in test-dense); promoted all from support[>={MAX_SUPPORT}]")


if __name__ == "__main__":
    main()
