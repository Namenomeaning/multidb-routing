"""Finalize MultiDB-Route splits — 0 LLM, leakage-safe. Addresses validator findings #2,#3,#4.

Operations (all on data/multidb/splits/):
  1. DEDUP support-balanced.jsonl — remove duplicate (iid,question) occurrences ONLY at file
     index >= MAX_SUPPORT (=5). The first-5 raw rows per DB are the card's provenance (the card was
     built reading rows[:5]); they are NEVER touched so the leakage audit stays valid.
  2. CARVE dev.jsonl — from card-unseen support spare (index>=5) NOT already in the final test set,
     up to DEV_CAP per DB. Guards: dev ∩ test = 0, dev ∩ card-seen = 0, no dups.
  3. FINALIZE test — overwrite test.jsonl with the densified set (test-dense.jsonl content, which is a
     superset of original test minus 7 dups), then remove test-dense.jsonl. Single canonical test.

card-seen anchor = ORIGINAL support file first-5 per DB (read before any dedup).

Usage: python src/routing/clean_splits_finalize.py --set <route_dir> [--dev-cap 5]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MAX_SUPPORT = 5


def load(p):
    return [json.loads(l) for l in open(p)]


def write(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--dev-cap", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.set)
    sp = root / "splits"

    db_eng = {d["instance_id"]: d["engine"] for d in load(root / "databases.jsonl")}
    support = load(sp / "support-balanced.jsonl")
    dense = load(sp / "test-dense.jsonl")

    # ---- card-seen anchor: ORIGINAL file order, first-5 per DB (immutable provenance)
    by_db_order = defaultdict(list)
    for i, q in enumerate(support):
        by_db_order[q["instance_id"]].append(q)
    card_seen = set()
    for iid, rows in by_db_order.items():
        for r in rows[:MAX_SUPPORT]:
            card_seen.add((iid, r["question"]))

    # ---- 1. DEDUP support: keep first occurrence; drop later dup ONLY if its index >= MAX_SUPPORT
    seen_pair = set()
    seen_count_per_db = defaultdict(int)
    sup_clean, dropped = [], 0
    for q in support:
        iid = q["instance_id"]
        key = (iid, q["question"])
        idx = seen_count_per_db[iid]
        seen_count_per_db[iid] += 1
        if key in seen_pair:           # duplicate occurrence
            if idx >= MAX_SUPPORT:     # safe to drop (outside card-seen prefix)
                dropped += 1
                continue
            # else: dup sits within first-5 raw prefix -> keep (card actually saw this slot)
        seen_pair.add(key)
        sup_clean.append(q)
    # verify first-5 prefix per DB unchanged
    chk = defaultdict(list)
    for q in sup_clean:
        chk[q["instance_id"]].append(q)
    for iid in by_db_order:
        before = [r["question"] for r in by_db_order[iid][:MAX_SUPPORT]]
        after = [r["question"] for r in chk[iid][:MAX_SUPPORT]]
        assert before == after, f"PREFIX CHANGED for {iid} — dedup unsafe"

    # ---- 2. CARVE dev from spare not in test-dense, not card-seen
    dense_keys = {(q["instance_id"], q["question"]) for q in dense}
    dev, dev_per_db = [], defaultdict(int)
    dev_keys = set()
    for iid, rows in by_db_order.items():
        for r in rows[MAX_SUPPORT:]:           # card-unseen spare only
            if dev_per_db[iid] >= args.dev_cap:
                break
            key = (iid, r["question"])
            if key in dense_keys or key in card_seen or key in dev_keys:
                continue
            r = dict(r); r["_split"] = "dev_card_unseen"
            dev.append(r); dev_keys.add(key); dev_per_db[iid] += 1

    # ---- GUARDS
    assert not (dev_keys & dense_keys), "LEAK: dev ∩ test"
    assert not (dev_keys & card_seen), "LEAK: dev ∩ card-seen"
    assert not (dense_keys & card_seen), "LEAK: test ∩ card-seen"
    assert len(dev_keys) == len(dev), "dup in dev"

    # ---- 3. FINALIZE: test.jsonl := dense ; remove test-dense.jsonl
    write(sp / "support-balanced.jsonl", sup_clean)
    write(sp / "dev.jsonl", dev)
    write(sp / "test.jsonl", dense)
    (sp / "test-dense.jsonl").unlink(missing_ok=True)

    # ---- report
    import statistics
    print(f"support dedup: {len(support)} -> {len(sup_clean)} (-{dropped} dups at idx>=5; first-5 prefixes intact)")
    print(f"dev carved: {len(dev)} q over {len(dev_per_db)} DB (cap {args.dev_cap}/DB)")
    print(f"test finalized: test.jsonl = {len(dense)} q (dense); test-dense.jsonl removed")
    print(f"GUARDS PASS: dev∩test=0, dev∩card-seen=0, test∩card-seen=0, no dups")
    for label, rows in [("test", dense), ("dev", dev)]:
        c = defaultdict(int)
        for q in rows:
            c[q["instance_id"]] += 1
        for e in ["postgresql", "mongodb", "neo4j"]:
            v = sorted(c[d] for d in db_eng if db_eng[d] == e and c[d] > 0)
            if v:
                print(f"  {label:4} {e:11} #DB={len(v):3} med={int(statistics.median(v))} min={v[0]} total={sum(v)}")


if __name__ == "__main__":
    main()
