"""Build Setup B: ours_multidb — our multi-type routing benchmark for the main flow.

Source = standard3_scale_v1 (208 DB: PG 94 = Spider 68 + BIRD 26, Mongo 87, Neo4j 27).
Base splits: test.jsonl (Jun-9 rebuild, BIRD-included, stratified ≥1/DB, leak-free vs
support-balanced) + support-balanced.jsonl (rebuilt today, BIRD covered).

Rebalance: PG is one merged pool; cap BIRD test queries per-DB so Spider:BIRD query
ratio = 72:28 (matches the DB-count ratio, ~Sudarshan 206:80). Keep all Spider, keep
>=1 query per BIRD DB. Mongo/Neo4j test unchanged. Metric = DB-macro (per-DB equal weight).

Outputs under benchmark/v2/ours_multidb/.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "benchmark" / "standard3_scale_v1"
OUT = ROOT / "benchmark" / "v2" / "ours_multidb"
SPIDER_SHARE = 0.72  # Spider fraction of PG queries (BIRD = 0.28)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def rebalance_pg(test: list[dict], src_of: dict[str, str], seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    spider, bird, other = [], defaultdict(list), []
    for q in test:
        eng = q["engine"]
        if eng != "postgresql":
            other.append(q)
            continue
        s = src_of.get(q["instance_id"])
        if s == "Spider":
            spider.append(q)
        elif s == "BIRD":
            bird[q["instance_id"]].append(q)
        else:
            other.append(q)  # any non-Spider/BIRD PG (none expected)

    n_spider = len(spider)
    target_bird = round(n_spider / SPIDER_SHARE * (1 - SPIDER_SHARE))

    # distribute target_bird across BIRD DBs round-robin, >=1/DB, capped at availability
    bird_dbs = sorted(bird)
    for db in bird_dbs:
        rng.shuffle(bird[db])
    kept_bird: list[dict] = []
    picked = defaultdict(int)
    round_idx = 0
    while len(kept_bird) < target_bird:
        progressed = False
        for db in bird_dbs:
            if len(kept_bird) >= target_bird:
                break
            if round_idx < len(bird[db]):
                kept_bird.append(bird[db][round_idx])
                picked[db] += 1
                progressed = True
        if not progressed:
            break
        round_idx += 1

    new_test = other + spider + kept_bird
    stats = {
        "pg_spider": n_spider,
        "pg_bird_before": sum(len(v) for v in bird.values()),
        "pg_bird_after": len(kept_bird),
        "pg_bird_dbs_kept": sum(1 for db in bird_dbs if picked[db] > 0),
        "pg_bird_dbs_total": len(bird_dbs),
        "pg_ratio_spider_bird": f"{n_spider/(n_spider+len(kept_bird))*100:.0f}:{len(kept_bird)/(n_spider+len(kept_bird))*100:.0f}",
        "target_bird": target_bird,
    }
    return new_test, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260610)
    args = ap.parse_args()

    databases = load_jsonl(SRC / "databases.jsonl")
    src_of = {d["instance_id"]: d["source_dataset"] for d in databases}

    test = load_jsonl(SRC / "splits" / "test.jsonl")
    support = load_jsonl(SRC / "splits" / "support-balanced.jsonl")

    new_test, stats = rebalance_pg(test, src_of, args.seed)

    write_jsonl(OUT / "databases.jsonl", databases)
    write_jsonl(OUT / "splits" / "test.jsonl", new_test)
    write_jsonl(OUT / "splits" / "support-balanced.jsonl", support)

    # engine + coverage summary for the rebalanced test
    from collections import Counter
    eng = Counter(q["engine"] for q in new_test)
    cov = len(set(q["instance_id"] for q in new_test))
    summary = {
        "setup": "ours_multidb",
        "source": str(SRC),
        "db_count": len(databases),
        "db_by_engine": dict(Counter(d["engine"] for d in databases)),
        "test_total": len(new_test),
        "test_by_engine": dict(eng),
        "test_dbs_covered": cov,
        "support_total": len(support),
        "pg_rebalance": stats,
        "seed": args.seed,
        "note": "PG = merged Spider+BIRD pool, query ratio rebalanced to 72:28; metric = DB-macro",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
