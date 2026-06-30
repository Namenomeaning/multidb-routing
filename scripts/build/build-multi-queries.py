"""Build canonical benchmark: databases.jsonl + queries.jsonl + splits/.

Sources:
  PG:    Spider train+validation queries filtered to PG partition db_ids.
  Mongo: DocSpider train + dev (combined; complementary coverage).
  Neo4j: CypherBench test (7 graphs) + train (4 graphs) — covers all 11.

Each query maps to an instance_id via partition.json + engine prefix.

Outputs:
  benchmark/multi/databases.jsonl   (171 corpus instances; symlink to instances.jsonl)
  benchmark/multi/queries.jsonl     (full query pool, ~17.8k)
  benchmark/multi/splits/test.jsonl (stratified ~1k, agent-eval target)
  benchmark/multi/splits/train.jsonl (remainder, threshold tuning / future use)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MULTI_DIR = BENCHMARK_DIR / "multi"
SPLITS_DIR = MULTI_DIR / "splits"


def load_instance_map() -> dict[tuple[str, str], str]:
    """Map (engine, db_id) → instance_id from databases.jsonl."""
    mapping = {}
    src = MULTI_DIR / "databases.jsonl"
    if not src.exists():
        src = MULTI_DIR / "instances.jsonl"  # fallback to legacy filename
    with open(src) as f:
        for line in f:
            inst = json.loads(line)
            mapping[(inst["engine"], inst["db_id"])] = inst["instance_id"]
    return mapping


def build_pg_queries(inst_map: dict) -> list[dict]:
    partition_path = MULTI_DIR / "partition.json"
    with open(partition_path) as f:
        pg_ids = set(json.load(f)["pg"])

    queries = []
    with open(BENCHMARK_DIR / "queries" / "spider-queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["db_id"] in pg_ids:
                instance_id = inst_map.get(("postgresql", q["db_id"]))
                if instance_id:
                    queries.append({
                        "question": q["question"],
                        "instance_id": instance_id,
                        "engine": "postgresql",
                        "db_id": q["db_id"],
                        "source": f"spider-{q.get('split', 'train')}",
                    })
    return queries


def build_mongo_queries(inst_map: dict) -> list[dict]:
    """Combine train + dev for full 50-db coverage (dev-only dbs missing from train)."""
    queries = []
    for split in ("train", "dev"):
        path = DATASET_DIR / "mongodb" / f"{split}.json"
        with open(path) as f:
            data = json.load(f)
        for d in data:
            instance_id = inst_map.get(("mongodb", d["db_id"]))
            if instance_id:
                queries.append({
                    "question": d["question"],
                    "instance_id": instance_id,
                    "engine": "mongodb",
                    "db_id": d["db_id"],
                    "source": f"docspider-{split}",
                })
    return queries


def build_neo4j_queries(inst_map: dict) -> list[dict]:
    """Combine test (7 graphs) + train (4 graphs) — disjoint, full 11-graph coverage."""
    queries = []
    for split in ("test", "train"):
        path = DATASET_DIR / "neo4j" / "queries" / f"{split}.json"
        with open(path) as f:
            data = json.load(f)
        for d in data:
            instance_id = inst_map.get(("neo4j", d["graph"]))
            if instance_id:
                queries.append({
                    "question": d["nl_question"],
                    "instance_id": instance_id,
                    "engine": "neo4j",
                    "db_id": d["graph"],
                    "source": f"cypherbench-{split}",
                })
    return queries


def stratified_test_split(queries: list[dict], target: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Per-engine round-robin sample to ~target queries.
    Each engine target = target // 3. Round-robin over instances within engine
    until per-engine cap reached. Remainder goes to train.
    """
    rng = random.Random(seed)
    per_engine = target // 3

    # Bucket by engine then instance
    by_engine: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in queries:
        by_engine[q["engine"]][q["instance_id"]].append(q)

    test_qids = set()
    for engine, inst_buckets in by_engine.items():
        # Shuffle each instance's queries deterministically
        for inst_id in inst_buckets:
            rng.shuffle(inst_buckets[inst_id])
        # Round-robin: pop one query per instance per round until cap
        instances = list(inst_buckets.keys())
        rng.shuffle(instances)
        picked = 0
        round_idx = 0
        while picked < per_engine:
            progressed = False
            for inst_id in instances:
                if picked >= per_engine:
                    break
                bucket = inst_buckets[inst_id]
                if round_idx < len(bucket):
                    test_qids.add(bucket[round_idx]["query_id"])
                    picked += 1
                    progressed = True
            if not progressed:
                break  # all buckets exhausted
            round_idx += 1

    test = [q for q in queries if q["query_id"] in test_qids]
    train = [q for q in queries if q["query_id"] not in test_qids]
    return test, train


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=1000,
                        help="Target test split size (stratified per engine)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inst_map = load_instance_map()

    pg_qs = build_pg_queries(inst_map)
    mongo_qs = build_mongo_queries(inst_map)
    neo4j_qs = build_neo4j_queries(inst_map)

    all_queries = []
    for i, q in enumerate(pg_qs + mongo_qs + neo4j_qs, 1):
        q["query_id"] = f"q_{i:05d}"
        all_queries.append(q)

    # Full pool
    write_jsonl(MULTI_DIR / "queries.jsonl", all_queries)

    # Stratified splits
    test, train = stratified_test_split(all_queries, args.test_size, args.seed)
    write_jsonl(SPLITS_DIR / "test.jsonl", test)
    write_jsonl(SPLITS_DIR / "train.jsonl", train)

    # Print summary
    engine_counts = Counter(q["engine"] for q in all_queries)
    print(f"Full pool: {len(all_queries)} queries")
    for engine, count in sorted(engine_counts.items()):
        print(f"  {engine}: {count}")

    test_engine = Counter(q["engine"] for q in test)
    test_inst = Counter(q["instance_id"] for q in test)
    print(f"\nTest split: {len(test)} queries, {len(test_inst)} instances")
    for engine, count in sorted(test_engine.items()):
        print(f"  {engine}: {count}")
    print(f"\nTrain split: {len(train)} queries (remainder)")

    print("\nFiles written:")
    print(f"  {MULTI_DIR / 'queries.jsonl'}")
    print(f"  {SPLITS_DIR / 'test.jsonl'}")
    print(f"  {SPLITS_DIR / 'train.jsonl'}")


if __name__ == "__main__":
    main()
