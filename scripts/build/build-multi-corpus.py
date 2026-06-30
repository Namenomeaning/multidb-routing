"""Build unified instances.jsonl for multi-engine routing benchmark.

Reads partition.json + source data, generates engine-native schema_text
for each instance, outputs benchmark/multi/instances.jsonl.

Schema formats (engine-native):
  PG:    DDL (CREATE TABLE ...)
  Mongo: Collection/field listing
  Neo4j: Cypher-like notation (Node labels, Relationships)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MULTI_DIR = BENCHMARK_DIR / "multi"


def load_partition() -> dict[str, list[str]]:
    with open(MULTI_DIR / "partition.json") as f:
        return json.load(f)


def build_pg_instances(pg_ids: list[str]) -> list[dict]:
    corpus = {}
    with open(BENCHMARK_DIR / "corpus.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            corpus[entry["db_id"]] = entry

    instances = []
    for i, db_id in enumerate(sorted(pg_ids), 1):
        entry = corpus.get(db_id)
        if not entry:
            schema_path = DATASET_DIR / "sql" / db_id / "schema.sql"
            if schema_path.exists():
                ddl = schema_path.read_text().strip()
            else:
                ddl = f"-- No DDL available for {db_id}"
        else:
            ddl = entry["ddl_text"]

        instances.append({
            "instance_id": f"pg_{i:03d}_{db_id}",
            "engine": "postgresql",
            "db_id": db_id,
            "schema_text": ddl,
        })
    return instances


def build_mongo_instances(mongo_ids: list[str]) -> list[dict]:
    with open(DATASET_DIR / "mongodb" / "collections.json") as f:
        docs = json.load(f)

    db_map = {d["db_id"]: d for d in docs}
    instances = []

    for i, db_id in enumerate(sorted(mongo_ids), 1):
        d = db_map[db_id]
        collections = d["collection_names"]
        cols_by_collection = defaultdict(list)
        for col_idx, col_name in d["column_names"]:
            if col_name != "_id":
                cols_by_collection[col_idx].append(col_name)

        lines = [f"Collections: {', '.join(collections)}"]
        lines.append("Fields:")
        for idx, coll_name in enumerate(collections):
            fields = cols_by_collection.get(idx, [])
            lines.append(f"  {coll_name}: {', '.join(fields)}")

        instances.append({
            "instance_id": f"mongo_{i:03d}_{db_id}",
            "engine": "mongodb",
            "db_id": db_id,
            "schema_text": "\n".join(lines),
        })
    return instances


def _extract_neo4j_properties() -> dict[str, dict[str, set[str]]]:
    """Extract node properties per graph from gold Cypher in train+test queries."""
    props: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for split in ("train.json", "test.json"):
        path = DATASET_DIR / "neo4j" / "queries" / split
        if not path.exists():
            continue
        with open(path) as f:
            queries = json.load(f)
        for q in queries:
            cypher = q.get("gold_cypher", "")
            for m in re.finditer(r"\((\w+):(\w+)", cypher):
                var, label = m.group(1), m.group(2)
                for pm in re.finditer(rf"{re.escape(var)}\.(\w+)", cypher):
                    props[q["graph"]][label].add(pm.group(1))
    return props


def build_neo4j_instances(neo4j_ids: list[str]) -> list[dict]:
    with open(DATASET_DIR / "neo4j" / "neo4j.json") as f:
        data = json.load(f)

    all_props = _extract_neo4j_properties()

    instances = []
    for i, db_id in enumerate(sorted(neo4j_ids), 1):
        graph = data[db_id]
        node_labels = set()
        rel_lines = []

        for rel in graph["relations"]:
            pairs = []
            for variant in rel["variants"]:
                parts = variant.split("#")
                if len(parts) == 3:
                    node_labels.add(parts[1])
                    node_labels.add(parts[2])
                    pairs.append((parts[1], parts[2]))
            for subj, obj in sorted(set(pairs)):
                rel_lines.append(f"  {rel['label']}: {subj} -> {obj}")

        graph_props = all_props.get(db_id, {})
        label_lines = []
        for label in sorted(node_labels):
            props = sorted(graph_props.get(label, set()))
            if props:
                label_lines.append(f"  {label}: {', '.join(props)}")
            else:
                label_lines.append(f"  {label}: (no properties extracted)")

        lines = ["Node labels:"]
        lines.extend(label_lines)
        lines.append("Relationships:")
        lines.extend(rel_lines)

        instances.append({
            "instance_id": f"neo4j_{i:03d}_{db_id}",
            "engine": "neo4j",
            "db_id": db_id,
            "schema_text": "\n".join(lines),
        })
    return instances


def main():
    partition = load_partition()

    pg = build_pg_instances(partition["pg"])
    mongo = build_mongo_instances(partition["mongo"])
    neo4j = build_neo4j_instances(partition["neo4j"])

    all_instances = pg + mongo + neo4j

    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MULTI_DIR / "instances.jsonl"
    with open(out_path, "w") as f:
        for inst in all_instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"Built {len(all_instances)} instances:")
    print(f"  PG:    {len(pg)}")
    print(f"  Mongo: {len(mongo)}")
    print(f"  Neo4j: {len(neo4j)}")
    print(f"\nSaved to {out_path}")

    for engine, items in [("PG", pg), ("Mongo", mongo), ("Neo4j", neo4j)]:
        sample = items[0]
        print(f"\n--- {engine} sample ({sample['instance_id']}) ---")
        print(sample["schema_text"][:300])
        if len(sample["schema_text"]) > 300:
            print("...")


if __name__ == "__main__":
    main()
