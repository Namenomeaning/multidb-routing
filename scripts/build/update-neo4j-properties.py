"""Update Neo4j schema_text in databases.jsonl to include node properties.

Properties are extracted from gold Cypher queries in train.json + test.json.
PG and Mongo entries are left untouched.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
DATABASES_PATH = Path(__file__).resolve().parent.parent / "benchmark" / "multi" / "databases.jsonl"


def extract_neo4j_properties() -> dict[str, dict[str, set[str]]]:
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


def rebuild_neo4j_schema(db_id: str, neo4j_graph: dict, graph_props: dict[str, set[str]]) -> str:
    node_labels: set[str] = set()
    rel_lines: list[str] = []

    for rel in neo4j_graph["relations"]:
        pairs: list[tuple[str, str]] = []
        for variant in rel["variants"]:
            parts = variant.split("#")
            if len(parts) == 3:
                node_labels.add(parts[1])
                node_labels.add(parts[2])
                pairs.append((parts[1], parts[2]))
        for subj, obj in sorted(set(pairs)):
            rel_lines.append(f"  {rel['label']}: {subj} -> {obj}")

    label_lines: list[str] = []
    for label in sorted(node_labels):
        p = sorted(graph_props.get(label, set()))
        if p:
            joined = ", ".join(p)
            label_lines.append(f"  {label}: {joined}")
        else:
            label_lines.append(f"  {label}: (no properties extracted)")

    lines = ["Node labels:"]
    lines.extend(label_lines)
    lines.append("Relationships:")
    lines.extend(rel_lines)
    return "\n".join(lines)


def main():
    all_props = extract_neo4j_properties()

    with open(DATASET_DIR / "neo4j" / "neo4j.json") as f:
        neo4j_data = json.load(f)

    with open(DATABASES_PATH) as f:
        all_dbs = [json.loads(line) for line in f]

    updated = 0
    for db in all_dbs:
        if db["engine"] != "neo4j":
            continue
        db_id = db["db_id"]
        graph = neo4j_data.get(db_id)
        if not graph:
            print(f"WARN: no neo4j.json entry for {db_id}")
            continue

        graph_props = all_props.get(db_id, {})
        db["schema_text"] = rebuild_neo4j_schema(db_id, graph, graph_props)
        updated += 1

    with open(DATABASES_PATH, "w") as f:
        for db in all_dbs:
            f.write(json.dumps(db, ensure_ascii=False) + "\n")

    print(f"Updated {updated} Neo4j instances in {DATABASES_PATH}")

    for db in all_dbs:
        if db["engine"] == "neo4j":
            print(f"\n=== {db['instance_id']} ===")
            for line in db["schema_text"].splitlines()[:10]:
                print(f"  {line}")


if __name__ == "__main__":
    main()
