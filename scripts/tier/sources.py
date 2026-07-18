"""Source loaders for the standard3 benchmark."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
from common import BENCH, DATASET, mongo_schema_text, read_jsonl, safe_slug
from bson_schema import schema_text_from_bson_samples
def load_pg(pg_limit: int) -> tuple[list[dict], list[dict]]:
    corpus_rows = [row for row in read_jsonl(BENCH / "multi" / "databases.jsonl") if row["engine"] == "postgresql"]
    queries = [row for row in read_jsonl(BENCH / "multi" / "queries.jsonl") if row["engine"] == "postgresql"]
    corpus = {row["db_id"]: row["schema_text"] for row in corpus_rows}
    counts = Counter(row["db_id"] for row in queries)
    db_ids = sorted([db for db, n in counts.most_common() if db in corpus and n >= 20][:pg_limit])
    id_map = {db: f"pgstd_{i:03d}_{db}" for i, db in enumerate(db_ids, 1)}
    instances = [
        {
            "instance_id": id_map[db],
            "engine": "postgresql",
            "db_id": db,
            "schema_text": corpus[db],
            "source_dataset": "Spider",
            "source_validity": "high_sql_native",
            "schema_source": "Spider DDL",
        }
        for db in db_ids
    ]
    out_queries = [
        {
            "question": row["question"],
            "instance_id": id_map[row["db_id"]],
            "engine": "postgresql",
            "db_id": row["db_id"],
            "source": row.get("source", "spider"),
            "source_validity": "high_sql_native",
            "original_query": row.get("query", ""),
        }
        for row in queries
        if row["db_id"] in id_map
    ]
    return instances, out_queries
def load_mongo_schema_datasets() -> tuple[list[dict], list[dict]]:
    rows = []
    for path in [
        DATASET / "mongodb_native" / "kylemesh" / "train.parquet",
        DATASET / "mongodb_native" / "kylemesh" / "test.parquet",
        DATASET / "mongodb_native" / "kylemesh" / "validation.parquet",
        DATASET / "mongodb_native" / "skshmjn" / "train.parquet",
    ]:
        if path.exists():
            frame = pd.read_parquet(path)
            frame["__source_file"] = path.name
            rows.extend(frame.to_dict("records"))

    by_db: dict[str, dict[str, Any]] = {}
    q_rows: list[dict[str, Any]] = []
    for row in rows:
        schema_raw = row.get("Schema")
        prompt = row.get("prompts")
        if not schema_raw or not isinstance(schema_raw, str) or not prompt:
            continue
        try:
            schema = json.loads(schema_raw)
            db_name = schema.get("database", {}).get("name")
        except Exception:
            continue
        if not db_name:
            continue
        iid = f"mongo_schema_{safe_slug(db_name)}"
        by_db.setdefault(iid, {"db_name": db_name, "schema": schema})
        q_rows.append({
            "question": str(prompt).strip(),
            "instance_id": iid,
            "engine": "mongodb",
            "db_id": db_name,
            "source": f"mongosh-schema-instruction/{row.get('__source_file')}",
            "source_validity": "medium_existing_mongo_schema_instruction",
            "original_query": str(row.get("query") or ""),
        })

    instances = [
        {
            "instance_id": iid,
            "engine": "mongodb",
            "db_id": item["db_name"],
            "schema_text": mongo_schema_text(item["db_name"], item["schema"]),
            "source_dataset": "kylemesh19/cleaned-mongosh-instructions + skshmjn/mongo_prompt_query",
            "source_validity": "medium_existing_mongo_schema_instruction",
            "schema_source": "dataset Schema column",
        }
        for iid, item in sorted(by_db.items())
    ]
    return instances, q_rows
def load_mongo_eai() -> tuple[list[dict], list[dict]]:
    path = DATASET / "mongodb_native" / "atlas_sample_data_benchmark.flat.csv"
    if not path.exists():
        return [], []
    rows = list(csv.DictReader(path.open(newline="")))
    by_db: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    q_rows = []
    for row in rows:
        db = row["input.databaseName"]
        query = row["expected.dbQuery"] or ""
        for coll in re.findall(r"\bdb\.([A-Za-z_][\w]*)\.", query):
            by_db[db][coll].add("observed_from_mql")
        for field in re.findall(r"['\"]([A-Za-z_][\w.]+)['\"]\s*:", query):
            coll = next(iter(by_db[db]), "unknown_collection")
            by_db[db][coll].add(field)
        q_rows.append({
            "question": row["input.nlQuery"],
            "instance_id": f"mongo_eai_{db}",
            "engine": "mongodb",
            "db_id": db,
            "source": "mongodb-eai/natural-language-to-mongosh",
            "source_validity": "high_mongo_native_queries",
            "original_query": query,
        })
    return _mongo_eai_instances(by_db), q_rows

def _mongo_eai_instances(by_db: dict[str, dict[str, set[str]]]) -> list[dict]:
    instances = []
    for db, collections in sorted(by_db.items()):
        schema_text = schema_text_from_bson_samples(DATASET / "mongodb_native" / "databases" / db, db)
        schema_source = "MongoDB EAI BSON sample files and/or collection index metadata"
        source_validity = "high_mongo_native_queries_bson_or_index_schema"
        if not schema_text:
            lines = [f"Database: {db}", "Collections:"]
            for coll, fields in sorted(collections.items()):
                lines.append(f"  {coll}: {', '.join(sorted(fields))}")
            schema_text = "\n".join(lines)
            schema_source = "field surface extracted from MQL answers; BSON sample unavailable"
            source_validity = "high_mongo_native_queries_low_schema_surface"
        instances.append({
            "instance_id": f"mongo_eai_{db}",
            "engine": "mongodb",
            "db_id": db,
            "schema_text": schema_text,
            "source_dataset": "mongodb-eai/natural-language-to-mongosh",
            "source_validity": source_validity,
            "schema_source": schema_source,
        })
    return instances

def load_neo4j() -> tuple[list[dict], list[dict]]:
    graph_data = json.loads((DATASET / "neo4j" / "neo4j.json").read_text())
    all_queries = []
    for split in ("train", "test"):
        for row in json.loads((DATASET / "neo4j" / "queries" / f"{split}.json").read_text()):
            all_queries.append(row | {"__split": split})
    props = _neo4j_props(all_queries)
    instances = [_neo4j_instance(i, db, graph, props) for i, (db, graph) in enumerate(sorted(graph_data.items()), 1)]
    id_map = {row["db_id"]: row["instance_id"] for row in instances}
    queries = [
        {
            "question": row["nl_question"],
            "instance_id": id_map[row["graph"]],
            "engine": "neo4j",
            "db_id": row["graph"],
            "source": f"cypherbench-{row['__split']}",
            "source_validity": "high_graph_native",
            "original_query": row.get("gold_cypher", ""),
        }
        for row in all_queries
    ]
    return instances, queries

def _neo4j_props(rows: list[dict]) -> dict[str, dict[str, set[str]]]:
    props: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        for var, label in re.findall(r"\((\w+):(\w+)", row.get("gold_cypher", "")):
            for prop in re.findall(rf"{re.escape(var)}\.(\w+)", row.get("gold_cypher", "")):
                props[row["graph"]][label].add(prop)
    return props

def _neo4j_instance(i: int, db: str, graph: dict, props: dict[str, dict[str, set[str]]]) -> dict:
    labels, rels = set(), []
    for rel in graph["relations"]:
        for variant in rel["variants"]:
            parts = variant.split("#")
            if len(parts) == 3:
                labels.update(parts[1:])
                rels.append((rel["label"], parts[1], parts[2]))
    lines = ["Node labels:"]
    lines.extend(f"  {label}: {', '.join(sorted(props[db].get(label, [])))}" for label in sorted(labels))
    lines.append("Relationships:")
    lines.extend(f"  {name}: {left} -> {right}" for name, left, right in sorted(set(rels)))
    return {
        "instance_id": f"neo4jstd_{i:03d}_{db}",
        "engine": "neo4j",
        "db_id": db,
        "schema_text": "\n".join(lines),
        "source_dataset": "CypherBench",
        "source_validity": "high_graph_native",
        "schema_source": "CypherBench graph metadata + gold Cypher property surface",
    }
