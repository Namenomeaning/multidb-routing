"""Build Setup A: Sudarshan-faithful routing benchmarks (spider_route + bird_route).

Recipe (Sudarshan arXiv:2601.19825): merge all DBs of a dataset into one pool,
split each DB's queries 50/50 into train/test, index = raw DDL text per DB,
route = predict correct db_id. SQL-only (engine=postgresql for both routes).

Outputs under benchmark/v2/sudarshan_repro/{spider_route,bird_route}/:
  databases.jsonl              one row/DB {instance_id, engine, db_id, schema_text, source_dataset}
  splits/train.jsonl           50% queries/DB (Sudarshan provides train; method is training-free)
  splits/test.jsonl            50% queries/DB (eval: R@1/R@3/mAP)
  summary.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent
SPIDER = THESIS / "dataset" / "_spider_dl" / "extracted" / "spider_data"
BIRD = THESIS / "dataset" / "sql_bird"
OUT = ROOT / "benchmark" / "v2" / "sudarshan_repro"


def ddl_from_tables(t: dict) -> str:
    """Render a Spider/BIRD tables.json entry into CREATE TABLE DDL text."""
    tables = t["table_names_original"]
    cols = t["column_names_original"]        # [[tbl_idx, col_name], ...] col 0 = [-1, '*']
    types = t["column_types"]                # parallel to cols
    pks: set[int] = set()                    # column indices (BIRD may nest composite PKs)
    for pk in t.get("primary_keys", []):
        if isinstance(pk, list):
            pks.update(pk)
        else:
            pks.add(pk)
    fks = t.get("foreign_keys", [])          # [[col_idx, ref_col_idx], ...]

    # group columns by table
    by_table: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(tables))}
    for ci, (ti, name) in enumerate(cols):
        if ti < 0:
            continue
        by_table[ti].append((ci, name, types[ci]))

    # foreign keys grouped by owning table
    fk_lines: dict[int, list[str]] = {i: [] for i in range(len(tables))}
    for col_idx, ref_idx in fks:
        ti = cols[col_idx][0]
        col_name = cols[col_idx][1]
        ref_ti = cols[ref_idx][0]
        ref_col = cols[ref_idx][1]
        if ti < 0 or ref_ti < 0:
            continue
        fk_lines[ti].append(
            f'  FOREIGN KEY ("{col_name}") REFERENCES "{tables[ref_ti]}"("{ref_col}")'
        )

    blocks = []
    for ti, tname in enumerate(tables):
        lines = []
        for ci, cname, ctype in by_table[ti]:
            pk = " PRIMARY KEY" if ci in pks else ""
            lines.append(f'  "{cname}" {ctype}{pk}')
        lines.extend(fk_lines[ti])
        body = ",\n".join(lines)
        blocks.append(f'CREATE TABLE "{tname}" (\n{body}\n);')
    return "\n\n".join(blocks)


def load_json(p: Path) -> list | dict:
    return json.loads(p.read_text())


def split_5050(queries: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """50/50 per-DB split. Group by db_id, shuffle, first half -> train, rest -> test."""
    by_db: dict[str, list[dict]] = {}
    for q in queries:
        by_db.setdefault(q["instance_id"], []).append(q)
    rng = random.Random(seed)
    train, test = [], []
    for db in sorted(by_db):
        items = by_db[db]
        rng.shuffle(items)
        half = len(items) // 2
        train.extend(items[:half])
        test.extend(items[half:])
    return train, test


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build_route(name: str, schema_entries: list[dict], query_files: list[tuple[Path, str]],
                prefix: str, source_dataset: str, seed: int) -> dict:
    # databases.jsonl (one schema per db_id; later schema entries override duplicates)
    schema_by_db: dict[str, dict] = {}
    for t in schema_entries:
        schema_by_db[t["db_id"]] = t
    db_ids = sorted(schema_by_db)
    id_map = {db: f"{prefix}_{i:03d}_{db}" for i, db in enumerate(db_ids, 1)}
    databases = [
        {
            "instance_id": id_map[db],
            "engine": "postgresql",
            "db_id": db,
            "schema_text": ddl_from_tables(schema_by_db[db]),
            "source_dataset": source_dataset,
        }
        for db in db_ids
    ]

    # queries (only keep those whose db_id has a schema)
    queries = []
    dropped = 0
    seen_q: set[tuple[str, str]] = set()  # (instance_id, question) — dedup exact dups so the
    dup = 0                               # 50/50 split cannot leak a repeated question across sides
    for qf, sql_key in query_files:
        for q in load_json(qf):
            db = q["db_id"]
            if db not in id_map:
                dropped += 1
                continue
            key = (id_map[db], q["question"].strip())
            if key in seen_q:
                dup += 1
                continue
            seen_q.add(key)
            queries.append({
                "question": q["question"],
                "instance_id": id_map[db],
                "engine": "postgresql",
                "db_id": db,
                "source_dataset": source_dataset,
                "original_query": q.get(sql_key, ""),
            })
    train, test = split_5050(queries, seed)

    out = OUT / name
    write_jsonl(out / "databases.jsonl", databases)
    write_jsonl(out / "splits" / "train.jsonl", train)
    write_jsonl(out / "splits" / "test.jsonl", test)
    summary = {
        "route": name,
        "source_dataset": source_dataset,
        "db_count": len(databases),
        "query_total": len(queries),
        "train_count": len(train),
        "test_count": len(test),
        "dropped_no_schema": dropped,
        "dropped_exact_dup_question": dup,
        "split_rule": "per_db_50_50_random",
        "seed": seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260610)
    args = ap.parse_args()

    # Spider-Route: 206 DB schemas live in test_tables.json (superset incl train+dev+test)
    spider_schemas = load_json(SPIDER / "test_tables.json")
    spider_summary = build_route(
        "spider_route",
        spider_schemas,
        [
            (SPIDER / "train_spider.json", "query"),
            (SPIDER / "train_others.json", "query"),
            (SPIDER / "dev.json", "query"),
            (SPIDER / "test.json", "query"),
        ],
        prefix="sr",
        source_dataset="Spider",
        seed=args.seed,
    )

    # Bird-Route: 80 DB = train_tables (69) + dev_tables (11)
    bird_schemas = load_json(BIRD / "train_tables.json") + load_json(BIRD / "dev_tables.json")
    bird_summary = build_route(
        "bird_route",
        bird_schemas,
        [
            (BIRD / "train.json", "SQL"),
            (BIRD / "dev.json", "SQL"),
        ],
        prefix="br",
        source_dataset="BIRD",
        seed=args.seed,
    )

    print(json.dumps({"spider_route": spider_summary, "bird_route": bird_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
