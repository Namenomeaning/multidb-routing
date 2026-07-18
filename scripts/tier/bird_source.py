"""BIRD (Li et al., NeurIPS 2024) source loader for the standard3 PostgreSQL tier.

Adds BIRD databases alongside Spider so the PG tier mirrors Sudarshan's
Spider-Route (206 DB) + BIRD-Route (80 DB) composition. BIRD provides 80 DBs
(69 train + 11 dev). We keep the PG instance budget fixed and split it Spider :
BIRD ≈ 206 : 80 (≈ 72 : 28) by DB count.

Schema source = BIRD's *_tables.json (Spider-style table/column metadata); we
render it as CREATE TABLE DDL to match the existing Spider schema_text format.
Only the schema + questions are used (no sqlite content), fetched via
fetch_bird_json_remote.py.

This module is import-safe (no side effects) and has a __main__ dry-run that
prints the selected DBs and the resulting PG composition without writing
anything or making any LLM call.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BIRD_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "sql_bird"


def _load(name: str) -> Any:
    return json.loads((BIRD_DIR / name).read_text())


def bird_schema_text(tab: dict) -> str:
    """Render BIRD *_tables.json entry as CREATE TABLE DDL (Spider-style)."""
    tnames = tab["table_names_original"]
    cols = tab["column_names_original"]  # [[tbl_idx, name], ...]; index 0 = [-1, '*']
    ctypes = tab["column_types"]
    fks = tab.get("foreign_keys", [])  # [[from_col_idx, to_col_idx], ...]

    pk_cols: set[int] = set()
    for pk in tab.get("primary_keys", []):
        if isinstance(pk, list):
            pk_cols.update(pk)
        else:
            pk_cols.add(pk)

    per_tbl: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for ci, (tidx, cname) in enumerate(cols):
        if tidx < 0:
            continue
        per_tbl[tidx].append((ci, cname, ctypes[ci]))

    stmts = []
    for tidx, tname in enumerate(tnames):
        defs = [f'"{cname}" {ctype}' for ci, cname, ctype in per_tbl[tidx]]
        tbl_pk = [cname for ci, cname, _ in per_tbl[tidx] if ci in pk_cols]
        if tbl_pk:
            defs.append("primary key (" + ", ".join(f'"{c}"' for c in tbl_pk) + ")")
        for fc, tc in fks:
            if cols[fc][0] == tidx and cols[tc][0] >= 0:
                defs.append(
                    f'foreign key("{cols[fc][1]}") references "{tnames[cols[tc][0]]}"("{cols[tc][1]}")'
                )
        stmts.append(f'CREATE TABLE IF NOT EXISTS "{tname}" ( {", ".join(defs)} );')
    return " ".join(stmts)


def load_bird_all() -> tuple[dict[str, dict], list[dict]]:
    """Return (schema_by_db, questions) over the full 80-DB BIRD set."""
    schema_by_db: dict[str, dict] = {}
    for tf in ("train_tables.json", "dev_tables.json"):
        for tab in _load(tf):
            schema_by_db[tab["db_id"]] = tab
    questions: list[dict] = []
    for qf in ("train.json", "dev.json"):
        split = "train" if qf.startswith("train") else "dev"
        for row in _load(qf):
            questions.append({"db_id": row["db_id"], "question": row["question"],
                              "SQL": row.get("SQL", ""), "__split": split})
    return schema_by_db, questions


def select_bird_dbs(n: int, min_q: int = 20) -> list[str]:
    """Top-n BIRD DBs by question count (>= min_q), deterministic by (-count, db_id)."""
    _, questions = load_bird_all()
    counts = Counter(r["db_id"] for r in questions)
    eligible = sorted([db for db, c in counts.items() if c >= min_q],
                      key=lambda db: (-counts[db], db))
    return eligible[:n]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bird", type=int, default=26, help="BIRD DBs to add (PG budget fixed)")
    ap.add_argument("--pg-total", type=int, default=94)
    args = ap.parse_args()

    schema_by_db, questions = load_bird_all()
    counts = Counter(r["db_id"] for r in questions)
    chosen = select_bird_dbs(args.n_bird)
    spider_keep = args.pg_total - args.n_bird
    ratio = args.n_bird / args.pg_total

    print(f"BIRD available: {len(schema_by_db)} DBs, {len(questions)} questions")
    print(f"Sudarshan ratio Spider:BIRD = 206:80 = {80/286:.1%} BIRD")
    print(f"Proposed PG (total {args.pg_total}): {spider_keep} Spider + {args.n_bird} BIRD "
          f"= {ratio:.1%} BIRD\n")
    print(f"Selected {len(chosen)} BIRD DBs (top by #q, >=20):")
    for db in chosen:
        n_tbl = len(schema_by_db[db]["table_names_original"])
        print(f"  {db:<34} {counts[db]:>4} q | {n_tbl} tables")
    print(f"\nTotal BIRD q in selected DBs: {sum(counts[db] for db in chosen)}")
    print("\n--- sample schema_text (first DB) ---")
    print(bird_schema_text(schema_by_db[chosen[0]])[:700])
