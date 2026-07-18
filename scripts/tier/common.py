"""Shared helpers for the standard3 benchmark scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "dataset"
BENCH = ROOT / "benchmark"
OUT = BENCH / "standard3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[?!.]+$", "", text)


def template_key(text: str) -> str:
    text = normalize(text)
    text = re.sub(r"['\"][^'\"]+['\"]", "<value>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    return re.sub(r"\s+", " ", text).strip()


def flatten_fields(obj: Any, prefix: str = "") -> list[str]:
    if not isinstance(obj, dict):
        return [prefix] if prefix else []
    out: list[str] = []
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.extend(flatten_fields(value, name))
        else:
            out.append(name)
    return out


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def mongo_schema_text(db_name: str, schema: dict[str, Any]) -> str:
    database = schema.get("database", schema)
    collections = database.get("collections", [])
    lines = [f"Database: {db_name}", "Collections:"]
    for coll in collections:
        fields = flatten_fields(coll.get("fields", {}))
        indexes = coll.get("indexes", [])
        lines.append(f"  {coll.get('name')}: {', '.join(fields)}")
        if indexes:
            lines.append(f"    indexes: {json.dumps(indexes, ensure_ascii=False)}")
    return "\n".join(lines)
