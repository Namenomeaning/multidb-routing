"""Derive compact MongoDB schema text from sampled BSON dump files."""

from __future__ import annotations

import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BSON_TYPES = {
    0x01: "double",
    0x02: "string",
    0x03: "object",
    0x04: "array",
    0x05: "binary",
    0x07: "objectId",
    0x08: "bool",
    0x09: "date",
    0x0A: "null",
    0x10: "int32",
    0x11: "timestamp",
    0x12: "int64",
    0x13: "decimal128",
}


def schema_text_from_bson_samples(db_dir: Path, db_name: str) -> str | None:
    metadata_files = sorted(db_dir.glob("*.metadata.json"))
    if not metadata_files:
        return None
    lines = [f"Database: {db_name}", "Collections:"]
    for meta_path in metadata_files:
        collection = meta_path.name.removesuffix(".metadata.json")
        if collection.startswith("temp_") or collection.startswith("average_"):
            continue
        field_counts = _field_counts(meta_path.with_name(f"{collection}.bson.sample"))
        index_fields = _index_fields(meta_path)
        if not field_counts and not index_fields:
            continue
        fields = _format_fields(field_counts)
        lines.append(f"  {collection}: {', '.join(fields)}")
        if index_fields:
            lines.append(f"    indexes: {', '.join(index_fields)}")
    return "\n".join(lines) if len(lines) > 2 else None


def _field_counts(path: Path) -> dict[str, Counter[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = path.read_bytes()
    pos = 0
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    docs = 0
    while pos + 4 <= len(data) and docs < 200:
        size = _int32(data, pos)
        if size < 5 or pos + size > len(data):
            break
        _parse_doc(data[pos:pos + size], "", counts)
        pos += size
        docs += 1
    return counts


def _parse_doc(data: bytes, prefix: str, counts: dict[str, Counter[str]]) -> None:
    pos = 4
    while pos < len(data) - 1:
        kind = data[pos]
        pos += 1
        name, pos = _cstring(data, pos)
        path = _join(prefix, name)
        counts[path][BSON_TYPES.get(kind, f"type_{kind:x}")] += 1
        pos = _skip_value(data, pos, kind, path, counts)


def _skip_value(data: bytes, pos: int, kind: int, path: str, counts: dict[str, Counter[str]]) -> int:
    if kind == 0x01:
        return pos + 8
    if kind == 0x02 or kind == 0x0D or kind == 0x0E:
        size = _int32(data, pos)
        return pos + 4 + max(size, 0)
    if kind in (0x03, 0x04):
        size = _int32(data, pos)
        end = pos + size
        if 5 <= size and end <= len(data):
            nested_prefix = f"{path}[]" if kind == 0x04 else path
            _parse_doc(data[pos:end], nested_prefix, counts)
        return end
    if kind == 0x05:
        size = _int32(data, pos)
        return pos + 5 + max(size, 0)
    if kind == 0x07:
        return pos + 12
    if kind == 0x08:
        return pos + 1
    if kind in (0x09, 0x11, 0x12):
        return pos + 8
    if kind == 0x0A:
        return pos
    if kind == 0x0B:
        _, pos = _cstring(data, pos)
        _, pos = _cstring(data, pos)
        return pos
    if kind == 0x0F:
        size = _int32(data, pos)
        return pos + max(size, 4)
    if kind == 0x10:
        return pos + 4
    if kind == 0x13:
        return pos + 16
    return len(data)


def _index_fields(meta_path: Path) -> list[str]:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return []
    fields = []
    for index in meta.get("indexes", []):
        key = index.get("key", {})
        if isinstance(key, dict):
            fields.extend(str(name) for name in key)
    return sorted(set(fields))


def _format_fields(field_counts: dict[str, Counter[str]], limit: int = 80) -> list[str]:
    ranked = sorted(field_counts.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    fields = []
    for path, type_counts in ranked[:limit]:
        if _skip_path(path):
            continue
        type_name = type_counts.most_common(1)[0][0]
        fields.append(f"{path}<{type_name}>")
    return fields


def _skip_path(path: str) -> bool:
    for part in path.replace("[]", "").split("."):
        if part.isdigit():
            return True
        if len(part) >= 24 and all(char in "0123456789abcdef" for char in part.lower()):
            return True
    return False


def _join(prefix: str, name: str) -> str:
    if name.isdigit():
        return prefix
    return f"{prefix}.{name}" if prefix else name


def _cstring(data: bytes, pos: int) -> tuple[str, int]:
    end = data.find(b"\x00", pos)
    if end < 0:
        return "", len(data)
    return data[pos:end].decode("utf-8", errors="replace"), end + 1


def _int32(data: bytes, pos: int) -> int:
    if pos + 4 > len(data):
        return 0
    return struct.unpack_from("<i", data, pos)[0]
