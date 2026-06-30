"""E3c obfuscation control for v2 (memory semantic-beats-raw-obfuscated; BENCHMARK.md §5).

Mask schema IDENTIFIER tokens (entity + field names) -> stable per-DB pseudonyms,
consistently across raw schema_text, inventory, and card. Queries stay CLEAN (real words).

  raw-obf  : schema_text with identifiers pseudonymised -> collapses if it rode on name overlap
  card-obf : card domain_description prose (real domain words, NOT exact identifiers) survives;
             only identifier surface forms (entity/field/glossary keys) get masked

Re-embed + re-eval: card-obf still > raw-obf and significant => gain is real query-time enrichment,
not query<->schema-name lexical coincidence. Card prose was written at BUILD time from real schema
(legitimate documentation) — this isolates QUERY-TIME name independence only (claim stays scoped).

Usage:
  python obfuscate.py --bench <route_dir> --semantic <sem_dir> --out <obf_route_dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# SQL/engine keywords + types never masked (else structure breaks / unfair to raw)
KEEP = {
    "CREATE", "TABLE", "IF", "NOT", "EXISTS", "PRIMARY", "FOREIGN", "KEY", "REFERENCES",
    "NULL", "UNIQUE", "CONSTRAINT", "INDEX", "DEFAULT", "AUTOINCREMENT", "ON", "DELETE",
    "UPDATE", "CASCADE", "INTEGER", "INT", "TEXT", "VARCHAR", "CHAR", "REAL", "FLOAT",
    "DOUBLE", "DECIMAL", "NUMERIC", "DATE", "DATETIME", "TIME", "TIMESTAMP", "BOOLEAN",
    "BOOL", "BLOB", "SERIAL", "BIGINT", "SMALLINT", "DATABASE", "COLLECTIONS", "INDEXES",
    "NODE", "LABELS", "RELATIONSHIPS", "PROPERTIES", "STRING", "NUMBER", "ARRAY", "OBJECT",
    "TRUE", "FALSE", "AND", "OR", "PK", "FK", "VARYING", "PRECISION", "WITHOUT", "ZONE",
}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build_name_map(inv: dict) -> dict[str, str]:
    """Collect entity + field identifiers, map each to a stable pseudonym."""
    ents = [e["name"] for e in inv["entities"]]
    fields = sorted({f["name"] for e in inv["entities"] for f in e["fields"]})
    m: dict[str, str] = {}
    for i, e in enumerate(ents, 1):
        m[e] = f"Entity{i:02d}"
    for i, f in enumerate(fields, 1):
        if f not in m:  # a field sharing an entity's exact name keeps the entity pseudonym
            m[f] = f"attr{i:03d}"
    return m


def mask_text(text: str, name_map: dict[str, str]) -> str:
    """Replace whole-word identifier tokens (longest first). Dotted paths handled segment-wise."""
    if not text:
        return text
    # longest identifiers first so 'movie_id' is masked before 'movie'
    for ident in sorted(name_map, key=len, reverse=True):
        if ident.upper() in KEEP or not ident:
            continue
        # word-ish boundary that also breaks on '.' so a.b.c segments mask independently
        text = re.sub(rf"(?<![\w]){re.escape(ident)}(?![\w])", name_map[ident], text)
    return text


def mask_inventory(inv: dict, m: dict[str, str]) -> dict:
    ents = []
    for e in inv["entities"]:
        ents.append({"name": m.get(e["name"], e["name"]),
                     "fields": [{**f, "name": m.get(f["name"], mask_text(f["name"], m))} for f in e["fields"]]})
    rels = []
    for r in inv["declared_relations"]:
        rr = dict(r)
        rr["from_entity"] = m.get(r["from_entity"], r["from_entity"])
        rr["to_entity"] = m.get(r["to_entity"], r["to_entity"])
        rr["via"] = m.get(r["via"], mask_text(r.get("via", ""), m))
        if "to_field" in rr:
            rr["to_field"] = m.get(r["to_field"], mask_text(r.get("to_field", ""), m))
        rels.append(rr)
    return {**inv, "entities": ents, "declared_relations": rels}


def mask_card(card: dict, m: dict[str, str]) -> dict:
    desc = mask_text(card.get("domain_description", ""), m)
    gloss = {m.get(k, mask_text(k, m)): mask_text(v, m) for k, v in (card.get("term_glossary") or {}).items()}
    return {**card, "domain_description": desc, "term_glossary": gloss}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench, sem, out = Path(args.bench), Path(args.semantic), Path(args.out)
    dbs = load(bench / "databases.jsonl")
    invs = {i["instance_id"]: i for i in load(sem / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(sem / "cards.jsonl")}

    obf_dbs, obf_inv, obf_cards = [], [], []
    sample = None
    for d in dbs:
        iid = d["instance_id"]
        inv = invs.get(iid)
        if inv is None:
            continue
        m = build_name_map(inv)
        obf_dbs.append({**d, "schema_text": mask_text(d["schema_text"], m)})
        obf_inv.append(mask_inventory(inv, m))
        if iid in cards:
            obf_cards.append(mask_card(cards[iid], m))
        if sample is None and len(m) > 4:
            sample = (iid, d["schema_text"][:200], obf_dbs[-1]["schema_text"][:200])

    write(out / "databases.jsonl", obf_dbs)
    write(out / "semantic" / "inventory.jsonl", obf_inv)
    write(out / "semantic" / "cards.jsonl", obf_cards)
    # copy splits (queries stay clean)
    for sp in (bench / "splits").glob("*.jsonl"):
        write(out / "splits" / sp.name, load(sp))

    print(f"{bench.name}: obfuscated {len(obf_dbs)} DBs, {len(obf_cards)} cards -> {out}")
    if sample:
        print(f"\nSAMPLE {sample[0]}:\n  CLEAN: {sample[1]}\n  OBF:   {sample[2]}")


if __name__ == "__main__":
    main()
