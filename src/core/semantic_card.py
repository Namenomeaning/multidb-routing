"""build semantic layer (A inventory -> B card + C adjacency) per BENCHMARK.md spec.

Self-contained — no imports from old experiment code. Data source = benchmark jsonl only.
Spec: BENCHMARK.md §2-§4 (locked 2026-06-10).
  A = code parse (per-engine parsers -> unified inventory)
  B = LLM call 1: card {domain_description, term_glossary}; input = A + <=5 support NL questions
  C = LLM call 2: implicit_edges {from,to,via,kind,reason}; input = A only (fill-in, closed index)
All LLM via OpenRouter (CLAUDE.md §11). Temp 0, JSON-only, sequential + retry-backoff.

Usage:
  python build_semantic.py --limit-per-engine 4,3,3        # 10-DB trial
  python build_semantic.py                                  # full 208
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
from src.prompts import load as load_prompt  # noqa: E402
HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent.parent / ".env")  # repo-root shared key (OPENROUTER_API_KEY)

BENCH = HERE.parent.parent / "data" / "_build" / "standard3_scale_v1"
DATABASES = BENCH / "databases.jsonl"
SUPPORT = BENCH / "splits" / "support-balanced.jsonl"
# NOTE: DATABASES/SUPPORT/outdir overridable via CLI (--databases/--questions/--outdir)
# so the same builder serves the build source and data/{multidb,spider,bird}.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_CHAT_MODEL = "deepseek/deepseek-v4-flash"          # OpenRouter id
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"               # DeepSeek-direct id (same weights)
MAX_SUPPORT_QUESTIONS = 5
SLEEP_BETWEEN_CALLS = 3.0  # RPM-gentle


def _provider() -> str:
    # LLM_PROVIDER=deepseek -> DeepSeek-direct (faster, same model); else OpenRouter (default).
    return os.environ.get("LLM_PROVIDER", "openrouter").lower()


def chat_model() -> str:
    if _provider() == "deepseek":
        return os.environ.get("DEEPSEEK_CHAT_MODEL") or DEFAULT_DEEPSEEK_MODEL
    return os.environ.get("OPENROUTER_CHAT_MODEL") or DEFAULT_CHAT_MODEL


def client() -> OpenAI:
    if _provider() == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY not set")
        return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


# ---------------------------------------------------------------- B0: parsers

def _split_top_level(body: str) -> list[str]:
    """Split DDL body on commas at paren depth 0 (handles 1-line BIRD DDL)."""
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def parse_postgresql(schema_text: str) -> tuple[list[dict], list[dict]]:
    entities, relations = [], []
    for m in re.finditer(
            r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[`\"']?([\w ]+?)[`\"']?\s*\((.*?)\)\s*(?:;|$)",
            schema_text, re.S | re.I):
        table, body = m.group(1).strip(), m.group(2)
        fields, pk = [], []
        for raw in _split_top_level(body):
            line = raw.strip().rstrip(",")
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("PRIMARY KEY"):
                pk = re.findall(r"[`\"](\w+)[`\"]", line) or re.findall(r"(\w+)", line[len("PRIMARY KEY"):])
                continue
            if upper.startswith("FOREIGN KEY"):
                fk = re.match(
                    r"FOREIGN KEY\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)\s*REFERENCES\s*[`\"]?(\w+)[`\"]?\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)",
                    line, re.I)
                if fk:
                    relations.append({"from_entity": table, "via": fk.group(1),
                                      "to_entity": fk.group(2), "to_field": fk.group(3), "kind": "fk"})
                continue
            if upper.startswith(("UNIQUE", "CONSTRAINT", "KEY ", "INDEX")):
                continue
            cm = re.match(r"[`\"]?(\w+)[`\"]?\s+(.+)", line)
            if cm:
                f = {"name": cm.group(1), "type": cm.group(2).split()[0].lower()}
                if "PRIMARY KEY" in upper:
                    f["pk"] = True
                fields.append(f)
        for f in fields:
            if f["name"] in pk:
                f["pk"] = True
        entities.append({"name": table, "fields": fields})
    return entities, relations


def _mongo_field(token: str) -> dict:
    """'address.location.coordinates[]<double>' -> name='address.location.coordinates', type='double'."""
    token = token.strip()
    tm = re.match(r"(.+?)<(\w+)>$", token)
    name, ftype = (tm.group(1), tm.group(2)) if tm else (token, "")
    return {"name": name.replace("[]", "").strip(), "type": ftype}


def parse_mongodb(schema_text: str) -> tuple[list[dict], list[dict]]:
    entities = []
    in_collections = False
    for line in schema_text.splitlines():
        if line.strip().startswith("Collections:"):
            in_collections = True
            continue
        if in_collections and line.startswith("  ") and ":" in line:
            name, fields = line.strip().split(":", 1)
            name = name.strip()
            if name.lower() == "indexes":
                continue  # system index metadata leaked into schema dump (15/87 DBs) — not a data collection
            entities.append({"name": name,
                             "fields": [_mongo_field(f) for f in fields.split(",") if f.strip()]})
    return entities, []  # no declared references in this schema format


def parse_neo4j(schema_text: str) -> tuple[list[dict], list[dict]]:
    if "Node properties:" in schema_text:
        return _parse_neo4j_t2c24(schema_text)
    entities, relations = [], []
    section = None
    for line in schema_text.splitlines():
        s = line.strip()
        if s.startswith("Node labels:"):
            section = "nodes"
            continue
        if s.startswith("Relationships:"):
            section = "rels"
            continue
        if section == "nodes" and line.startswith("  ") and ":" in s:
            name, fields = s.split(":", 1)
            entities.append({"name": name.strip(),
                             "fields": [{"name": f.strip(), "type": ""} for f in fields.split(",") if f.strip()]})
        elif section == "rels" and line.startswith("  ") and ":" in s:
            rel, ends = s.split(":", 1)
            em = re.match(r"\s*(\w+)\s*->\s*(\w+)", ends)
            if em:
                relations.append({"from_entity": em.group(1), "via": rel.strip(),
                                  "to_entity": em.group(2), "kind": "relationship"})
    return entities, relations


def _parse_neo4j_t2c24(schema_text: str) -> tuple[list[dict], list[dict]]:
    """text2cypher-2024 markdown format: Node properties / Relationship properties /
    'The relationships:' with (:A)-[:REL]->(:B) patterns."""
    entities, relations = [], []
    section, current = None, None
    for line in schema_text.splitlines():
        s = line.strip()
        if s.startswith("Node properties"):
            section = "nodes"
            continue
        if s.startswith("Relationship properties"):
            section = "rel_props"
            continue
        if s.startswith("The relationships"):
            section = "rels"
            continue
        if section == "nodes":
            bm = re.match(r"(\w+)\s*\{(.*)\}\s*$", s)  # 'Label {prop: TYPE, ...}' variant
            if bm:
                fields = []
                for tok in bm.group(2).split(","):
                    fm = re.match(r"\s*([\w:]+)\s*:\s*(\w+)", tok)
                    if fm:
                        fields.append({"name": fm.group(1), "type": fm.group(2).lower()})
                entities.append({"name": bm.group(1), "fields": fields})
                continue
            lm = re.match(r"-\s*\*\*(\w+)\*\*", s)
            if lm:
                current = {"name": lm.group(1), "fields": []}
                entities.append(current)
                continue
            pm = re.match(r"-\s*`([\w.]+)`?\s*:?\s*`?:?\s*(\w+)?", s)
            if pm and current is not None and s.startswith("-"):
                prop = re.match(r"-\s*`([\w.]+)[`:]\s*:?\s*(\w+)?", s)
                if prop:
                    current["fields"].append({"name": prop.group(1), "type": (prop.group(2) or "").lower()})
        elif section == "rels":
            rm = re.match(r"\(:(\w+)\)-\[:(\w+)\]->\(:(\w+)\)", s)
            if rm:
                relations.append({"from_entity": rm.group(1), "via": rm.group(2),
                                  "to_entity": rm.group(3), "kind": "relationship"})
    return entities, relations


PARSERS = {"postgresql": parse_postgresql, "mongodb": parse_mongodb, "neo4j": parse_neo4j}


def build_inventory(row: dict) -> dict:
    entities, relations = PARSERS[row["engine"]](row["schema_text"])
    return {"instance_id": row["instance_id"], "engine": row["engine"],
            "entities": entities, "declared_relations": relations}


# ---------------------------------------------------------------- A renderers

def render_entities(inv: dict) -> str:
    lines = []
    for e in inv["entities"]:
        parts = [f["name"] + (f" {f['type']}" if f["type"] else "") + (" PK" if f.get("pk") else "")
                 for f in e["fields"]]
        lines.append(f"{e['name']}({', '.join(parts)})")
    return "\n".join(lines)


def render_relations(inv: dict) -> str:
    if not inv["declared_relations"]:
        return "(none declared)"
    out = []
    for r in inv["declared_relations"]:
        if r["kind"] == "fk":
            out.append(f"{r['from_entity']}.{r['via']} -> {r['to_entity']}.{r.get('to_field','')} (FK)")
        else:
            out.append(f"{r['from_entity']} -[{r['via']}]-> {r['to_entity']} (relationship)")
    return "\n".join(out)


def entity_index(inv: dict) -> list[str]:
    return [e["name"] for e in inv["entities"]]


def render_indexed_entities(inv: dict) -> str:
    return "\n".join(f"[{i}] {n}" for i, n in enumerate(entity_index(inv)))


def render_confirmed_edges(inv: dict) -> str:
    idx = {n: i for i, n in enumerate(entity_index(inv))}
    out = []
    for r in inv["declared_relations"]:
        a, b = idx.get(r["from_entity"]), idx.get(r["to_entity"])
        if a is None or b is None:
            continue
        out.append(f"[{a}]->[{b}] via {r['via']} ({r['kind']})")
    return "\n".join(out) if out else "(none)"


# ---------------------------------------------------------------- prompts (spec BENCHMARK.md §3, §4)

CARD_PROMPT = load_prompt("card")

QUESTION_SECTION = """
## Sample questions users have asked this database
{questions}
"""

# Adapted from Sudarshan et al. (arXiv:2601.19825) adjacency_list_prompt.txt — the
# implicit-edge logic ONLY. Their Priority 1 (explicit FK) is already done by our code parser
# (the confirmed edges in the SCHEMA section); this prompt reproduces their Priority 2 (naming-
# convention + data-type implied FK) and Priority 3 (simple name/type match, USE CAUTIOUSLY)
# plus reciprocity, mapped to our fill-in output (BENCHMARK §4). Generalized from SQL tables to
# entities/collections/node-labels for Mongo/Neo4j. Verbatim source:
# refs/sudarshan/DB-Routing-prompts/adjacency_list_prompt.txt
#
# SECTION ORDER (KV-cache optimization): the entire instruction block below is STATIC and
# byte-identical for every DB → DeepSeek auto prefix-cache reuses it across all adjacency
# calls; only the variable SCHEMA section is appended LAST so it never breaks the cached
# prefix (DeepSeek context caching: full-prefix match, prompt_cache_hit_tokens). Content is
# faithful to Sudarshan; only the placement of the schema (moved to the end) differs.
ADJ_PROMPT = load_prompt("adjacency")


# ---------------------------------------------------------------- LLM call

# KV-cache meter: accumulates DeepSeek prompt_cache_hit/miss tokens across calls so any script
# can verify the prefix-cache is actually firing (print build_semantic.CACHE_STATS).
CACHE_STATS = {"hit": 0, "miss": 0, "calls": 0, "fail": 0}


def call_json(oai: OpenAI, prompt: str, *, retries: int = 4, thinking: bool = False) -> dict:
    """DeepSeek thinking is ON by default upstream and ~7x slower with NO recall gain on the
    extraction/triage steps (probe 2026-06-14: GT-kept identical off vs on). So thinking defaults
    OFF here for the whole pipeline (card build, triage, parse, map). Pass thinking=True ONLY for
    the final agent tie-break, where reasoning over close candidates may help.

    Anti-hang contract: after exhausting retries (stalled/empty/malformed upstream — measured: the
    provider occasionally locks into an empty-output loop on a specific prompt at temp 0), this
    returns {} instead of raising, so ONE bad query degrades to a miss (callers use `.get(...)`
    fallbacks) rather than crashing the run OR blocking the pool in shutdown-wait. Failures are
    counted in CACHE_STATS['fail'] and logged loudly, so a systematic outage stays visible."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = oai.chat.completions.create(
                # attempt 0 = temp 0 (deterministic, preserves cached results); retries nudge temp up
                # so a query that DETERMINISTICALLY emits malformed/empty JSON at temp 0 can escape it.
                model=chat_model(), temperature=(0.0 if attempt == 0 else 0.5),
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
                timeout=60,  # per-request cap: a stalled call raises fast → retry, never ties up a worker
            )
            _record_cache(resp)
            text = resp.choices[0].message.content or ""
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001 - transient upstream / parse; back off
            last = exc
            wait = min(8.0 * (2 ** attempt), 20.0)  # capped: don't add minutes of sleep on a stuck call
            print(f"  [retry {attempt+1}/{retries}] {type(exc).__name__}: {str(exc)[:120]} -> sleep {wait:.0f}s", flush=True)
            time.sleep(wait)
    CACHE_STATS["fail"] += 1
    print(f"  [GAVE UP after {retries+1} attempts -> return {{}} (miss); "
          f"total fails={CACHE_STATS['fail']}] last={type(last).__name__}: {str(last)[:100]}", flush=True)
    return {}


def _record_cache(resp) -> None:
    """Pull DeepSeek cache fields from usage (hit = prompt_cache_hit_tokens; OpenRouter exposes
    the same via prompt_tokens_details.cached_tokens). Best-effort; ignores absence."""
    try:
        u = resp.usage
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        if hit is None:
            det = getattr(u, "prompt_tokens_details", None)
            hit = getattr(det, "cached_tokens", 0) if det else 0
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if miss is None:
            miss = max(0, (getattr(u, "prompt_tokens", 0) or 0) - (hit or 0))
        CACHE_STATS["hit"] += int(hit or 0)
        CACHE_STATS["miss"] += int(miss or 0)
        CACHE_STATS["calls"] += 1
    except Exception:  # noqa: BLE001 - metering must never break the call
        pass


# ---------------------------------------------------------------- C merge + validation

INFERRED_KINDS = {"ref_field", "name_type_match"}


def declared_graph_connected(inv: dict) -> bool:
    """True if declared edges already connect ALL entities (single component).
    Then implicit edges provably cannot change any BFS-connectivity outcome -> skip LLM."""
    names = entity_index(inv)
    if len(names) <= 1:
        return True
    idx = {n: i for i, n in enumerate(names)}
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in inv["declared_relations"]:
        a, b = idx.get(r["from_entity"]), idx.get(r["to_entity"])
        if a is not None and b is not None:
            parent[find(a)] = find(b)
    return len({find(i) for i in range(len(names))}) == 1


def merge_adjacency(inv: dict, llm_out: dict) -> dict:
    names = entity_index(inv)
    idx = {n: i for i, n in enumerate(names)}
    fields_of = [ {f["name"].lower() for f in e["fields"]} for e in inv["entities"] ]
    edges, dropped = [], []

    for r in inv["declared_relations"]:
        a, b = idx.get(r["from_entity"]), idx.get(r["to_entity"])
        if a is None or b is None:
            continue
        edges.append({"from": a, "to": b, "via": r["via"], "kind": r["kind"],
                      "reason": "declared in schema"})

    confirmed_pairs = {(e["from"], e["to"]) for e in edges} | {(e["to"], e["from"]) for e in edges}
    for e in llm_out.get("implicit_edges", []):
        try:
            a, b = int(e["from"]), int(e["to"])
        except (KeyError, TypeError, ValueError):
            dropped.append({"edge": e, "why": "bad indices"})
            continue
        if not (0 <= a < len(names) and 0 <= b < len(names)) or a == b:
            dropped.append({"edge": e, "why": "index out of range"})
            continue
        if e.get("kind") not in INFERRED_KINDS:
            dropped.append({"edge": e, "why": f"bad kind {e.get('kind')}"})
            continue
        if str(e.get("via", "")).lower() not in (fields_of[a] | fields_of[b]):
            dropped.append({"edge": e, "why": f"via not a field of either endpoint: {e.get('via')}"})
            continue
        if (a, b) in confirmed_pairs:
            dropped.append({"edge": e, "why": "repeats confirmed edge"})
            continue
        edges.append({"from": a, "to": b, "via": e["via"], "kind": e["kind"],
                      "reason": str(e.get("reason", ""))[:200]})

    n = len(names)
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    density = len({(min(e['from'], e['to']), max(e['from'], e['to'])) for e in edges}) / max_edges
    return {"instance_id": inv["instance_id"], "engine": inv["engine"],
            "entity_index": names, "edges": edges,
            "n_declared": sum(1 for e in edges if e["kind"] not in INFERRED_KINDS),
            "n_inferred": sum(1 for e in edges if e["kind"] in INFERRED_KINDS),
            "n_dropped": len(dropped), "dropped": dropped,
            "density": round(density, 3), "density_flag": density > 0.8 and n > 4}


def validate_card(inv: dict, card: dict) -> dict:
    known = {e["name"].lower() for e in inv["entities"]} | \
            {f["name"].lower() for e in inv["entities"] for f in e["fields"]}
    glossary = card.get("term_glossary") or {}
    bad = [k for k in glossary if k.lower() not in known]
    return {"instance_id": inv["instance_id"], "engine": inv["engine"],
            "domain_description": str(card.get("domain_description", "")),
            "term_glossary": {k: v for k, v in glossary.items() if k.lower() in known},
            "glossary_dropped_unknown_names": bad,
            "desc_words": len(str(card.get("domain_description", "")).split())}


# ---------------------------------------------------------------- main

def load_support_questions(path: Path = SUPPORT) -> dict[str, list[str]]:
    by_db: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_db.setdefault(r["instance_id"], []).append(r["question"])
    return {k: v[:MAX_SUPPORT_QUESTIONS] for k, v in by_db.items()}


def select_dbs(rows: list[dict], limits: dict[str, int], support: dict) -> list[dict]:
    """Deterministic trial pick: per engine, first by instance_id; for PG force-include
    one cold-start DB (0 support queries) to exercise the schema-only fallback."""
    out = []
    for engine, limit in limits.items():
        pool = sorted((r for r in rows if r["engine"] == engine), key=lambda r: r["instance_id"])
        picked = pool[:limit]
        if engine == "postgresql":
            cold = next((r for r in pool if not support.get(r["instance_id"])), None)
            if cold and cold["instance_id"] not in {p["instance_id"] for p in picked}:
                picked = picked[:limit - 1] + [cold]
        out.extend(picked)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-engine", default="",
                    help="pg,mongo,neo4j counts e.g. 4,3,3 ; empty = all DBs")
    ap.add_argument("--databases", default=str(DATABASES), help="databases.jsonl path")
    ap.add_argument("--questions", default=str(SUPPORT),
                    help="question source for query-informed cards (support/train split)")
    ap.add_argument("--outdir", default=str(HERE), help="dir for inventory/cards/adjacency jsonl")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    args = ap.parse_args()

    with open(args.databases) as f:
        rows = [json.loads(line) for line in f]
    support = load_support_questions(Path(args.questions))

    if args.limit_per_engine:
        pg, mongo, neo = (int(x) for x in args.limit_per_engine.split(","))
        rows = select_dbs(rows, {"postgresql": pg, "mongodb": mongo, "neo4j": neo}, support)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inv_path, card_path, adj_path = outdir / "inventory.jsonl", outdir / "cards.jsonl", outdir / "adjacency.jsonl"
    done_cards = {json.loads(l)["instance_id"] for l in open(card_path)} if card_path.exists() else set()
    done_adj = {json.loads(l)["instance_id"] for l in open(adj_path)} if adj_path.exists() else set()

    oai = client()
    print(f"model={chat_model()}  dbs={len(rows)}  resume: cards={len(done_cards)} adj={len(done_adj)}")

    with open(inv_path, "w") as f_inv:
        inventories = []
        for row in rows:
            inv = build_inventory(row)
            inventories.append(inv)
            f_inv.write(json.dumps(inv) + "\n")

    f_card = open(card_path, "a")
    f_adj = open(adj_path, "a")
    try:
        for inv in inventories:
            iid = inv["instance_id"]
            if not inv["entities"]:
                print(f"\n!!! {iid}: parse produced 0 entities — SKIP LLM (no card from empty input)", flush=True)
                continue
            qs = support.get(iid, [])
            print(f"\n=== {iid} ({inv['engine']}) entities={len(inv['entities'])} "
                  f"declared={len(inv['declared_relations'])} support_q={len(qs)}", flush=True)

            if iid not in done_cards:
                qsec = QUESTION_SECTION.format(questions="\n".join(qs)) if qs else "\n"
                card_raw = call_json(oai, CARD_PROMPT.format(
                    engine=inv["engine"], entity_block=render_entities(inv),
                    relations_block=render_relations(inv), question_section=qsec))
                card = validate_card(inv, card_raw)
                card["query_informed"] = bool(qs)
                f_card.write(json.dumps(card) + "\n")
                f_card.flush()
                print(f"  card: {card['desc_words']}w gloss={len(card['term_glossary'])} "
                      f"dropped_gloss={card['glossary_dropped_unknown_names']}", flush=True)
                time.sleep(args.sleep)

            if iid not in done_adj:
                if declared_graph_connected(inv):
                    adj_raw = {"implicit_edges": []}
                    adj = merge_adjacency(inv, adj_raw)
                    adj["llm_skipped"] = "declared graph already connected — implicit edges cannot change BFS"
                    f_adj.write(json.dumps(adj) + "\n")
                    f_adj.flush()
                    print(f"  adj: SKIP LLM (declared connected) declared={adj['n_declared']}", flush=True)
                    continue
                adj_raw = call_json(oai, ADJ_PROMPT.format(
                    engine=inv["engine"], indexed_entity_block=render_indexed_entities(inv),
                    confirmed_edges_block=render_confirmed_edges(inv),
                    surface_block=render_entities(inv)))
                adj = merge_adjacency(inv, adj_raw)
                f_adj.write(json.dumps(adj) + "\n")
                f_adj.flush()
                print(f"  adj: declared={adj['n_declared']} inferred={adj['n_inferred']} "
                      f"dropped={adj['n_dropped']} density={adj['density']}"
                      f"{' DENSITY_FLAG' if adj['density_flag'] else ''}", flush=True)
                time.sleep(args.sleep)
    finally:
        f_card.close()
        f_adj.close()
    print("\ndone.")


if __name__ == "__main__":
    main()
