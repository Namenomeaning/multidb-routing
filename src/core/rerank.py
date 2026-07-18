"""Deterministic re-ranking primitives (final-routing layer, NOT retrieval).

Pure functions shared by the routing pipeline and baselines:
  coverage / connectivity / score_candidate  Coverage(e^{-n·x}) × Connectivity(BFS) scoring
                                              (Sudarshan arXiv:2601.19825; LLM extracts, code scores)
  field2ent / resolve_nodes                   map extracted entity strings to adjacency node indices
  top_pool                                    cosine top-k retrieval pool

Coverage = exp(-n·x), x = unmatched/(matched+unmatched)        [n = thesis hyperparam]
Connectivity = 1 if matched entities form one connected component over the candidate's
               adjacency graph (declared+inferred edges) via BFS, else 0  [≤1 node ⇒ 1]
score = Coverage × Connectivity  (multiplicative; no retrieval term; engine-neutral)

Invariant (m3-design-decisions): the LLM only maps query phrases → schema entities per candidate;
the score is deterministic code over those mappings — no LLM self-confidence. EXTRACT_PROMPT is the
per-candidate mapping prompt (Sudarshan phrase_mapping, generalized to Entity.field).
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
from src.prompts import load as load_prompt  # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KMAX_HOLISTIC = 5  # top-5 rerank scope (Sudarshan)


# ---------------------------------------------------------------- prompts (extraction-only, engine-neutral)

# Adapted from Sudarshan et al. (arXiv:2601.19825) phrase_mapping_prompt.txt — verbatim
# instruction structure (column-level mapping, multiple mappings per phrase, the
# include/exclude lists, the identifier-vs-counting special rule, N/A sparingly).
# Generalized from SQL TableName.ColumnName to Entity.field so it applies to Mongo
# collections + Neo4j node labels/properties. JSON output (robust parse) instead of plain
# 'phrase - Table.Column' lines. Verbatim source:
# refs/sudarshan/DB-Routing-prompts/phrase_mapping_prompt.txt
#
# SECTION ORDER (KV-cache optimization): instructions + output spec are STATIC (identical
# every call) and the SCHEMA block is identical for every query hitting the same DB, so both
# form a long cacheable prefix; the variable QUERY is appended LAST. DeepSeek context caching
# (full-prefix match) then charges only the short query tail on repeat calls over the same DB
# — prompt_cache_hit_tokens covers instructions+schema. Content faithful to Sudarshan; only
# the placement of the query (moved to the end) differs from their layout.
EXTRACT_PROMPT = load_prompt("extract")


# ---------------------------------------------------------------- deterministic scoring

def coverage(matched: int, unmatched: int, n: float) -> float:
    total = matched + unmatched
    if total == 0:
        return 0.0
    return math.exp(-n * (unmatched / total))


def field2ent(inv: dict) -> dict[str, int]:
    """field-name (lower) -> owning entity index; entity order == adjacency entity_index order."""
    m: dict[str, int] = {}
    for i, e in enumerate(inv["entities"]):
        for f in e["fields"]:
            m.setdefault(f["name"].lower(), i)
    return m


def resolve_nodes(tokens: list[str], names: list[str], f2e: dict[str, int]) -> set[int]:
    """Map LLM 'entity' strings (bare entity, 'Entity.field', or bare field) to entity indices."""
    pos = {nm.lower(): i for i, nm in enumerate(names)}
    nodes: set[int] = set()
    for t in tokens:
        low = t.strip().lower()
        if not low:
            continue
        if low in pos:
            nodes.add(pos[low]); continue
        if "." in low:                                   # 'Entity.field' -> Entity
            pre, suf = low.split(".", 1)
            if pre in pos:
                nodes.add(pos[pre]); continue
            if suf in f2e:
                nodes.add(f2e[suf]); continue
        if low in f2e:                                   # bare field -> owning entity
            nodes.add(f2e[low])
    return nodes


def connectivity(matched_entities: list[str], adj: dict, f2e: dict[str, int]) -> int:
    """1 if resolved matched entities form a single connected component over the adjacency graph."""
    names = adj.get("entity_index", [])
    nodes = resolve_nodes(matched_entities, names, f2e)
    if len(nodes) <= 1:
        return 1
    g = defaultdict(set)
    for e in adj.get("edges", []):
        g[e["from"]].add(e["to"]); g[e["to"]].add(e["from"])
    seen, dq = set(), deque([next(iter(nodes))])
    while dq:
        x = dq.popleft()
        if x in seen:
            continue
        seen.add(x)
        dq.extend(g[x] - seen)
    return 1 if nodes <= seen else 0


def score_candidate(extr: dict, adj: dict, inv: dict, n: float) -> float:
    """Sudarshan-faithful: phrase->multiple Entity.field mappings (column-level).
    Coverage over phrases (matched = has >=1 target, N/A = empty targets); Connectivity
    BFS over ALL mapped target entities. Multiplicative."""
    maps = extr.get("mappings", []) or []
    matched = [m for m in maps if isinstance(m, dict) and (m.get("targets") or [])]
    na = [m for m in maps if isinstance(m, dict) and not (m.get("targets") or [])]
    cov = coverage(len(matched), len(na), n)
    targets: list[str] = []
    for m in matched:
        targets.extend(t for t in m["targets"] if isinstance(t, str))
    conn = connectivity(targets, adj, field2ent(inv))
    return cov * conn


# ---------------------------------------------------------------- pool

def top_pool(qv: np.ndarray, index: np.ndarray, ids: list[str], k: int) -> list[list[str]]:
    sims = qv @ index.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return [[ids[j] for j in row] for row in order]
