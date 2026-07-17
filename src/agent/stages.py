"""Standalone pipeline stages for the routing agent (`agent_router.py`).

Thin wrappers around the LOCKED deterministic pipeline so the agent reuses the EXACT verified
components — no new prompt, no new scoring. Everything below is imported from the existing modules:

  retrieve : card-embedding top-k            (agent_rerank.top_pool + clients.openrouter.embed)
  parse    : schema-blind query decomposition (decomp_card_eval.PARSE_PROMPT)   -> shared phrases
  map      : per-candidate phrase->field map  (decomp_card_eval.MAP_PROMPT_CAL) -> LLM extraction only
  score    : Coverage(e^-n·x) x Connectivity  (decomp_card_eval.score_fixed)    -> deterministic
  evidence : covered / N/A phrase->field list (agent_flow_eval.map_evidence)    -> tie-break context

Design + rationale: plans/260711-1221-agent-router-architecture + plan keen-forging-boole.md.
No benchmark/eval is run here; this module only exposes callable stages.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_RR = next(p for p in _P(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(_RR) not in _sys.path:
    _sys.path.insert(0, str(_RR))
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # multidb-routing/
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.retrieval import load  # noqa: E402
from src.core.rerank import top_pool  # noqa: E402
from src.core.index import card_text  # noqa: E402
from src.core.semantic_card import client, call_json  # noqa: E402
from src.core.scoring import PARSE_PROMPT, MAP_PROMPT_CAL, score_fixed  # noqa: E402
from src.workflow.final_routing import map_evidence, N as COV_N  # noqa: E402
from src.core.openrouter import embed  # noqa: E402


@dataclass
class SetData:
    """Everything a route dir provides, loaded once (databases / semantic / index)."""
    root: Path
    dbs: dict
    invs: dict
    cards: dict
    adjs: dict
    ids: list
    card_index: np.ndarray


def load_set(route_dir) -> SetData:
    root = Path(route_dir)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    card_index = np.load(root / "index" / "card.npy")
    return SetData(root, dbs, invs, cards, adjs, ids, card_index)


# ---------------------------------------------------------------- retrieval (card embedding top-k)

def retrieve_topk(question: str, data: SetData, k: int) -> list[str]:
    """Embed the question ONCE and return the fixed top-k candidate pool (no widening).

    NB: there is deliberately NO separate deterministic triage stage here. In the paper's fixed
    pipeline, S2 domain triage is a standalone LLM step that pre-filters the pool. In this agent
    the triage function is MOVED INTO the agent itself: it reads the top-k cards and selects which
    candidates to inspect, so the shortlist IS the domain triage. Re-adding a fixed pre-triage
    would duplicate that and reduce the agent's agency."""
    qv = np.asarray(embed([question]), dtype=np.float32)
    return top_pool(qv, data.card_index, data.ids, k)[0]


# ---------------------------------------------------------------- parse (schema-blind, shared)

def parse_query(oai, question: str) -> list[str]:
    """Decompose the question into phrases using PARSE_PROMPT (schema-blind, question-only).
    thinking=False per the locked extraction config. Returns ONE shared phrase set."""
    out = call_json(oai, PARSE_PROMPT.format(question=question), thinking=False)
    return [p for p in (out.get("phrases") or []) if isinstance(p, str)]


# ---------------------------------------------------------------- map + deterministic score

def map_phrases(oai, phrases: list[str], cand_id: str, data: SetData) -> dict:
    """LLM maps the shared phrases -> this candidate's actual schema fields (extraction only).
    Uses MAP_PROMPT_CAL verbatim; thinking=False."""
    out = call_json(oai, MAP_PROMPT_CAL.format(
        engine=data.invs[cand_id]["engine"],
        schema_block=card_text(data.cards[cand_id], data.invs[cand_id]),
        phrases=json.dumps(phrases)), thinking=False)
    return out


def score(phrases: list[str], mapping: dict, cand_id: str, data: SetData, n: float = COV_N) -> float:
    """Deterministic Coverage x Connectivity over the fixed phrase set (score_fixed)."""
    return score_fixed(phrases, mapping, data.adjs.get(cand_id, {}), data.invs[cand_id], n)


def evidence(phrases: list[str], mapping: dict) -> tuple[str, str]:
    """(covered, na) phrase->field strings — the tie-break context for why coverage scored as it did."""
    return map_evidence(phrases, mapping)


# ---------------------------------------------------------------- card views (brief vs full)

def card_brief(cand_id: str, data: SetData, max_chars: int = 320) -> str:
    """Lightweight metadata for the retrieve pool: engine + trimmed domain description."""
    card = data.cards.get(cand_id, {})
    desc = (card.get("domain_description") or "").strip()
    if len(desc) > max_chars:
        desc = desc[:max_chars].rsplit(" ", 1)[0] + "…"
    return f"{data.invs[cand_id]['engine']} | {desc}"


def card_full(cand_id: str, data: SetData) -> str:
    """Full card + structured schema (card_text) — the detail an inspect/tie-break needs."""
    return card_text(data.cards[cand_id], data.invs[cand_id])


def new_client():
    return client()
