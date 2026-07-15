"""Agent-driven database router (LangGraph) wrapping the locked deterministic pipeline.

ARCHITECTURE BOUNDARY (read first — the paper and this file are two SEPARATE stages of the work):
  * DONE (paper.tex, "Proposed Method") = a FIXED pipeline: semantic search → deterministic domain
    triage → parse once → map+score EVERY candidate → deterministic rank → tie-break. No loop; the
    LLM never decides when to stop. The paper's reported numbers come from THAT pipeline
    (agent_flow_eval.py / decomp_card_eval.py), not this file.
  * NEXT WORK (this file) = the agent contribution: show the routing task can be driven by an LLM
    AGENT and evaluated on the same benchmark. It reuses the verified primitives (parse / map /
    deterministic Coverage x Connectivity score) but MOVES the control into the agent — including the
    domain triage, which is no longer a separate deterministic step: the agent reads the pool cards,
    drops out-of-domain candidates, and selects which to inspect (the shortlist IS the triage). The
    agent also decides when it has enough evidence to answer. Divergence from the fixed pipeline is
    the POINT of this stage, not a fidelity defect.

The verified pipeline components — card retrieval, one shared schema-blind parse, per-candidate LLM
phrase->field mapping, deterministic Coverage x Connectivity scoring — are exposed as tools an LLM
agent drives. The agent's genuine decisions are (a) domain triage + selective schema reading (which
candidates to inspect) and (b) tie-break adjudication over full evidence when structural scores
saturate. Pool is fixed (retrieve once, no expansion). Running it on the benchmark requires an
explicit user "chạy"/"run" (project hard gate).

Invariants (graph-enforced, not left to the LLM):
  - phrases parsed ONCE (schema-blind), frozen, shared across all inspects;
  - scoring stays deterministic (agent reads scores / selects; never emits a confidence number);
  - answer id must be in the retrieved pool; max_turns caps the loop; a fallback always answers.

No API call happens at import; run the CLI (needs OPENROUTER_API_KEY) to execute a live demo.
See plan keen-forging-boole.md + plans/260711-1221-agent-router-architecture/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_stages as ps  # noqa: E402
from src.clients.openrouter import DEFAULT_OPENROUTER_CHAT_MODEL  # noqa: E402
from retrieval_eval import load  # noqa: E402
from build_semantic import chat_model  # noqa: E402


TIE_MARGIN = 0.2  # near-tie band (locked pipeline default): scores within this of the top → tie-break

_SYSTEM_PROMPT = """You route ONE natural-language question to the single best database instance.

You are given a FIXED set of query phrases (already extracted from the question alone — do NOT
re-derive or change them). Every candidate is judged against this SAME phrase set.

Tools (call them yourself, in this order):
1. retrieve — load the fixed candidate pool (top-k by semantic similarity) with short domain cards.
   Call it once. There is no pool expansion.
2. inspect — you first TRIAGE the pool by domain: from the cards, drop the candidates clearly outside
   the question's subject domain and keep the rest, then pick a SHORT LIST of the most promising to
   read in full. inspect reads each one's full schema, maps the shared phrases onto its actual
   fields, and returns a deterministic structural score (Coverage x Connectivity) plus the
   phrase->field evidence (which phrases were COVERED and onto which fields, which were N/A). Inspect
   the few that the cards suggest actually match; you do not have to inspect all of them.
3. answer — commit ONE instance_id from the pool, then stop.

Deciding:
- The structural score is your measuring stick. If one inspected candidate clearly has the highest
  score, answer it.
- If several inspected candidates are (near-)tied on score — they all cover the phrases — the score
  can no longer separate them. Break the tie using the evidence, applying these criteria IN ORDER:
    (1) DOMAIN: is the database actually about what the question asks? Reject a merely adjacent domain.
    (2) ENTITIES: does it have the specific entities/attributes the question needs? A phrase forced
        onto a generic/thematic field is WEAKER than a genuine field for it.
    (3) RELATIONSHIPS: can the matched entities be joined/traversed to answer the question?
- Never output a confidence value or a numeric score of your own. Use only what inspect returns and
  what each card states. Judge engines neutrally (an inferred link, a foreign key, and a graph edge
  are equally valid). If evidence is thin, inspect another pool candidate before answering.
"""


class RouterState(TypedDict, total=False):
    question: str
    phrases: list[str]
    pool_ids: list[str]
    inspected: dict            # id -> {score, covered, na}
    messages: list[dict]
    last_tool_calls: list[dict]
    agent_calls: int
    final_id: str
    log: list[str]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Load the fixed candidate pool (top-k by semantic similarity) with short "
                           "domain cards. Call once; there is no expansion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why retrieval is needed (first step)."},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "Read the FULL schema of the given pool candidates, map the shared phrases "
                           "onto their fields, and return a deterministic Coverage x Connectivity "
                           "score plus phrase->field evidence for each. Phrases are fixed; you only "
                           "choose which ids to inspect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why these candidates were shortlisted."},
                    "ids": {"type": "array", "items": {"type": "string"},
                            "description": "instance_ids from the retrieved pool to inspect fully."},
                },
                "required": ["reason", "ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Commit the final database choice (one instance_id from the pool) and stop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this candidate best answers the "
                               "question, citing fields/relationships or the tie-break criteria."},
                    "instance_id": {"type": "string"},
                },
                "required": ["reason", "instance_id"],
            },
        },
    },
]


def _assistant_message_dict(msg: Any, tool_calls: list[Any]) -> dict:
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ] if tool_calls else None,
    }


def _serialized_tool_calls(tool_calls: list[Any]) -> list[dict]:
    return [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in tool_calls]


class AgentRouter:
    """LangGraph tool-calling router over the locked pipeline stages."""

    def __init__(self, route_dir, *, model: str | None = None, max_turns: int = 6,
                 pool_k: int = 10):
        self.data = ps.load_set(route_dir)
        self.oai = ps.new_client()
        self.model = model or chat_model() or DEFAULT_OPENROUTER_CHAT_MODEL
        self.max_turns = max_turns
        self.pool_k = pool_k
        self.graph = self._build_graph()

    # -- nodes -----------------------------------------------------------------
    def _chat_tools(self, messages: list[dict], retries: int = 3):
        """Provider-aware tool-calling turn on self.oai (OpenRouter OR DeepSeek-native, both
        OpenAI-compatible). thinking OFF on every agent step (per config; parse/map already off)."""
        last = None
        for attempt in range(retries):
            try:
                resp = self.oai.chat.completions.create(
                    model=self.model, messages=messages, tools=TOOL_SCHEMAS,
                    tool_choice="auto", temperature=0.0, max_tokens=700,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if not getattr(resp, "choices", None):
                    raise RuntimeError("empty choices")
                return resp.choices[0].message
            except Exception as exc:  # noqa: BLE001 - transient upstream; back off
                last = exc
                time.sleep(4.0 * (attempt + 1))
        raise last

    def _agent_node(self, state: RouterState) -> dict:
        msg = self._chat_tools(state["messages"])
        tool_calls = getattr(msg, "tool_calls", None) or []
        return {
            "messages": state["messages"] + [_assistant_message_dict(msg, tool_calls)],
            "last_tool_calls": _serialized_tool_calls(tool_calls),
            "agent_calls": state.get("agent_calls", 0) + 1,
        }

    def _tools_node(self, state: RouterState) -> dict:
        messages = list(state["messages"])
        pool_ids = list(state.get("pool_ids", []))
        inspected = dict(state.get("inspected", {}))
        phrases = state.get("phrases", [])
        final_id = state.get("final_id", "")
        log = list(state.get("log", []))

        for tc in state.get("last_tool_calls", []):
            name = tc["name"]
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "retrieve":
                if pool_ids:
                    content = "pool already loaded (retrieve is called once)."
                else:
                    pool_ids = ps.retrieve_topk(state["question"], self.data, self.pool_k)
                    log.append(f"retrieve → pool[{len(pool_ids)}]")
                    lines = [f"[{i+1}] {cid} | {ps.card_brief(cid, self.data)}"
                             for i, cid in enumerate(pool_ids)]
                    content = ("Candidate pool (fixed). Read the cards, drop those clearly outside the "
                               "question's domain, and shortlist the rest to inspect.\n" + "\n".join(lines))

            elif name == "inspect":
                ids = [i for i in (args.get("ids") or []) if i in pool_ids]
                if not ids:
                    content = "no valid pool ids to inspect (ids must come from the retrieved pool)."
                else:
                    for cid in ids:
                        if cid not in inspected:
                            mp = ps.map_phrases(self.oai, phrases, cid, self.data)
                            sc = ps.score(phrases, mp, cid, self.data)
                            cov, na = ps.evidence(phrases, mp)
                            inspected[cid] = {"score": sc, "covered": cov, "na": na}
                    log.append(f"inspect → {', '.join(ids)}")
                    content = self._inspect_report(ids, inspected)

            elif name == "answer":
                iid = args.get("instance_id", "")
                if iid and iid in pool_ids:
                    final_id = iid
                    log.append(f"answer → {iid}")
                    content = f"committed: {iid}"
                else:
                    content = (f"invalid instance_id '{iid}'. Must be one of the retrieved pool ids.")
            else:
                content = f"unknown tool: {name}"

            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

        return {"messages": messages, "pool_ids": pool_ids, "inspected": inspected,
                "final_id": final_id, "log": log, "last_tool_calls": []}

    def _inspect_report(self, ids: list[str], inspected: dict) -> str:
        """Evidence package for the inspected ids: score + covered/na + full card/schema, with a
        near-tie flag so the agent knows when the score can no longer separate candidates."""
        top = max((inspected[c]["score"] for c in ids), default=0.0)
        blocks = []
        for i, cid in enumerate(ids, 1):
            ev = inspected[cid]
            tied = top > 0 and ev["score"] > 0 and (top - ev["score"]) <= TIE_MARGIN
            flag = "  [NEAR-TIE with top — use tie-break criteria]" if tied and ev["score"] < top else \
                   ("  [current top]" if ev["score"] == top and top > 0 else "")
            blocks.append(
                f"[{i}] {cid}  score={ev['score']:.3f}{flag}\n"
                f"COVERED: {ev['covered']}\n"
                f"N/A: {ev['na']}\n"
                f"{ps.card_full(cid, self.data)}")
        return "Inspection evidence (deterministic score = measuring stick):\n\n" + "\n\n".join(blocks)

    # -- edges -----------------------------------------------------------------
    def _after_agent(self, state: RouterState) -> str:
        return "tools" if state.get("last_tool_calls") else "end"

    def _after_tools(self, state: RouterState) -> str:
        if state.get("final_id"):
            return "end"
        if state.get("agent_calls", 0) >= self.max_turns:
            return "end"
        return "agent"

    def _build_graph(self):
        graph = StateGraph(RouterState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._after_agent, {"tools": "tools", "end": END})
        graph.add_conditional_edges("tools", self._after_tools, {"agent": "agent", "end": END})
        return graph.compile()

    # -- public ----------------------------------------------------------------
    def route(self, question: str) -> dict:
        phrases = ps.parse_query(self.oai, question)  # ONE shared, schema-blind phrase set (frozen)
        user_msg = (f"Question: {question}\n"
                    f"Fixed query phrases (use exactly these for every candidate): {phrases}")
        initial: RouterState = {
            "question": question, "phrases": phrases,
            "pool_ids": [], "inspected": {},
            "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                         {"role": "user", "content": user_msg}],
            "last_tool_calls": [], "agent_calls": 0, "final_id": "", "log": [],
        }
        result = self.graph.invoke(initial)
        final_id = result.get("final_id", "")
        inspected = result.get("inspected", {})
        pool_ids = result.get("pool_ids", [])
        if not final_id:  # fallback: best inspected, else top of pool
            if inspected:
                final_id = max(inspected.items(), key=lambda kv: kv[1]["score"])[0]
            elif pool_ids:
                final_id = pool_ids[0]
        return {"best": final_id, "phrases": phrases, "pool_ids": pool_ids,
                "inspected": inspected, "agent_calls": result.get("agent_calls", 0),
                "log": result.get("log", [])}


# ---------------------------------------------------------------- CLI demo (no aggregate metric)

def _print_trace(q: str, res: dict, gt: str | None = None):
    print("=" * 88)
    print(f"Q: {q}")
    print(f"phrases: {res['phrases']}")
    print("trace: " + "  →  ".join(res["log"]))
    if res["inspected"]:
        print("inspected:")
        for cid, ev in sorted(res["inspected"].items(), key=lambda kv: -kv[1]["score"]):
            print(f"   {ev['score']:.3f}  {cid}   covered=[{ev['covered']}]")
    verdict = ""
    if gt is not None:
        verdict = "  ✓" if res["best"] == gt else f"  ✗ (GT={gt})"
    print(f"PICK: {res['best']}{verdict}")


def main():
    ap = argparse.ArgumentParser(description="Agent router demo (single query or a few from test).")
    ap.add_argument("--set", required=True, help="route dir (databases/semantic/index/splits)")
    ap.add_argument("--query", default="", help="single question to route")
    ap.add_argument("--n", type=int, default=0, help="instead: route N questions from splits/test.jsonl")
    ap.add_argument("--seed", type=int, default=260713)
    ap.add_argument("--pool-k", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()

    router = AgentRouter(args.set, max_turns=args.max_turns, pool_k=args.pool_k)

    if args.query:
        _print_trace(args.query, router.route(args.query))
        return
    if args.n > 0:
        import numpy as np
        qs = load(Path(args.set) / "splits" / "test.jsonl")
        qs = [q for q in qs if q["instance_id"] in router.data.cards]
        idx = np.random.default_rng(args.seed).permutation(len(qs))[:args.n]
        for j in idx:
            q = qs[int(j)]
            _print_trace(q["question"], router.route(q["question"]), gt=q["instance_id"])
        return
    ap.error("give --query \"...\" or --n N")


if __name__ == "__main__":
    main()
