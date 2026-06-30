#!/usr/bin/env python3
"""Faithful Sudarshan adjacency reproduction (arXiv:2601.19825).

Runs the ORIGINAL adjacency_list_prompt.txt (cache: refs/sudarshan/DB-Routing-prompts/)
VERBATIM, per DB, over the DB's raw DDL (databases.jsonl `schema_text`) -> directed adjacency list
`idx:targets`. NOTHING substituted: original prompt text, original strict text output format,
0-based table indices in schema appearance order, reciprocity as the prompt mandates.

Output -> <set>/semantic/adjacency_sudarshan.jsonl, mirroring the structure of the existing
adjacency.jsonl (instance_id, engine, entity_index[], edges[{from,to}]) so the connectivity BFS
can consume it by swapping the file path -- no scoring code change needed.

Resumable: skips DBs already in the output file. Raw chat (no json response_format) because the
faithful output is plain text lines, not JSON. thinking OFF (extraction step, rule m3-design).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_semantic import client, chat_model, _record_cache  # noqa: E402

PROMPT_PATH = (HERE.parent.parent / "refs" / "sudarshan" /
               "DB-Routing-prompts" / "adjacency_list_prompt.txt")
PROMPT_TMPL = PROMPT_PATH.read_text()
assert "[Schema Information]" in PROMPT_TMPL, "prompt placeholder missing"

CREATE_RE = re.compile(r'CREATE\s+TABLE\s+"?([^"\s(]+)"?', re.IGNORECASE)
LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*([\d,\s]*)\s*$")


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def table_order(schema_text: str) -> list[str]:
    """0-based index order = CREATE TABLE appearance order (what the LLM sees)."""
    return CREATE_RE.findall(schema_text)


def raw_chat(oai, prompt: str, retries: int = 4) -> str:
    """Faithful: plain-text completion, temp 0, thinking disabled. No JSON coercion."""
    import time
    last = None
    for attempt in range(retries + 1):
        try:
            resp = oai.chat.completions.create(
                model=chat_model(), temperature=0,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "disabled"}},
            )
            _record_cache(resp)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 8.0 * (2 ** attempt)
            print(f"  [retry {attempt+1}/{retries}] {type(exc).__name__}: {str(exc)[:100]} -> {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise last


def parse_adj(text: str, n_tables: int) -> dict[int, list[int]]:
    """Parse strict `idx:targets` lines. Tolerates stray markdown fences."""
    adj: dict[int, list[int]] = {}
    for ln in text.splitlines():
        ln = ln.strip().strip("`")
        m = LINE_RE.match(ln)
        if not m:
            continue
        src = int(m.group(1))
        tgts = [int(t) for t in m.group(2).split(",") if t.strip().isdigit()]
        if 0 <= src < n_tables:
            adj[src] = sorted({t for t in tgts if 0 <= t < n_tables and t != src})
    return adj


def to_edges(adj: dict[int, list[int]]) -> list[dict]:
    """Directed idx:targets -> edge list [{from,to}] (kept directed; BFS treats as undirected)."""
    edges = []
    for src, tgts in sorted(adj.items()):
        for t in tgts:
            edges.append({"from": src, "to": t})
    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all DBs")
    args = ap.parse_args()

    root = Path(args.set)
    dbs = load(root / "databases.jsonl")
    if args.limit:
        dbs = dbs[:args.limit]
    out_p = root / "semantic" / "adjacency_sudarshan.jsonl"
    done = {r["instance_id"] for r in load(out_p)} if out_p.exists() else set()
    todo = [d for d in dbs if d["instance_id"] not in done]
    print(f"{root.name}: {len(dbs)} DBs, {len(done)} cached, {len(todo)} to infer, model={chat_model()}",
          flush=True)
    if not todo:
        print("all cached; nothing to do.")
        return

    oai = client()

    def infer(d: dict) -> dict:
        order = table_order(d["schema_text"])
        prompt = PROMPT_TMPL.replace("[Schema Information]", d["schema_text"])
        text = raw_chat(oai, prompt)
        adj = parse_adj(text, len(order))
        return {
            "instance_id": d["instance_id"],
            "engine": d["engine"],
            "entity_index": order,
            "edges": to_edges(adj),
            "n_edges": sum(len(v) for v in adj.values()),
            "raw": text.strip(),
        }

    n = 0
    with open(out_p, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(infer, todo):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0 or n == len(todo):
                print(f"  adj {n}/{len(todo)}", flush=True)

    print(f"wrote {out_p} (+{n} DBs)", flush=True)


if __name__ == "__main__":
    main()
