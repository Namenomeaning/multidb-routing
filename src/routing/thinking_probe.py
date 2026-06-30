"""Probe: does DeepSeek thinking mode (default ENABLED) change triage picks vs disabled, and how much
slower is it? Runs N test-slice queries through the triage prompt twice — thinking off vs on — and
compares the picked set + wall-clock. DeepSeek OpenAI-format param: extra_body={"thinking":{"type":...}}.

Usage: LLM_PROVIDER=deepseek python thinking_probe.py --set <route_dir> --n 25
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all
from agent_rerank import top_pool
from build_semantic import client, chat_model
from triage_eval import TRIAGE_PROMPT, candidate_block
from retrieval_pipeline_benchmark import split_slice


def triage_call(oai, prompt, thinking: bool):
    t0 = time.time()
    resp = oai.chat.completions.create(
        model=chat_model(), temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )
    dt = time.time() - t0
    try:
        rel = json.loads(resp.choices[0].message.content or "{}").get("relevant") or []
    except Exception:
        rel = []
    return rel, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    root = Path(args.set)
    man = json.loads((root / "index" / "manifest.json").read_text())
    ids = man["instance_ids"]
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    card = np.load(root / "index" / "card.npy")

    all_q = [q for q in load(root / "splits" / "test.jsonl") if q["instance_id"] in cards]
    qs = split_slice(all_q, "test", 10, 3)
    # engine-balanced first-N
    from collections import defaultdict
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    by = defaultdict(list)
    for q in qs:
        by[dbs[q["instance_id"]]["engine"]].append(q)
    engs = sorted(by); pick, i = [], 0
    while len(pick) < args.n and any(by.values()):
        e = engs[i % len(engs)]
        if by[e]:
            pick.append(by[e].pop(0))
        i += 1

    qv = embed_all([q["question"] for q in pick])
    pools = top_pool(qv, card, ids, 10)
    oai = client()
    print(f"{root.name}: {len(pick)} test queries | model={chat_model()}\n")

    diff = 0; toff = ton = 0.0
    gt_off = gt_on = 0
    for q, pool in zip(pick, pools):
        cb = candidate_block("desc", pool, cards, invs)
        prompt = TRIAGE_PROMPT.format(question=q["question"], candidate_block=cb)
        roff, doff = triage_call(oai, prompt, False)
        ron, don = triage_call(oai, prompt, True)
        toff += doff; ton += don
        soff = {pool[i-1] for i in roff if isinstance(i, int) and 1 <= i <= len(pool)}
        son = {pool[i-1] for i in ron if isinstance(i, int) and 1 <= i <= len(pool)}
        gt = q["instance_id"]
        gt_off += gt in soff; gt_on += gt in son
        same = soff == son
        diff += not same
        if not same:
            print(f"DIFF q='{q['question'][:55]}' GT={gt in pool and 'in10' or 'out'}")
            print(f"   off({len(soff)}): GT={'Y' if gt in soff else 'N'}  on({len(son)}): GT={'Y' if gt in son else 'N'}")
    n = len(pick)
    print(f"\n=== {n} queries ===")
    print(f"picks differ off-vs-on: {diff}/{n} ({diff/n*100:.0f}%)")
    print(f"GT-kept  off={gt_off}/{n}  on={gt_on}/{n}")
    print(f"avg latency  off={toff/n:.2f}s  on={ton/n:.2f}s  (thinking {ton/toff:.1f}x slower)")


if __name__ == "__main__":
    main()
