#!/usr/bin/env python3
"""PURE Sudarshan reproduction (arXiv:2601.19825) — NOTHING substituted in the pipeline.

Faithful end-to-end Sudarshan rerank:
  retrieval : dense top-K=5 over RAW-DDL schema embedding (raw.npy)            ← their retrieval
  mapping   : VERBATIM phrase_mapping_prompt.txt over the candidate's raw DDL  ← their extractor
              (text output `phrase - Table.Column`, per-candidate extract)
  adjacency : LLM-inferred join graph from adjacency_list_prompt.txt           ← their join graph
              (semantic/adjacency_sudarshan.jsonl, built by reproduce_sudarshan_adj.py)
  scoring   : coverage = e^(-n·x), x = NA-phrases/total-phrases ; connectivity = faithful
              (∃ one-entity-per-phrase combination connected) ; score = cov × conn
  tie-break : average cosine(query-phrase emb, matched-entity emb)             ← their tie-break

Reported two-layer (CLAUDE.md §3): pool recall@5 + R@1|in-pool + R@1(all). McNemar pure-Sudarshan
vs OURS arm-C (card+fixed-parse+agent-tie), reconstructed from decomp_card_cache on the SAME query
slice (identical sampling) so the comparison is paired and honest.

Caches in pure_sud_cache/: map_raw.jsonl {qid,cand,out}. Resumable; --score-only re-scores no LLM.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, embed_all, mcnemar          # noqa: E402
from agent_rerank import coverage, field2ent, top_pool       # noqa: E402
from decomp_card_eval import connectivity_faithful, score_fixed, K  # noqa: E402
from build_semantic import client, chat_model, _record_cache  # noqa: E402

MAP_PROMPT = (HERE.parent.parent / "refs" / "sudarshan" /
              "DB-Routing-prompts" / "phrase_mapping_prompt.txt").read_text()
assert "{query}" in MAP_PROMPT and "{Database Information}" in MAP_PROMPT


# ---------------------------------------------------------------- faithful map call + parse

def raw_chat(oai, prompt: str, retries: int = 4) -> str:
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
            print(f"  [retry {attempt+1}/{retries}] {type(exc).__name__}: {str(exc)[:90]} -> {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise last


NA_RE = re.compile(r"\bN/?A\b", re.IGNORECASE)


def parse_mapping(text: str) -> dict[str, list[str]]:
    """Sudarshan strict output `word_or_phrase - Table.Column` (one mapping per line; N/A = unmapped).
    Group by phrase -> list of targets (empty list = N/A)."""
    by: dict[str, list[str]] = defaultdict(list)
    for ln in text.splitlines():
        ln = ln.strip().strip("-• \t`")
        if " - " not in ln and " – " not in ln:
            continue
        sep = " - " if " - " in ln else " – "
        phrase, tgt = ln.rsplit(sep, 1)
        phrase, tgt = phrase.strip(), tgt.strip().strip(".")
        if not phrase:
            continue
        by.setdefault(phrase, [])
        if tgt and not NA_RE.fullmatch(tgt) and not NA_RE.match(tgt):
            by[phrase].append(tgt)
    return dict(by)


def score_pure(by: dict[str, list[str]], adj: dict, inv: dict, n: float) -> float:
    """Coverage over the candidate's own phrase set (per-candidate denom = Sudarshan original) ×
    connectivity-faithful over matched Table.Column targets."""
    phrases = list(by.keys())
    if not phrases:
        return 0.0
    matched = [p for p in phrases if by[p]]
    na = len(phrases) - len(matched)
    cov = coverage(len(matched), na, n)
    by_phrase = [by[p] for p in matched]
    conn = connectivity_faithful(by_phrase, adj, field2ent(inv))
    return cov * conn


# ---------------------------------------------------------------- sampling (mirror decomp_card_eval)

def sample_slice(root: Path, dbs: dict, cards: set, max_dbs: int, cap: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_db = defaultdict(list)
    for q in load(root / "splits" / "test.jsonl"):
        if q["instance_id"] in cards:
            by_db[q["instance_id"]].append(q)
    by_engine = defaultdict(list)
    for d in sorted(by_db):
        by_engine[dbs[d]["engine"]].append(d)
    engines = sorted(by_engine)
    pick_dbs, ei = [], 0
    while len(pick_dbs) < max_dbs and any(by_engine.values()):
        eng = engines[ei % len(engines)]
        if by_engine[eng]:
            pick_dbs.append(by_engine[eng].pop(0))
        ei += 1
    qs = []
    for d in pick_dbs:
        items = by_db[d]
        idx = rng.permutation(len(items))[:cap]
        qs.extend(items[i] for i in idx)
    for q in qs:
        q["_qid"] = q.get("query_id") or hashlib.md5(
            (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]
    return qs


# ---------------------------------------------------------------- ours arm-C reconstruction

def ours_picks(root: Path, qs: list[dict], invs: dict, n: float) -> tuple[dict, dict]:
    """Reconstruct OURS arm-C (card + fixed-parse + agent tie-break) picks on card-pool top-5,
    from decomp_card_cache. Returns (pick_by_qid, inpool_by_qid)."""
    cdir = root / "decomp_card_cache"
    parse = {r["qid"]: r["out"] for r in load(cdir / "parse.jsonl")}
    mapc = {(r["qid"], r["cand"]): r["out"] for r in load(cdir / "map_card.jsonl")}
    tie = {r["qid"]: r["pick"] for r in load(cdir / "tiebreak.jsonl")}
    adjs = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency.jsonl")}
    ids = json.loads((root / "index" / "manifest.json").read_text())["instance_ids"]
    card_idx = np.load(root / "index" / "card.npy")
    qv = embed_all([q["question"] for q in qs])
    pools = top_pool(qv, card_idx, ids, K)
    pick, inpool = {}, {}
    for q, pool in zip(qs, pools):
        qid, gt = q["_qid"], q["instance_id"]
        inpool[qid] = gt in pool
        ph = parse.get(qid, {}).get("phrases") or []
        sc = [(score_fixed(ph, mapc.get((qid, c), {}), adjs.get(c, {}), invs[c], n), c) for c in pool]
        best = max(s for s, _ in sc)
        ties = [c for s, c in sc if s == best]
        pick[qid] = tie.get(qid, ties[0]) if len(ties) > 1 else ties[0]
    return pick, inpool


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--max-dbs", type=int, default=210)
    ap.add_argument("--per-db-cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=260613)
    ap.add_argument("--n", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    adj_sud = {a["instance_id"]: a for a in load(root / "semantic" / "adjacency_sudarshan.jsonl")}
    ids = json.loads((root / "index" / "manifest.json").read_text())["instance_ids"]
    raw_idx = np.load(root / "index" / "raw.npy")

    qs = sample_slice(root, dbs, set(invs), args.max_dbs, args.per_db_cap, args.seed)
    gts = {q["_qid"]: q["instance_id"] for q in qs}
    qv = embed_all([q["question"] for q in qs])
    pools = {q["_qid"]: pool for q, pool in zip(qs, top_pool(qv, raw_idx, ids, K))}
    print(f"{root.name}: {len(qs)} q / {len({q['instance_id'] for q in qs})} GT-DB, "
          f"RAW-pool top-{K}, n={args.n}, model={chat_model()}", flush=True)

    cdir = root / "pure_sud_cache"
    cdir.mkdir(exist_ok=True)
    map_p = cdir / "map_raw.jsonl"
    mp = {(r["qid"], r["cand"]): r["out"] for r in (load(map_p) if map_p.exists() else [])}

    # ---- phrase-mapping: verbatim prompt, per (query, candidate) over candidate raw DDL
    if not args.score_only:
        todo = [(q["_qid"], q["question"], c) for q in qs for c in pools[q["_qid"]]
                if (q["_qid"], c) not in mp]
        oai = client()
        print(f"  [phrase-map] {len(todo)} calls @ {args.workers}w", flush=True)

        def do_map(t):
            qid, question, cand = t
            prompt = (MAP_PROMPT.replace("{query}", question)
                      .replace("{Database Information}", dbs[cand]["schema_text"]))
            return t, parse_mapping(raw_chat(oai, prompt))

        done = 0
        with open(map_p, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
            for (qid, _, cand), out in ex.map(do_map, todo):
                mp[(qid, cand)] = out
                f.write(json.dumps({"qid": qid, "cand": cand, "out": out}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"  map {done}/{len(todo)}", flush=True)

    # ---- score + argmax + cosine tie-break
    def pick_for(qid: str) -> str:
        pool = pools[qid]
        sc = [(score_pure(mp.get((qid, c), {}), adj_sud.get(c, {}), invs[c], args.n), c) for c in pool]
        best = max(s for s, _ in sc)
        ties = [c for s, c in sc if s == best]
        if len(ties) == 1:
            return ties[0]
        return cosine_tiebreak(qid, ties, mp)

    # cosine tie-break: avg cosine(query-phrase emb, matched-entity emb)
    emb_cache: dict[str, np.ndarray] = {}

    def get_emb(strings: list[str]) -> dict[str, np.ndarray]:
        miss = [s for s in strings if s not in emb_cache]
        if miss:
            vs = embed_all(miss)
            for s, v in zip(miss, vs):
                emb_cache[s] = v
        return {s: emb_cache[s] for s in strings}

    def cosine_tiebreak(qid: str, ties: list[str], mp: dict) -> str:
        # collect phrases + matched entity tokens needing embeddings
        need = set()
        per = {}
        for c in ties:
            by = mp.get((qid, c), {})
            pairs = [(p, t) for p, ts in by.items() if ts for t in ts]
            per[c] = pairs
            for p, t in pairs:
                need.add(p); need.add(t)
        if not need:
            return ties[0]
        E = get_emb(sorted(need))
        def cs(a, b):
            va, vb = E[a], E[b]
            d = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
            return float(va @ vb) / d
        best_c, best_s = ties[0], -1.0
        for c in ties:
            pairs = per[c]
            s = np.mean([cs(p, t) for p, t in pairs]) if pairs else -1.0
            if s > best_s:
                best_s, best_c = s, c
        return best_c

    sud_pick = {qid: pick_for(qid) for qid in gts}
    sud_inpool = {qid: gts[qid] in pools[qid] for qid in gts}

    # ---- ours arm-C (paired, same slice)
    our_pick, our_inpool = ours_picks(root, qs, invs, args.n)

    # ---- metrics
    qids = list(gts)
    N = len(qids)
    rec5 = sum(sud_inpool.values())
    def r1_all(pick): return sum(pick[q] == gts[q] for q in qids) / N
    def r1_inp(pick, inp):
        ip = [q for q in qids if inp[q]]
        return sum(pick[q] == gts[q] for q in ip) / max(len(ip), 1), len(ip)

    print(f"\n== PURE Sudarshan vs OURS (K={K}, n={args.n}) ==")
    print(f"pure-Sud RAW-pool recall@5 = {rec5}/{N} = {rec5/N:.3f}")
    s_in, s_n = r1_inp(sud_pick, sud_inpool)
    o_in, o_n = r1_inp(our_pick, our_inpool)
    print(f"{'arm':<24}{'R@1(all)':>10}{'R@1|in-pool':>14}{'pool-rec':>10}")
    print(f"{'PURE-Sudarshan':<24}{r1_all(sud_pick):>10.3f}{s_in:>14.3f}{rec5/N:>10.3f}")
    print(f"{'OURS (card+parse+tie)':<24}{r1_all(our_pick):>10.3f}{o_in:>14.3f}"
          f"{sum(our_inpool.values())/N:>10.3f}")

    # McNemar (all-queries, end-to-end system comparison)
    a = [1 if our_pick[q] == gts[q] else 0 for q in qids]
    b = [1 if sud_pick[q] == gts[q] else 0 for q in qids]
    m = mcnemar(a, b, N)
    print(f"\n-- McNemar (all {N}q, end-to-end): OURS vs PURE-Sudarshan --")
    print(f"  ours_only={m['raw_only'] if 'raw_only' in m else m.get('a_only')} "
          f"sud_only={m.get('card_only', m.get('b_only'))} p={m['p']:.4g}")

    # diagnostics: how often connectivity=1 (degenerate?) for pure arm
    conn1 = tot = 0
    for q in qids:
        for c in pools[q]:
            by = mp.get((q, c), {})
            if not by:
                continue
            matched = [p for p in by if by[p]]
            cv = connectivity_faithful([by[p] for p in matched], adj_sud.get(c, {}), field2ent(invs[c]))
            conn1 += cv; tot += 1
    print(f"\npure-Sud connectivity=1 rate: {conn1}/{tot} = {conn1/max(tot,1):.3f}  "
          f"(→1.0 = connectivity inert, score=coverage-only)")


if __name__ == "__main__":
    main()
