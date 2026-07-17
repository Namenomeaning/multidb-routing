"""Baseline 2 — LLM zero-shot routing, size sweep, factorial representation {card, raw}.

Per query, per level N, per representation: show the LLM a candidate set of N databases that ALWAYS
contains the GT + (N-1) random distractors (seeded by (qid,N), reproducible), shuffled. The LLM
SELECTS one (no examples, no schema inspection beyond the shown text, no scoring -> a pure zero-shot
selection baseline; thinking OFF). Correct iff chosen id == GT.

Because GT is always present, this measures the DECISION layer with N distractors and NO retrieval
miss -> an upper bound for a blind LLM. The acc-vs-N curve is the motivating result (does blind LLM
collapse as the registry grows). Levels include the full registry (N=208).

Representations (compact, both fit N=208):
  card : cards.jsonl domain_description (truncated)        -- our semantic side
  raw  : entity/field name-list from inventory (no prose)  -- DDL-derived structure side

Usage (multi-model R@1, whole-registry read, via OpenRouter — default drops PostgreSQL so the
run is MongoDB+Neo4j; the SQL case is validated separately by the Sudarshan Setup-A repro):
  python baseline_zeroshot_sweep.py --set <dir>          # reps=both, levels=full, R@1 2-col table
  # --levels 5,10,20,50,100 -> size-sweep curve; --exclude-engines '' -> keep all engines
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.core.retrieval import load, stratified, embed_all  # noqa: E402
from src.core.semantic_card import client, call_json, chat_model, CACHE_STATS  # noqa: E402

HEAD = load_prompt("zeroshot_head")


def render_namelist(inv: dict, cap_ent: int = 14, cap_field: int = 8) -> str:
    parts = []
    for e in inv["entities"][:cap_ent]:
        fs = ", ".join(f["name"] for f in e["fields"][:cap_field])
        parts.append(f"{e['name']}({fs})")
    return "; ".join(parts)


def qid_of(q: dict) -> str:
    return q.get("query_id") or hashlib.md5(
        (q["instance_id"] + "|" + q["question"]).encode()).hexdigest()[:12]


def cand_seed(qid: str, n: int) -> int:
    return int(hashlib.md5(f"{qid}|{n}".encode()).hexdigest()[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rep", default="raw", choices=["card", "raw", "both"],
                    help="raw [default] = LLM zero-shot reads the raw schema (name-list); this is "
                         "the LLM-read baseline. (card-read available via --rep card; the card "
                         "baseline proper is the embedding one in baseline_embed_routing.py)")
    ap.add_argument("--levels", default="full",
                    help="'full' = whole (filtered) registry only [default, R@1]; or a comma list "
                         "like 5,10,20,50,100 for the size-sweep curve")
    ap.add_argument("--per-db-cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0=all, for timing probe)")
    ap.add_argument("--desc-chars", type=int, default=320)
    ap.add_argument("--distractor", choices=["random", "hard"], default="random",
                    help="random = off-domain easy; hard = query-nearest neighbours (same-domain, realistic)")
    ap.add_argument("--exclude-engines", default="",
                    help="engines dropped from BOTH registry and queries. Default '' = keep all "
                         "3 engines (the multi-model registry is PostgreSQL+MongoDB+Neo4j). The "
                         "SQL-only Sudarshan head-to-head is a SEPARATE single-model run.")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    suf = "" if args.distractor == "random" else "_hard"

    root = Path(args.set)
    dbs = {d["instance_id"]: d for d in load(root / "databases.jsonl")}
    cards = {c["instance_id"]: c for c in load(root / "semantic" / "cards.jsonl")}
    invs = {i["instance_id"]: i for i in load(root / "semantic" / "inventory.jsonl")}
    excl = {e.strip().lower() for e in args.exclude_engines.split(",") if e.strip()}
    all_ids = [i for i in cards if i in invs and dbs[i]["engine"].lower() not in excl]
    keep_ids = set(all_ids)
    n_db = len(all_ids)
    # Cache files are scoped by the engine filter: dropping an engine changes the candidate sets,
    # so reusing the old PG-inclusive caches would be WRONG. New filter -> new cache namespace.
    excl_tag = "" if not excl else "_excl-" + "-".join(sorted(e[:4] for e in excl))
    levels = [n_db] if args.levels.strip().lower() == "full" \
        else sorted({min(int(x), n_db) for x in args.levels.split(",")})

    qs = [q for q in load(root / "splits" / "test.jsonl") if q["instance_id"] in keep_ids]
    qs = stratified(qs, args.per_db_cap, args.seed)
    if args.limit:
        qs = qs[:args.limit]
    for q in qs:
        q["_qid"] = qid_of(q)
    reps = ["card", "raw"] if args.rep == "both" else [args.rep]
    eng_left = sorted({dbs[i]["engine"] for i in all_ids})
    print(f"{root.name} reps={reps}: {n_db} DBs (engines={eng_left}, excluded={sorted(excl) or '-'}), "
          f"{len(qs)} q, levels={levels}, distractor={args.distractor}, model={chat_model()}")

    # hard distractors = query-nearest DBs by card embedding (same-domain, the realistic pool)
    near_ids = {}
    if args.distractor == "hard":
        man = json.loads((root / "index" / "manifest.json").read_text())
        idx_ids0 = man["instance_ids"]
        C0 = np.load(root / "index" / "card.npy")
        rows = [j for j, i in enumerate(idx_ids0) if i in keep_ids]  # restrict index to kept engines
        idx_ids = [idx_ids0[j] for j in rows]
        C = C0[rows]
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        qv = embed_all([q["question"] for q in qs])
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        order = np.argsort(-(qv @ C.T), axis=1)
        for q, row in zip(qs, order):
            near_ids[q["_qid"]] = [idx_ids[j] for j in row if idx_ids[j] in keep_ids]

    cache_dir = root / "zeroshot_cache"
    cache_dir.mkdir(exist_ok=True)
    oai = None if args.score_only else client()
    lock = threading.Lock()

    def build_candset(qid: str, gt: str, n: int) -> list[str]:
        rng = np.random.default_rng(cand_seed(qid, n))
        k = min(n - 1, len(all_ids) - 1)
        if args.distractor == "hard":
            # GT + nearest-by-query DBs (same-domain confusable), excl GT; pad random if short
            dist = [i for i in near_ids[qid] if i != gt][:k]
            if len(dist) < k:
                extra = [i for i in all_ids if i != gt and i not in dist]
                dist += list(rng.choice(np.asarray(extra), size=k - len(dist), replace=False))
        else:
            pool = [i for i in all_ids if i != gt]
            dist = list(rng.choice(np.asarray(pool), size=k, replace=False)) if k > 0 else []
        # CANONICAL sort (NOT per-query shuffle): makes the candidate/description block
        # byte-identical across queries that share the set -> DeepSeek prefix-cache fires.
        # At N=full-registry every query shows all DBs in the same order -> ~full cache hit.
        # No position bias: GT differs per query, so its sorted position is uncorrelated with
        # the answer (a fixed canonical order, not a fixed GT slot).
        return sorted([gt] + [str(x) for x in dist])

    def run_rep(rep: str) -> dict:
        def rep_text(i: str) -> str:
            if rep == "card":
                return cards[i].get("domain_description", "")[:args.desc_chars]
            # raw = compact entity/field name-list (DDL-derived structure, no prose). Lightweight so
            # the whole registry fits one prompt without the full-DDL token blow-up.
            return render_namelist(invs[i])[:args.desc_chars]

        accs: dict = {}
        for n in levels:
            path = cache_dir / f"zs_{rep}{suf}{excl_tag}_N{n}.jsonl"
            done = {r["qid"]: r for r in (load(path) if path.exists() else [])}
            tasks = [q for q in qs if q["_qid"] not in done]

            def do_one(q):
                qid, gt = q["_qid"], q["instance_id"]
                cand = build_candset(qid, gt, n)
                block = "\n".join(f"[{j}] {c} ({dbs[c]['engine']}) — {rep_text(c)}"
                                  for j, c in enumerate(cand, 1))
                out = call_json(oai, HEAD + "\n## DATABASES\n" + block +
                                f"\n\n## QUESTION\n{q['question']}\n", thinking=False)
                ch = out.get("choice")
                pick = cand[ch - 1] if isinstance(ch, int) and 1 <= ch <= len(cand) else cand[0]
                return {"qid": qid, "gt": gt, "pick": pick, "correct": int(pick == gt),
                        "engine": dbs[gt]["engine"]}

            if tasks and not args.score_only:
                with open(path, "a") as fh:
                    cnt = 0
                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        futs = {ex.submit(do_one, q): q for q in tasks}
                        for fut in as_completed(futs):
                            rec = fut.result()
                            with lock:
                                fh.write(json.dumps(rec) + "\n"); fh.flush()
                            done[rec["qid"]] = rec
                            cnt += 1
                            if cnt % 50 == 0:
                                print(f"  [{rep}] N={n} {cnt}/{len(tasks)}", flush=True)
            elif tasks and args.score_only:
                print(f"  [{rep}] N={n}: {len(tasks)} q NOT cached (score-only) — skipped")

            recs = [done[q["_qid"]] for q in qs if q["_qid"] in done]
            if not recs:
                continue
            acc = float(np.mean([r["correct"] for r in recs]))
            accs[n] = acc
            per_eng = defaultdict(lambda: [0, 0])
            for r in recs:
                per_eng[r["engine"]][0] += r["correct"]; per_eng[r["engine"]][1] += 1
            eng_str = "  ".join(f"{ge}={per_eng[ge][0]/max(per_eng[ge][1],1):.3f}" for ge in sorted(per_eng))
            print(f"== ZS [{rep}] N={n:3d}  R@1={acc:.3f}  (N_q={len(recs)})  | {eng_str}")
        return accs

    rep_accs = {rep: run_rep(rep) for rep in reps}

    # Final 2-column table at the full (filtered) registry — R@1 only, no in-pool.
    full = max(levels)
    label = {"raw": "read raw schema", "card": "read card"}
    print(f"\n=== Multi-model routing (read whole registry, N={full}) — R@1 ===")
    print(f"{'Method':20s} {'R@1':>7}")
    for rep in reps:
        if full in rep_accs[rep]:
            print(f"{label[rep]:20s} {rep_accs[rep][full]:>7.3f}")
    h, m = CACHE_STATS["hit"], CACHE_STATS["miss"]
    if h + m:
        print(f"prefix-cache: hit={h/(h+m)*100:.0f}% ({CACHE_STATS['calls']} calls cumulative)")


if __name__ == "__main__":
    main()
