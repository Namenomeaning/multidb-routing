"""Assemble baseline comparison: zero-shot acc-vs-N curve + paired McNemar vs OURS + bootstrap CI.

Inputs (produced by the baseline runs):
  --embed   /tmp/embed_perqid.jsonl      {qid,rep,correct}        (1040 q × {raw,card})
  --ours    /tmp/ours_perqid.jsonl       {qid,correct,in_pool,engine}  (OURS full flow, 312 q)
  --zsdir   <set>/zeroshot_cache         zs_{rep}_N{n}.jsonl      {qid,correct,engine}

Layers (two-layer discipline):
  embed-top1 = real routing  -> compare to OURS R@1(all)      (no GT guarantee)
  zero-shot  = GT guaranteed -> compare to OURS R@1|in-pool   (GT present both sides)
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def ci(hits, iters=2000, seed=7):
    a = np.asarray(hits, float)
    if len(a) == 0: return (0., 0.)
    r = np.random.default_rng(seed)
    m = [a[r.integers(0, len(a), len(a))].mean() for _ in range(iters)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    ao = sum(1 for x, y in zip(a, b) if x and not y)
    bo = sum(1 for x, y in zip(a, b) if y and not x)
    n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n)
    return ao, bo, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="/tmp/embed_perqid.jsonl")
    ap.add_argument("--ours", default="/tmp/ours_perqid.jsonl")
    ap.add_argument("--zsdir", required=True)
    ap.add_argument("--levels", default="5,10,20,50,100,208")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    embed = defaultdict(dict)   # rep -> qid -> correct
    for r in load(args.embed):
        embed[r["rep"]][r["qid"]] = r["correct"]
    ours = {r["qid"]: r for r in load(args.ours)}
    zsdir = Path(args.zsdir)
    zs = {}                     # (rep,N) -> qid -> correct
    for rep in ("card", "raw"):
        for n in levels:
            p = zsdir / f"zs_{rep}_N{n}.jsonl"
            if p.exists():
                zs[(rep, n)] = {r["qid"]: r["correct"] for r in load(p)}

    # ---- 1. acc-vs-N curve (full 1040), per engine ----
    eng_of = {}
    for n in levels:
        for r in load(zsdir / f"zs_card_N{n}.jsonl"):
            eng_of[r["qid"]] = r["engine"]
    print("=== Zero-shot acc vs N (full slice, GT always present) ===")
    print(f"{'N':>4} | {'card':>6} {'card CI':>15} | {'raw':>6} {'raw CI':>15}")
    for n in levels:
        out = []
        for rep in ("card", "raw"):
            d = zs.get((rep, n), {})
            hits = list(d.values())
            lo, hi = ci(hits)
            out.append((np.mean(hits) if hits else 0, lo, hi, len(hits)))
        c, r = out
        print(f"{n:>4} | {c[0]:.3f} [{c[1]:.3f}-{c[2]:.3f}] | {r[0]:.3f} [{r[1]:.3f}-{r[2]:.3f}]  (Nq {c[3]})")

    # per-engine curve (card)
    print("\n=== Zero-shot card acc vs N, per GT-engine ===")
    print(f"{'N':>4} | {'mongodb':>8} {'neo4j':>8} {'postgresql':>10}")
    for n in levels:
        d = zs.get(("card", n), {})
        pe = defaultdict(lambda: [0, 0])
        for q, ok in d.items():
            pe[eng_of[q]][0] += ok; pe[eng_of[q]][1] += 1
        print(f"{n:>4} | " + " ".join(f"{pe[e][0]/max(pe[e][1],1):.3f}" for e in ("mongodb", "neo4j", "postgresql")))

    # ---- 2. headline R@1 + CI ----
    print("\n=== Headline routing R@1 (1040 q unless noted) ===")
    for rep in ("raw", "card"):
        h = list(embed[rep].values())
        lo, hi = ci(h)
        print(f"  embed-top1 [{rep}]  R@1(all)={np.mean(h):.3f} [{lo:.3f}-{hi:.3f}]  N={len(h)}")
    o_all = [r["correct"] for r in ours.values()]
    o_in = [r["correct"] for r in ours.values() if r["in_pool"]]
    lo, hi = ci(o_all); print(f"  OURS full flow    R@1(all)={np.mean(o_all):.3f} [{lo:.3f}-{hi:.3f}]  N={len(o_all)} (312 triage-aligned)")
    lo, hi = ci(o_in); print(f"  OURS full flow    R@1|in-pool={np.mean(o_in):.3f} [{lo:.3f}-{hi:.3f}]  N={len(o_in)}")

    # ---- 3. paired McNemar on the 312 intersection ----
    oq = list(ours.keys())
    print("\n=== Paired McNemar on 312 triage-aligned qids ===")
    # OURS R@1(all) vs embed-top1 (real routing, both no-GT-guarantee)
    for rep in ("raw", "card"):
        common = [q for q in oq if q in embed[rep]]
        a = [ours[q]["correct"] for q in common]
        b = [embed[rep][q] for q in common]
        ao, bo, p = mcnemar(a, b)
        print(f"  OURS(all) vs embed-{rep}(all): OURS={np.mean(a):.3f} embed={np.mean(b):.3f} "
              f"ours_only={ao} embed_only={bo} p={p:.2e}  (N={len(common)})")
    # OURS in-pool vs zero-shot@N (both GT-present): restrict to OURS in_pool qids
    inq = [q for q in oq if ours[q]["in_pool"]]
    for n in (5, 208):
        for rep in ("card", "raw"):
            d = zs.get((rep, n), {})
            common = [q for q in inq if q in d]
            a = [ours[q]["correct"] for q in common]
            b = [d[q] for q in common]
            ao, bo, p = mcnemar(a, b)
            print(f"  OURS(in-pool) vs ZS-{rep}@{n}: OURS={np.mean(a):.3f} ZS={np.mean(b):.3f} "
                  f"ours_only={ao} zs_only={bo} p={p:.2e}  (N={len(common)})")


if __name__ == "__main__":
    main()
