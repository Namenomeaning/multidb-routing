"""Stratified benchmark for the LangGraph agent router (`agent_router.py`), locked config.

Two metric layers kept SEPARATE:
  * retrieval  = pool-recall  (was the GT DB in the dense top-k pool)
  * final route = R@1         (agent committed the correct instance_id)

Per-case wall-clock timeout -> retry -> drop (daemon thread; an abandoned slow call never blocks
process exit). Dropped cases are re-run once at low concurrency (--recover) to remove the cold-start
burst that shows up when many workers hit the provider simultaneously. Per-query rows are written to
a jsonl for later bootstrap CI / error analysis.

Provider + keys come from the environment (build_semantic loads the bundle .env):
  LLM_PROVIDER = openrouter (default) | deepseek   + the matching *_API_KEY.
Embeddings always route through OpenRouter (OPENROUTER_API_KEY), independent of LLM_PROVIDER.

Usage:
  LLM_PROVIDER=deepseek python run_agent_benchmark.py --cap 5
  LLM_PROVIDER=deepseek python run_agent_benchmark.py --cap 2 --max-dbs 12 --dup 6   # quick validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_BUNDLE = _HERE.parent.parent  # multidb-routing/
for _p in (str(_HERE), str(_BUNDLE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_router import AgentRouter  # noqa: E402
from retrieval_eval import load  # noqa: E402
import build_semantic as bs  # noqa: E402


def stratified_sample(route_dir: Path, cap: int, max_dbs: int, seed: int, cards: set) -> list[dict]:
    """cap queries per GT-DB (min(cap, available)). max_dbs>0 = engine-balanced round-robin subset."""
    dbs = {d["instance_id"]: d for d in load(route_dir / "databases.jsonl")}
    by_db: dict[str, list] = defaultdict(list)
    for q in load(route_dir / "splits" / "test.jsonl"):
        if q["instance_id"] in cards:
            by_db[q["instance_id"]].append(q)
    db_ids = sorted(by_db)
    if max_dbs > 0:
        by_eng: dict[str, list] = defaultdict(list)
        for d in db_ids:
            by_eng[dbs[d]["engine"]].append(d)
        engines = sorted(by_eng)
        picked, ei = [], 0
        while len(picked) < max_dbs and any(by_eng.values()):
            e = engines[ei % len(engines)]
            if by_eng[e]:
                picked.append(by_eng[e].pop(0))
            ei += 1
        db_ids = picked
    rng = np.random.default_rng(seed)
    sample = []
    for d in db_ids:
        items = by_db[d]
        for i in rng.permutation(len(items))[:cap]:
            q = items[int(i)]
            sample.append({"question": q["question"], "gt": d, "engine": dbs[d]["engine"]})
    return sample


def route_timed(router: AgentRouter, question: str, timeout: float) -> dict:
    """Run route() in a daemon thread; raise TimeoutError if it exceeds `timeout`."""
    box: dict = {}

    def run():
        try:
            box["r"] = router.route(question)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError
    if "e" in box:
        raise box["e"]
    return box["r"]


def run_case(router, rec: dict, timeout: float, retries: int) -> dict:
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            res = route_timed(router, rec["question"], timeout)
            rec.update(best=res["best"], ok=(res["best"] == rec["gt"]),
                       in_pool=(rec["gt"] in res["pool_ids"]), turns=res["agent_calls"],
                       n_inspect=len(res["inspected"]), dt=time.time() - t0, dropped=False, err=None)
            return rec
        except TimeoutError:
            if attempt < retries:
                continue
            rec.update(best=None, ok=False, in_pool=None, turns=0, n_inspect=0,
                       dt=time.time() - t0, dropped=True, err=None)
        except Exception as exc:  # noqa: BLE001
            rec.update(best=None, ok=False, in_pool=None, turns=0, n_inspect=0,
                       dt=time.time() - t0, dropped=False, err=f"{type(exc).__name__}: {exc}")
            return rec
    return rec


def report(rows: list[dict]) -> None:
    done = [r for r in rows if not r["dropped"] and not r["err"]]
    dropped = sum(1 for r in rows if r["dropped"])
    err = sum(1 for r in rows if r["err"])
    if not done:
        print("no completed cases")
        return
    ok = sum(r["ok"] for r in done)
    in_pool = sum(1 for r in done if r["in_pool"])
    p95 = np.percentile([r["dt"] for r in done], 95)
    print("=" * 74)
    print(f"completed={len(done)}/{len(rows)}  dropped={dropped}  errors={err}")
    print(f"POOL-RECALL (GT in pool)   = {in_pool}/{len(done)} = {in_pool/len(done):.3f}")
    print(f"FINAL R@1 (overall)        = {ok}/{len(done)} = {ok/len(done):.3f}")
    print(f"FINAL R@1 | GT in pool     = {ok}/{in_pool} = {ok/in_pool:.3f}")
    print("R@1 per engine:")
    for e in sorted({r["engine"] for r in done}):
        de = [r for r in done if r["engine"] == e]
        oke = sum(r["ok"] for r in de)
        print(f"   {e:11s} {oke}/{len(de)} = {oke/len(de):.3f}")
    print(f"avg_turns={np.mean([r['turns'] for r in done]):.1f}  "
          f"avg_inspect={np.mean([r['n_inspect'] for r in done]):.1f}  p95_latency={p95:.1f}s")
    h, m = bs.CACHE_STATS["hit"], bs.CACHE_STATS["miss"]
    if h + m:
        print(f"call_json cache (parse+map): calls={bs.CACHE_STATS['calls']} hit_ratio={100*h/(h+m):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default=str(_BUNDLE / "data" / "multidb"))
    ap.add_argument("--cap", type=int, default=5, help="queries per GT-DB")
    ap.add_argument("--max-dbs", type=int, default=0, help="0=all; >0=engine-balanced subset (validation)")
    ap.add_argument("--seed", type=int, default=260714)
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--retry", type=int, default=1)
    ap.add_argument("--recover", action="store_true", default=True,
                    help="re-run drops at low concurrency (default on)")
    ap.add_argument("--no-recover", dest="recover", action="store_false")
    ap.add_argument("--dup", type=int, default=0, help="re-run first N completed to check determinism")
    ap.add_argument("--out", default=str(_BUNDLE / "results" / "agent-bench.jsonl"))
    a = ap.parse_args()

    router = AgentRouter(a.set, max_turns=6, pool_k=10)
    sample = stratified_sample(Path(a.set), a.cap, a.max_dbs, a.seed, router.data.cards)
    for i, s in enumerate(sample):
        s["i"] = i
    n = len(sample)
    print(f"provider={os.environ.get('LLM_PROVIDER', 'openrouter')} model={router.model}  "
          f"cap={a.cap}/DB  N={n}  workers={a.workers}  timeout={a.timeout}s retry={a.retry}")
    print(f"engines: {dict(Counter(s['engine'] for s in sample))}", flush=True)

    lock = threading.Lock()
    rows: list[dict] = []

    def work(s):
        rec = run_case(router, s, a.timeout, a.retry)
        with lock:
            rows.append(rec)
            k = len(rows)
            if k % 25 == 0 or rec["dropped"] or rec["err"]:
                tag = "DROP" if rec["dropped"] else ("ERR " if rec["err"] else "ok")
                print(f"[{k:4d}/{n}] {tag} {rec['engine'][:5]} {rec['dt']:.1f}s {rec['err'] or ''}",
                      flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in as_completed([ex.submit(work, s) for s in sample]):
            pass

    if a.recover:
        drops = [r for r in rows if r["dropped"]]
        if drops:
            print(f"\nrecovering {len(drops)} dropped cases at low concurrency (8 workers, 120s)...",
                  flush=True)
            with ThreadPoolExecutor(max_workers=8) as ex:
                for _ in as_completed([ex.submit(run_case, router, r, 120.0, 0) for r in drops]):
                    pass

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in sorted(rows, key=lambda x: x["i"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    report(rows)

    if a.dup > 0:
        done = [r for r in rows if not r["dropped"] and not r["err"]][:a.dup]
        print(f"\n== determinism check: re-run {len(done)} queries ==")
        same = 0
        for r in done:
            try:
                r2 = route_timed(router, r["question"], 120.0)
                hit = r2["best"] == r["best"]
                same += hit
                print(f"   {'==' if hit else '!='}  {r['best']}  vs  {r2['best']}")
            except Exception as exc:  # noqa: BLE001
                print(f"   ERR {exc}")
        print(f"stable picks: {same}/{len(done)}")


if __name__ == "__main__":
    main()
