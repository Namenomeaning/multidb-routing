"""Throughput test — DeepSeek DIRECT API (NOT via OpenRouter).

One-off measurement to estimate benchmark-build time if the card-generation LLM flow
runs on DeepSeek directly instead of OpenRouter. Mirrors build_semantic call shape:
temperature 0, response_format=json_object, single user message, representative prompt size.

Tests escalating concurrency to find the practical parallel ceiling (where latency stops
improving or errors appear). Does NOT touch the eval path — measurement only.

CONFLICTS with CLAUDE.md §11 (OpenRouter-only) — run only as user-authorized throughput test.

Env: DEEPSEEK_API_KEY (loaded via dotenv, never printed). Model: DEEPSEEK_CHAT_MODEL or default.
Usage: python deepseek_throughput.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
CONCURRENCIES = [4, 8, 16, 32]
CALLS_PER_LEVEL = 16

# representative card-build prompt (~schema -> JSON card), moderate size
PROMPT = """You are given a database schema. Produce a JSON object describing it.

Schema (PostgreSQL):
CREATE TABLE singer (singer_id INT PRIMARY KEY, name TEXT, country TEXT, song_name TEXT, age INT);
CREATE TABLE stadium (stadium_id INT PRIMARY KEY, location TEXT, name TEXT, capacity INT, highest INT, lowest INT, average INT);
CREATE TABLE concert (concert_id INT PRIMARY KEY, concert_name TEXT, theme TEXT, stadium_id INT REFERENCES stadium, year INT);
CREATE TABLE singer_in_concert (concert_id INT REFERENCES concert, singer_id INT REFERENCES singer);

Sample user questions this DB should answer:
- How many singers are from each country?
- Which stadium hosted the most concerts in 2014?
- List concert themes for singers older than 30.

Return JSON with keys:
  "domain_description": one paragraph natural-language description of what this database is about,
  "term_glossary": object mapping 5-8 important domain terms to short plain definitions.
Return ONLY the JSON object."""


def client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set in env")
    return OpenAI(api_key=key, base_url=BASE_URL)


def one_call(oai: OpenAI) -> tuple[bool, float, int, str]:
    t0 = time.perf_counter()
    try:
        resp = oai.chat.completions.create(
            model=MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT}],
        )
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        json.loads(text)  # validate parseable
        toks = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0
        return True, dt, toks, ""
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return False, dt, 0, f"{type(exc).__name__}: {str(exc)[:100]}"


def run_level(oai: OpenAI, workers: int, n: int):
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_call, oai) for _ in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0
    ok = [r for r in results if r[0]]
    fail = [r for r in results if not r[0]]
    lat = sorted(r[1] for r in ok)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
    cpm = len(ok) / wall * 60 if wall else 0.0
    toks = sum(r[2] for r in ok)
    tpm = toks / wall * 60 if wall else 0.0
    print(f"workers={workers:3d}  n={n}  ok={len(ok)} fail={len(fail)}  "
          f"wall={wall:6.1f}s  p50={p50:5.1f}s p95={p95:5.1f}s  "
          f"throughput={cpm:6.1f} call/min  {tpm:7.0f} tok/min")
    for r in fail[:3]:
        print(f"    FAIL: {r[3]}")
    return cpm


def main():
    oai = client()
    print(f"DeepSeek DIRECT  base={BASE_URL}  model={MODEL}")
    print("warmup 1 call...")
    ok, dt, toks, err = one_call(oai)
    print(f"  warmup ok={ok} dt={dt:.1f}s tokens={toks} {err}")
    if not ok:
        raise SystemExit(f"warmup failed: {err}")
    print()
    best = 0.0
    for w in CONCURRENCIES:
        cpm = run_level(oai, w, CALLS_PER_LEVEL)
        best = max(best, cpm)
    print(f"\nbest throughput = {best:.1f} call/min")
    print(f"build estimate for 494 DB cards: {494 / best * 60:.0f}s = {494 / best:.1f} min at best level")


if __name__ == "__main__":
    main()
