# Multi-DB Routing Benchmark

Instance-level routing for natural-language queries over a **multi-engine** database registry
(PostgreSQL · MongoDB · Neo4j). Given one NL question and a registry of database instances, the
system routes the question to the single best-matching instance.

Pipeline (V2): semantic-card embedding → dense top-K candidate pool → LLM domain triage →
deterministic `Coverage × Connectivity` rerank → LLM tie-break. The LLM only **extracts/selects**;
all scoring is deterministic code (no LLM self-reported confidence).

This folder is **self-contained**. It ships the benchmark data, the LLM/embedding caches, the
embedding index, the pipeline code, the Sudarshan grounding references, and the result reports —
so the evaluation can be re-run without rebuilding anything from scratch.

---

## Layout

```
multidb-routing/
  src/routing/        the pipeline + all eval scripts (one flat, interdependent module set)
  src/clients/        openrouter.py — the single LLM/embedding client (all calls via OpenRouter)
  data/
    multidb/          MAIN set — 208 DB, 3 engines (Setup B, "ours")
    spider/           Setup A — Spider-Route (206 PG), Sudarshan-faithful repro
    bird/             Setup A — BIRD-Route (80 PG), Sudarshan-faithful repro
      each set has: databases.jsonl, splits/{test,dev,support-balanced}.jsonl,
                    semantic/{cards,adjacency,inventory}.jsonl, index/{card,raw}.npy,
                    and committed *_cache/ dirs (LLM extraction / triage / tie-break replay)
  refs/sudarshan/     third-party Sudarshan routing prompts (grounding; arXiv:2601.19825)
  results/            result reports (narrative source of truth)
  BENCHMARK.md        locked benchmark spec
  scripts/
    download_datasets.py   from-scratch raw fetch (provenance; unverified end-to-end)
    build/                 original build-chain scripts (provenance; see scripts/build/README.md)
  pyproject.toml  uv.lock  .env.example
```

## Prerequisites

```bash
uv sync                       # installs runtime deps (numpy, openai, python-dotenv, ...)
cp .env.example .env          # then edit .env
```

`.env` needs:

- `OPENROUTER_API_KEY=...` — **all** LLM + embedding calls route through OpenRouter.
- optional `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY=...` — to call DeepSeek directly (used for
  the cache-optimized build/triage paths).

Models are configured in code/env (chat: a DeepSeek model; embedding: `text-embedding-3-large`),
not hardcoded across files — change the model = change config.

## Reproduce the published numbers (PRIMARY path)

The data + index + caches are prebuilt, so you do **not** download or rebuild anything. Run the
eval scripts directly with `--set` (a route dir). Query embeddings are recomputed at eval time
(a few hundred cheap embedding calls per set); the LLM extraction / triage / tie-break steps
**replay from the committed `*_cache/` dirs**, so LLM cost is near-zero on a clean replay.

```bash
cd src/routing

# 1) Retrieval layer — card vs raw-DDL, R@k + McNemar (per set)
python retrieval_eval.py --bench ../../data/multidb --index ../../data/multidb/index
python retrieval_eval.py --bench ../../data/spider  --index ../../data/spider/index
python retrieval_eval.py --bench ../../data/bird    --index ../../data/bird/index

# 2) Full pipeline (retrieval -> rerank) on the main set
python full_eval_v1.py --set ../../data/multidb

# 3) Final-routing agent flow (triage -> deterministic rerank -> tie-break)
python agent_flow_eval.py --set ../../data/multidb

# 4) Hybrid gate + agent
python hybrid_v2_eval.py --set ../../data/multidb

# 5) Sudarshan-faithful baseline arm (Setup A)
python pure_sudarshan_eval.py --set ../../data/spider
python pure_sudarshan_eval.py --set ../../data/bird
```

Scripts that take `--score-only` (e.g. `agent_flow_eval.py`, `agent_rerank.py`,
`pure_sudarshan_eval.py`) re-score purely from cache with **no** API calls — use that for a
zero-cost integrity check that the caches and scoring still line up.

Cross-check the printed metrics against the reports in `results/`. Discrepancies mean the
data/cache wiring drifted (wrong set dir, missing cache, etc.) — investigate before trusting.

> **Caveat (re-plumbed layout).** This folder was assembled from the original research tree;
> path plumbing was edited but scoring/prompts/formulas are byte-identical. The committed caches
> still need one real replay per set to confirm the rename didn't break cache lookup or set
> selection. That replay is the true acceptance test — run the commands above and diff against
> `results/`.

## Agent routing benchmark (live LLM — costs API credit)

`run_agent_benchmark.py` drives the LangGraph agent (`agent_router.py`) over a stratified query
sample and reports the two metric layers separately: **pool-recall** (was the GT DB in the dense
top-k pool) vs **final routing R@1**. Unlike the cache-replay scripts above, this makes **live**
LLM calls, so it needs a working key and spends credit (locked config: `deepseek-v4-flash`,
thinking off, pool_k=10, max_turns=6, Coverage N=2.0, tie-break δ=0.2).

```bash
cd src/routing

# Quick validation slice (12 DBs x 2 = 24 queries) + determinism re-check
LLM_PROVIDER=deepseek python run_agent_benchmark.py --cap 2 --max-dbs 12 --dup 6

# Full stratified benchmark (5 queries per GT-DB = 1040 on data/multidb)
LLM_PROVIDER=deepseek python run_agent_benchmark.py --cap 5 --out ../../results/agent-bench-5db.jsonl
```

Per-case guard: a wall-clock `--timeout` (default 45s) → `--retry` → drop; dropped cases are
re-run once at low concurrency (`--recover`, on by default) to absorb provider cold-start bursts.
Per-query rows are written to `--out` (jsonl, gitignored) for bootstrap CI / error analysis.
Reference run (2026-07-14, 5/DB, DeepSeek-direct): pool-recall 0.948, final R@1 0.770.

## Build from scratch (SECONDARY path — unverified)

Only needed to regenerate the benchmark from raw sources. **Not** required for reproduction.

```bash
python scripts/download_datasets.py --list      # see the source manifest (CLAUDE.md §10)
python scripts/download_datasets.py             # fetch automatable sources into data/_raw/
# then follow scripts/build/README.md to run the build chain, then:
#   LLM_PROVIDER=deepseek python src/routing/build_semantic.py   # semantic cards + adjacency
#   python src/routing/build_index.py                            # embedding index
```

⚠️ The from-scratch path is **unverified end-to-end**: external endpoints/formats change, some
sources need a manual/license download, and the build scripts assume the original research
directory layout (see `scripts/build/README.md`). The build scripts also need **pandas**
(`uv pip install pandas`), which the runtime path does not. Spot-check one source before relying
on it.

## What is NOT shipped (and why)

- **Secrets** — only `.env.example` ships; never the live `.env`.
- **Raw source datasets** (`data/_raw/`, multi-GB) — regenerate via `scripts/download_datasets.py`.
- **Obfuscation control artifacts** (`*_obf` / name-masked variants) — deprecated control, omitted.
- **A second, dead routing implementation** and the scripts that import it
  (`build-semantic-layer.py`, `bucket-scenarios.py`, `baseline-embedding-scenario60.py`,
  `eval-m3-agent.py`) — not part of this pipeline.

## Citation

Reproducibility bundle for the paper *Agent-Based Natural Language Query Routing to Heterogeneous
Databases*. The method builds on Sudarshan et al., *Routing End User Queries to Enterprise
Databases* (<https://arxiv.org/html/2601.19825v1>).

## Licensing / redistribution

- **This repository's own code** (`src/`, `scripts/`) is released under the MIT License (see
  `LICENSE`).
- **`refs/sudarshan/`** holds third-party prompts (not under this repo's license); baseline paper:
  <https://arxiv.org/html/2601.19825v1>.
- **`data/`** contains artifacts derived from Spider, BIRD, CypherBench, Text2Cypher, DocSpider,
  and MongoDB-EAI, each under its own upstream license:
  - Spider — <https://yale-lily.github.io/spider> (CC BY-SA 4.0)
  - BIRD — <https://bird-bench.github.io/> (CC BY-SA 4.0)
  - DocSpider — <https://www.cambridge.org/core/journals/natural-language-processing/article/docspider-a-dataset-of-crossdomain-natural-language-querying-for-mongodb/1E35B1DBF843B9E0F444B595B975695A>
  - MongoDB-EAI — <https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh>
  - CypherBench — <https://huggingface.co/datasets/megagonlabs/cypherbench>
  - Text2Cypher — <https://huggingface.co/datasets/neo4j/text2cypher-2024v1>

  Derived splits are redistributed for reproducibility under the source terms; confirm each
  source's redistribution conditions before any further use.
