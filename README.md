# MultiDB-Route

**Natural-language query routing over heterogeneous (multi-model) databases.**

Given a natural-language question and a registry of database instances spanning different data
models — relational (PostgreSQL), document (MongoDB), and graph (Neo4j) — the system routes the
question to the single instance best able to answer it, *before* any query is generated. A fast
semantic-retrieval + domain-triage stage narrows the registry, then a deterministic
`Coverage × Connectivity` score ranks the survivors; the LLM only *extracts and selects* and never
emits a confidence number (the no-self-scoring invariant), so every route is reproducible.

This repository is the reproducibility bundle for the accompanying paper: it ships the benchmark
(208 databases, ground-truth labels inherited deterministically from the source datasets), the
pipeline, and the extraction/retrieval caches, so every number in `results/RESULTS.md` can be
re-derived without rebuilding anything.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green)
![Paper: under review](https://img.shields.io/badge/paper-under%20review-orange)

## Install

```bash
uv sync
cp .env.example .env          # then set OPENROUTER_API_KEY (all LLM + embedding calls route through OpenRouter)
```

## Reproduce the results

Run from the repo root. `--score-only` replays LLM extraction/triage/tie-break entirely from the
committed caches (no API calls); only query embeddings are recomputed (a few hundred cheap calls
per set). Cross-check printed metrics against `results/RESULTS.md`.

```bash
# RQ1 — retrieval: semantic card vs raw DDL, recall@k
python -m src.core.retrieval --bench data/multidb --index data/multidb/index
python -m src.core.retrieval --bench data/spider  --index data/spider/index
python -m src.core.retrieval --bench data/bird    --index data/bird/index

# RQ2 — single-model SQL head-to-head (Spider / BIRD reconstruction)
python -m src.workflow.sudarshan --set data/spider
python -m src.workflow.sudarshan --set data/bird

# RQ3 — multi-model routing on the 1,040-question slice (R@1 0.749)
python -m src.workflow.final_routing --set data/multidb --seed 260714 --n-queries 1040 --score-only
```

### RQ4 — agentic variant (optional, live LLM)

A LangGraph agent issues the `retrieve` / `inspect` / `answer` tool calls itself while reusing the
exact same deterministic scoring code. Unlike the cache-replay scripts above, this makes **live**
LLM calls (spends credit).

```bash
LLM_PROVIDER=deepseek python -m src.agent.benchmark --cap 5 --out results/agent-bench.jsonl
```

## Citation

```bibtex
@misc{multidbroute2026,
  title  = {Natural Language Query Routing in Heterogeneous Databases},
  author = {Nguyen, Tran Minh Thu and Phan, Vu Anh Quang and Dao, Ngoc Hung},
  year   = {2026},
  note   = {Under review},
  url    = {https://github.com/Namenomeaning/multidb-routing}
}
```

The method builds on Sudarshan et al., *Routing End User Queries to Enterprise Databases*
(arXiv:2601.19825).

## License

- **Code** (`src/`, `scripts/`) — MIT (see `LICENSE`).
- **`refs/sudarshan/`** — third-party prompts, under their original terms.
- **`data/`** — derived from Spider, BIRD, DocSpider, MongoDB-EAI, CypherBench, and Text2Cypher,
  each under its own upstream license (CC BY-SA 4.0 for Spider/BIRD; see each source for the rest).
  Redistributed for reproducibility under the source terms — confirm each source's conditions
  before further use.
