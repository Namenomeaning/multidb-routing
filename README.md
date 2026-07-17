# MultiDB-Route

**Natural-language query routing over heterogeneous (multi-model) databases.**

Given a natural-language question and a registry of database instances spanning different
data models — relational (PostgreSQL), document (MongoDB), and graph (Neo4j) — the system
routes the question to the single instance best able to answer it, *before* any query is
generated. This repository is the reproducibility bundle for the accompanying paper: it ships
the benchmark, the pipeline, the extraction/retrieval caches, and the result reports, so every
reported number can be re-derived without rebuilding anything.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green)
![Paper: under review](https://img.shields.io/badge/paper-under%20review-orange)

---

## Highlights

- **First multi-model routing benchmark.** *MultiDB-Route* — 208 databases across 3 data
  models, with ground-truth labels inherited deterministically from the source datasets
  (no hand annotation).
- **Cascade router, deterministic by construction.** Fast semantic retrieval + domain triage
  narrow the registry; a deterministic `Coverage × Connectivity` score ranks the survivors.
  The LLM only *extracts and selects* — it never emits a confidence number (the
  **no-self-scoring invariant**), so every route is reproducible.
- **Name-independent retrieval.** Each database is summarized into a *semantic card* that
  replaces raw DDL, making retrieval robust to schema-vocabulary drift and query rephrasing.
- **Connectivity beyond SQL.** The structural-connectivity signal is defined for document and
  graph schemas too, not just the relational foreign-key graph.
- **Reproducible.** LLM extraction / triage / tie-break steps replay from committed caches, so
  a clean reproduction costs only a few hundred cheap query embeddings.

## Method

The pipeline runs in two phases (shaded steps are LLM extract/select; the rest is
deterministic code):

**Offline (per database, once).** Parse the schema into entities + declared relations, then
build (i) a **semantic card** — a short domain description plus a glossary of ambiguous names —
and (ii) an **adjacency graph** of neighbour relations between entities.

**Online (per query).**

```text
question
  → parse into a shared phrase set (once)
  → retrieve: embed → top-10 candidate pool (semantic cards)
  → triage: LLM drops out-of-domain candidates → shortlist
  → score each candidate:  Coverage(e^{-n·x}) × Connectivity(∈{0,1})   [deterministic]
  → decide: clear leader → answer;  near-tie (δ) → LLM tie-break → answer
  → route to the single destination
```

## Repository layout

```text
multidb-routing/
├── src/                pipeline code, grouped by concern (run as `python -m src.<group>.<module>`)
│   ├── core/           building blocks: semantic_card · index · retrieval · rerank · scoring · openrouter (embed client)
│   ├── agent/          LangGraph agent: router · stages · benchmark
│   ├── workflow/       deterministic evals: final_routing · full_pipeline · hybrid · sudarshan · prompt_variants
│   ├── baselines/      comparison baselines: embedding · zeroshot
│   └── prompts/        every LLM prompt as a standalone .txt (loaded verbatim; cite-friendly)
├── data/
│   ├── multidb/        MAIN registry — 208 DBs, 3 models
│   ├── spider/         Spider-Route (206 PG) — single-model SQL reconstruction
│   └── bird/           BIRD-Route (80 PG)   — single-model SQL reconstruction
│                       each set: databases.jsonl, splits/, semantic/{cards,adjacency,inventory}.jsonl,
│                       index/{card,raw}.npy, and committed *_cache/ (LLM replay)
├── refs/sudarshan/     third-party baseline prompts (grounding)
├── results/            RESULTS.md (all reported numbers, one file) + figs/ (pipeline figure)
├── scripts/            dataset download + from-scratch build chain (provenance)
├── BENCHMARK.md        locked benchmark specification
└── pyproject.toml · uv.lock · .env.example
```

## Installation

```bash
uv sync                       # numpy, openai, python-dotenv, ...
cp .env.example .env          # then add your key
```

`.env` requires:

- `OPENROUTER_API_KEY=...` — **all** LLM and embedding calls route through OpenRouter.
- *(optional)* `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY=...` — to call DeepSeek directly for
  the cache-optimized extraction paths.

Model names are set via env/config, not hardcoded across files — change the model by changing
config, not code.

## Reproducing the results

The data, embedding index, and extraction caches are prebuilt — **nothing needs to be
downloaded or rebuilt.** Query embeddings are recomputed at eval time (a few hundred cheap
calls per set); the LLM extraction / triage / tie-break steps **replay from the committed
`*_cache/` directories**, so LLM cost on a clean replay is near zero.

Run everything from the repo root (`multidb-routing/`) with `python -m`:

```bash
# Retrieval layer — card vs. raw-DDL, recall@k + significance
python -m src.core.retrieval --bench data/multidb --index data/multidb/index
python -m src.core.retrieval --bench data/spider  --index data/spider/index
python -m src.core.retrieval --bench data/bird    --index data/bird/index

# Full pipeline (retrieval → deterministic rerank)
python -m src.workflow.full_pipeline  --set data/multidb
python -m src.workflow.final_routing  --set data/multidb   # final routing: triage → score → tie-break

# Single-model SQL baseline reconstruction
python -m src.workflow.sudarshan --set data/spider
python -m src.workflow.sudarshan --set data/bird
```

Scripts that accept `--score-only` re-score **entirely from cache with no API calls** — use that
for a zero-cost integrity check. Cross-check the printed metrics against `results/`; a
divergence means the data/cache wiring drifted (wrong set dir, missing cache) and should be
investigated before the numbers are trusted.

> **Reproducibility note.** This bundle was assembled from the original research tree: path
> plumbing was re-wired, but scoring, prompts, and formulas are byte-identical. The committed
> caches still warrant one real replay per set to confirm the rename did not break cache lookup
> or set selection — that replay is the true acceptance test.

## Agent routing (optional, live LLM)

An agentic realization of the same pipeline, built on **LangGraph**: the LLM issues the
`retrieve` / `inspect` / `answer` tool calls itself, selects which candidates to inspect, and
performs the tie-break — while reusing the *exact same deterministic scoring code* as above
(no forked scoring). It is provided as a feasibility demonstration, not as a separate reported
metric; because the agent controls candidate selection itself, its end-to-end numbers differ
from the deterministic pipeline.

```bash
# Quick validation slice (24 queries) + determinism re-check
LLM_PROVIDER=deepseek python -m src.agent.benchmark --cap 2 --max-dbs 12 --dup 6

# Full stratified benchmark (5 queries per gold DB = 1,040 on data/multidb)
LLM_PROVIDER=deepseek python -m src.agent.benchmark --cap 5 --out results/agent-bench-5db.jsonl
```

Unlike the cache-replay scripts, this makes **live** LLM calls (spends credit). Locked config:
`deepseek-v4-flash`, thinking off, `pool_k=10`, `max_turns=6`, Coverage `n=2.0`, tie `δ=0.2`.
Per-query rows go to `--out` (gitignored) for bootstrap CI / error analysis.

## Building from scratch (secondary path — unverified end-to-end)

Only needed to regenerate the benchmark from raw sources; **not** required for reproduction.

```bash
python scripts/download_datasets.py --list        # source manifest
python scripts/download_datasets.py               # fetch automatable sources → data/_raw/
# then follow scripts/build/README.md, then:
LLM_PROVIDER=deepseek python -m src.core.semantic_card   # semantic cards + adjacency
python -m src.core.index                                 # embedding index
```

⚠️ External endpoints/formats change, some sources need a manual/license download, and the
build scripts assume the original research layout and need `pandas`. Spot-check one source
before relying on this path.

## What is not shipped (and why)

- **Secrets** — only `.env.example`; never a live `.env`.
- **Raw source datasets** (`data/_raw/`, multi-GB) — regenerate via `scripts/download_datasets.py`.
- **Obfuscation-control artifacts** (name-masked variants) — deprecated control, omitted.
- **A second, dead routing implementation** and the scripts that import it — not part of this pipeline.

## Citation

If you use this benchmark or code, please cite:

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
- **`data/`** — artifacts derived from Spider, BIRD, DocSpider, MongoDB-EAI, CypherBench, and
  Text2Cypher, each under its own upstream license (CC BY-SA 4.0 for Spider/BIRD; see each
  source for the rest). Derived splits are redistributed for reproducibility under the source
  terms — confirm each source's conditions before further use.
