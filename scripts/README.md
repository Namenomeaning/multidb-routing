# Build-from-scratch scripts (provenance / audit)

These are the **original research build scripts**, copied here with their **contents verbatim**
(only the filenames were normalized to `snake_case`) so the dataset construction is auditable. They are **not a turnkey rebuilder** and were **not re-run** to
produce the artifacts shipped in `../data/` — those artifacts predate this folder.

> **The primary reproduction path is the prebuilt artifacts + committed caches** under
> `../data/{multidb,spider,bird}/`. See the top-level `README.md`. You only need anything
> here if you want to regenerate the benchmark from the raw source datasets.

## ⚠️ These scripts assume the ORIGINAL research layout

They use hardcoded relative paths from when the pipeline lived under `experiment/`:

- `tier/common.py`: `DATASET = <repo>/dataset`, `BENCH = <repo>/benchmark`
- `setups/build_ours_multidb.py`, `setups/build_sudarshan_repro.py`: read `benchmark/standard3_scale_v1`, write `benchmark/v2/...`
- `registry/augment_schemas.py`, `registry/update_neo4j_properties.py`: read/write `benchmark/multi/...`

Dropped into this clean repo those paths **dangle** (`dataset/`, `benchmark/` do not exist
here). To actually run them you must either recreate that directory layout or edit the path
constants. They are kept unedited on purpose — editing them un-run would risk silently
diverging from what produced the published numbers. Treat them as the **specification of how
the benchmark was built**, not as a script you can `python X.py` cold.

## The build chain (stage order)

The scripts are grouped into three functional stages; run order is top-to-bottom.
`download_datasets.py` (step 0) sits at the `scripts/` root.

```text
raw sources (data/_raw/ via download_datasets.py)
   │
   ├─ registry/   the 208-DB multi-model registry -> benchmark/multi/
   │    build_multi_corpus.py      raw + partition.json  -> benchmark/multi/instances.jsonl
   │    build_multi_queries.py     raw queries           -> benchmark/multi/queries.jsonl
   │    augment_schemas.py         LLM domain summaries  -> benchmark/multi/augmented-schemas.jsonl
   │    update_neo4j_properties.py enrich Neo4j schema_text from gold Cypher
   │
   ├─ tier/       merge PG (Spider+BIRD) + Mongo + Neo4j -> benchmark/standard3 -> standard3_scale_v1
   │    sources.py                 (uses common, bson_schema, bird_source)
   │    bird_source.py             pulls BIRD schema/questions via fetch_bird_json_remote.py
   │
   └─ setups/     cut the final Setup A/B benchmarks -> benchmark/v2/
        build_ours_multidb.py      standard3_scale_v1 -> benchmark/v2/ours_multidb   (Setup B, 208 DB)
        build_sudarshan_repro.py   spider+bird raw    -> benchmark/v2/sudarshan_repro/{spider_route,bird_route} (Setup A)
            │
            └─ then the semantic layer + embedding index are built by the REAL builders that
               ship in the pipeline (not here):
                 LLM_PROVIDER=deepseek python -m src.core.semantic_card   # cards + adjacency + inventory
                 python -m src.core.index                            # card.npy + raw.npy
               (these read a build dir under data/_build/standard3_scale_v1)
```

## Extra runtime dependency

`tier/sources.py` imports **pandas**, which is not in the top-level `pyproject.toml`
(the eval/runtime path does not need it). Install it before running the build chain:
`uv pip install pandas` (or `pip install pandas`).

## Excluded on purpose

The original `experiment/scripts/` also contained `build-semantic-layer.py`,
`bucket-scenarios.py`, `baseline-embedding-scenario60.py`, and `eval-m3-agent.py`. Those import
a second, **dead** routing implementation (`src/matching/ (removed)`) that is not part of this
published pipeline, so they are not copied here. The real semantic-layer builder is
``python -m src.core.semantic_card``.
