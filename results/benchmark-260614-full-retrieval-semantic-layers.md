# Full benchmark report — semantic-layer + retrieval-pipeline evidence (v2)

Date: 2026-06-14 · benchmark `experiment/benchmark/v2` · embedder `text-embedding-3-large`
(OpenRouter, both arms — only indexed TEXT/pipeline differ) · semantic-card + triage LLM
`deepseek-v4-flash`. This consolidates the two benchmark LAYERS into one picture for review. Final
routing (agent decision) is a third, separate layer (CLAUDE.md §3) — included only as context.

Registry: 208 multi-type DBs (PG/Mongo/Neo4j) in `ours_multidb`; Sudarshan-repro SQL sets
`spider_route` (206 DB) + `bird_route` (80 DB). Slices stratified (cap N queries / GT-DB, every GT-DB
covered → DB-macro honest, per eval-slice-bias rule).

---

## LAYER 1 — Semantic card vs raw DDL (representation)

**Claim under test:** embedding a query-informed semantic card (domain_description + term_glossary +
entities + declared relations) retrieves the right DB better than embedding raw DDL (Sudarshan's
representation), and the gain is NOT a schema-name lexical-overlap artifact.

### 1a. Clean head-to-head (report 260611, stratified ≤5 q/GT-DB)

| Set | metric | raw | card | Δ | McNemar p |
|---|---|---|---|---|---|
| spider (206 DB, 1026 q) | R@1 | .614 | .644 | +.030 | .020 |
| | R@5 | .867 | .887 | +.019 | .036 |
| bird (80 DB, 398 q) | R@1 | .706 | .789 | +.083 | 3e-5 |
| | R@5 | .925 | .935 | +.010 | .50 (ceiling) |
| ours_multidb (208 DB, 743 q) | DBmacro R@1 | .566 | .613 | +.047 | .10 (borderline) |
| | DBmacro R@5 | .797 | .869 | +.072 | 8e-5 |

Card ≥ raw on every set/metric. raw-arm spider R@5 .867 ≈ Sudarshan published 87.0% → faithful
baseline reproduction.

### 1b. Obfuscation control — is the card edge just name leakage? (report 260611) — PASS

Mask schema identifier tokens (entity+field names) consistently across schema_text + card + inventory;
queries stay clean. Rebuild + re-eval same slice:

| Set | metric | raw clean | raw obf | card clean | card obf |
|---|---|---|---|---|---|
| spider | R@5 | .867 | **.043** | .887 | **.864** |
| bird | R@5 | .925 | **.123** | .935 | **.937** |
| ours | DBmacro R@5 | .797 | **.343** | .869 | **.836** |

Raw collapses (rides on name overlap); card is name-independent (drop ≈ 0). McNemar card_obf vs
raw_obf @5 all p≪1e-10. → the clean card edge is honest, not a name artifact.

### 1c. Query-side robustness — real peer-reviewed benchmarks (report 260611, SQL/Spider)

- **Spider-Syn** (synonym-substitute schema words in question): gap card−raw widens under synonyms;
  R@5 raw .824 / card .884; McNemar @1 p=1.4e-12.
- **Spider-Realistic** (drop explicit column mentions): R@1 raw .565 / card **.707** (+.142); p=8.7e-12.

→ card's value rises as query↔schema lexical overlap weakens — a real, benchmarked problem (the axis
DBCopilot tests on the query side). Complements 1b's schema-side masking.

**Scope (not overclaimed):** proves query-time name independence. Card is BUILT from clean schema
(legitimate build-time documentation), so this is NOT a "proxy-free end-to-end" claim. card-from-masked
(harsher) not run here.

---

## LAYER 2 — Retrieval pipeline: OURS (card + domain triage) vs SUDARSHAN (raw dense top-K)

**Claim under test:** our full retrieval pipeline hands the downstream reranker a candidate pool that
is higher-recall AND smaller than Sudarshan's plain top-5. (report 260614, stratified cap5/GT-DB.)

OURS = card embed top-10 → recall-protective domain-relatedness triage (agent keeps every
topically-related candidate, drops only clearly-unrelated; selects subset, no rank/winner → invariant)
→ picked subset. Triage input = domain_description only (ablation: desc keeps GT; glossary/entities
only tighten, no recall gain).

Recall reported as micro / DB-macro (≈ equal under the balanced cap-5 slice; DB-macro = headline).

| Set | method | avg pool | micro R | DB-macro R | McNemar OURS vs raw@5 |
|---|---|---|---|---|---|
| ours_multidb (743 q) | Sudarshan raw top-5 | 5.00 | .797 | .797 | — |
| | card top-10 | 10.0 | .917 | .925 | — |
| | **OURS card10→triage** | **4.28** | **.902** | **.914** | 95 vs 17, **p<1e-4** |
| spider (1026 q) | Sudarshan raw top-5 | 5.00 | .867 | .868 | — |
| | **OURS card10→triage** | **4.27** | **.942** | **.942** | 91 vs 15, **p<1e-4** |
| bird (398 q) | Sudarshan raw top-5 | 5.00 | .925 | .925 | — |
| | **OURS card10→triage** | **2.69** | **.947** | **.948** | 14 vs 5, p=.064 (ceiling) |

Triage gate-recall (GT kept | GT in card top-10): ours .984, spider .992, bird .974 — triage drops GT
in <3% of cases (real cost, reported; net recall still beats raw@5 because card@10 base recall is far
higher). Triage prompt = general keep-all rules (same-kind consistency / broader-contains / when-unsure-
keep), abstract only, NO test-derived examples. Remaining drops = benchmark single-destination ambiguity
+ label quirks + proper-noun/contentless queries (need entity-level, not prompt-fixable).

**Reading:** OURS recall ≥ Sudarshan raw@5 on all three, at a SMALLER pool (2.5–3.9 vs 5). Multi-type
set (thesis target): +9.5 pp recall with fewer candidates, significant. Two stacked sources: (1)
representation card>raw (Layer 1), (2) pipeline — wide net (top-10) recovers GTs top-5 missed, triage
narrows back to ~3 without re-losing them.

---

## LAYER 3 (context only) — final routing over the triaged pool

Behavior probe (100 q/set, directional): over the ~3 triaged candidates, agent-reads-cards (V3) beats
deterministic Coverage×Connectivity (V1) on all 3 DB types (ours .898 vs .739, spider .916 vs .747,
bird .935 vs .891 | GT-in-pool). Coverage saturates (25–47% ties) post-triage → does not add value;
kept as baseline. NOT a headline claim (needs full slice). See agent-flow-260614 report.

---

## Methodology summary (for review)

- Same embedder both arms; only indexed text differs (clean representation isolation).
- Stratified slices, every GT-DB covered; DB-macro headline on multi-type set.
- Significance: McNemar exact, paired; ceiling cases flagged (not hidden).
- Two-layer separation: retrieval recall never conflated with final-routing R@1.
- Controls: obfuscation (schema-side) + Spider-Syn/Realistic (query-side, peer-reviewed).
- Claim scoping: query-time name independence, NOT proxy-free end-to-end.
- Triage cost (gate-recall < 1.0) reported, not hidden.

## Review resolution (code-reviewer audit 2026-06-14)

Audit verified the comparisons hold up adversarially: same embedder + L2-normalized cosine both arms,
same DB pool (missing_semantic=[]), stratification applied + all GT-DBs covered, two-layer separation
clean, Layer-2 McNemar exact-binomial correct, gate-recall conditioning correct, raw-arm reproduces
Sudarshan 87.0% (baseline credible), claim scoping (query-time name independence, not proxy-free) correct.

Issues found + resolved:
- **F1 HIGH (fixed):** Layer-1 McNemar p-values lacked source code → added `retrieval_eval.py:mcnemar_p()`
  (exact two-sided binomial), emitted in `main()`.
- **F2 HIGH (fixed):** micro vs DB-macro recall now both reported + labeled in Layer-2 table; they
  coincide under the balanced cap-5 slice (so the 0.797 cross-layer number is consistent).
- **F3 MED (scoped):** desc-only triage sufficiency bounded to the 36-q probe; full-slice info-level
  verification flagged open.
- **F4/F5/F6 (disclosed):** triage empty-pool counted as conservative OURS miss; obfuscation masking is
  case-sensitive hence conservative (does not inflate card); question-last prompt ordering justified
  (full-context read before output, gate-recall .96–.99).

## Artifacts

- Reports: `retrieval-260611-0512-v2-headtohead-card-vs-raw.md`,
  `retrieval-260611-robustness-query-side-spidersyn-realistic.md`,
  `retrieval-260614-pipeline-triage-vs-sudarshan.md`, `agent-flow-260614-final-routing-variants.md`.
- Scripts: `exp_v2/{build_semantic,build_index,retrieval_eval,obfuscate,e3c_eval,robustness_query_eval,
  retrieval_pipeline_benchmark,triage_eval,agent_flow_eval}.py`.
- Indexes/caches: `benchmark/v2/<set>/{index,semantic,triage_cache,agent_flow_cache}/`.
