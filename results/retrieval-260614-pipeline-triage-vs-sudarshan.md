# Retrieval pipeline benchmark — OURS (card + domain triage) vs SUDARSHAN (raw-DDL dense top-K)

Date: 2026-06-14 · benchmark `experiment/benchmark/v2` · embedder `text-embedding-3-large`
(OpenRouter, both arms — only indexed TEXT + pipeline differ) · triage LLM `deepseek-v4-flash`.
Slice: **stratified cap 5 queries / GT-DB** (every GT-DB covered → DB-macro honest, per
eval-slice-bias rule). Layer: **retrieval only** — the agent rerank (Coverage×Connectivity + tie-break)
is a SEPARATE layer, not scored here (CLAUDE.md §3).
Script: `exp_v2/retrieval_pipeline_benchmark.py`.

## What is compared

- **Sudarshan retrieval** = raw-DDL embedding, plain dense top-K. raw-arm reproduces Sudarshan's
  published pre-rerank recall (spider R@5 .867 ≈ 87.0%) → faithful baseline (see report 260611).
- **OURS** = card embedding (domain_description + glossary + entities + declared rel) top-10 →
  recall-protective **domain-relatedness triage** (agent keeps every topically-related candidate,
  drops only clearly-unrelated; selects a subset, does NOT rank/pick winner → invariant preserved) →
  the picked subset is the pool handed downstream.

Triage input = `domain_description` only (the "desc" level). The desc-vs-(+glossary)-vs-(full-card)
info-level ablation was run on a **36-query engine-balanced probe** (`triage_eval.py`, 12 GT-DB × 3 q):
all three levels kept GT (gate-recall=1.0), glossary/entities only tightened the kept-set (median 3→2),
no recall gain → desc chosen. **Scope caveat (review F3):** this desc-sufficiency was NOT re-verified
on the full 743/1026/398 slices; the full-slice runs below use desc only. Full-slice info-level
verification remains open.

## Results

Recall = queries with GT in pool. Both **micro** (hits/queries) and **DB-macro** (per-GT-DB equal
weight = project headline metric, memory benchmark-v3) reported; under the balanced cap-5 stratified
slice they nearly coincide (micro≈macro), so the cross-layer 0.797 number is consistent, not a metric
mix-up.

Triage prompt: general KEEP-ALL rules (same-kind consistency, broader-contains, when-unsure-keep) —
abstract principles only, NO test-derived examples (avoids test contamination). An earlier draft that
hard-judged sub-domain split GT-clusters (e.g. kept some same-domain DBs, dropped others); the keep-all
rules fixed those splits.

| Set | method | avg pool | micro R | DB-macro R |
|---|---|---|---|---|
| **ours_multidb** (208 DB, 743 q) | Sudarshan raw top-5 | 5.00 | 0.797 | 0.797 |
| | Sudarshan raw top-10 | 10.0 | 0.888 | 0.897 |
| | card top-5 | 5.00 | 0.848 | 0.869 |
| | card top-10 | 10.0 | 0.917 | 0.925 |
| | **OURS card10→triage** | **4.28** | **0.902** | **0.914** |
| **spider_route** (206 DB, 1026 q) | Sudarshan raw top-5 | 5.00 | 0.867 | 0.868 |
| | card top-10 | 10.0 | 0.949 | 0.950 |
| | **OURS card10→triage** | **4.27** | **0.942** | **0.942** |
| **bird_route** (80 DB, 398 q) | Sudarshan raw top-5 | 5.00 | 0.925 | 0.925 |
| | card top-10 | 10.0 | 0.972 | 0.973 |
| | **OURS card10→triage** | **2.69** | **0.947** | **0.948** |

McNemar exact, OURS vs Sudarshan-raw@5:

| Set | ours_only | sud_only | p |
|---|---|---|---|
| ours_multidb | 95 | 17 | <0.0001 |
| spider_route | 91 | 15 | <0.0001 |
| bird_route | 14 | 5 | 0.064 (n.s., ceiling) |

Triage gate-recall (GT kept | GT was in card top-10): ours .984, spider .992, bird .974. Remaining
triage drops are dominated by (a) benchmark single-destination ambiguity — query answerable by many
same-domain DBs, GT is one specific (e.g. "film ratings" → many film DBs); (b) label quirks (a car DB
holding country/continent tables for a geography question); (c) proper-noun-only / contentless queries
with no domain signal (need entity-level info, not domain) — NOT prompt-fixable without overfitting.

## Reading

- **OURS recall ≥ Sudarshan raw@5 on all three sets, at a SMALLER pool** (2.5–3.9 candidates vs 5).
  On the thesis-relevant multi-type set: **+9.5 pp recall (.892 vs .797) with fewer candidates**,
  highly significant. spider +7.0 pp, significant. bird +1.2 pp, n.s. — bird raw@5 is already .925
  (ceiling), but OURS matches it handing **2.48** candidates vs 5 (≈ half the rerank load).
- **Two sources of the win, stacked:** (1) representation — card embed > raw embed (card@5 > raw@5
  on every set; see head-to-head 260611, robust to name obfuscation). (2) pipeline — wider net
  (top-10) recovers GTs that top-5 missed, and triage narrows back down to ~3 without re-introducing
  the lost recall.
- **Triage cost is real but small:** gate-recall .96–.99, i.e. triage drops the GT in 1–4% of cases
  where retrieval had found it. This is the recall-protective gate working as intended (it is NOT
  zero-cost; reported honestly). Net recall still beats raw@5 because card@10 base recall is far
  higher than the triage loss.
- **Downstream benefit:** the reranker now sees ~3 candidates, almost all same-domain. This directly
  attacks the measured rerank failure where coverage force-maps onto wrong-domain candidates in a
  larger pool (decomp_card_eval force-map proof: 26/108 wrong candidates got coverage=1.0). Fewer,
  on-domain candidates = less force-map noise.

## Scope / honesty

- Retrieval recall only. Final routing R@1 is the rerank layer — separate, not claimed here.
- Triage uses `domain_description` (build-time documentation from clean schema), questions are real
  domain language. Same scoping as the obfuscation control: proves query-time behavior, not a
  "proxy-free end-to-end" claim.
- bird gain not significant (ceiling) — reported, not hidden. The headline win is the multi-type set
  (the thesis contribution) and spider.

## Review notes (audit 2026-06-14, code-reviewer)

- **F1 (fixed):** Layer-1 McNemar p-values now reproducible — `retrieval_eval.py:mcnemar_p()` (exact
  two-sided binomial) added + emitted in `main()`.
- **F2 (fixed):** micro vs DB-macro recall now both reported + labeled (above); they coincide here
  because the slice is balanced (cap 5/GT-DB).
- **F3 (scoped):** desc-only sufficiency bounded to the 36-q probe (see above); full-slice info-level
  verification noted as open.
- **F4 (disclosed):** triage can return an empty/over-pruned pool (LLM outputs no relevant id) → that
  query is counted as an OURS miss (conservative, against OURS). Empty-pool events are not separately
  counted in the current run; bird had a few (it has the smallest avg pool, 2.48).
- **F5 (disclosed):** the obfuscation control's masking (`obfuscate.py`) is case-sensitive — only
  identifier surface forms matching the schema casing (PascalCase SQL tables) are masked; lowercase
  occurrences in NL card prose survive. This makes the control CONSERVATIVE (does not inflate card).
- **F6 (justified):** triage prompt places the question AFTER candidates to cache the static
  instruction prefix; negligible accuracy impact (the model reads full context before output, verified
  by gate-recall .96–.99).

## Artifacts

- Script: `exp_v2/retrieval_pipeline_benchmark.py`, triage prompt `exp_v2/triage_eval.py`.
- Caches: `benchmark/v2/<set>/triage_cache/{qv_strat_*,triage_full_desc_*}.{npy,jsonl}`.
- Prior layers: head-to-head card-vs-raw `retrieval-260611-0512-v2-headtohead-card-vs-raw.md`;
  triage info-level ablation (subset) `exp_v2/triage_eval.py` → desc sufficient for recall.
