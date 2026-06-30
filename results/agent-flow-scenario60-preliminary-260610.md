# Agent Routing Flow — Design + Preliminary Scenario Evaluation (2026-06-10)

**Scope.** One end-to-end run of the redesigned agent routing flow on
`standard3_scale_v1` (208 DB: PostgreSQL 94 / MongoDB 87 / Neo4j 27), evaluated on
60 queries split into three routing-difficulty scenarios (20 each). This is a
**single-arm preliminary run** (no baseline arm yet) — read accordingly.

Model: default OpenRouter chat model (DeepSeek v4 Flash). Retrieval: semantic
layer (LLM-generated cards + pseudo-queries), dense cosine, card vectors embedded
once and disk-cached.

---

## 1. Designed flow (each step developed from Sudarshan 2601.19825)

Sudarshan base pipeline = `embed -> top-K -> LLM decomposed Coverage x Connectivity
rerank -> top-1`. Our flow keeps that spine and extends each stage:

| Stage | Sudarshan base | Our flow | Grounding for the change |
|---|---|---|---|
| 0. Intent | (none — embeds raw question) | LLM schema-blind parse: phrases, value_literals, operations, relationship_required, domain_guess, engine_guess, expansion_terms | HyDE 2212.10496 / Query2Doc 2303.07678 / CRUSH4SQL 2311.01173 |
| 1. Recall | embed -> top-K cosine | enrich query (+expansion_terms +domain_guess) over the **semantic layer** (dense), coverage-guaranteed pool | HyDE + RouterRetriever 2409.02685 (soft, coverage-kept) |
| 2. Rerank | LLM extract -> Coverage e^(-n·x) × Connectivity BFS {0,1}, top-5 scope | same, multiplicative; intent reused from Stage-0 LLM (not regex); engine-neutral multi-type prompt | Sudarshan 2601.19825 |
| 3. Tie-break | average cosine | **agent reads finalists' evidence and decides**; refuses if none validated | thesis extension |
| 4. Rescue | (none) | on no validated candidate: expand_query + re-search (<=2 rounds), else abstain | FLARE 2305.06983 trigger analogy |

Invariant preserved: LLM does extraction only; all scoring is deterministic code
(`validated_score = Coverage × Connectivity`); no LLM self-confidence.

---

## 2. Scenario construction

Buckets defined on the **raw-question** semantic retrieval (the recall layer),
20 queries each:
- **T1 normal** — GT ranks #1 with a clear cosine margin.
- **T2 siblings** — GT in pool but ambiguous (rank > 1, or #1 with a tight margin
  to a same-domain lookalike).
- **T3 out-of-pool** — GT NOT in the raw retrieval pool → only enrichment +
  agent rescue can recover it.

**Confound (flagged):** engine and scenario are correlated. T1 = mongo/neo4j only;
T3 = 19/20 PostgreSQL (the cold-start PG instances with no support pseudo-queries).
So T3's failures cannot be cleanly separated from "PG cold-start + engine effect."

---

## 3. Preliminary results (two-layer, never conflated)

| Scenario | n | recall pool (GT in top-30) | recall package (GT in top-5) | final routing R@1 |
|---|---|---|---|---|
| T1 normal | 20 | 100.0% | 100.0% | 85.0% |
| T2 siblings | 20 | 100.0% | 95.0% | 70.0% |
| T3 out-of-pool | 20 | 90.0% | 5.0% | 0.0% |
| **Overall** | 60 | 96.7% | 66.7% | 51.7% |

Cost: 4.6 LLM calls/query. Abstentions (refused rather than guess): 16/60.

### Baseline comparison — M1 embedding similarity (added 2026-06-10, post-review)

Baseline #1 of the thesis (embedding cosine over the **same** semantic layer, raw
question, **no LLM at inference**) on the identical 60 queries. The regex path is
NOT a baseline (no scientific basis) and is not used.

| Scenario | M1 embedding final R@1 | M3 agent final R@1 | Δ (M3−M1) |
|---|---|---|---|
| T1 normal | 100.0% | 85.0% | **−15.0** |
| T2 siblings | 75.0% | 70.0% | **−5.0** |
| T3 out-of-pool | 0.0% | 0.0% | 0.0 |
| **Overall** | **58.3%** | **51.7%** | **−6.7** |

M1 recall: R@5 = 66.7%, R@30 = 96.7% — **identical** to M3's package/pool recall.

**Two hard, falsifying facts:**
1. **The agent flow is currently WORSE than pure embedding on final routing**
   (51.7% vs 58.3%; it loses on T1 and T2, ties on T3). The Devil's-Advocate
   counter-argument is confirmed by data: the added machinery is net-negative as
   calibrated. The cause is Bottleneck B — the strict validation gate refuses
   correct GTs that argmax-cosine picks (16 abstentions, mostly the difference).
2. **Enrichment shows NO recall gain.** M1 (raw, no enrich) R@30 = 96.7% equals
   M3 (enriched) recall pool = 96.7%; T3 R@30 = 90% in both. The earlier
   "enrichment recovers 18/20" reading was an artifact of the bucket definition
   (rank-in-pool vs rank>10), not a real lift over the raw embedding baseline.

### Findings

1. **Recall lever works (recall layer).** Enrichment + semantic retrieval put the
   GT into the 30-candidate pool for 96.7% overall — including **90% (18/20) of the
   T3 bucket where the raw-question retrieval had placed it out of pool**. This is
   the clearest positive signal: enrichment recovers GT the base retrieval missed.

2. **Bottleneck A — top-5 scope too tight on hard cases.** T3 recall pool 90% but
   package (top-5) only 5%: **17/20 T3 queries have GT in the top-30 pool but not in
   the top-5** fed to the reranker. The recall gain does not reach the LLM because
   Sudarshan's K=5 cosine cut excludes it. Final R@1 = 0% on T3 follows directly.

3. **Bottleneck B — strict validation over-rejects valid GT.** On T1/T2 the GT is
   in the top-5 with high `validated_score` (0.72–0.82) but is downgraded to
   "partial" by a single mapping-validation nitpick (e.g. LLM maps "campus"→Campus,
   validator finds no exact entity; `same_as` typed `unknown`; a relation edge not
   in the graph). Because `can_execute_query` requires answerability=="complete"
   AND zero validation errors, one rejection makes the GT non-answerable and the
   agent **refuses a correct answer**. This drives the T1 validation_error losses
   (3) and most T2 losses (4). The strict gate, at n=1 with binary connectivity and
   exact-match validation over schema-blind phrases, is mis-calibrated.

4. **Agent tie-break is sound where GT reaches it.** T1 85%, T2 70% final R@1 with
   the agent making the pick; it abstains rather than guessing when no candidate
   validates (16 abstentions, almost all T3) — desirable routing discipline.

5. **Rescue currently ineffective.** expand_query + re-search triggered on 16
   queries (9 of them T3) but recovered the GT into the final pick **0 times**.

---

## 4. What this does and does NOT show

**Shows:** the flow is wired end-to-end and Sudarshan-faithful; the recall
extension (LLM-parse enrich + semantic) measurably lifts GT-in-pool, including on
the bucket where base retrieval failed; the agent tie-break and abstention behave
sensibly; two concrete, fixable bottlenecks are localized with numbers.

**Does NOT show (honest limits):**
- **No "better than Sudarshan" claim is supported yet.** Only one arm was run.
  There is no baseline arm (raw question / no-enrich / multichannel / regex intent)
  on the same 60, so no Δ, no paired test, no CI. "Better" is unproven.
- **Engine↔scenario confound** means T3=0% conflates difficulty with PG cold-start.
- **Single model, n=20/bucket, no obfuscation control, no bootstrap CI.** All
  numbers are preliminary and would move under those controls.
- The strict-scoring and tight-K bottlenecks mean current final R@1 understates the
  design's ceiling; they are calibration/wiring issues, not evidence against the
  hypothesis.

**Next to actually prove "better":** baseline arm A0 (raw+no-enrich+regex) vs full
flow on the same 60 (paired, two-layer); widen package K or feed the recall pool
the GT reaches; loosen the answerability gate (vscore threshold instead of
zero-error); de-confound by adding non-PG T3 and warm PG T3.
