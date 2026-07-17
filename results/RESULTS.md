# MultiDB-Route — Results

Single source of truth for the numbers reported in the paper. Every figure below is
reproducible from the shipped data, index, and caches (see the repo `README.md`).
Two levels are always reported separately:

- **Retrieval** — `recall@k` (DB-macro), paired with the average candidate-set size (recall
  rises with pool size, so the two are read together).
- **Final routing** — `R@1` and `R@1 | in-pool`. Relation: `R@1 ≈ gold-in-pool × R@1|in-pool`.

**Configuration.** LLM `deepseek-v4-flash` (temperature 0), embeddings `text-embedding-3-large`,
both via OpenRouter. Coverage penalty `n=2`, tie margin `δ=0.2`, retrieval top-10 triaged to ~5.
Scoring is deterministic and exactly reproducible given the same extracted input (the LLM only
extracts/selects; it never emits a confidence number). Significance = paired McNemar, 95% CIs by
bootstrap; with four baseline comparisons on one slice the Bonferroni threshold is ≈0.0125.

---

## Benchmark composition

**MultiDB-Route** — 208 databases over three data models (routing GT inherited deterministically
from the source datasets; no hand annotation).

| Metric | PostgreSQL | MongoDB | Neo4j | Total |
|---|--:|--:|--:|--:|
| Databases | 94 | 87 | 27 | **208** |
| Retrieval queries | 894 | 863 | 266 | **2,023** |
| Routing queries | 470 | 435 | 135 | **1,040** |

Sources — PostgreSQL: Spider 68 + BIRD 26; MongoDB: DocSpider 72 + MongoDB-EAI 8 +
mongosh-instructions 7; Neo4j: Text2Cypher-2024 16 + CypherBench 11.

**Reconstructed SQL setting** (paired head-to-head vs the baseline, distinct from the balanced
MultiDB-Route slice): Spider-Route = 206 DBs / 5,939 test queries; BIRD-Route = 80 DBs / 5,501
test queries.

Slices are stratified by database (≥10 questions per DB) so DB-macro accuracy is not dominated by
question-rich databases. A disjoint 876-question dev set tunes prompts, then is frozen. The
support set seeds the offline phase only — no parameters are learned from it; ~20 PostgreSQL DBs
without support questions fall back to schema-only cards.

---

## RQ1 — Retrieval representation (semantic card vs raw schema)

`recall@5`, no triage (Raw schema = the top-5 baseline). *Full flow* = recall of top-10 with
triage; *Avg. cand.* = average candidates handed to scoring.

| Set | Raw schema | Card | Full flow | Avg. cand. |
|---|:--:|:--:|:--:|:--:|
| Spider | 0.876 | 0.899 | **0.945** | 5.45 |
| BIRD | 0.943 | 0.954 | **0.959** | 2.70 |
| Multi-model | 0.832 | 0.880 | **0.928** | 4.88 |

The card beats raw schema on all three sets; significant on Spider and multi-model (`p<10⁻⁴`),
BIRD near the ceiling (0.011 gap, `p=0.053`, n.s.). The advantage is semantic, not name-overlap:
under noise the gap **widens**. Under synonyms the card holds `recall@5` 0.884 vs 0.824 (clean
0.960 / 0.938 on the same questions) — raw schema's drop (−0.114) is nearly twice the card's
(−0.076). With column names removed, the card reaches `R@1` 0.707 vs 0.565. Robustness evidence
is so far SQL-only.

---

## RQ2 — Pipeline design on SQL (vs baseline)

**Retrieval.** Top-10-with-triage gives scoring a richer set at comparable size while retaining
the gold (triage keeps the gold: 0.991 Spider, 0.979 BIRD, 0.989 multi-model). Full flow wins
6.9 points on Spider and 9.6 on multi-model over raw top-5.

**Final routing.** After triage the set is in-domain, so Coverage × Connectivity saturates and
25–47% of questions tie — most of the remaining difference falls to the tie-break. On 402
tie-only questions, the LLM reading cards reaches 0.668 vs the baseline's cosine tie-break 0.498
(`p=2.2×10⁻⁴`). Head-to-head, isolated rerank (shared top-5, triage off both sides), `R@1|in-pool`:

| Set | Baseline | Ours |
|---|:--:|:--:|
| Spider | 0.707 | **0.808** |
| BIRD | 0.793 | 0.817 |

Spider: clear and significant (`p=4.3×10⁻⁶`). BIRD: both at the ceiling, 0.024 gap **not**
significant (`p=0.358`) — reported as parity, not a win. With triage back on, `R@1|in-pool` is
essentially unchanged while overall `R@1` rises (Spider 0.715→0.769) from better retrieval, not
the decision layer. Baseline fidelity check: reconstruction reaches raw `recall@5` 0.874 on
Spider, near the original ~87%.

---

## RQ3 — Multi-model routing (the main task)

One rule routes a query over the **whole 208-database repository**, no per-type branch. Compared
against two spectrum-end baselines on the same 1,040-question slice (paired McNemar).

| Method | R@1 |
|---|:--:|
| Embedding, raw schema | 0.568 |
| Embedding, card | 0.653 |
| Read whole repository (N=208), raw schema | 0.692 |
| Read whole repository (N=208), card | 0.697 |
| **Proposed (full pipeline)** | **0.727** |

The proposed pipeline beats both embedding baselines (`p=2.6×10⁻⁶` vs card, `1.2×10⁻¹⁹` vs raw).
Against reading the whole repository it is statistically **level** under the 0.0125 threshold
(0.030 gap vs card `p=0.055`; 0.035 vs raw `p=0.034`) — even though that baseline is handed the
gold on every query while ours can still lose it at retrieval. The two levels combine as
predicted: gold-in-pool 0.941 × `R@1|in-pool` 0.772 ≈ `R@1` 0.727.

**Cost.** The accuracy tie hides a cost gap: the baseline reads all 208 cards on every query;
ours reads only ~5 triaged candidates. On the slice, a token proxy puts ours at ≈9.6M tokens vs
the baseline's ≈20.0M (roughly half). Per-query context stays bounded at ~5 cards, while the
baseline grows linearly with the repository and its accuracy decays from 0.981 at N=5 to 0.697 at
N=208. Routing through retrieval + triage matches whole-repository accuracy while scaling with
repository size.

**By data model** (1,040-question slice):

| Data model | Queries | R@1 | R@1 \| in-pool |
|---|:--:|:--:|:--:|
| PostgreSQL | 470 | 0.800 | 0.837 |
| Neo4j | 135 | 0.815 | 0.853 |
| MongoDB | 435 | 0.621 | 0.673 |
| **Total** | **1,040** | **0.727** | **0.772** |

One rule routes strongly on PostgreSQL and Neo4j but drops on MongoDB. Most errors are at the
decision layer, not retrieval misses: the gold is in the candidate set but ranked below a
same-domain database of another model. Neo4j is indicative only (~80% of its questions are
machine-generated, a possible stylistic signal).

---

*Figures for the pipeline are in `figs/` (`fig1-overall-flow`). The paper's main pipeline figure
is `our-routing-pipeline.pdf` at the thesis root.*
