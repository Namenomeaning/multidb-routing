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

The deterministic two-tier flow routes a query over the **whole 208-database repository** with one
rule, no per-type branch. It is compared against two spectrum-end single-tier baselines on the same
1,040-question slice (paired McNemar).

| Method | R@1 |
|---|:--:|
| Embedding, raw schema | 0.568 |
| Embedding, card | 0.637 |
| Read whole repository (N=208), raw schema | 0.710 |
| Read whole repository (N=208), card | 0.703 |
| **Proposed — deterministic flow** | **0.749** |
| **Proposed — agentic variant (RQ4)** | **0.770** |

The deterministic flow is the significantly best single decision procedure. It beats embedding-card
by 0.112 (`p=3.2×10⁻¹²`) and beats reading the whole repository by 0.046 (`p=2.0×10⁻³`, under the
0.0125 threshold) — even though that baseline is handed the gold on every query while ours can lose
it at retrieval. Reading the whole repository in turn beats embedding-card by 0.066 (`p=3.9×10⁻⁶`).
Representation barely matters once the model reasons over full schemas: for the read-whole baseline
raw vs card differ by only 0.006 (`p=0.59`, n.s.), whereas at the embedding layer the card is
decisive (RQ1). The two levels combine as predicted: gold-in-pool 0.938 × `R@1|in-pool` 0.799 ≈
`R@1` 0.749.

**Cost.** The whole-repository baseline reads all 208 cards on every query; the flow reads only ~5
triaged candidates. On the slice, a token proxy puts the flow at ≈9.6M tokens vs the baseline's
≈20.0M (roughly half). Per-query context stays bounded at ~5 cards, while the baseline grows
linearly with the repository and its accuracy decays from 0.981 at N=5 to 0.703 at N=208. Routing
through retrieval + triage matches whole-repository accuracy at a fraction of the cost and scales
with repository size.

**By data model** (deterministic flow, 1,040-question slice):

| Data model | Queries | R@1 | R@1 \| in-pool |
|---|:--:|:--:|:--:|
| PostgreSQL | 470 | 0.830 | 0.874 |
| Neo4j | 135 | 0.756 | 0.810 |
| MongoDB | 435 | 0.660 | 0.712 |
| **Total** | **1,040** | **0.749** | **0.799** |

One rule routes strongly on PostgreSQL and Neo4j but drops on MongoDB. Most errors are at the
decision layer, not retrieval misses: the gold is in the candidate set but ranked below a
same-domain database of another model. Neo4j is indicative only (~80% of its questions are
machine-generated, a possible stylistic signal).

---

## RQ4 — Agentic variant

The deterministic flow is re-expressed as an LLM agent (a two-node LangGraph loop over retrieve /
inspect-schema / answer tools) while keeping the **no-self-scoring invariant** — the agent chooses
which candidates to inspect and when to stop, but Coverage × Connectivity still scores
deterministically; the answer tool only rejects out-of-pool ids. On the same 1,040-question slice
the agent reaches `R@1` **0.770**, above the deterministic flow's 0.749 and above both single-tier
baselines, without relaxing the invariant. The agentic path is run via the live agent runner
(`src/agent/`), so — unlike the cached deterministic figures — reproducing it issues live LLM calls.

---

*Figures for the pipeline are in `figs/` (`fig1-overall-flow`). The paper's main pipeline figure
is `our-routing-pipeline.pdf` at the thesis root.*
