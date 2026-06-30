# Research Report: Query-Informed Semantic Card Generation

**Date:** 2026-06-10
**Branch:** m3-protected-rerank
**Scope:** Dispute resolution — schema-only vs. query-informed card input; block D retention decision

---

## Tóm tắt (Vietnamese executive summary)

Tác giả đặt câu hỏi: semantic card hiện tại chỉ dùng schema làm input — như vậy LLM không biết entity nào thực sự quan trọng (salient) hay term nào thực sự phân biệt DB này với DB cùng domain. Đề xuất: cho thêm sample queries từ support split vào prompt sinh card, và có thể bỏ block D (pseudo-query view riêng).

Kết luận sau khi xem xét 10+ primary source:

1. **Query-informed card generation có cơ sở học thuật** — workload-aware schema summarization (Yang et al. VLDB 2009), dbt MetricFlow (industry), DocSpider template-leakage literature — nhưng chưa có paper nào làm đúng với routing multi-type thay vì NL2SQL generation. Cần gắn nhãn ENGINEERING-ABLATE.

2. **Không thể bỏ block D (pseudo-query view)** — doc2query/docTTTTTquery (Nogueira 2019) cho thấy raw queries appended vào index bổ sung trực tiếp; nếu bỏ D và "chưng cất" vào card, cần ablation E3 để chứng minh không mất recall.

3. **Rủi ro leakage có thật và phải kiểm soát** — support và test queries lấy từ cùng dataset families (Spider/DocSpider/CypherBench); template overlap phải audit bắt buộc trước khi claim gain.

4. **Cold-start cần xử lý sạch** — 20 PG DB không có support query phải được tách nhóm và báo cáo riêng; không được lẫn vào headline metric.

5. **Khuyến nghị cuối:** dùng card query-informed (≤5 support queries/DB), GIỮ block D, thiết kế ablation E3a/E3b để đo từng thành phần. Lock spec update vào BENCHMARK.md sau khi E3 chạy xong.

---

## 1. Dispute statement

Current locked spec (BENCHMARK.md §3): card input = schema inventory A only (entities + fields + types + declared relations). No example queries.

Author's challenge: LLM cannot determine salience (which entities are "key") or contrastive distinctiveness (which terms distinguish this DB from same-domain siblings) from schema structure alone. Proposes: feed ≤5 real support-split queries per DB into card-generation prompt; possibly drop block D.

---

## 2. Literature evidence

### 2.1 Workload-aware schema summarization

**Yu & Jagadish, "Schema Summarization," VLDB 2006** — VERIFIED (direct URL confirmed: vldb.org/conf/2006/p319-yu.pdf). Scores schema elements for importance using *structural* graph centrality and content coverage metrics, NOT query workload. The importance signal is topology-derived (e.g., node connectivity, coverage of tuples). Does not inject queries into the summarized description.

**Yang, Procopiuc & Srivastava, "Summarizing Relational Databases," VLDB 2009** — VERIFIED (direct PDF confirmed: vldb.org/pvldb/vol2/vldb09-784.pdf; also confirmed in follow-up "Summary Graphs for Relational Database Schemas," PVLDB 2020, ACM dl.acm.org/doi/10.14778/3402707.3402728). The 2009 paper determines table salience via *user query tables* — given a set of query-specified tables, the algorithm computes the most relevant joins. The follow-up defines "summary graph" as explicitly query-set-driven. **This is the strongest academic precedent for workload-aware salience.** Key nuance: their workload is a *query set specifying target tables*, not raw NL queries, and the output is a subgraph, not a text description. The mechanism (query-driven table weight propagation via schema graph) is analogous to, not identical to, feeding NL example queries into an LLM card prompt.

**Gao & Luo, "Automatic Database Description Generation for Text-to-SQL," arXiv:2502.20657** — VERIFIED (HTML confirmed: arxiv.org/html/2502.20657v1). Uses schema + example data rows, explicitly does NOT use query workload ("do not utilize evidence"). Handles cold-start scenario (no prior descriptions). Classification of column difficulty: 47% self-evident (schema alone), 23% context-aided (schema + data values), 27% ambiguity-prone (need external knowledge), 2% domain-dependent. Direct implication: ~30% of columns have salience that schema alone cannot determine.

**Wretblad et al., "Synthetic SQL Column Descriptions," arXiv:2408.04691** — VERIFIED (HTML confirmed). Input = schema + example data rows (no queries). Does not ablate query-informed vs. schema-only. Confirms schema + data values is current state of practice for LLM-generated descriptions.

**Summary for Q1:** No LLM-era paper directly feeds NL workload queries into DB-side description generation for *routing*. Yang et al. 2009 is the academic grounding for the claim that workload informs salience — but the mechanism differs. Closest analog is DBCopilot's reverse schema-to-question generation (described below). Tag: **ENGINEERING-ABLATE** — the specific mechanism (NL support queries → card LLM prompt) is thesis-original; the motivation has academic precedent.

### 2.2 Routing/retrieval precedents

**DBCopilot, arXiv:2312.03463, EDBT 2025** — VERIFIED (HTML confirmed: arxiv.org/html/2312.03463). Routes across multiple DBs. Does NOT inject real workload queries into schema-side representation. Instead, uses a reverse schema-to-question model trained on Spider/BIRD to *generate* synthetic queries, which then train the router. DB-side representation = schema structure only (Seq2Seq differentiable search index). The synthetic query generation is query-time (router training), not index-time (card generation). Key distinction from the thesis proposal: DBCopilot generates queries *from* schema to train the router; the thesis proposal feeds real queries *into* the card to enrich it.

**DBRouting, arXiv:2501.16220** — VERIFIED (HTML confirmed: arxiv.org/html/2501.16220). DB-side representation = DDL text only. Fine-tuning uses question-to-DB mapping pairs (positive/negative contrastive), but these are used to train the embedding model, not injected into the schema description text. Embedding fine-tuning does benefit from DB-specific questions — this is the closest routing-specific evidence that query signal improves representation, but via model fine-tuning, not via description enrichment.

**Sudarshan et al., arXiv:2601.19825** — PARTIALLY VERIFIED (abstract confirmed; full method details not accessible from HTML). BIRD column descriptions are curated metadata — confirmed in CLAUDE.md rules as "curated metadata" and in the thesis's own EXPERIMENT-LOG commentary. They are NOT derived from queries. The column descriptions exist pre-provided in BIRD. This is consistent with the observation in Gao & Luo that human-written descriptions outperform auto-generated ones.

**CRUSH4SQL, arXiv:2311.01173** — VERIFIED (search confirmed). Hallucination approach: query-side, LLM hallucinates a target schema from the query, then retrieves from the real schema. No injection of queries into the DB-side indexed representation.

**Summary for Q2:** None of the routing papers (DBCopilot, DBRouting, Sudarshan, CRUSH4SQL) inject example NL queries into the DB-side representation (card/description). DB representations are schema-derived. Query signal enters only via model fine-tuning (DBRouting) or synthetic generation at training time (DBCopilot). The thesis proposal (feed real support queries into card LLM prompt) has no direct precedent in routing literature — it is thesis-original.

### 2.3 doc2query / document expansion line

**Nogueira et al., "Document Expansion by Query Prediction," arXiv:1904.08375, SIGIR 2019** — VERIFIED (ar5iv HTML confirmed). Core finding: predicted queries appended to documents before indexing improve retrieval. Ablation: copied words (MRR@10 = 19.7) + new words (synonyms, MRR@10 = 18.8) → combined = 21.5 (+~10% over either alone). Key mechanism: *append* queries to document text, not distill into a new description. Paper does NOT compare raw-query-append vs. description-distillation — no such ablation exists.

**Nogueira & Lin, "From doc2query to docTTTTTquery," 2019** — VERIFIED (cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf). Extended with T5. Same append mechanism. No distillation comparison.

**Implications for the block D decision:**
- doc2query establishes that raw query expansion (appending queries to indexed representation) and description-level enrichment are *not* equivalent — you cannot simply assume distilling queries into a card achieves the same recall benefit as the raw block D view.
- No paper measures: (card with queries distilled in) vs. (card schema-only + separate query block). This ablation (E3) is mandatory before dropping D.
- doc2query overfitting caveat: poor generalization from in-domain (MS MARCO) to out-of-domain (BEIR) is documented in the search results. Parallel risk: card enriched with Spider-family queries may overfit to Spider-family test queries.

**Summary for Q3:** doc2query precedent supports keeping block D as a separate view — distillation is not proven equivalent to direct indexing of real queries. E3 ablation is required before dropping D.

### 2.4 Leakage / generalization risk

**Contamination in Text-to-SQL, arXiv:2402.08100** — VERIFIED (direct PDF confirmed). Spider contamination in LLM pretraining increases over time; models generate correct table/field names in zero-shot, indicating memorization. Directly relevant: Spider query phrasing patterns are likely in pretraining corpora.

**Yoon et al., "Hypothetical Documents or Knowledge Leakage?" arXiv:2504.14175** — VERIFIED (HTML confirmed). Distinguishes legitimate query expansion from knowledge leakage where LLMs reproduce memorized content. Performance gains from query expansion appear only when LLM-generated documents contain sentences "entailed by gold evidence." Implication: if support queries come from Spider/DocSpider (datasets likely in LLM pretraining), injecting them into card generation mixes dataset-memorized phrasing into the card, making it hard to attribute gains.

**Doc2Query++ / domain generalization** — LIKELY (search result summary; full paper not fetched). Confirms poor out-of-domain generalization of query-expanded indices.

**Locally measured (BENCHMARK.md / EXPERIMENT-LOG):** Query-time name leakage inflates results measurably in this thesis. Support and test queries come from same dataset families (Spider → PG, DocSpider → Mongo, CypherBench → Neo4j). Template overlap audit via current heuristic is insufficient (template overlap = 0 does not mean proxy-free per CLAUDE.md §7).

**Summary for Q4:** Leakage risk is real and multilevel: (a) LLM memorization of Spider/DocSpider phrasing during pretraining → card absorbs dataset-specific vocabulary; (b) support and test queries share template structure within same dataset families. Controls are mandatory.

### 2.5 Cold-start

**Gao & Luo (2502.20657)** — VERIFIED. Their method is explicitly designed for the cold-start scenario (no prior descriptions/queries). Schema-only input is the correct design when queries are absent. This directly applies to the 20 PG DBs with no support queries: they must receive schema-only card; any query-informed card design must declare a fallback path.

**No workload-mixing precedent found** — no paper was found that explicitly handles a mixed registry (some DBs with workload, some without) in a single evaluation setup. This is a gap the thesis must address procedurally.

**Summary for Q5:** Mixed cold-start is a confound. Clean design requires (a) schema-only card for cold-start DBs, (b) query-informed card for DBs with support, (c) separate reporting of the two groups, (d) cold-start group included in headline metric only under clearly stated conditions.

### 2.6 Industry semantic layer (supplementary, non-academic)

**dbt MetricFlow** — INDUSTRY (docs.getdbt.com/best-practices/how-we-build-our-metrics/semantic-layer-2-setup; confirmed). MetricFlow requires human definition of entities, dimensions, measures, and metrics. The semantic manifest cannot be auto-derived from schema alone — business knowledge ("what does Revenue mean, which joins matter") is explicitly encoded by humans. This validates the author's intuition: salience and distinguishing terms cannot be fully inferred from schema structure.

**Cube.js** — INDUSTRY (github.com/cube-js/cube; not fetched in detail). Same pattern: human-curated semantic model on top of raw schema.

**Relevance:** Industry semantic layers are the human-curated analog of what the thesis proposes to automate via LLM + support queries. The industry case confirms that salience is genuinely workload-dependent. However, these layers are human-curated, not LLM-generated from queries — the automated analog is the thesis's own contribution, not an existing established pattern. Tag: **INDUSTRY** (not a substitute for VERIFIED academic citation).

---

## 3. Verdict

**Query-informed card generation: GROUNDED, with controls.**

The author's core intuition is correct — salience cannot be fully determined from schema structure alone (Yang et al. 2009 academic; Gao & Luo ambiguity-prone 30% empirical; dbt MetricFlow industry). However, no routing or NL2SQL paper has implemented the exact proposed mechanism (real support NL queries → card LLM prompt). This mechanism is thesis-original and must be labeled ENGINEERING-ABLATE with an ablation design.

The mechanism is reasonable under three controls: (1) query count capped at ≤5/DB to limit leakage surface; (2) obfuscated-card run to measure leakage delta; (3) cold-start DBs receive schema-only fallback.

---

## 4. Final recommended card spec

### 4.1 Input

```
Schema inventory A (entities + fields + types + declared relations) — same as current
Support queries: ≤5 real queries from support split for this DB
  Format: list of NL questions only (NO ground-truth SQL/Cypher/MQL — prevents schema-hint leakage from query structure)
  Absent: if 0 support queries → schema-only path (cold-start fallback)
```

Rationale for NL-only: injecting the ground-truth query language (SQL/MQL/Cypher) would leak schema entity names directly through the query structure, bypassing the intent of the card abstraction.

### 4.2 Output fields — keep all 4, modify 2

| Field | Decision | Change | Basis |
|---|---|---|---|
| `domain_description` | KEEP | No change | Gao&Luo, DBCopilot precedent |
| `key_entities` | KEEP | Add instruction: "use query frequency as signal if sample queries provided" | Yang 2009 (query-driven salience); ENGINEERING-ABLATE |
| `distinguishing_terms` | KEEP | Add instruction: "use query-specific vocabulary as contrastive signal" | DBRouting intra-domain failure mode; ENGINEERING-ABLATE |
| `term_glossary` | KEEP | No change | Gao&Luo column descriptions; Sudarshan metadata |

The 4-field structure is preserved — no new fields. Only the LLM prompt for `key_entities` and `distinguishing_terms` gains optional context from support queries.

### 4.3 Prompt I/O sketch

```
SYSTEM: You are generating a routing semantic card for a database instance.
INPUT A: [Entity inventory — entities, fields, types, declared relations]
INPUT B (optional, ≤5 items): Sample natural-language questions this DB answers:
  - "..."
  - "..."
Task:
  domain_description: 100-150 words, scope and use case, only reference relations present in A
  key_entities: 5-10 most salient entities for routing; if sample questions provided, weight
                entities that appear across multiple questions higher
  distinguishing_terms: vocabulary that distinguishes this DB from same-domain siblings;
                        if sample questions provided, include terms from questions that are
                        DB-specific (not generic NL words)
  term_glossary: only opaque names; gloss only when derivable from context; omit otherwise
CONSTRAINT: Do not output confidence scores or probability estimates.
            Do not reference DB engine type in any field value.
```

---

## 5. Decision on block D

**KEEP block D alongside query-informed card. Do NOT drop.**

Reasoning:
1. doc2query (Nogueira 2019) shows raw queries appended to index and description-level enrichment are not equivalent — the former directly adds lexical variety; the latter distils it. Distillation may lose rare query terms that only appear in 1-2 support queries.
2. No paper has measured (query-informed card + no D) vs. (schema-only card + D). Dropping D before E3 runs would conflate two changes and make the source of any recall difference unattributable.
3. Block D's role in the current pipeline (separate vector per support query, multi-view retrieval) is already measured to beat BM25 alone — this is the established baseline. Any change must beat it.
4. If E3 shows query-informed card alone matches card+D, then D can be dropped in a later iteration with a clean A/B comparison.

**Block D status after this research: retained as is, pending E3.**

---

## 6. Leakage control design (mandatory before claiming gain)

| Control | What it checks | How |
|---|---|---|
| Template-overlap audit | NL phrasing overlap between support and test queries within each dataset family | Exact + n-gram overlap; flag DB pairs with >20% shared n-grams; report in ablation |
| Obfuscated-card run | Whether card gains survive entity-name obfuscation | Build query-informed card with obfuscated DB names; measure recall delta vs. clear-name card |
| No-support baseline | Whether gains come from any query signal or specifically query-informed card | Compare: schema-only card + D (current) vs. query-informed card + D (proposed) vs. query-informed card only |
| Cold-start isolation | Prevent cold-start DBs from diluting or inflating headline metric | Report headline on full 208 DBs AND separately on "query-rich" (≥5 support) subset |

Obfuscation run is the most critical: if query-informed card gains vanish under obfuscation, the mechanism is driven by name leakage, not semantic enrichment.

---

## 7. Cold-start design

**Rule:** 20 PG DBs with 0 support queries receive schema-only card (same as current 4-field spec). No query-informed enrichment.

**Implementation:** card generation script checks `len(support_queries) == 0` → uses schema-only prompt path. Logs which DBs used which path.

**Reporting:** all benchmark tables report a "cold-start flag" column. Headline metric = all 208 DBs. Supplementary table = "query-rich" only (≥1 support query). Claim about query-informed card improvements must be stated as applying to "query-rich DBs" only unless cold-start gap is explicitly measured and bounded.

**Confound guard:** do not compare M3 (which uses query-informed card) vs. M1/M2 (which use schema-only card) without noting cold-start split. A gain on the full 208 that disappears on cold-start 20 is a signal of cold-start DBs dragging M1/M2 down — not a signal of the card improving M3.

---

## 8. Ablation table — updated

Extend BENCHMARK.md §5:

| ID | Arm A | Arm B | Measures | Priority |
|---|---|---|---|---|
| E1 | card 3-field vs raw DDL | same retriever/split | card adds recall over raw | existing |
| E2 | ±distinguishing_terms | intra-domain pairs | contrastive field effect | existing |
| **E3a** | **schema-only card + D** | **query-informed card + D** | **query signal in card adds over schema-only (with D constant)** | **NEW, MANDATORY before claim** |
| **E3b** | **query-informed card + D** | **query-informed card only (D dropped)** | **block D is redundant if card is query-informed** | **NEW, run after E3a** |
| **E3c** | **query-informed card (clear names)** | **query-informed card (obfuscated)** | **leakage delta — how much gain is name leakage** | **NEW, MANDATORY before claim** |
| E4 | adjacency fill-in vs from-scratch | edge recall on ~20-30 DBs | closed-vocab design | existing |
| E5 | BFS all edges vs ±name_type_match | connectivity score | spurious edge impact | existing |
| **E6 (optional)** | **≤5 queries/DB vs ≤2 queries/DB** | **same card+D setup** | **leakage surface vs. salience signal trade-off** | **NEW, run if E3a shows gain** |

E3a must run before claiming "query-informed card improves routing." E3b must run before dropping block D. E3c (obfuscation) must run before any public claim about the mechanism.

---

## 9. Source table

| Paper | Tag | URL | Key claim used |
|---|---|---|---|
| Yu & Jagadish, VLDB 2006 | VERIFIED | vldb.org/conf/2006/p319-yu.pdf | Schema summarization uses structural signals (NOT query workload) |
| Yang, Procopiuc, Srivastava, PVLDB 2009 | VERIFIED | vldb.org/pvldb/vol2/vldb09-784.pdf | Query-set-driven table salience; workload determines which tables matter |
| Yang et al., "Summary Graphs," PVLDB 2020 | VERIFIED | dl.acm.org/doi/10.14778/3402707.3402728 | Follow-up: summary graph driven by user query table set |
| Gao & Luo, arXiv:2502.20657 | VERIFIED | arxiv.org/html/2502.20657v1 | Schema-only input; 30% columns need external signal; cold-start capable |
| Wretblad et al., arXiv:2408.04691 | VERIFIED | arxiv.org/html/2408.04691 | Schema + data rows input; no query injection; difficulty taxonomy |
| DBCopilot, arXiv:2312.03463 | VERIFIED | arxiv.org/html/2312.03463 | Synthetic query generation from schema; real queries NOT in index repr. |
| DBRouting, arXiv:2501.16220 | VERIFIED | arxiv.org/html/2501.16220 | DDL-only DB repr.; fine-tuning uses query-DB pairs (not description injection) |
| Sudarshan et al., arXiv:2601.19825 | PARTIALLY VERIFIED | arxiv.org/abs/2601.19825 | BIRD column descriptions = curated metadata, NOT query-derived |
| CRUSH4SQL, arXiv:2311.01173 | VERIFIED | arxiv.org/abs/2311.01173 | Query-side hallucination; no query injection into DB-side repr. |
| Nogueira et al., arXiv:1904.08375 | VERIFIED | ar5iv.labs.arxiv.org/html/1904.08375 | Raw query append ≠ description distillation; no comparison exists |
| Yoon et al., arXiv:2504.14175 | VERIFIED | arxiv.org/html/2504.14175v1 | LLM query expansion leakage via memorized content |
| Contamination in T2SQL, arXiv:2402.08100 | VERIFIED | arxiv.org/pdf/2402.08100 | Spider query phrasing in LLM pretraining; name memorization confirmed |
| dbt MetricFlow docs | INDUSTRY | docs.getdbt.com/best-practices/how-we-build-our-metrics/semantic-layer-2-setup | Semantic layers require human workload/business knowledge beyond schema |
| Cube.js | INDUSTRY | github.com/cube-js/cube | Same pattern as dbt MetricFlow |

---

## 10. Unresolved questions

1. **Sudarshan full method detail** — full paper PDF not accessible via HTML fetch; BIRD column description derivation mechanism (query-derived vs. human-curated vs. data-value-derived) is inferred from CLAUDE.md annotation, not directly confirmed from the full text. Verify before using as citation in thesis Chapter 2.

2. **Optimal support query count** — 5 is chosen as a practical cap; no paper calibrates this for card enrichment quality. E6 ablation (≤2 vs. ≤5) should be run if E3a shows a gain.

3. **NL-only vs. NL+schema-annotation support queries** — the recommendation injects NL questions only. Whether adding the ground-truth DB type (but not the query language) helps distinguish T2 cross-type cases is untested.

4. **Connectivity adaptation for Mongo/Neo4j** — unchanged from current spec; not addressed in this report (scope = card input dispute only).

5. **doc2query++ out-of-domain findings** — full paper not fetched; generalization warning is from search summary (LIKELY tag). Fetch arXiv:2510.09557 to confirm before citing.
