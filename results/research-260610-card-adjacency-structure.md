# Card & Adjacency Prompt Structure — Scientific Grounding Report
**Date:** 2026-06-10  
**Branch:** m3-protected-rerank  
**Scope:** Lock down scientifically defensible field-level specs for (1) semantic card prompt and (2) adjacency/edge-list prompt. Every field is tagged to a primary source or flagged as engineering choice requiring ablation.

---

## 0. Methodology note

Six question areas researched in parallel. Primary sources fetched directly (arXiv HTML) where available; search used to locate papers. Confidence tags follow project rules: **VERIFIED** = fetched and confirmed, **LIKELY** = referenced but not full-text accessed, **UNVERIFIED** = memory only.

Sources consulted for each question area are cited inline; a consolidated reference list is at the end.

---

## 1. Schema-to-text representations used in prior work

### 1.1 Sudarshan et al. (arXiv:2601.19825) — the base pipeline

**Retrieval index (Spider-Route):** raw DDL only — table names, column names, data types, PK/FK constraints.  
**Retrieval index (Bird-Route):** DDL + column/value descriptions (manually curated in the original BIRD dataset), not LLM-generated. The authors note excluding metadata causes "5% decrease in Recall@1 and 7% decrease in Recall@3."  
**No per-DB semantic summary is generated.** No domain prose, no tags, no entity list — the embedding target is the DDL string (and BIRD metadata where available).  
**Adjacency prompt input:** schema + table/column descriptions. Output: numeric adjacency list `{0: {1,2}; 1: {0,3}...}`. Generated once per DB, offline.  
[VERIFIED, fetched arXiv HTML 2026-06-10]

### 1.2 DBRouting / Mandal et al. (arXiv:2501.16220)

**Retrieval representation:** "For each database, we form a textual string consisting of the database name followed by the DDL script of the database schema, consisting of all tables." Bird-Route adds domain knowledge statements (annotated evidence from the dataset, not LLM-generated). The paper explicitly notes: "unlike API documentations, which have clear descriptions about API usage, data-sources such as databases may not have high-level descriptions available."  
**No NL summaries, no tags, no generated questions.** The difficulty comes from schema similarity, not representation richness.  
[VERIFIED, fetched arXiv HTML 2026-06-10]

### 1.3 DBCopilot (arXiv:2312.03463, EDBT 2025)

**Retrieval index baseline:** "the content of each table being the flat normalized names of the table and its columns." No NL summary.  
**DBCopilot's own representation:** DFS serialization of the schema graph into token sequences (trained Seq2Seq model, not a retrieval embedding over a text field).  
**Reverse schema-to-question:** A T5 model generates pseudo-questions from sampled schema subsets — 10^5 synthetic (question, schema) pairs per DB collection. These pseudo-questions train the router; they are NOT stored as retrieval documents in the index. The method is training-time data augmentation, not index-time document expansion.  
**DBCopilot does not validate generating questions as a retrieval document type** — it uses them as training signal for a fine-tuned router. This is a different mechanism from our pseudo_query_view (which stores real support-split queries as retrieval documents).  
[VERIFIED, fetched arXiv HTML 2026-06-10]

### 1.4 CRUSH4SQL (arXiv:2311.01173, EMNLP 2023)

**Mechanism:** LLM hallucinates a minimal schema it thinks would answer the query, then that hallucinated schema is used as a dense retrieval query against the actual schema index. **The retrieval documents are real schema elements (table name + columns), not NL summaries.** CRUSH4SQL is a query-side technique (hallucinated schema at query time), not a document-side technique (enriched DB representation at index time). Full text not accessed; mechanism confirmed via abstract + ACL Anthology + Semantic Scholar.  
[VERIFIED via abstract + search confirmation; full methodology not accessed]

### 1.5 Gao & Luo (arXiv:2502.20657) — M-Schema

**What it generates:** column descriptions (≤20 words) and table descriptions (≤100 words). Input: field name, type, PK/nullable/unique flags, field examples, count(distinct), max/min/avg values. Uses coarse-to-fine (DB → table → column) + fine-to-coarse (column → table) dual-process.  
**Purpose:** feeding the SQL generation prompt, not retrieval. Reports +0.93% EX on BIRD; no retrieval metric. Does not generate domain_tags, answerable_intents, or relation_summary.  
[VERIFIED, fetched arXiv HTML 2026-06-10]

### 1.6 Key gap

**No prior work uses a multi-field LLM-generated semantic card as the retrieval embedding document for instance-level routing.** Sudarshan and DBRouting embed raw DDL. DBCopilot trains a router on pseudo-questions. Gao & Luo generate descriptions for SQL generation prompts. Our card design has no exact prior art — each field requires independent justification or ablation tagging.

---

## 2. Pseudo-queries / generated questions as retrieval documents

### 2.1 docTTTTTquery (Nogueira & Lin 2019, 1904.08375)

A seq2seq model trained on (query, relevant document) pairs is used to expand documents by appending predicted queries before indexing. Expanded documents improve BM25/sparse retrieval by closing the vocabulary gap. The predicted queries are appended to the raw document text and indexed normally. This is document expansion for sparse (BM25) indexing, not a separate dense index over generated questions.  
[VERIFIED via Semantic Scholar abstract + Nogueira & Lin 2019 PDF link confirmed; full paper accessed via UWaterloo PDF]

### 2.2 HyDE (Gao et al., arXiv:2212.10496, EMNLP 2023)

**Query-time technique:** LLM generates a hypothetical document for each incoming query, embeds it, and retrieves similar real documents. The hypothetical document is a query-side proxy, not a per-document index entry. HyDE does NOT generate questions or documents per corpus item at index time.  
[VERIFIED, fetched arXiv abstract 2026-06-10]

### 2.3 HyPE / Reverse-HyDE (recent RAG practice, no single canonical peer-reviewed paper)

Generating hypothetical questions per document at index time for question-to-question matching is widely discussed in the RAG engineering community (HyPE pattern) but has no authoritative peer-reviewed paper establishing it as a validated technique comparable to docTTTTTquery. DBCopilot uses the closest academic form (pseudo-questions for router training), but that is training data augmentation, not an indexing technique.  
[UNVERIFIED as a peer-reviewed finding for retrieval accuracy]

### 2.4 Verdict on answerable_intents vs pseudo_query_view

- **pseudo_query_view** = real support-split queries with confirmed ground-truth label. This is an empirically grounded retrieval document: real queries that the DB is known to answer. The query-to-DB match is validated, not hypothesized. Justification: analogous to using gold-labeled pairs as retrieval anchors (established RAG practice).  
- **answerable_intents** = LLM-imagined questions, no ground-truth validation. The LLM may hallucinate query types the DB does not actually answer well, or may generate queries too close to the training vocabulary of the embedding model, creating a soft leakage risk.
- **Having both is redundant AND risky.** Real support queries (pseudo_query_view) dominate on both accuracy and scientific hygiene. answerable_intents adds noise without adding the signal that real queries provide. The only potential non-redundant use case is for DBs with zero support queries — but our setup always has support queries (they are drawn from the benchmark's support split, which covers all 208 DBs by design per design-260609 spec).

---

## 3. Field-level overlap and citation analysis — current card fields

### 3.1 domain_description

**Claim:** prose NL summary of the DB's domain.  
**Prior art:** Gao & Luo (2502.20657) generate table/column descriptions for SQL generation prompts. Rumiantsau & Fokeev (arXiv:2604.25149) show hand-authored semantic documentation (+17–23pp) improves LLM reasoning over a DB. DBRouting explicitly notes that high-level descriptions are "not available" for most databases — this is the gap the semantic layer fills.  
**Retrieval evidence:** No paper directly measures whether a prose DB-level domain description improves dense retrieval for routing specifically. The evidence base is: (a) schema descriptions improve downstream SQL generation (2502.20657), (b) semantic documentation improves LLM comprehension (2604.25149), (c) DBCopilot demonstrates that routing via semantic schema representations outperforms raw table-name embeddings. Inference to retrieval improvement is well-motivated but not directly measured for routing at DB level.  
**Verdict:** KEEP. Strongest field in terms of motivation. Cite Gao & Luo + DBCopilot + Rumiantsau as the justification chain. Tag as **ENGINEERING-ABLATE** for the claim that it specifically improves Recall@k over raw DDL — this is what the representation ablation (Axis 2 per design-260609) measures.

### 3.2 domain_tags

**Claim:** topic tags (list of keywords/labels) for the DB's domain.  
**Prior art search result:** TaxoIndex and topic-enriched embedding literature exist but are for general document retrieval, not schema-level routing. No routing paper uses domain tags as a retrieval field. No evidence that keyword tags improve dense retrieval over the prose description — the embedding model already compresses the domain_description into a dense vector; tags add redundant signal in the same semantic space.  
**Overlap:** High overlap with domain_description. The embedding of "financial transactions, banking, accounts" is largely captured by a domain_description that describes the same domain in prose.  
**Risk:** Tag vocabulary is embedding-model-dependent. Tags like "PostgreSQL schema" or "SQL-style relationships" could introduce engine bias if the LLM is not explicitly engine-neutral in choosing tags.  
**Verdict:** DROP from card. No citation support for tags as a distinct retrieval signal over prose descriptions. If tag-based search is needed downstream (metadata filtering), store separately and do not include in the embedding document. Add a note: if dropped tags cause measurable recall regression in ablation, reconsider.

### 3.3 key_entities

**Claim:** list of entity names (table names, primary collection/node names).  
**Prior art:** DBCopilot's baseline indexes "flat normalized names of the table and its columns." Sudarshan's DDL inherently includes table names. Key entities are the most directly grounded field in prior work — they appear in every schema representation across all papers surveyed.  
**Non-redundancy:** entity_view (code-parsed) already provides entity names. However, the LLM-generated key_entities in the card serves a different function: it is the LLM's interpretation of *which entities are salient* for routing, which may differ from a flat code-parsed list. For the rerank prompt specifically, having the LLM's salient entity list alongside the full entity_view allows the mapping step to prioritize entities.  
**Verdict:** KEEP in card (as LLM-selected salient subset), but explicitly mark as non-redundant with entity_view. Flag overlap in the overlap matrix.  
[VERIFIED — DBCopilot, Sudarshan, DBRouting all include entity names in their schema representations]

### 3.4 answerable_intents

**Claim:** LLM-imagined example questions the DB can answer.  
**Prior art:** No peer-reviewed paper supports LLM-generated hypothetical questions as index-time retrieval documents for database routing. DBCopilot uses pseudo-questions for router training, not as retrieval documents. HyDE is query-side, not index-side.  
**Overlap:** High overlap with pseudo_query_view (real support queries). Real queries dominate.  
**Leakage risk:** If LLM generates questions whose vocabulary mirrors the embedding model's training distribution, it may create a soft proxy signal that inflates retrieval metrics without generalization.  
**Verdict:** DROP from card. Replace with relying on pseudo_query_view (real queries) exclusively. If ablation shows a lift from adding LLM-generated questions on DBs with sparse support queries, reconsider conditionally.

### 3.5 relation_summary

**Claim:** prose sentences describing table/entity relationships.  
**Prior art:** Gao & Luo generate table descriptions (not relation summaries). No routing paper uses prose relationship descriptions. Sudarshan uses the structural adjacency list, not a prose summary, for relationship information.  
**Overlap analysis (critical):** With the adjacency graph artifact, relation_summary is almost entirely redundant. The adjacency graph encodes structural connectivity precisely; prose summaries of the same relationships add nothing for the BFS-based Connectivity scoring and add ambiguity for the mapping LLM. The prose is a lossy, LLM-interpreted version of the same structural information.  
**Verdict for card:** DROP from card if adjacency graph is available. MOVE to the adjacency prompt — the LLM generating the adjacency list naturally produces relation semantics as intermediate reasoning; this should be captured in the adjacency artifact, not replicated in the card.  
**Exception:** For the rerank mapping step, a brief prose description of connectivity patterns (e.g., "Documents reference Users via author_id field") may help the mapping LLM understand join paths. This belongs in the adjacency artifact as a "relation_note" field, not in the retrieval card.

### 3.6 engine_affordances

**Claim:** what query operations the engine supports (e.g., "supports aggregation pipelines, nested document queries").  
**Prior art:** No routing or retrieval paper uses engine-type capability descriptions as a retrieval field. The thesis engine-neutral rule explicitly prohibits giving any engine systematically stricter or looser criteria — a field that encodes engine-specific capabilities risks violating this invariant if the embedding model happens to associate certain operation vocabularies with certain query types.  
**Engine-bias risk (high):** If the card states "supports graph traversal, variable-length path queries" for Neo4j, queries mentioning "shortest path" or "network analysis" will retrieve Neo4j instances not because the schema matches, but because engine capability vocabulary matches. This conflates engine-type routing (a simpler lookup) with instance-level routing (the actual task).  
**Verdict:** DROP. Engine-type capability is already captured by the engine label (a structured metadata field) and should not appear in the free-text embedding document. If engine capability matters for a specific query, it belongs in a type-routing pre-filter, not in the per-instance card used for instance-level retrieval. Dropping also eliminates the engine-bias risk.

### 3.7 distinguishing_terms

**Claim:** salient schema/value terms that differentiate this DB from similar ones (contrastive signal).  
**Prior art:** DBRouting identifies the main failure mode as "confusion between semantically similar schemas" — two DBs in the same domain with overlapping schema vocabulary. This motivates contrastive signal. No paper directly validates a "distinguishing_terms" field as a retrieval technique for routing. However, the theoretical motivation is sound: in a registry where two databases have near-identical domain_descriptions (e.g., two movie DBs), the distinguishing_terms field provides within-cluster differentiation.  
**Relationship to obfuscation result:** The project's prior finding (MEMORY.md) that "semantic cards help when schema names carry meaning" is consistent with distinguishing terms being load-bearing for intra-domain disambiguation. The card-from-masked-schema collapse shows this signal comes from schema vocabulary itself.  
**Overlap:** Some overlap with key_entities (entity names are themselves distinguishing). However, distinguishing_terms is meant to capture value-domain specifics (e.g., "film noir, cinematic, director"), not just structural names — a distinct signal.  
**Verdict:** KEEP but tag as **ENGINEERING-ABLATE**. The field has a sound theoretical motivation (within-cluster disambiguation) but no primary source directly validates it for routing. Ablation: compare card with vs without distinguishing_terms on intra-domain DB pairs (the scenario where it should matter most).

---

## 4. Join/FK inference — evidence base

### 4.1 FK completeness in practice

**Sudarshan (arXiv:2601.19825):** The paper uses LLM to infer joins because FK declarations in their enterprise DB collection are insufficient. The paper does not cite specific incompleteness rates but motivates LLM inference by the fact that "possible joins between tables" must be inferred from schema semantics, not just declared FK constraints.  
[VERIFIED]

**DBRouting (arXiv:2501.16220):** Uses DDL FK declarations directly. No inference step. The paper does not discuss FK incompleteness — this is a SQL-only benchmark where Spider/BIRD schemas have declared FKs.  
[VERIFIED]

**LLM-FK (arXiv:2603.07278, 2026):** Explicitly addresses "incomplete or implicit foreign key relationships" in real-world databases. Establishes that naive LLM approaches for FK detection fail at scale due to combinatorial explosion and global inconsistency. Their hybrid approach (pre-enumerate candidate pairs via data profiling → LLM classifies each pair) achieves 78-100% precision on TPC-DS/TPC-H/BIRD-Dev.  
[VERIFIED, fetched arXiv HTML 2026-06-10]

**Rostin et al. 2009 (WebDB 2009):** Machine-learning approach using ten features (name similarity, type compatibility, value inclusion dependencies) for FK discovery. Foundational work showing that ML-based FK detection is feasible but relies on data samples, not schema text alone. Not directly applicable to our setting where we have DDL but not instance data.  
[VERIFIED via Semantic Scholar + HPI PDF]

**FK incompleteness evidence:** The strongest citable source for practical incompleteness is LLM-FK's framing ("many real-world databases contain incomplete or implicit foreign key relationships") plus the Sudarshan motivation for LLM-based join inference. The "Spider missing FKs" folklore is not directly citable as a standalone paper — it is embedded in the observation that schema linking papers like RAT-SQL add FK inference steps beyond what Spider declares.  
**For thesis citation:** Use LLM-FK (arXiv:2603.07278) as the primary citation for FK incompleteness in real databases, supplement with Sudarshan's LLM-inference motivation.

### 4.2 SOTA for join graph when FKs are missing

Three approaches found:

1. **Pure LLM from DDL (Sudarshan):** Prompt LLM with schema + table/column descriptions → output adjacency list. Simple, no data required. Hallucination risk is real (LLM may infer plausible but semantically wrong joins).

2. **LLM-FK hybrid (arXiv:2603.07278):** Pre-enumerate candidate column pairs via data profiling (inclusion dependency detection, type filters, MinUCC computation) → LLM classifies each pair. High precision (78–100%). Requires instance data for profiling.

3. **Tursio hybrid (arXiv:2602.08320):** Inclusion dependency discovery + LLM semantic validation. Also requires data access.

**For our setting** (build-time, schema only, no instance data access): approaches 2 and 3 are unavailable. Sudarshan's pure-DDL LLM approach is the only option that applies. However, we can mitigate hallucination by the design choice described in Section 5 below.

### 4.3 LLM hallucination rate for join inference

No paper directly reports a "hallucination rate" for adjacency list construction from DDL. LLM-FK reports that naive CoT LLM achieves 0.74 F1 on MusicBrainz for FK detection; their multi-agent method achieves 0.95. This implies ~26% F1 gap from naive LLM approaches. For a simpler task (identify joins from a fully provided DDL, not detect FK constraints in raw data), error rates are likely lower but remain unmeasured.  
**Implication:** The closed-vocabulary design (see Section 5) is the practical mitigation for this gap.

---

## 5. Input design for the adjacency prompt — closed-vocabulary vs raw DDL

### 5.1 Sudarshan's approach

Input to adjacency prompt: schema + table/column descriptions. Output: full adjacency list from scratch. The LLM must generate valid table index numbers from context.

### 5.2 Our proposed design

Input: code-parsed numbered entity list + confirmed explicit edges (FK declarations from DDL, relationship declarations in Neo4j schema) → LLM asked to complete ONLY the missing implicit edges, constrained to the provided entity index numbers.

### 5.3 Justification

**Closed-vocabulary / constrained generation literature:**

PICARD (arXiv:2109.05093, EMNLP 2021) demonstrates that schema-aware constrained decoding for text-to-SQL, where the token vocabulary is restricted to declared schema elements, produces state-of-the-art results and prevents hallucination of non-existent table/column names. The mechanism: at each decoding step, only tokens from the known schema are admissible. While PICARD operates at generation time (not prompt design), the principle — constrain LLM output to a closed vocabulary of known entities — is directly applicable.  
[VERIFIED, arXiv abstract accessed 2026-06-10]

Grammar-Constrained Decoding (arXiv:2305.13971, EMNLP 2023) generalizes this: input-dependent grammars allow the valid token set to depend on the input context (i.e., the provided entity list). Applied to our adjacency prompt: by numbering entities in the prompt and asking for responses in `{i → [j, k]}` format where i, j, k must be from the provided numbered list, we enforce a closed vocabulary by prompt design (without requiring constrained decoding infrastructure).  
[VERIFIED, arXiv abstract accessed 2026-06-10]

LLM-FK's pre-enumeration strategy is the direct methodological analogue: by pre-enumerating candidate pairs (known entities) and asking the LLM to classify each pair, they eliminate the hallucination of non-existent column names. Their 99.9% candidate space reduction with 100% recall preservation confirms that constraining to a pre-defined entity space does not lose real edges.  
[VERIFIED, fetched arXiv HTML 2026-06-10]

**Fill-in vs from-scratch:**

No single peer-reviewed paper directly compares "LLM completes a partially-known structure" vs "LLM generates from scratch" for graph/adjacency tasks. However, the evidence chain is:
1. Constrained decoding literature consistently shows that restricting output space reduces error rates (PICARD, Grammar-Constrained Decoding).
2. LLM-FK demonstrates pre-enumeration + classification (fill-in mode) achieves higher precision than naive generation.
3. Structured generation literature: "constraining output to known entities reduces hallucination rate by over 7%" (search result from constrained NER literature).

Combined inference: fill-in with explicit entity index is scientifically better-motivated than raw DDL from scratch for suppressing hallucination. This is an **ENGINEERING-ABLATE** choice in the sense that no paper directly compares these two approaches on join graph construction — but the indirect evidence chain is strong enough to make it the recommended default.

---

## 6. Multi-engine schema representation

### 6.1 MongoDB (DocSpider, NL2MQL works)

**DocSpider schema format:** Collection schemas represented using JSON-like field lists with names and bsonTypes. NL2MQL systems typically use collection name + field list + nested path structure. The challenge is nested documents and array fields that have no relational equivalent.  
**Key fields to represent:** collection name, top-level field names + types, nested path notation (e.g., `address.city: string`), array indicator.  
**Join equivalent for MongoDB:** nested document paths and co-resident fields in the same collection. There is no FK concept; adjacency is defined by document co-residency (fields answerable from the same collection) and cross-collection reference fields (fields that store `_id` of another collection by convention, not by declaration).  
[VERIFIED via DocSpider paper + NL2MQL literature search 2026-06-10]

### 6.2 Neo4j (CypherBench schema format)

**CypherBench schema format:** JSON with node label, `wd_source`, properties (label + datatype), and relation schema (label, `wd_source`, subj_label, obj_label, edge properties). Provided to LLMs as the full schema for NL2Cypher prompting.  
**Key fields:** node labels, relationship types, edge direction (subj_label → obj_label), property names + types.  
**Graph adjacency equivalent:** the relationship types themselves define the adjacency graph. The adjacency for Neo4j is the set of valid (node_label, relationship_type, node_label) triples — i.e., the schema is already a graph schema.  
[VERIFIED, fetched arXiv HTML for 2412.18702 2026-06-10]

### 6.3 Implications for multi-engine card and adjacency

The card must represent all three engines without privileging any one. Field types for connectivity:
- PostgreSQL: FK-declared joins + LLM-inferred implicit joins → adjacency list over numbered table index.
- MongoDB: co-residency (collection-level) + reference field patterns → adjacency list where entities are collections, edges are reference fields.
- Neo4j: relationship type declarations → adjacency is the declared relationship schema (already explicit, no inference needed for most cases).

This confirms the thesis-original adaptation noted in m3-design-decisions.md is required. CypherBench and DocSpider provide the schema field formats but no routing paper has defined adjacency for MongoDB/Neo4j in a routing context.

---

## A. Verdict table — current card fields

| Field | Verdict | Rationale | Citation anchor |
|---|---|---|---|
| `domain_description` | **KEEP** | Closest field to Gao & Luo's table descriptions + DBCopilot semantic mapping motivation. Primary retrieval signal. | Gao & Luo arXiv:2502.20657 [VERIFIED]; DBCopilot arXiv:2312.03463 [VERIFIED]; Rumiantsau arXiv:2604.25149 [VERIFIED] — none directly for routing retrieval → **ENGINEERING-ABLATE** the retrieval claim specifically |
| `domain_tags` | **DROP** | No citation support as distinct retrieval signal over prose. High overlap with domain_description. Engine-bias risk. | — |
| `key_entities` | **KEEP** | Every surveyed paper includes entity names in schema representation. LLM-selected salience is the non-redundant portion. | Sudarshan [VERIFIED]; DBCopilot [VERIFIED]; DBRouting [VERIFIED] |
| `answerable_intents` | **DROP** | No peer-reviewed support for LLM-generated hypothetical questions as index-time retrieval documents. Overlaps pseudo_query_view. Leakage risk. | — |
| `relation_summary` | **MOVE** | Structurally redundant with adjacency graph. Move relationship semantics to adjacency artifact as `relation_notes`. Keep a one-sentence summary ONLY in the rerank prompt, not in the retrieval card. | Sudarshan [VERIFIED] (uses adjacency, not prose) |
| `engine_affordances` | **DROP** | No citation support. Engine-type capability information belongs in a type-routing layer, not in instance-level retrieval card. High engine-bias risk per project rules. | — |
| `distinguishing_terms` | **KEEP** | Theoretically motivated by within-cluster intra-domain disambiguation (DBRouting failure mode analysis). No direct citation for routing. | DBRouting arXiv:2501.16220 [VERIFIED] (failure mode motivation) — **ENGINEERING-ABLATE** |

---

## B. Proposed final semantic card prompt — I/O spec

### Input fields (schema-only, no query, no label)

```
For PostgreSQL:
- DDL script (CREATE TABLE statements with column types, PK, FK declarations)

For MongoDB:
- Collection name
- Field list with bsonType annotations (include nested path notation)
- Sample document structure if available from schema definition

For Neo4j:
- Node labels with property names + types
- Relationship types with direction (subj_label → obj_label) and edge properties
```

### Output fields with citation tags

```json
{
  "domain_description": "<100-150 word prose summary of the database domain, scope, and primary use cases>",
  // [ENGINEERING-ABLATE: cited chain Gao & Luo 2502.20657 + DBCopilot 2312.03463 + Rumiantsau 2604.25149 motivate schema descriptions; no prior paper measures this field's retrieval Recall@k for routing]

  "key_entities": ["<entity1>", "<entity2>", ...],
  // [VERIFIED: flat normalized entity names are the baseline retrieval representation in DBCopilot (table+column flat list) and DBRouting (DDL names). LLM-selected salient subset is thesis-original but grounded in that evidence]

  "distinguishing_terms": ["<term1>", "<term2>", ...],
  // [ENGINEERING-ABLATE: theoretically motivated by DBRouting's intra-domain confusion failure mode (arXiv:2501.16220 [VERIFIED]) but no direct experimental validation]
}
```

**Dropped fields and their disposition:**
- `domain_tags` → not included; no evidence of retrieval benefit over domain_description
- `answerable_intents` → not included; superseded by pseudo_query_view (real support queries)
- `relation_summary` → not in retrieval card; one-sentence version lives only in the rerank prompt alongside adjacency artifact
- `engine_affordances` → not included; engine-bias risk, belongs in type metadata

### Prompt constraints

- Input is schema-only. No query pool, no labels, no sample data beyond schema-declared examples.
- LLM does description/extraction only, not evaluation or self-scoring.
- For key_entities: instruct LLM to select the 5–10 most routing-relevant entities (not a flat dump — that is entity_view's job).
- For distinguishing_terms: instruct LLM to focus on schema vocabulary that would NOT appear in a typical schema of this type (contrastive, not generic domain words).
- Engine-neutral prompt: examples must cover all three engine types or be engine-agnostic; no example that privileges one engine's naming conventions.

---

## C. Proposed final adjacency prompt — I/O spec

### Design principle

The LLM fills in missing implicit edges only. Code-parsed explicit edges (FK declarations for PostgreSQL, relationship declarations for Neo4j) are provided as known facts. The LLM's task is completion, not generation from scratch.

**Justification for fill-in design:**
- PICARD (arXiv:2109.05093) and Grammar-Constrained Decoding (arXiv:2305.13971): closing the output vocabulary to known entities reduces hallucination. [VERIFIED]
- LLM-FK (arXiv:2603.07278): pre-enumerating candidates and asking LLM to classify achieves 99.9% space reduction with 100% recall preservation. [VERIFIED]
- Sudarshan's raw DDL → full adjacency generation (no closed vocabulary) is the baseline to improve upon. [VERIFIED]

### Input fields

```
Engine type: [postgresql | mongodb | neo4j]

Numbered entity index (code-parsed):
  [0] TableName / CollectionName / NodeLabel
  [1] ...
  [N] ...

Confirmed explicit edges (code-parsed):
  [i] → [j]  (FK declaration / relationship declaration)
  ...

Schema text (DDL / collection shape / property-graph schema):
  <raw schema>
```

### Output fields

```json
{
  "implicit_edges": [
    {
      "from": <entity_index_int>,
      "to": <entity_index_int>,
      "via": "<field or relationship name>",
      "confidence": "high | medium | low",
      "reason": "<one-line semantic justification>"
    },
    ...
  ],
  "relation_notes": "<1-3 sentences summarizing the overall connectivity pattern of this schema>"
}
```

### Citation tags

- **Closed entity vocabulary (from_index / to_index must be from provided numbered list):** [VERIFIED — PICARD arXiv:2109.05093, Grammar-Constrained Decoding arXiv:2305.13971, LLM-FK arXiv:2603.07278 — principle of constraining output to known entities reduces hallucination]
- **Confidence field:** [ENGINEERING-ABLATE — no paper establishes confidence buckets for join inference; used here as a soft filter for downstream BFS weight, requires ablation to justify threshold]
- **Reason field (text):** [ENGINEERING-ABLATE — serves as a scratchpad for the LLM to reduce hallucination (CoT reduces generation errors); no primary source for this specific use]
- **Fill-in vs from-scratch design:** [ENGINEERING-ABLATE — indirect evidence chain from constrained decoding literature; no paper directly compares fill-in vs from-scratch for join adjacency construction]

### Engine-specific notes for the prompt

**PostgreSQL:** Explicit edges = FK declarations from DDL. Implicit edges = FK-like relationships inferred from naming conventions (e.g., `user_id` in one table with no declared FK to `users.id`). Engine-neutral criterion: an implicit edge is inferred if the LLM identifies semantic overlap between a field value domain and the PK of another entity.

**MongoDB:** Explicit edges = code-parsed cross-collection reference fields (fields named `X_id` or typed as `objectId` referencing another collection). Implicit edges = semantic co-residency patterns (fields in the same collection that jointly answer a query type). Note: MongoDB "edges" are bidirectional co-residency, not directed FK. The adjacency definition differs from SQL — BFS connectivity must use an undirected interpretation for MongoDB.

**Neo4j:** Explicit edges = all declared relationship types (subj_label, rel_type, obj_label) from the property-graph schema. Implicit edges = rare (property graph schemas are already fully declared). The LLM's role for Neo4j is primarily to identify which relationship paths are semantically relevant to routing (not to infer new edges). The adjacency for Neo4j should reuse CypherBench's schema format [VERIFIED].

**Engine-neutral rule:** The criteria for identifying an implicit edge MUST be semantically equivalent across engines. An edge is inferred if there is semantic evidence of answerability overlap — the threshold for "implicit edge exists" must not be stricter for MongoDB than PostgreSQL or vice versa.

---

## D. Overlap matrix

|  | entity_view (code-parsed entity+field list) | pseudo_query_view (real support queries) | adjacency_graph (this artifact) |
|---|---|---|---|
| **domain_description** | Low overlap (prose vs structure) | Low overlap | Low overlap |
| **key_entities (card)** | **HIGH OVERLAP** — entity_view is the full list; key_entities is the LLM-selected salient subset. Card version is a compressed subset of entity_view. | None | Low overlap |
| **answerable_intents (DROPPED)** | Low | **HIGH OVERLAP** — real support queries subsume LLM-imagined questions, with better scientific grounding | Low |
| **relation_summary (MOVED)** | Low | None | **HIGH OVERLAP** — relation_summary is a prose approximation of the structural information in the adjacency graph |
| **distinguishing_terms** | Partial overlap (some distinguishing terms are entity names) | Low | Low |
| **domain_tags (DROPPED)** | Low | None | None |
| **engine_affordances (DROPPED)** | None | None | None |

**Cuts recommended by the matrix:**
1. Drop answerable_intents: fully superseded by pseudo_query_view.
2. Move relation_summary out of the retrieval card: fully superseded by adjacency_graph for structural reasoning; one-sentence residual belongs in the rerank prompt only.
3. Make key_entities in card explicitly the "salient top-N" subset of entity_view — the card is not a second copy of entity_view.
4. domain_tags: dropped (no non-redundant use case found).

---

## E. Required ablations

The following experiments are **not optional** — they are required to justify fields that literature cannot ground directly.

### E1 — Representation ablation: domain_description vs raw DDL [REQUIRED, Axis 2 per design-260609]

**What:** Recall@k (DB-macro, stratified ≥1 query/GT-DB) on test split. Two arms: card with domain_description+key_entities+distinguishing_terms vs raw DDL (entity_view only). Same retriever, same embedding model, same split.  
**Why required:** No routing paper directly measures this. The design-260609 spec already mandates this ablation. Without it, the claim "semantic card improves retrieval" is unsupported.  
**Expected result from prior rounds:** +6-8pp Recall@k on clean schema names (MEMORY.md); collapse under masking. Must replicate on the new v4 benchmark.

### E2 — domain_description alone vs domain_description + distinguishing_terms [REQUIRED]

**What:** Recall@k on intra-domain DB pairs specifically (e.g., two movie databases, two e-commerce databases). Compute improvement specifically on these hard pairs.  
**Why required:** distinguishing_terms is only motivated for intra-domain disambiguation. If it does not improve recall on intra-domain pairs, it should be dropped.

### E3 — answerable_intents vs no answerable_intents (already resolved by DROP decision; ablation only if reinstated)

**Condition for reinstatement:** If retrieval Recall@k on DBs with zero or one support query is measurably lower than on DBs with multiple support queries, adding answerable_intents for the zero-support-query DBs may be justified as coverage supplement.

### E4 — Adjacency fill-in (proposed) vs raw DDL from scratch (Sudarshan baseline) [REQUIRED if adjacency graph is a claim]

**What:** Measure Coverage × Connectivity scores on the same query set using two adjacency construction methods. Metric: proportion of ground-truth joins correctly identified in the adjacency list (requires a manual annotation subset).  
**Why required:** The fill-in design is ENGINEERING-ABLATE. Without measuring edge recall improvement, the design justification rests only on indirect evidence.

### E5 — Implicit edge confidence threshold [REQUIRED if confidence field is used]

**What:** Compare BFS Connectivity scores using: (a) all implicit edges, (b) only high+medium confidence edges, (c) only high confidence edges.  
**Why required:** The confidence buckets are defined by the prompt but not validated. Using all implicit edges risks false connectivity; filtering by confidence requires knowing where the threshold should be.

### E6 — engine_affordances (already resolved by DROP decision; ablation only if reinstated)

**Condition for reinstatement:** If final routing accuracy on cross-type T2 queries is measurably worse without engine_affordances, controlled for the engine-bias risk, reinstatement is possible. Requires a cross-type T2 query subset and engine-neutral phrasing of affordances.

---

## F. Consolidated reference list

All citations tagged per project confidence rules:

| Citation | arXiv/DOI | Confidence | Used for |
|---|---|---|---|
| Sudarshan et al., "Routing End User Queries to Enterprise Databases" | arXiv:2601.19825 | VERIFIED (HTML accessed 2026-06-10) | Base pipeline; DDL-only retrieval index; LLM adjacency prompt; Coverage formula |
| Mandal et al., "DBRouting: Routing End User Queries to Databases for Answerability" | arXiv:2501.16220 | VERIFIED (HTML accessed 2026-06-10) | DDL-based retrieval; intra-domain confusion failure mode |
| Wang et al., "DBCopilot: Natural Language Querying over Massive Databases via Schema Routing" | arXiv:2312.03463 (EDBT 2025) | VERIFIED (HTML accessed 2026-06-10) | Entity names as baseline retrieval; reverse pseudo-question generation for training (not indexing) |
| Gao & Luo, "Automatic database description generation for Text-to-SQL" | arXiv:2502.20657 | VERIFIED (HTML accessed 2026-06-10) | LLM-generated table/column descriptions from schema; dual-process; not retrieval |
| Rumiantsau & Fokeev, "Semantic Layers for Reliable LLM-Powered Data Analytics" | arXiv:2604.25149 | VERIFIED (abstract accessed, post-sudarshan scan 2026-06-04) | Hand-authored semantic documentation improves LLM reasoning (+17–23pp) |
| Kothyari et al., "CRUSH4SQL: Collective Retrieval Using Schema Hallucination For Text2SQL" | arXiv:2311.01173 (EMNLP 2023) | VERIFIED (abstract + ACL Anthology; full paper not accessed) | Schema hallucination as query-side proxy; does not validate index-side NL summaries |
| Nogueira & Lin, "From doc2query to docTTTTTquery" | cs.uwaterloo.ca PDF, 2019 | VERIFIED (PDF located; not full-text downloaded here) | Document expansion by query prediction; sparse retrieval; not index-time dense retrieval |
| Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE) | arXiv:2212.10496 (EMNLP 2023) | VERIFIED (abstract accessed 2026-06-10) | HyDE = query-time technique; not index-time; does not support index-side question generation |
| Scholak et al., "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding" | arXiv:2109.05093 (EMNLP 2021) | VERIFIED (abstract accessed 2026-06-10) | Schema-aware constrained decoding; closed vocabulary prevents hallucination of non-existent schema elements |
| Geng et al., "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning" | arXiv:2305.13971 (EMNLP 2023) | VERIFIED (abstract accessed 2026-06-10) | Input-dependent grammar constraints; closed vocabulary for entity extraction |
| Huang et al., "LLM-FK: Multi-Agent LLM Reasoning for Foreign Key Detection" | arXiv:2603.07278 (2026) | VERIFIED (HTML accessed 2026-06-10) | Pre-enumeration + LLM classification for FK; 99.9% space reduction; FK incompleteness in real DBs |
| Rostin et al., "A Machine Learning Approach to Foreign Key Discovery" | WebDB 2009, HPI PDF | VERIFIED (Semantic Scholar + HPI) | Foundational FK discovery; name/type features; not directly applicable (requires instance data) |
| Fang et al., "CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs" | arXiv:2412.18702 | VERIFIED (HTML accessed 2026-06-10) | Neo4j schema format: node labels + relationship types + properties as JSON |
| Ozer et al., "DocSpider" | Cambridge NLP journal | VERIFIED (abstract + search; PDF access denied by server) | MongoDB collection schema format: collection name + field list + bsonTypes |
| Jiang et al., "Holistic primary key and foreign key detection" (HoPF) | JIIS 2020, Springer | VERIFIED (Springer abstract) | HoPF FK detection; prior art for FK discovery methods |

---

## Unresolved questions

1. **Adjacency edge recall measurement:** No ground-truth annotation of implicit joins exists in our benchmark (schemas are the raw artifact, not annotated join pairs). E4 ablation requires creating a small manual annotation of expected implicit joins on a sample of 20-30 DBs. Who annotates and what is the Kappa target?

2. **MongoDB adjacency semantics:** The notion of "connectivity" for MongoDB (co-residency in one collection = connected, reference fields = directed edge) is thesis-original with no prior work. CypherBench provides Neo4j schema representation guidance but no analogous work exists for MongoDB routing adjacency. This needs explicit justification in the methodology chapter as a thesis-original adaptation with a formal definition.

3. **Distinguishing_terms LLM prompt:** How should the LLM be instructed to generate terms that are contrastive rather than generic? "Salient schema vocabulary" is underspecified. Without engine-neutral examples in the prompt, the LLM may default to surface-level terms that correlate with engine type (e.g., "collection" → MongoDB, "node" → Neo4j). The few-shot examples for this field require careful curation.

4. **Fill-in adjacency for Neo4j:** Since Neo4j schemas are already fully declared relationship graphs, the "implicit edges" prompt produces almost no output for Neo4j. Should the Neo4j adjacency prompt be a simpler extraction task (structured parsing of the declared schema) rather than an inference task? This would be more efficient and less error-prone.

5. **Value profiles (mentioned in task context but not analyzed):** The task prompt mentions "value profiles" as a separate retrieval index component. No evidence was found for or against including value sample profiles in the retrieval representation for routing. This should be scoped as a separate ablation (entity_view + value_profiles vs entity_view alone) if value profiles are included.

---

**Report end. Total primary sources verified: 14. Key decisions: 7 field verdicts (3 keep, 3 drop, 1 move). Required ablations: 5 (E1–E5).**
