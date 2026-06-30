# Ablation Design + Results — Connectivity Scoring + Adjacency Construction Extensions (#2)

**Date:** 2026-06-18
**Status:** TESTED (free re-score, no LLM). **DECISION: keep #1 connectivity-faithful; Nhóm 1 + Nhóm 2
REJECTED as negative results.** See §8.

---

## 8. RESULTS — measured 2026-06-18 (free re-score from cache, no LLM)

**Decision: direction #1 (connectivity-faithful) is the locked config. Soft connectivity (Factor A)
and adjacency extensions (Factor B) tested and REJECTED — marginal/unreachable gain.**

### 8.1 Factor A — soft connectivity (V1 | in-pool, scoring layer)

| set | A0 binary (locked) | A1 ratio | A2 decay λ1 | GT-lockout A0→A1 |
|---|---|---|---|---|
| ours | 0.761 | 0.769 | 0.766 | 50→42 |
| spider | 0.766 | 0.765 | 0.766 | 53→53 |
| bird | 0.902 | 0.905 | 0.905 | 18→17 |

Marginal: ours +0.8pp / 8 fewer lock-outs; spider flat (even −0.1); bird +0.3pp. The faithful fix
already captured the connectivity win. Soft conn gives GT partial credit but competitors keep high
scores → argmax rarely flips. tie-rate unchanged (no flooding). **NOT worth the LLM cost of a V2
confirmation. REJECTED.**

### 8.2 Factor B — adjacency extensions

- **Neo4j (B2): already done.** Relationships are declared in the schema → adjacency already takes
  them directly (0% zero-edge Mongo... err Neo4j; median 7 edges). Nothing to improve.
- **Mongo (B1): theoretical max = 7/50 ours lock-outs (~7 queries, ~+1pp), and unreachable.** Of 10 zero-edge
  Mongo DBs: 3 are single-collection (connectivity trivially 1, not broken); the 7 genuine
  multi-collection lock-outs need SEMANTIC joins (`people.People_ID = gymnast.Gymnast_ID`,
  `election.Party = party.Party_ID`) where field names DON'T follow id-conventions → deterministic
  shared-id/ref-name heuristic catches 1/8, and that one (`_id`) is a SPURIOUS edge. Stronger LLM
  inference risks FALSE edges (wrong DBs also get conn=1), violating scoring discipline. **REJECTED.**

### 8.3 Scientific conclusion

The real connectivity gain was fully captured by the faithful fix (#1, V2 ours 0.782→0.803, all 3
sets up, primary-source-grounded). The residual lock-out is NOT a scoring defect addressable by soft
connectivity or richer adjacency — it is **benchmark-inherent near-duplicate-instance ambiguity**
resolved (or not) at the tie-break layer (many same-domain instances with overlapping
descriptions/schemas). Nhóm 1 + Nhóm 2 are recorded as negative results: obvious next levers tried,
gain not worth the cost/risk.

---

## (original design below — retained for record)

**Date:** 2026-06-18
**Status:** DESIGN ONLY — no runs executed. Requires explicit "chạy" before any LLM/eval.
**Depends on:** connectivity-faithful fix LOCKED 2026-06-18 (`.claude/rules/m3-design-decisions.md`
§Connectivity = Sudarshan-faithful). This doc covers the *residual* after that fix.

---

## 1. Motivation (measured)

Headline cal-v2, δ=0.2, R@1 | GT-in-pool, after connectivity-faithful fix:

| set | V1 | V2 |
|---|---|---|
| ours_multidb | 0.761 | 0.803 |
| spider_route | 0.766 | 0.814 |
| bird_route | 0.902 | 0.899 |

Established earlier this session: **the residual V2 error splits into (a) scoring lock-out — GT
excluded from the near-set before the agent ever sees it, and (b) tie-break ambiguity — GT in the
near-set but a near-duplicate DB chosen.** When the agent is allowed to see GT it usually picks it
correctly; the lever is therefore the scoring step that locks GT out. The remaining lock-out (V2
scoring-faults): ours 50, spider 53, bird 18. Of these, the dominant mechanism is **Connectivity = 0
on the correct DB** (coverage never reaches 0).

Connectivity-faithful already removed the "flatten-all" over-strictness (rescued 31/50 on ours). The
residual conn=0 has two sources, both OUT of scope for the faithful fix:

1. **Binary brittleness on incomplete adjacency (all engines):** Connectivity is 0/1 and
   "immediately invalidates the DB" (Sudarshan). One missing edge in the LLM-inferred join graph
   zeroes the correct DB. The more thorough the (correct) mapping, the more entities, the higher the
   chance one edge is missing → conn=0. Multi-table queries (the hard, interesting ones) are punished
   on the right DB.
2. **Adjacency gaps for non-relational stores:** Sudarshan's adjacency prompt is SQL/FK-only.

### 1.1 Current adjacency density (ours_multidb, diagnostic)

| engine | #DB | 0-edge DBs | edges/DB (median) | entities/DB (median) | edges/entity |
|---|---|---|---|---|---|
| mongodb | 87 | 10 (11%) | 6 | 4 | 1.45 |
| neo4j | 27 | 0 (0%) | 7 | 5 | 1.88 |
| postgresql | 94 | 0 (0%) | 3 | 4 | 0.94 |

**Nuance:** Mongo is NOT uniformly edgeless — only 11% of Mongo DBs have zero edges (mostly tiny
2-collection / EAI-sample DBs). So the residual is **mostly binary-brittleness (Factor A)**, with a
**smaller zero-edge Mongo subset (Factor B)**. Both factors must be isolated, not conflated.

---

## 2. Two factors to ablate

### Factor A — Connectivity scoring (binary → graded)

| Arm | definition | invariant note |
|---|---|---|
| **A0** binary faithful (CURRENT LOCKED) | conn ∈ {0,1}: ∃ component covering all phrases | baseline |
| **A1** largest-component coverage ratio | conn = (max over components C of #phrases with ≥1 candidate in C) / #matched-phrases | graded, ∈ (0,1] |
| **A2** path-cost (graded reachability) | conn = e^(−λ·h), h = min total join-path length (hops) connecting one-per-phrase selection; no path → small floor, not hard 0 | graded by join complexity |

- A1 gives partial credit when *most* phrases sit in one component but a few stragglers don't —
  directly the soft analog of the BFS check. Cheap, deterministic.
- A2 borrows the keyword-search/Steiner-tree tradition (path length = join complexity).
- **λ (A2) and the floor are thesis hyperparameters → must be calibrated/ablated**, not magic.
- **MUST stay engine-neutral:** one formula for all engines. No per-engine constant, no per-engine
  threshold (engine-bias anti-pattern). The graded score reads the same on a PG FK edge, a Mongo
  `$lookup` edge, a Neo4j relationship edge.

### Factor B — Adjacency construction per engine

| Arm | PostgreSQL | MongoDB | Neo4j |
|---|---|---|---|
| **B0** current (CURRENT) | FK-style (Sudarshan-faithful) | current builder (11% 0-edge) | current builder |
| **B1** Mongo-aware | = B0 | + `$lookup` localField↔foreignField edges, + shared id-field naming heuristic (`<coll>_id`/ObjectId-typed), + nested-path co-residency (fields on same document path = connected) | = B0 |
| **B2** Neo4j-native | = B0 | = B1 | edges taken DIRECTLY from relationship types (graph schema = adjacency); bounded hop cap (ablate 3 vs 4) |

- B1/B2 only ADD edges from each engine's *native* join mechanism — they do not loosen the
  connectivity *criterion* (that stays the Factor-A formula). This keeps engines comparable: every
  engine's real join paths get represented, none gets a looser gate.
- B1 Mongo `$lookup`/shared-id edge model = **THESIS-ORIGINAL** (no paper formalizes a
  cross-collection join graph — see §4). Label explicitly; support by ablation.
- B2 Neo4j = clean adaptation (graph schema IS the adjacency), low originality risk.

---

## 3. Design matrix (factor isolation)

Run the cross of {A0,A1,A2} × {B0,B1}, plus A0×B2 and best-A×B2, on **ours_multidb** (the only
multi-engine set; spider/bird are PG-only so Factor B is inert there → use them only as A-factor
controls + regression guard that A1/A2 don't HURT the SQL sets).

| | B0 | B1 (Mongo) | B2 (Neo4j) |
|---|---|---|---|
| A0 binary | current locked | isolates Mongo adjacency | isolates Neo4j adjacency |
| A1 ratio | isolates soft-conn | A1 + Mongo | — |
| A2 path-cost | isolates soft-conn | best combo | best combo |

- **A0→A1, A0→A2 (B fixed):** effect of soft connectivity alone.
- **B0→B1 (A fixed):** effect of Mongo adjacency alone (expect gain concentrated on the 10 zero-edge
  Mongo DBs + Mongo multi-collection queries).
- **B0→B2 (A fixed):** effect of Neo4j native adjacency.
- Best A × best B = candidate new locked config.

---

## 4. Academic grounding (research 2026-06-18)

- **Soft/graded connectivity is the established norm**, not binary: keyword-search-over-RDB lineage
  (DISCOVER VLDB'02; BANKS VLDB/ICDE'02 — partial keyword coverage ranked, not hard-invalidated;
  survey IEEE DEBull'10) all score by Steiner-tree / join-path cost. Modern text-to-SQL:
  **SchemaGraphSQL** (arXiv 2505.18363, SOTA BIRD'25) replaces binary BFS with Dijkstra/A* path cost;
  **xDBTagger** (VLDB-J'23) ranks by shortest join path.
- **Why binary is risky here:** **EDBT 2026** [In-depth Analysis of LLM-based Schema Linking] —
  join-path errors are a distinct measurable failure category; **RSL-SQL** (arXiv 2411.00073) — ~6%
  omission at 94% recall → one missing edge zeroes the correct DB ~6% baseline. Grounds Factor A.
- **Neo4j adjacency = property-graph schema:** node labels = nodes, relationship types = edges —
  **Text2Cypher Schema Filtering** (arXiv 2505.05118), **CypherBench** (arXiv 2412.18702),
  **PG-Schema** (PACMMOD'23), **DiscoPG** (PVLDB'22). Grounds Factor B2.
- **Mongo cross-collection graph:** NO paper formalizes it — **TEND** (arXiv 2502.11201) and
  **DocSpider** (Cambridge NLP'25) treat each collection as a nested-path tree; cross-collection =
  `$lookup`/shared-key. Factor B1 = THESIS-ORIGINAL (label explicitly).
- **Heterogeneous schema connectivity gap:** only **ConnectionLens** (VLDB'18) spans
  relational+document+graph, but at DATA level (NER edges), not SCHEMA level → genuine contribution
  gap, not just engineering.

---

## 5. Metrics & rigor

- Two-layer: report retrieval recall@pool separately; judge arms on **R@1 | GT-in-pool**.
- **Per-engine breakdown mandatory** (Mongo/Neo4j are the Factor-B targets) + DB-macro headline.
- Stratified ≥1 query/GT-DB; bootstrap CI; **McNemar exact** before any "better" claim.
- Regression guard: A1/A2 and B1/B2 must NOT reduce PG R@1 on spider/bird.
- Error re-decomposition after each arm: scoring-fault (GT locked out) vs tie-fault, to confirm the
  lever moved the intended bucket.
- Invariant guards: scoring stays deterministic over LLM extraction; no LLM self-confidence; no
  per-engine constants/thresholds in the connectivity formula (engine-bias anti-pattern).

---

## 6. Run protocol (credit-thrifty — NO auto-run)

1. **Adjacency rebuild (Factor B):** B1/B2 need new adjacency.jsonl per engine. B2 (Neo4j) can be
   derived from existing schema (relationship types already in inventory) — likely no LLM. B1 (Mongo)
   `$lookup`/shared-id edges = derivable from schema/field names (heuristic, likely no LLM) +
   optionally LLM for nested-path co-residency. Audit first whether edges are derivable
   deterministically before spending LLM.
2. **Factor A is pure scoring (no LLM):** A1/A2 re-score from existing parse_v2 + map_cal_v2 caches →
   `--score-only` style, free. Run A-sweep first, read result, THEN decide if B is worth the
   adjacency rebuild.
3. Subset (≤12 GT-DB) directional first; scale to headline ~1000 only if subset promising
   (eval-slice-bias rule).
4. Tie-break cache invalidates whenever the score/near-set changes → bump a connectivity tag or move
   stale tie caches aside (as done for the faithful fix).

**Order of expected value:** A1/A2 (free, broad) → B2 Neo4j (cheap, clean) → B1 Mongo (original,
targets 11% subset). Start with the free A-sweep.

---

## 7. Open questions

- A1 vs A2: ratio is simpler + no hyperparameter; path-cost is more principled but adds λ. Prefer A1
  unless A2 measurably wins (Occam + fewer knobs to defend).
- Bounded hop for A2/B2: property graphs are near-fully-connected → need a hop cap or path-cost decay
  or connectivity is trivially 1. No principled default → ablate (3 vs 4) + acknowledge.
- Does soft connectivity re-introduce the saturation problem (more candidates near top → bigger
  near-sets → more agent load)? Monitor tie-rate per arm; soft conn may widen near-sets.
