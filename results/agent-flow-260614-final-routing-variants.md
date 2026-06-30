# Agent-flow behavior probe — final routing over the triaged pool (100 q/set, 3 DB types)

Date: 2026-06-14 · benchmark `experiment/benchmark/v2` · triage+decision LLM `deepseek-v4-flash`.
Behavior study (≈100 engine-balanced queries per set, DIRECTIONAL per eval-slice-bias rule — not a
headline claim). Script `exp_v2/agent_flow_eval.py`. Retrieval layers (card top-10 → domain triage)
reused from the pipeline-benchmark cache; this probes only the FINAL routing decision over the ~3
triaged candidates.

## Question

After triage narrows the pool to ~3 same-domain candidates, how should the single final DB be chosen?
Three variants, same triaged pool + same parsed phrases:

- **V1 cov-argmax** — Coverage(e^−n·x) × Connectivity(BFS over per-engine adjacency); top score wins,
  ties broken by pool order (arbitrary). = Sudarshan-style deterministic rerank, post-triage.
- **V2 cov+agent-tiebreak** — V1, but when the top score TIES, an agent reads the tied candidates'
  full cards + why-they-tied (matched entities) + ordered criteria (domain → entities → relationships)
  and picks one. (hybrid gate+agent.)
- **V3 agent-on-picks** — agent reads ALL ~3 triaged cards and picks directly; coverage unused.

Coverage/Connectivity are engine-agnostic at scoring time; per-engine adaptation (PG FK / Mongo
doc-path co-residence / Neo4j graph rels) lives in the prebuilt adjacency graph → one flow, 3 types.
INVARIANT: every variant's agent makes a routing CHOICE among given options, never a self-reported
confidence number fed into a formula (m3-design-decisions preserved).

## Results — R@1 | GT-in-triaged-pool (routing layer isolated)

| Set | ties% | avg tie | V1 cov-argmax | V2 cov+tiebreak | V3 agent-on-picks |
|---|---|---|---|---|---|
| ours_multidb (multi-type, 88 in-pool) | 47% | 3.36 | 0.739 | 0.807 | **0.898** |
| spider_route (SQL, 95 in-pool) | 42% | 3.98 | 0.747 | 0.811 | **0.916** |
| bird_route (SQL, 92 in-pool) | 25% | 2.92 | 0.891 | 0.902 | **0.935** |

(R@1 over all 100 incl. triage misses: ours .650/.710/.790, spider .710/.770/.870, bird .820/.830/.860.)

## Reading

- **V3 > V2 > V1 on all three DB types, consistently.** Agent reading the triaged cards directly is
  the best final-routing method; the deterministic Coverage×Connectivity (V1) is the weakest.
- **Why coverage loses: saturation.** 25–47% of queries end in a score TIE (avg 3–4 candidates at the
  max score), so the deterministic rank can't separate same-domain candidates — exactly the force-map
  failure measured earlier (coverage credits a candidate for mapping peripheral phrases regardless of
  whether the central entity truly exists). Post-triage the pool is ALL same-domain, so coverage's
  one discriminator (N/A failure-to-map) rarely fires.
- **Tie-break agent (V2) helps but is dominated.** V2 only overrides coverage on ties; it still trusts
  coverage's non-tie #1, which is itself unreliable. Letting the agent decide the whole ~3-way choice
  (V3) is better than trusting coverage anywhere.
- **Negative result, reported honestly:** Coverage×Connectivity (the Sudarshan-grounded deterministic
  mechanism) does NOT add value once triage has narrowed to same-domain candidates. The thesis's
  structural score is best framed as the BASELINE that triage + agent-card-reading beats, not as the
  winning decision rule.

## Recommended complete flow

```
query → card embed top-10 → domain-relatedness triage (desc) → ~3 same-domain candidates
      → agent reads the ~3 cards (domain + entities + relations) → picks the single DB     [V3]
```

Coverage×Connectivity retained as a reported baseline/ablation (V1), and the criteria-driven
tie-break (V2) documented as the invariant-preserving hybrid — but V3 is the empirically optimal
decision rule across all three DB types.

## Caveats

- 100-query behavior probes, directional only (eval-slice-bias rule). A headline R@1 claim needs the
  full stratified slice; this study answers "which decision rule" and "what behavior", not the final
  number.
- V3 is the agent-judgment arm — must be presented as a final-routing CHOICE (invariant-preserving),
  with the deterministic V1/V2 alongside for transparency.
- Triage misses (empty/over-pruned pool) cap V3 too: bird had a few empty-pool queries (triage dropped
  all) → counted as misses in R@1(all). Triage recall is the upstream ceiling (Phase-2 benchmark).
