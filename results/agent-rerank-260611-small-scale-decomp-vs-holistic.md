# Agent rerank (final-routing layer) — small-scale, decomp vs holistic vs retrieval

Date 2026-06-11 · sets `benchmark/v2/{sudarshan_repro/spider_route, ours_multidb}` · build LLM
deepseek-v4-flash · cards via `exp_v2/build_semantic.py` · harness `exp_v2/agent_rerank.py`
(extractions cached → n/K sweep + reruns cost no LLM calls).

**Layer = final routing only** (CLAUDE.md §3), reported separately from retrieval recall.
Small slice: 12 GT-DBs/set (engine-balanced for multi-DB), ≤3 queries/DB (36 q/set).
Sample is small (in-pool n=35 spider / 27 multi-DB) — directional, NOT a claim.

## Arms

- **retrieval** — top-1 by card cosine (no rerank). Baseline.
- **holistic** — one LLM call sees the top-5 candidate CARD descriptions + their deterministic
  scores, picks one. Reproduces the user-observed "LLM-as-ranker" setup.
- **decomp** — per-candidate LLM extraction (schema-aware, extraction-only: map query phrases →
  this schema's entities, list unmatched) → deterministic Coverage(e^−n·x)×Connectivity(BFS) →
  argmax. Ours / Sudarshan 2601.19825. Tie-break = cosine order.

## Results (R@1 conditioned on GT-in-pool = rerank quality)

| Set | K | poolRecall* | retrieval | decomp | holistic | hybrid |
|---|---|---|---|---|---|---|
| spider | 5 | .889 | .556 | .625 | .625 | **.656** |
| spider | 8 | .972 | .556 | .571 | .571 | **.600** |
| multi-DB | 5 | .694 | — | .680 | **.840** | **.840** |
| multi-DB | 8 | .750 | — | .704 | .778 | .778 |

n ∈ {1,2,3} → identical at every row (coverage saturates; penalty irrelevant).
\* poolRecall on this 12-DB mini-slice is biased low; proper full-slice card recall (743/398/1026 q,
≤5/GT-DB) = multi-DB R@5 .85 / R@8 .90 / R@10 .92; spider R@5 .89 / R@8 .94. Retrieval is NOT the
final-routing bottleneck — the rerank layer is.

- **hybrid** = deterministic gate (drop connectivity=0 candidates) → agent picks among survivors via
  cards. Matches the best arm (holistic) on multi-DB, edges it on spider, AND keeps the invariant
  (agent SELECTS within a gated set, emits no confidence). Recommended design.

## Bigger validation (25 GT-DB/set, ≤5 q/DB, K=5) — trustworthy headline

In-pool n = 114 (spider) / 88 (multi-DB). R@1 | GT-in-pool:

| set | decomp | holistic | hybrid |
|---|---|---|---|
| spider | .614 | .763 | .746 |
| multi-DB | .545 | .795 | .761 |

McNemar exact (in-pool, K=5):
- hybrid vs decomp: spider p=.0315 (29 vs 14), multi-DB **p=.0013** (26 vs 7) — agent-card beats deterministic, significant.
- holistic vs decomp: spider p=.0023, multi-DB p<.0001 — same direction, stronger.
- hybrid vs holistic: spider p=.81, multi-DB p=.61 — **no significant difference** (the gate neither helps nor hurts accuracy; it buys invariant-compliance for free).

Diagnostic (multi-DB, n_in=88): 26 GT-alone-at-top, 32 ties (avg 3.0), **30 GT-below-top** — extraction noise pushes GT below a sibling on a third of cases; this is the decomp ceiling the agent recovers.

**Verdict:** scale **hybrid** — statistically tied with the best arm (holistic) while keeping the
"LLM never self-evaluates" invariant (deterministic gate + agent selection, no confidence number).
Deterministic Coverage×Connectivity argmax alone is the worst arm and should NOT be the final picker.
Gate cost: when extraction wrongly marks GT disconnected (conn=0) the gate drops GT (~2pp, n.s.) —
a softer gate is a tunable follow-up.

## decomp diagnostic (K=8, n=2, in-pool only) — initial 36-q slice

| | spider (n_in=35) | multi-DB (n_in=27) |
|---|---|---|
| GT alone at top score (decomp right) | 6 | 9 |
| GT **tied** at top (tie-break decides) | 21 (avg tie 3.8) | 11 (avg tie 4.3) |
| GT below top (a sibling scores higher) | 8 | 7 |

## Reading

1. **The deterministic score is non-discriminative on a top-K same-domain pool.** Once retrieval
   has narrowed to K near-duplicates, they ALL cover the query and ALL connect → 3-4 way ties at
   score 1.0. ~60% (spider) / ~40% (multi-DB) of in-pool decisions are pure ties; the tie-break,
   not the score, decides. This — not "the agent ignores the score" — is the mechanistic reason
   scoring feels useless: in the tie regime there is no signal in it to respect. Tuning n/θ/δ on
   this score is futile (coverage = 1.0 for every top candidate regardless of n).

2. **With semantic cards, the LLM selector (holistic) beats deterministic argmax on multi-engine**
   (.84 vs .68 in-pool, K=5). The card domain prose carries the discriminative signal that
   Coverage×Connectivity discards; the LLM uses it. This REVERSES Sudarshan's holistic<decomposed
   finding — because Sudarshan's holistic baseline read raw DDL, ours reads cards. Pro-card result.

3. **Where cards themselves are near-identical, both arms cap together** (spider academic cluster:
   holistic = decomp ≈ .57). The GT is genuinely indistinguishable from siblings by the question
   alone = Sudarshan's intra-domain confusion (86.2% of their failures). No rerank mechanism fixes
   an undistinguishable-by-design case.

4. **Both arms are bounded by retrieval recall.** multi-DB pool recall only .69 (K5)/.75 (K8) on
   this slice → 25-30% of queries the GT is never retrieved → unfixable downstream. Improving the
   retrieval pool (or K) is the larger lever for multi-DB right now.

## Proposed design — gate-then-agent-tiebreak (needs GVHD/user sign-off)

The score IS discriminative at the LOW end (connectivity=0 / low coverage cleanly kills wrong-domain
or disconnected candidates); it saturates only at the top. So:
- use deterministic Coverage×Connectivity to **filter** the pool (drop conn=0 and clearly-low
  coverage) — this is the threshold "not 100%-match-only" the user asked for;
- among the surviving high-coverage tie-set, let the **agent pick using card descriptions**
  (the holistic judgment that empirically wins).

Grounding: agent tie-break beat semantic-similarity tie-break before (.668 vs .498, McNemar
p=2.2e-4; EXPERIMENT-LOG Tier-2). Invariant check: the LLM does final SELECTION within a
deterministically-gated set, emits NO confidence number → consistent with the "no LLM
self-confidence" rule's letter. Pure holistic-over-full-pool drifts toward "LLM evaluates"; the
deterministic gate is what keeps it disciplined. NOT yet tested — next experiment.

## Caveats

- Small sample (≤36 q/set); magnitudes noisy, mechanism (ties) is the robust takeaway.
- `entity: null` from the extractor handled (filtered); a few JSON retries on deepseek-v4-flash.
- Artifacts: `…/agent_cache/{extractions,holistic}.jsonl` per set; script `exp_v2/agent_rerank.py`.
