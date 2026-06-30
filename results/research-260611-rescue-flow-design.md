# Rescue flow — research + grounded design

Date 2026-06-11. Question: when the first retrieval+rerank pass is weak, can a second-chance
"rescue" pass recover the ground-truth DB? Grounded in VERIFIED primary sources; mapped to the
gaps WE measured this session. NOT yet implemented — design + experiment proposal.

## The gap rescue targets (measured this session)

1. **GT outside the retrieval pool** — the hard ceiling. card recall multi-DB R@5 .82 / R@10 .92,
   spider R@5 .89 / R@8 .94. So ~8-18% (multi-DB) of queries the GT is NEVER retrieved → no rerank
   can fix it (Sudarshan: final R@1 capped by R@5). This is the single biggest lever left.
2. **GT dropped by the hybrid gate** (extraction marks GT connectivity=0) — ~2pp, secondary.
3. **Saturated tie / low max score** — score landscape flat, decision low-evidence.

## What rescue is, in the literature

A rescue flow = **trigger-conditioned second retrieval with query expansion**. Two parts:

### Trigger — WHEN to rescue (external signals only, invariant-safe)
- **FLARE** [arXiv:2305.06983, EMNLP 2023, VERIFIED]: re-retrieve when next-token confidence < θ
  (θ=0.8 in paper). Structural analog for us: trigger when an EXTERNAL signal is weak —
  - max rerank score < θ_score (no candidate covers the query well), OR
  - margin (score₁ − score₂) < δ (saturated tie — exactly our measured failure mode), OR
  - retrieval cosine margin small (Signal 2, m3-design-decisions).
- All are externally computed → NO LLM self-confidence → invariant preserved.

### Expansion — HOW to rescue (generate, then re-retrieve)
- **CRUSH4SQL** [arXiv:2311.01173, EMNLP 2023, VERIFIED] — closest analog: hallucinate a schema
  fragment for the question, retrieve schemas by it. Built for large/overlapping schema retrieval =
  our exact setting.
- **HyDE** [2212.10496, ACL 2023] / **Query2Doc** [2303.07678, EMNLP 2023]: generate a hypothetical
  answer/doc, embed it, retrieve. +6.8 NDCG@10 (HyDE BEIR); +3-15% (Query2Doc BM25).
- **DBCopilot** [2312.03463, EDBT 2025, VERIFIED 2026-06-11 via full HTML]: trained differentiable
  search index for schema routing. Main DB R@1: Spider 85.01%, Bird 88.92% (TRAINED — not directly
  comparable to our training-free dense). Robustness benchmarks = **Spidersyn** (synonym-substitute
  schema words in the QUESTION) + **Spiderreal** (remove explicit column-name mentions from the
  QUESTION) — QUERY-side lexical mismatch, NOT schema-name obfuscation. DBCopilot's lead over
  baselines: DB R@1 +14.89% (Spidersyn) / **+19.88% (Spiderreal)**; Table R@5 +2.83% / +6.94%.
  (CORRECTION: earlier notes mis-stated this as "+19.88% on obfuscated terminology" — it is the
  query-side robustness advantage, not schema obfuscation.) Relevance: query↔schema lexical mismatch
  is a real, peer-reviewed-benchmarked problem; our E3c tests the SCHEMA-side of the same axis.
- Mechanism for us: LLM emits expansion terms / a hypothetical schema fragment for the question →
  fold into the retrieval query → re-embed → pull a WIDER pool.

### Coverage / non-exclusion (our own measured lessons)
- **RouterRetriever** [2409.02685, AAAI 2025, VERIFIED]: hard routing leaves a 7.8 nDCG gap vs
  oracle; keep coverage ≥1/engine on the widened pool, never hard-filter.
- **DO NOT exclude the original pool ids from the re-search** — the removed `rescue exclude-ids` bug
  (m3-design-decisions, 2026-06-10) banned the GT from the 2nd search. The GT is often *just outside*
  top-K; excluding it defeats the rescue. Widen K instead.

## Proposed rescue flow (invariant-safe, grounded)

```
pass 1: retrieve top-K (card) → extract → gate → agent pick   (the hybrid flow)
compute external triggers: max_score < θ_score  OR  (s1 − s2) < δ
if NOT triggered → return pass-1 pick
if triggered → RESCUE:
  1. LLM expands query: hypothetical schema fragment + domain terms (CRUSH4SQL/HyDE), extraction-only
  2. re-embed expanded query → retrieve WIDER pool (K' > K, coverage ≥1/engine, NO id exclusion)
  3. re-run extract → gate → agent pick on the widened pool
  return rescued pick
```

## Counter-evidence to honor (mandatory in any claim)

- Query expansion HURTS on ambiguous/unfamiliar queries + popularity bias [arXiv:2505.12694].
- Multi-query fusion gains can vanish after reranking (Hit@10 51.3→47.8) [arXiv:2603.02153].
→ Must report: no-rescue baseline; rescue applied ONLY to the triggered subset; whether rescued GT
  was previously out-of-pool; net ΔR@1 with CI. Never claim rescue is free.

## Experiment plan (small-scale, when run)

- Arms: hybrid (no rescue) vs hybrid+rescue. Metrics two-layer: Δ pool-recall on triggered subset
  (did rescue pull GT in?) + Δ final R@1. Report trigger rate + cost (extra LLM calls/query).
- Calibrate θ_score, δ on a small slice (thesis hyperparams, no literature default → ablation).
- Primary target = the ~18% multi-DB out-of-pool queries; success = rescue lifts pool recall there
  without hurting the non-triggered majority.
