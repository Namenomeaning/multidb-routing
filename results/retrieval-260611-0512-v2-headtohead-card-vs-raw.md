# Retrieval head-to-head — card (ours) vs raw-DDL (Sudarshan), v2 benchmarks

Date: 2026-06-11 · benchmark `experiment/benchmark/v2` · embedder `text-embedding-3-large`
(OpenRouter, BOTH arms — only the indexed TEXT differs). Slice: stratified ≤5 queries/GT-DB.
Layer: **retrieval only** (1 view/DB, cosine top-k). Agent/rerank layer NOT included here.

## Setup

- **raw arm** = embed `schema_text` (raw DDL) = Sudarshan baseline representation.
- **card arm** = embed B4 render: card(domain_description + term_glossary) + A(entities/fields/declared rel).
- Same embedder + same DB pool + same queries → only representation differs (clean E1 / head-to-head).
- Build: `exp_v2/build_semantic.py` (card+adjacency, deepseek-v4-flash) → `build_index.py` → `retrieval_eval.py`.

## Results (clean condition)

| Set | metric | raw | card | Δ | McNemar p |
|---|---|---|---|---|---|
| spider_route (206 DB, 1026 q) | R@1 | .614 | .644 | +.030 | .020 |
| | R@5 | .867 | .887 | +.019 | .036 |
| | mAP | .726 | .751 | +.025 | — |
| bird_route (80 DB, 398 q) | R@1 | .706 | .789 | +.083 | 3e-5 |
| | R@5 | .925 | .935 | +.010 | .50 (ceiling) |
| | mAP | .804 | .857 | +.053 | — |
| ours_multidb (208 DB, 743 q) | DBmacro R@1 | .566 | .613 | +.047 | .10 (borderline) |
| | DBmacro R@5 | .797 | .869 | +.072 | 8e-5 |
| | mAP | .678 | .714 | +.036 | — |

GT_unfound = 0 everywhere (every GT DB has a vector).

## Reading

- **Card > raw on all 3 sets, all metrics.** Significant (McNemar, continuity-corrected) at the
  metrics that matter: spider R@1 & R@5, bird R@1 (strong), ours-multitype R@5 macro (strong).
  Not-significant cases are ceiling (bird R@5 ~93% both) or borderline (ours R@1 p=.10).
- **Reproduction credible**: raw-arm spider R@5 = .867 ≈ Sudarshan published 87.0% pre-rerank
  → our reconstruction + harness are faithful; the comparison baseline is trustworthy.

## E3c obfuscation control (RUN 2026-06-11) — PASS

Mask schema identifier tokens (entity + field names) → stable per-DB pseudonyms, consistently
across schema_text + inventory + card. Queries stay CLEAN (real domain words; cannot exploit
masked names). Rebuild raw+card indexes from obfuscated text, re-eval same stratified slice.
Scripts: `exp_v2/{obfuscate,e3c_eval}.py`. Obf benchmarks: `…_obf/` siblings.

| Set | metric | raw clean | raw obf | card clean | card obf | raw drop | card drop |
|---|---|---|---|---|---|---|---|
| spider | R@5 | .867 | **.043** | .887 | **.864** | −.825 | −.023 |
| spider | R@1 | .614 | .012 | .644 | .603 | −.602 | −.041 |
| bird | R@5 | .925 | **.123** | .935 | **.937** | −.802 | +.003 |
| bird | R@1 | .706 | .073 | .789 | .721 | −.633 | −.068 |
| ours | DBmacro R@5 | .797 | **.343** | .869 | **.836** | −.454 | −.033 |
| ours | DBmacro R@1 | .566 | .191 | .613 | .584 | −.341 | −.038 |

McNemar card_obf vs raw_obf @5: spider 847 vs 5, bird 325 vs 1, ours 328 vs 33 — card wins overwhelmingly under obfuscation (all p≪1e-10).

**Reading.**
- **Raw-DDL retrieval rides almost entirely on schema-name↔query lexical overlap.** Mask names →
  raw collapses (SQL sets −80 to −83 pp; multi-engine −45 pp, less because the `Database:`/engine
  structural tokens survive — conservative, favours raw).
- **Card retrieval is name-independent** (card_drop ≈ 0 everywhere; bird even +.003). Its signal lives
  in the NL domain prose, not in identifier matching.
- **Verdict:** the clean-condition card edge is NOT a name-leakage artifact — card does not use
  name-matching, so its clean number is honest. The card's decisive value is robustness to weak
  query↔schema lexical overlap = a real, benchmarked problem. DBCopilot (2312.03463, EDBT 2025,
  VERIFIED) tests the QUERY side of this axis via Spidersyn (synonym-substitute schema words in the
  question) + Spiderreal (drop explicit column mentions), where it leads baselines by DB R@1 +14.89%
  / +19.88%. Our E3c tests the SCHEMA side (mask schema names) — complementary. (Earlier draft
  mis-stated DBCopilot's +19.88% as "obfuscated terminology"; it is the query-side robustness lead.)
  Stronger external-validity move than synthetic masking: run card-vs-raw on Spidersyn/Spiderreal.

**Scope (per memory `semantic-beats-raw-obfuscated`, NOT overclaimed):** this proves QUERY-TIME name
independence. The card was still BUILT reading the real (clean) schema — legitimate build-time
documentation, not test leakage. NOT a "proxy-free end-to-end" claim. Variant where the card is
re-built FROM masked schema (card-from-masked) is a separate, harsher test (prior standard3 work:
collapses to .104) — not run here; this control answers the leakage question the head-to-head needed.

- This is retrieval recall only. Final routing (agent rerank: Coverage×Connectivity + agent pick)
  is a separate layer — do not conflate (CLAUDE.md §3).

## Artifacts

- Indexes: `benchmark/v2/{sudarshan_repro/spider_route,sudarshan_repro/bird_route,ours_multidb}/index/{raw,card}.npy`
- Semantic: same dirs `…/semantic/{cards,adjacency,inventory}.jsonl` (494 DB, 100% query-informed, 0 parse-fail, 1 benign density flag)
- Scripts: `exp_v2/{build_semantic,build_index,retrieval_eval,smoke_retrieval}.py`
