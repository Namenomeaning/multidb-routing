# Agent final-routing — full eval + engine split + prompt v1 (phiên 2026-06-11→12)

Tổng hợp toàn bộ thí nghiệm phiên này: từ retrieval head-to-head → robustness 2 trục → luồng
agent final-routing → mổ lỗi → các fix → prompt v1 validate full → engine-split → so Sudarshan.

Benchmark `experiment/benchmark/v2` (sudarshan_repro/{spider,bird}_route SQL-only; ours_multidb 208 DB
PG/Mongo/Neo4j). Embedder `text-embedding-3-large` (OpenRouter). Rerank/agent LLM `deepseek-v4-flash`
(DeepSeek-direct cho tốc độ, fallback OpenRouter — cùng model). Slice stratified ≤5 câu/GT-DB.
**Two-layer metrics (CLAUDE.md §3): retrieval recall TÁCH final-routing R@1.**

---

## 1. Retrieval layer — card (ours) vs raw-DDL (Sudarshan)

Embed card (domain_description + term_glossary + entities/fields/declared-rel) vs raw schema DDL.
Cùng embedder, cùng pool, chỉ khác TEXT đem embed.

| Set | metric | raw | card | McNemar p |
|---|---|---|---|---|
| spider (206 DB) | R@1 | .614 | **.644** | .020 |
| | R@5 | .867 | **.887** | .036 |
| bird (80 DB) | R@1 | .706 | **.789** | 3e-5 |
| ours (208 DB) | DBmacro R@1 | .566 | **.613** | .10 |
| | DBmacro R@5 | .797 | **.869** | 8e-5 |

**Card > raw mọi set.** Raw-arm spider R@5 .867 ≈ Sudarshan công bố 87% → harness tái dựng trung thực.

---

## 2. Robustness — 2 trục lệch-chữ query↔schema

### 2a. Schema-side (E3c) — che tên schema, query giữ nguyên
Mask entity/field names → pseudonym; rebuild raw+card; re-eval.

| Set | metric | raw clean→obf | card clean→obf |
|---|---|---|---|
| spider | R@5 | .867 → **.043** | .887 → **.864** |
| bird | R@5 | .925 → **.123** | .935 → **.937** |
| ours | DBmacro R@5 | .797 → **.343** | .869 → **.836** |

→ Raw cưỡi trên trùng-chữ tên schema (sập khi che). Card name-independent (gần như không đổi).
Clean-edge của card KHÔNG phải leakage artifact.

### 2b. Query-side (Spider-Syn / Spider-Realistic) — benchmark peer-reviewed, đổi câu hỏi
Cùng index 206 DB spider, chỉ đổi QUERY.

| điều kiện | raw R@1 | card R@1 | raw R@5 | card R@5 | McNemar p (R@1) |
|---|---|---|---|---|---|
| Spider-Syn clean (1034 q) | .652 | .752 | .938 | .960 | — |
| Spider-Syn synonym (1034 q) | .491 | **.597** | .824 | **.884** | 1.4e-12 |
| Spider-Realistic drop-column (508 q) | .565 | **.707** | .896 | **.953** | 8.7e-12 |

R@5 drop clean→synonym: card −.076 (.960→.884) < raw −.114 (.938→.824) → **card bền hơn**.
Chênh card−raw R@1 TĂNG khi câu càng lệch tên schema (clean +.10 → synonym +.106 → realistic +.142).
= bằng chứng "cần semantic" trên benchmark thật. McNemar R@5 cũng card>raw (SYN p=8.1e-8, REAL p=1.5e-5).

**Kết luận lớp retrieval: card hơn raw, bền cả 2 trục lệch-chữ. Đây là claim retrieval mạnh nhất.**

---

## 3. Agent final-routing — các arm chọn top-1

Cùng pool top-5 (card cosine). 4 cách chọn:
- **retrieval**: cosine top-1, không rerank.
- **holistic**: agent đọc 5 card + điểm xác định → chọn (1 LLM call).
- **decomp**: per-candidate LLM extract → deterministic Coverage×Connectivity → argmax (= Sudarshan method).
- **hybrid**: cổng deterministic loại conn=0 → agent chọn trong survivor.

Coverage = e^(−n·x), x = unmatched/(matched+unmatched); Connectivity = BFS adjacency 0/1; nhân.
LLM CHỈ extract + chọn ứng viên, scoring là code (giữ invariant "không tự khai confidence").

### Sơ bộ slice ours 30-DB (111 câu, in-pool 90):
| arm | inst-acc in-pool |
|---|---|
| cosine top-1 | 0.689 |
| holistic | **0.822** |
| decomp (deterministic thuần) | 0.622 |
| hybrid (gate+agent) | 0.789 |

→ **Điểm xác định thuần TỆ NHẤT** (bão hòa thành cụm hòa 3-4 way + xếp sibling cao hơn GT).
Agent đọc card hơn hẳn. Cho agent **xem điểm** (holistic .822) > không xem (.789).

---

## 4. Mổ lỗi (hybrid, 19 câu sai / 90 in-pool)

| pattern | số | bản chất |
|---|---|---|
| cross-engine pick | 8 | domain nhân bản (mondial PG↔Mongo...), nhãn GT gần tùy ý |
| tie saturation | 4 | GT & pick cùng điểm đỉnh, tie-break quyết sai |
| gate giết oan GT | 4 | conn=0 nhưng coverage hoàn hảo → adjacency thiếu cạnh |
| extract map nhầm sibling | 2 | |
| agent override điểm | 1 | |

→ ~63% lỗi = domain trùng cross-engine (ambiguity benchmark), không phải thuật toán yếu.

---

## 5. Các fix đã thử — KHÔNG cái nào vượt holistic

| fix | kết quả slice | verdict |
|---|---|---|
| cổng mềm (conn=0 phạt thay vì loại) + thu hẹp cụm | best 0.789 | thu hẹp HẠI (bỏ rơi GT); penalty vô tác dụng |
| agent phá hòa (chỉ cụm hòa) | 0.667 | REGRESS (det-pick không đáng tin cả khi không hòa) |
| evidence giàu (card + điểm + phrase matched/unmatched) | 0.700 | HẠI có ý nghĩa (p=0.019); phrase nhiễu → agent đếm match |

→ Nút thắt KHÔNG ở cơ chế gate/tie/evidence. Holistic (card+điểm) là tốt nhất.

---

## 6. Prompt v1 "khớp domain cụ thể nhất" — validate FULL

Đổi câu lệnh holistic (engine-neutral): "chọn DB có domain khớp CỤ THỂ nhất, đừng chọn cái chỉ
trùng từ chung chung". Cùng block (card+điểm).

### Full set (stratified ≤5/GT-DB):
| set | N | pool rec@5 | v0 R@1 | **v1 R@1** | v1 in-pool | engine-acc | DBmacro | McNemar |
|---|---|---|---|---|---|---|---|---|
| **spider** (SQL) | 1026 | .887 | .664 | **.683** | .770 | — | .683 | **p=.0495** |
| **ours** (đa-engine) | 743 | .848 | .592 | **.627** | .740 | .789 | .637 | **p=.006** |

**v1 > v0 significant cả 2 set.** Giữ ở full scale (slice trước 0.822 lạc quan; full .740 mới tin được
— bài học eval-slice-bias).

---

## 7. Engine-split — route INTRA-engine (cô lập cross-engine)

| set | #DB | N | pool rec@5 | v0 R@1 | v1 R@1 | v1 in-pool | McNemar |
|---|---|---|---|---|---|---|---|
| Mongo | 87 | 361 | .889 | .659 | .681 | .766 | p=.20 (n.s.) |
| Neo4j | 27 | 122 | .992 | .828 | .803 | .810 | p=.45 (n.s.) |
| (mixed đa-engine) | 208 | 743 | .848 | .592 | .627 | .740 | p=.006 |

**2 phát hiện:**
1. **Intra-engine DỄ hơn mixed.** Mixed (.627) THẤP hơn trung bình per-engine → phần lớn lỗi multi-DB
   là **nhầm cross-engine** (domain replicate), không phải phân biệt instance trong engine.
2. **Lợi ích v1 chỉ significant khi CÓ cross-engine** (spider/mixed). Trong 1 engine (mongo/neo4j)
   v1≈v0 n.s. → v1 đánh trúng đúng bẫy trùng-từ cross-engine.

⚠️ Neo4j 27 DB / 122 câu: pool nhỏ (dễ) + N nhỏ (CI rộng) → kết luận yếu. Mongo 87 mới đủ chắc.

---

## 8. So Sudarshan (.7865 final R@1 Spider) — thành thật

- v1 spider overall **.683 < .7865**.
- Retrieval **tái dựng đúng** (recall@5 .887 ≈ 87% họ).
- Gap ở **bước rerank**: họ chuyển 87%→78.65% (in-pool ~.90); ta in-pool .77.
- **KHÔNG apples-to-apples**: embedder khác (text-3-large vs gte-Qwen2-7B-instruct), rerank LLM khác
  (deepseek-v4-flash vs lớp GPT-4), ta đa-engine (khó hơn). .7865 chỉ là mốc tham chiếu;
  so có kiểm soát = raw-vs-card cùng-model (đã làm, card thắng).

---

## 9. Hạ tầng — DeepSeek-direct

- `LLM_PROVIDER=deepseek` → DeepSeek API trực tiếp, model `deepseek-v4-flash` (id thật, có cả `-pro`).
- Nhanh **~2.4×** (≈5s vs 12s/call OpenRouter), throughput ~120-300 call/min @16-32 workers, 0 lỗi.
- **Cùng model y hệt** OpenRouter → output tương đương, không cần re-check chất lượng.
- Hết balance giữa chừng → resume OpenRouter (cache valid, cùng model).
- §11 cần ghi ngoại lệ LLM-only (CLAUDE.md write-protected — sửa tay): build+agent LLM dùng
  DeepSeek-direct được; **embedding VẪN bắt buộc OpenRouter**.

---

## 10. Chốt + việc còn treo

**Đã chứng minh:**
- Retrieval: card > raw, bền 2 trục lệch-chữ (E3c schema-side + Spider-Syn/Realistic query-side).
- Final routing: agent-đọc-card+điểm (holistic) > deterministic Cov×Conn (bão hòa); prompt v1
  "khớp cụ thể" cải thiện significant ở set có cross-engine.
- Multi-DB: phần khó = nhầm cross-engine (domain replicate), intra-engine dễ hơn nhiều.

**Còn treo:**
- Gap rerank vs Sudarshan (.683 vs .7865) — đẩy in-pool conversion (.77→?).
- Rescue flow cho ~15% GT ngoài pool (design xong, chưa implement).
- Parity model Sudarshan (gte-Qwen2) cần đổi §11 (local embedding) — không bắt buộc cho claim.
- Dọn benchmark: gộp/đánh dấu DB cross-engine gần trùng (giảm nhãn tùy ý) → instance-acc bật gần engine-acc.
- Adaptive θ-gate (số ứng viên co giãn theo độ chắc) — chưa triển khai trong luồng này.
- Mở rộng Neo4j inventory (>27) cho kết luận chắc hơn.

---

## Scripts phiên này (exp_v2/)

`obfuscate.py`, `e3c_eval.py` (E3c) · `robustness_query_eval.py` (Spider-Syn/Realistic) ·
`agent_rerank.py` (4 arm) · `agent_error_analysis.py` (mổ lỗi) · `hybrid_v2_eval.py` (cổng mềm) ·
`holistic_evidence_eval.py` (evidence + engine/instance split) · `prompt_variant_eval.py` (v0-v3) ·
`full_eval_v1.py` (full + `--engine` intra) · `deepseek_throughput.py` (đo tốc độ).
Caches: mỗi set `agent_cache/{extractions,pv_*}.jsonl` (reuse, rerun không tốn LLM).
