# Đặc tả benchmark hoàn chỉnh — Định tuyến truy vấn đa-engine (so Sudarshan)

**Ngày:** 2026-06-18 · **Mục đích:** chốt cấu trúc + quy mô + giao thức benchmark đủ chặt để bảo vệ khoa học, trước khi chạy. Gộp: (a) source-of-truth `BENCHMARK.md`, (b) kết quả isolation P1 mục 4–5 (2026-06-18), (c) các control mà peer-review yêu cầu.

> Đây là SPEC để chốt — chưa phải lệnh chạy. Sau khi user confirm, mới execute (HARD GATE no-auto-run).

---

## 0. Kết quả P1 mục 4–5 (cache-only, chạy 2026-06-18) — đưa vào để hiệu chỉnh claim

**Mục 4 — cô lập đóng góp tầng lọc domain (triage):** so recall pool top-10 (chưa lọc) vs pool đã lọc, McNemar trên cùng query.

| Set | recall top-10 | recall sau lọc | GT bị bỏ | McNemar p | kết luận |
|---|---|---|---|---|---|
| MultiDB-Route | .903 | .882 | 8 q (2.4% GT-in-pool) | **.008** | lọc CÓ phí recall, nhỏ nhưng significant |
| Spider-Route | .960 | .950 | 5 q (1.0%) | .063 | không significant |
| Bird-Route | .956 | .937 | 4 q (2.0%) | .125 | không significant |

→ **Hiệu chỉnh claim RQ2:** tầng lọc domain KHÔNG phải bộ tăng recall. Recall tăng so SOTA là nhờ **card + lấy rộng top-10**; bản thân agent lọc chỉ **nén pool 10→4.3** và phải trả ~1–2.4% GT (trên MultiDB là có ý nghĩa thống kê). Đóng góp đúng của RQ2 = "thu nhỏ pool giữ ~98% GT-in-pool", KHÔNG phải "agent reasoning làm recall cao hơn".

**Mục 5 — ties có phải do lọc domain gom near-duplicate cùng domain không:**

| Set | tie-rate (δ=0.2) | tie set cùng-engine | tie | pool 1-engine | tie | pool đa-engine |
|---|---|---|---|---|
| MultiDB-Route | 359/570 = 63%* | 26.5% | 52.7% | **65.0%** |
| Spider-Route | 66% | 100% (toàn PG) | 66% | — |
| Bird-Route | 52% | 100% (toàn PG) | 52% | — |

*359/743 = 48% nếu lấy mẫu số toàn bộ query (khớp số §5 cũ; 359 = số lần tie-break nổ).

→ **Phản biện lại C5 (reviewer nghi RQ2 tự chế ra ties):** trên MultiDB, pool **đa-engine ties NHIỀU hơn** (65% vs 53%), và **73.5% nhóm tie trải nhiều engine**. Ties KHÔNG tập trung ở pool cùng-domain → không phải do lọc domain gom near-dup. Gốc ties = **coverage bão hòa** (tín hiệu NA-rate hiếm khi kích hoạt trên pool nhỏ), hiện tượng nền không phụ thuộc engine mix. Đây là điểm RQ3 (tie-break) thực sự can thiệp.

---

## 1. Nguyên tắc thiết kế (grounding học thuật)

| Nguyên tắc | Áp dụng | Căn cứ |
|---|---|---|
| **Registry chia sẻ, split theo QUERY (không theo DB)** | mọi DB hiện diện lúc infer; chia query support/dev/test | Bài toán routing = chọn DB từ registry đã biết (Sudarshan arXiv:2601.19825); khác Spider/BIRD chia DB-disjoint vì chúng là generation |
| **Support ⟂ test (no leakage)** | card chỉ sinh từ support; support ∩ test = 0 (đã verify 3 bộ) | Spider train/dev/test disjoint; chuẩn ML chống leakage |
| **Dev ⟂ test, test đo MỘT lần** | dev tinh chỉnh prompt/θ/δ; test khóa | Tránh test-tuning leakage; chuẩn benchmark (BIRD dev 1.534 tách test ẩn) |
| **DB-macro headline + stratified ≥1 q/GT-DB** | mỗi DB trọng số bằng nhau, chặn DB phổ biến chi phối | memory `eval-slice-bias`: slice lệch đảo ngược kết luận 2 lần |
| **Hai lớp metric tách bạch** | retrieval recall@k ⟂ final R@1 | CLAUDE.md §3 |
| **Power đủ phát hiện ~5pp** | test ~1.000/bộ; per-engine ≥150–300 | Spider dev 1.034, BIRD dev 1.534, MS MARCO dev 6.980; ~1.000 cho DB-macro CI ±~3pp |

---

## 2. Kiến trúc split 3 tầng (chốt)

```
REGISTRY (tất cả DB, hiện diện lúc infer)
   │
   ├── SUPPORT  ≤5 q/DB   → chỉ sinh semantic card (index reserve, KHÔNG train)
   ├── DEV      ~400–500   → tinh chỉnh prompt / θ / δ / n   (khóa khỏi test)
   └── TEST     ~1.000     → đo MỘT lần, stratified, DB-macro headline
        (3 tập rời nhau theo query; card không bao giờ thấy dev/test)
```

**Quy mô đề xuất mỗi benchmark:**

| Benchmark | #DB | engine | support (card) | dev | **test (headline)** | cap/DB test |
|---|---|---|---|---|---|---|
| **MultiDB-Route** | 208 | PG94/Mongo87/Neo27→~30 | ≤5/DB (support 9.646 deduped) | **dev.jsonl 876** | **test.jsonl 3.220 (≥10 q/DB)** | — |
| Spider-Route (repro) | 206 | PG | ≤5/DB | ~960 | ~1.026 | 5 |
| Bird-Route (repro) | 80 | PG | ≤5/DB | ~970 | ~1.026 | 13 |
| intra-mongo (slice của MultiDB) | 87 | Mongo | — | — | **≥300** (hiện 516) | — |
| intra-neo4j (slice của MultiDB) | 27 | Neo4j | — | — | **≥150** (hiện 181) | — |

---

## 3. Mật độ query/DB — ĐÃ SỬA (densify 2026-06-19)

**Vấn đề phát hiện (đo trên test.jsonl gốc):** mật độ test quá thưa → recall per-DB không ước lượng được.

| Engine | #DB | q/DB med (cũ → mới) | DB <3q (cũ → mới) |
|---|---|---|---|
| PostgreSQL | 94 | **3 → 10** | 45% → **0%** |
| MongoDB | 87 | 9 → 10 | 19% → **0%** |
| Neo4j | 27 | 31 → 31 | 7% → **0%** |
| **Tổng** | 208 | **4 → 10** | — → **0%** |

**Cách sửa (leakage-safe, 0 LLM):** card chỉ đọc 5 câu support ĐẦU mỗi DB (`build_semantic.MAX_SUPPORT_QUESTIONS=5`, deterministic theo thứ tự file). Mọi câu support index ≥5 = **card CHƯA bao giờ thấy** → promote sang test an toàn. `exp_v2/recarve_pg_test_dense.py` đẩy spare card-unseen vào test tới sàn ≥10 q/DB mọi engine → **test-dense.jsonl** (3.220 câu, +895 PG/Mongo/Neo4j, −7 dup data có sẵn được dọn). Guard: 0 câu card-seen lọt test (assert PASS). So Spider chuẩn ~54 q/DB → vẫn thưa hơn nhưng đủ ước lượng aggregate + per-engine.

**Còn lại (không sửa được bằng re-split) — KHAI limitation:**
1. **Neo4j 27 DB** = trần dữ liệu (CypherBench 11 + Text2Cypher demo 18 + tomasonjo 1 ≈ 30); thêm ~3 graph thật → ~30. **THÊM (2026-06-20): ~80% câu Neo4j test là máy-sinh** (Text2Cypher-2024 `neo4jLabs_synthetic_gemini/gpt4o/...` 842/1049; người-viết chỉ 207 = CypherBench+crowdsourced) — đặc tính NGUỒN GỐC, không phải mình sinh, nhưng văn phong đồng nhất. → Càng phải báo **directional, CI rộng, KHÔNG claim engine-macro riêng**. Nếu cần "clean cut": báo phụ trên subset người-viết (207 câu, CI rất rộng).
2. **Per-DB recall:** med 10 q/DB → ±~15pp/DB, vẫn nhiễu → headline = **DB-macro aggregate**, KHÔNG báo per-DB từng cái. Per-engine DB-macro giờ đủ ước lượng.
3. **21/94 PG DB còn 5–9 q** (trần nguồn Spider/BIRD max 16 q/DB cho các DB này) → per-DB PG ở sàn nhiễu hơn; báo aggregate.
4. **5/87 Mongo DB single-entity** (mongo_eai sample_airbnb/geospatial/guides/supplies/weatherdata) → 1 collection, 0 cạnh. **ĐÍNH CHÍNH (2026-06-20, verify code+empiric):** `connectivity_faithful` trả **1** cho single-entity (1 node = liên thông tầm thường, đúng ∃-một-tổ-hợp Sudarshan) — **KHÔNG phải 0**. Không có penalty connectivity. (Claim cũ "10 DB connectivity=0 vĩnh viễn" là STALE/sai — nhầm với số ca conn-fail "7/50" metric khác.) Khó khăn thật của các DB này chỉ là coverage nếu câu cần nhiều entity mà DB chỉ có 1 — đó là hành vi ĐÚNG (DB thật không trả lời được), không phải bias. Mongo có cạnh: 77/87 (median 6) — KHÔNG degenerate.
5. **Cold-start ~20 PG DB** (0 support → card schema-only): tách nhóm trong phân tích per-engine.
6. **Spider/Bird GIỮ NGUYÊN** test 1.026 (cap5) — set tái dựng SOTA, mật độ khớp giao thức Sudarshan để giữ calibration (.867≈87%). Densify chỉ áp MultiDB-Route (primary thesis).

**Validator-fix (2026-06-20, `exp_v2/clean_splits_finalize.py`, 0 LLM):**
- **dev.jsonl tạo mới** (876 q, PG311/Mongo435/Neo130) từ support card-unseen index≥5, leakage-safe (dev∩test=0, dev∩card-seen=0) → θ/δ tune có holdout thật, hết test-tuning.
- **support dedup** 9.858→9.646 (−212 dup index≥5; **prefix 5-câu card bất biến** → audit leakage còn hiệu lực).
- **test.jsonl = bản dense 3.220** (gộp test-dense, xóa test-dense.jsonl + 7 dup gốc) → 1 canonical test.

**Verify mapping GT (2026-06-20, reviewer M1): κ N/A đứng vững.** Kiểm `(source_dataset, db_id) → instance_id` cho 1.260 câu Mongo: **0 key ánh xạ ≥2 instance** — mapping là hàm xác định, instance_id chỉ là db_id thêm prefix (vd DocSpider `farm` → `mongoscale_doc_033_farm`). KHÔNG có matching mờ/phán đoán tay ở bất kỳ engine nào → **không annotation step → Fleiss κ N/A** (khớp DBRouting + Sudarshan). Xác nhận quyết định bỏ T1/T2 + κ N/A.

---

## 3b. Phản biện reviewer C1 — "tautology engine-routing" (rebuttal, đếm được, 0 model)

**Cáo buộc:** GT label = nguồn dataset = engine → nhìn chủ đề/bề mặt câu là biết loại → routing dễ giả tạo.

**Bác bỏ bằng dữ kiện đếm 208 DB (không cần classifier):** domain TRỘN CHÉO engine — **26 domain xuất hiện trên ≥2 loại**, nhiều cái cả 3:
- `movie` → PG (`movie_3`, `movie_platform`, `movies_4`) + Mongo (`movie_1`, `film_rank`, `tvshow`) + Neo4j (`movie`, `movies`, `eoflix`, `recommendations`)
- `soccer`, `company`, `flight` → cả PG + Mongo + Neo4j
- customer, department, student, retail, election, network, twitter, club, loan, game, player, product… → PG+Mongo (vài cái + Neo4j)

→ Cùng chủ đề nằm trên nhiều loại ⇒ **biết chủ đề KHÔNG suy ra loại** ⇒ **tautology theo chủ đề KHÔNG tồn tại.** Đây là deliverable đúng: chọn 1 instance trong 208 (đa loại), không phải chọn loại.

**Residual đã biết (không phải tautology chủ đề):** rò **văn phong** ở ~80% câu Neo4j máy-sinh (đặc tính nguồn Text2Cypher) — khai ở §3.1, xử bằng báo Neo4j directional. Câu Neo4j người-viết + domain chồng chéo ⇒ route loại khó thật, không giả-dễ.

*(Ghi chú phương pháp: tautology được kiểm sơ bộ bằng partial-input probe đếm-từ — chỉ dùng định hướng, KHÔNG đưa số vào claim. Bằng chứng chính thức = domain-overlap đếm được ở trên.)*

---

## 4. Kiểm tra chất lượng TỪNG BƯỚC (giao thức đo)

Mỗi bước có metric riêng + control riêng. Không gộp.

### Bước 1 — Biểu diễn (RQ1: card vs raw DDL)
- **Metric:** recall@1/@5, DB-macro + micro.
- **Control 1 — robustness query-side Spider-Syn/Spider-Realistic (ĐÃ CHẠY 2026-06-11, `exp_v2/robustness_query_eval.py`):** đổi CÂU HỎI, schema giữ nguyên, trên **benchmark external peer-reviewed** (chính bộ DBCopilot dùng) — chứng minh card dùng ngữ nghĩa chứ không khớp-tên. Spider-Syn [Gan ACL 2021, 1034 paired]: card R@1 .752 vs raw .652 (gốc), .597 vs .491 (synonym); tụt R@5 card −.076 vs raw −.114; McNemar @1 p=1.4e-12. Spider-Realistic [Deng NAACL 2021, 508, xóa tên cột]: card R@1 .707 vs raw .565 = **+.142**; p=8.7e-12. **Đọc:** chênh card−raw TO RA khi câu lệch chữ với schema (clean +.10 → realistic +.142) → card dùng ngữ nghĩa, raw bám trùng-chữ. Report `retrieval-260611-robustness-query-side-spidersyn-realistic.md`.
- **⚠️ SCOPE robustness (re-scope 2026-06-20 sau peer review):** robustness query-side chỉ chứng minh trên **SQL/Spider (206 DB), retrieval layer**. KHÔNG test robustness riêng Mongo/Neo4j (control obfuscation từng phủ 3 engine đã bỏ). Cơ chế card = dịch build-time, giống mọi engine → chuyển sang Mongo/Neo4j theo THIẾT KẾ chứ không phải đo. **Đóng góp multi-type chống lưng bằng recall + R@1 đo thật trên 3 engine, KHÔNG dựa robustness.** KHÔNG phát biểu "card bền cả 3 engine". (Giải quyết reviewer C2: hổng chỉ thành CRITICAL nếu claim robustness multi-type — đã re-scope để không claim.)
- **(ĐÃ BỎ) obfuscation che tên schema-side (`tok_NNNN`):** perturbation tự chế, claim phải scope hẹp (card-build-từ-schema-bị-che thì sụp → không proxy-free hoàn toàn) → khó defend. Bỏ làm control; robustness do Control 1 query-side (external peer-reviewed) gánh. Số cũ giữ trong lịch sử EXPERIMENT-LOG, không dùng làm bằng chứng claim.
- **Control 2 (MỚI, reviewer C6 yêu cầu, CHỜ CHẠY):** arm **"LLM-paraphrase thường"** — mô tả DDL do LLM viết KHÔNG theo cấu trúc routing (không domain_description/term_glossary, chỉ kể lại schema). So với card. Mục đích: tách "card routing-informed thắng" khỏi "mọi paraphrase LLM đều thắng raw". *Nếu card ≈ paraphrase-thường → RQ1 chỉ là hiệu ứng generic, phải hạ claim.*

### Bước 2 — Retrieval (lấy pool)
- **Metric:** recall@5 (so trực tiếp Sudarshan), recall@10 (pool ta lấy rộng). Hai lớp.
- **Calibration:** raw spider R@5 ≈ .867 khớp công bố Sudarshan 87% (đã đạt).

### Bước 3 — Lọc domain (RQ2, ĐÃ cô lập ở P1 mục 4)
- **Metric:** gate-recall = GT sống sót vào pick; McNemar top-10 vs pick.
- **Control (MỚI, reviewer):** baseline lọc **không-LLM** (BM25/classifier domain) vs agent lọc — chứng minh "agent reasoning" hơn lọc cơ học. Nếu không hơn → gọi là "domain filter", bỏ chữ "agent reasoning".
- **Claim đã hiệu chỉnh:** "nén pool 10→4.3 giữ ~98% GT", không phải "tăng recall".

### Bước 4 — Rerank/scoring (Coverage×Connectivity)
- **Metric:** tie-rate; connectivity **fire-rate PER ENGINE** (reviewer C4). **Số đo thật từ adjacency.jsonl (validator 2026-06-20):** Mongo **77/87 DB CÓ cạnh** (median 6 cạnh/DB, 854 cạnh inferred — KHÔNG degenerate; "7/50" cũ là số ca conn-fail adjacency cứu, metric khác), Neo4j native (median 7), PG FK (median 3). Residual fairness = **10 DB Mongo single-entity** (không quan hệ → connectivity=0 vĩnh viễn) — khai limitation, phân tích tách riêng.
- **Đã chốt:** connectivity-faithful (∃-tổ-hợp liên thông), không gộp-hết. Column-level map đã thử & REJECT (A9b: gt_cov .869→.526) — entity-level là lựa chọn đo được, ghi nhãn THESIS-VARIANT.

### Bước 5 — Quyết định cuối (RQ3: agent tie-break)
- **Metric:** R@1 | GT-in-pool, V1 (argmax thuần) vs V2 (+tie-break). McNemar.
- **GIAO THỨC THỐNG KÊ BẮT BUỘC (reviewer C1):**
  - **Pre-register primary = MultiDB-Route, α=.05** trước khi chạy test cuối.
  - Spider/Bird = secondary calibration.
  - Pooled cross-set = phân tích **post-hoc**, ghi rõ nhãn, KHÔNG dùng làm headline significance.
  - Báo R@1 **cả conditional (|in-pool) lẫn unconditional** + McNemar trên unconditional (đừng bỏ 8–10% query GT-out-pool — đó là ca khó nhất).
- **Hiệu chỉnh kỳ vọng:** hiện p = .07/.60/.24 → RQ3 directional. Nếu test cuối vẫn không significant → **re-frame RQ3 thành "exploratory: agent tie-break giải ties coverage-bão-hòa"**, không bán là kết quả đã chứng minh.

---

## 5. Ma trận đo (cái gì chạy trên cái gì)

| Bước | MultiDB-Route | Spider-Route | Bird-Route | intra-mongo | intra-neo4j |
|---|---|---|---|---|---|
| 1 biểu diễn (card/raw/+paraphrase; robustness Syn/Realistic SQL) | ✓ headline | ✓ | ✓ | ✓ | ✓ |
| 2 retrieval R@5/@10 | ✓ | ✓ calib | ✓ calib | ✓ | ✓ |
| 3 lọc domain (gate + non-LLM baseline) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 scoring (tie + conn fire-rate/engine) | ✓ | ✓ (PG) | ✓ (PG) | ✓ (Mongo) | ✓ (Neo4j) |
| 5 V1/V2 R@1 + McNemar | ✓ **primary** | ✓ | ✓ | ✓ | ✓ (CI rộng) |

---

## 6. Giao thức thống kê (tổng)
- Headline = **DB-macro** recall/R@1; micro phụ.
- **Bootstrap CI** 1.000 resample cho mọi headline.
- **McNemar exact two-sided** paired cho mọi so sánh "tốt hơn".
- Hai lớp (retrieval ⟂ final) không bao giờ gộp.
- Pre-register primary set + α; post-hoc gắn nhãn.

---

## 7. Quyết định đã CHỐT (2026-06-19)

1. **Neo4j = max nguồn sạch ~30 DB, báo directional.** Trần graph Neo4j sạch ≈ 30 (CypherBench 11 + Text2Cypher demo 18 + tomasonjo 1), đã dùng 27 → thêm ~3 graph thật còn lại (2 text2cypher chưa dùng + 1 tomasonjo) lên ~30. KHÔNG tới 40 với dữ liệu sạch; 40 chỉ đạt bằng schema GPT-synthetic (đã loại vì không phải graph thật). → Neo4j báo **directional, CI rộng, KHÔNG claim engine-macro significance riêng engine**. Headline đa-engine vẫn DB-macro toàn registry.
2. **Chạy CẢ HAI control mới:** (i) arm **LLM-paraphrase-thường** [bước 1] — tách card-routing khỏi paraphrase generic; (ii) baseline **lọc domain không-LLM** (BM25/classifier) [bước 3] — tách agent khỏi lọc cơ học.
3. **Pre-register:** primary = **MultiDB-Route, α=.05**; Spider/Bird secondary; pooled cross-set = post-hoc có nhãn. R@1 báo cả conditional lẫn unconditional + McNemar trên unconditional.
4. **Densify (ĐÃ LÀM 2026-06-19):** test MultiDB-Route → `test-dense.jsonl` (3.220 câu, ≥10 q/DB mọi engine, 0 DB <3q), leakage-safe từ support card-unseen. Headline chạy trên test-dense, DB-macro full pool (không down-sample 1.000 nữa).
5. **BỎ HẲN nhãn độ khó T1/T2 (2026-06-20).** GT routing = source DB của câu hỏi, inherit 100% từ dataset gốc (deterministic, không annotate) → **không annotation step, Fleiss κ N/A by design** — khớp y DBRouting (arXiv:2501.16220) + Sudarshan (cả 2 cũng inherit, 0 κ). Cascade "khó → gọi agent" định nghĩa lại bằng **proxy tính được** (embedding-margin δ thấp / score-tie), KHÔNG phải nhãn tay → giữ invariant (không người/LLM tự khai độ khó). Đóng góp agent đo bằng ablation V1-vs-V2, không cần subset khó. `scenario60` (bucket T1/T2/T3) = legacy diagnostic debug-only, không vào claim.

---

## 8. Kế hoạch thực thi (sau khi user nói "chạy")

Thứ tự + ước tính. Mọi LLM qua DeepSeek thinking-OFF (trừ tie-break); embedding qua OpenRouter.

| # | Việc | LLM/embed mới | Ước thời gian |
|---|---|---|---|
| 8.0 | **Densify test MultiDB → test-dense.jsonl** (≥10 q/DB, leakage-safe) | 0 (re-split) | ✅ ĐÃ XONG |
| 8.1 | **Build ~3 DB Neo4j mới:** parse inventory + card (1 call/DB) + adjacency (1 call/DB) + embed | ~6 call + 3 embed | < 1 phút |
| 8.2 | **Control paraphrase-thường:** sinh paraphrase card cho 208(+3) DB (1 call/DB) + embed + index | ~211 call + embed | ~5 phút |
| 8.3 | **Control lọc không-LLM:** BM25/classifier domain filter (code, 0 LLM) | 0 | giây |
| 8.4 | **Retrieval headline cap=8 full slice** 3 bộ (card/raw/paraphrase/+obf) — score-only nếu cache đủ | embed lại các arm mới | ~1–2 phút |
| 8.5 | **Triage gate** 3 bộ + intra-mongo/neo4j (đã có method P1) | tái dùng cache, bù phần mới | ~1 phút |
| 8.6 | **Full flow V1/V2** trên eval slice 3 bộ + 2 slice engine, connectivity fire-rate per-engine, McNemar pre-registered | parse+map cho DB mới + slice mới | ~5 phút |
| | **Tổng compute** | | **~15 phút** |

Sau khi chạy: cập nhật `benchmark-260614-status-...md` §3–§5 với số headline cap=8 + 2 control + per-engine connectivity + McNemar pre-registered; cập nhật `BENCHMARK.md` registry Neo4j 27→~30.

**Chưa execute — chờ user "chạy".**

---

## 9. So vị thế thiết kế vs paper cùng lĩnh vực (2026-06-20)

So với 2 paper instance-level routing gần nhất: **Sudarshan (arXiv:2601.19825)** + **DBRouting/Mandal (arXiv:2501.16220, "Routing End User Queries to Databases for Answerability", CIKM 2024 GenAI/RAG workshop)** — chính paper thesis EXTEND.

| Trục | DBRouting (2501.16220) | Sudarshan (2601.19825) | OURS |
|---|---|---|---|
| Loại routing | instance-level, **SQL-only** | instance-level, **SQL-only** | instance-level, **multi-type PG/Mongo/Neo4j** |
| Registry #DB | 160 / 80 | 206 / 80 | **208** (+ repro 206/80) |
| #test query | ~2.075 / ~3.013 | 5.939 / 5.501 | **3.220** (3 engine) |
| q/DB density | ~57 | ~57–137 | **~15.5** ⚠️ thấp hơn |
| Split/leakage | DB-disjoint, **không check** | query-level, **không check** | query-level, **support∩test=0 verified** |
| Hai lớp metric | gộp R@k | gộp R@k | **tách retrieval ⟂ R@1** |
| Significance test | **không** | **không** | **McNemar + bootstrap + robustness control (Spider-Syn/Realistic)** |
| Annotation κ | inherit GT, 0 κ | inherit GT, 0 κ | inherit GT, 0 κ (T1/T2 đã bỏ) |

**Verdict:** OURS **vượt chuẩn** ở multi-type scope, statistical rigor, tách 2 lớp metric, leakage check (cả 2 paper đều thiếu). **Ngang** registry scale + split protocol. **Thiếu:** Neo4j 27 DB (13% registry, directional), q/DB density thấp hơn (bù bằng DB-macro + bootstrap), chưa public release. Gap κ researcher nêu → **đã đóng bằng quyết định §7.5** (bỏ T1/T2 → GT inherit → κ N/A, đúng như chính 2 paper này).
