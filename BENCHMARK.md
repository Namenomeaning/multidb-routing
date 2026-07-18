# BENCHMARK.md — Benchmark & Semantic Layer Specification

Đặc tả **thiết kế + giao thức** của benchmark: registry, splits, cấu trúc semantic
card / adjacency đã chốt, và danh sách ablation bắt buộc. **Không chứa số kết quả** —
mọi con số đo (recall@k, R@1, p-value, kích thước pool) sống ở một nơi duy nhất:
[`results/RESULTS.md`](results/RESULTS.md).

---

## 1. Benchmark hiện hành — `standard3_scale_v1`

| Thành phần | Giá trị |
|---|---|
| Registry | 208 DB: PostgreSQL 94 / MongoDB 87 / Neo4j 27 |
| Nguồn | PG 94 = Spider 68 + **BIRD 26**; Mongo 87 = DocSpider 72 + MongoDB-EAI 8 + mongosh-instructions 7; Neo4j 27 = Text2Cypher-2024 16 + CypherBench 11 (đếm từ `source_dataset` trong databases.jsonl, verify 2026-06-10) |
| Splits | `test` (stratified), `support-balanced` = **index reserve** (nuôi pseudo-query view, KHÔNG phải train split) |
| Cold-start | ~20 PG DB không có support query → pseudo_query_view rỗng, retrieval chỉ dựa card. Mọi phân tích per-engine phải tách nhóm này |
| Metric chính | DB-macro recall (headline), stratified slice ≥1 query/GT-DB, bootstrap CI + McNemar trước khi claim |
| Hai lớp metric | Retrieval recall (GT-in-pool / GT-in-package) tách khỏi final routing R@1 — không bao giờ gộp |
| Metric chính thức (chốt 2026-06-10) | (1) **final R@1**; (2) **R-pool dynamic** = GT nằm trong pool LLM đã tiêu thụ thực tế (size adaptive 5→30) — BẮT BUỘC báo kèm avg/median pool size tiêu thụ, vì pool to recall tự tăng; (3) phụ: GT-in-top-5 cố định để so trực tiếp Sudarshan R@5 |
| `scenario60.jsonl` | 60 query chẩn đoán (bucket T1/T2/T3 × 20) — **legacy diagnostic, debug only, KHÔNG dùng cho claim** |
| Nhãn độ khó (T1/T2) | **ĐÃ BỎ khỏi benchmark (2026-06-20).** GT 100% inherit từ source dataset (deterministic, không annotate) → không cần Fleiss κ (κ N/A, như DBRouting + Sudarshan). "Độ khó" nếu cần = proxy tính được (embedding-margin δ / score-tie), KHÔNG phải nhãn tay. scenario60 ở trên là artifact cũ còn nhãn bucket, chỉ để debug. |
| κ N/A — verify mapping (2026-06-20) | Kiểm 1.260 câu Mongo: `(source_dataset,db_id)→instance_id` là hàm xác định, 0 key ánh xạ ≥2 instance (instance_id = db_id + prefix). Không matching mờ/phán đoán tay → κ N/A đứng vững. |
| Tautology engine-routing (rebuttal C1) | **Bác bằng domain-overlap đếm 208 DB:** 26 domain trên ≥2 loại; movie/soccer/company/flight trên CẢ 3 loại → biết chủ đề KHÔNG suy ra loại → không có tautology theo chủ đề. Residual: rò văn phong ~80% câu Neo4j máy-sinh (Text2Cypher gốc) → báo Neo4j directional. |

## 2. Luồng pipeline (M3 agent, base Sudarshan arXiv:2601.19825) — rev MINIMAL+QIC 2026-06-10

**Semantic layer per DB = 3 khối, nguồn tên duy nhất là A. Khối D (pseudo-query view) ĐÃ RÚT khỏi spec chính** — query mẫu chỉ sống duy nhất trong prompt tạo card (§3). D giữ làm arm ablation tùy chọn.

```
A. ENTITY INVENTORY (code parse): entities + fields + types + PK/FK/relationship
   tường minh — nguồn tên gốc; tên trong B, C đều phải khớp về A
B. CARD (LLM call 1, input = toàn bộ A + ≤5 câu hỏi NL từ support): 2 field — §3
C. ADJACENCY (code ∪ LLM call 2, fill-in chỉ số đóng trên A): edges {from,to,via,kind,reason} — §4
```

**BUILD FLOW CHỐT (2026-06-10) — A là gốc duy nhất, B1 ∥ B2 độc lập, không LLM nào ăn output LLM khác:**

```
[B0] schema gốc (databases.jsonl) ──code parse──▶ A
       entities + fields + types + declared_relations · deterministic, không LLM

[B1] A + ≤5 câu hỏi NL support ──LLM call 1 (prompt CARD)──▶ B = {domain_description, term_glossary}
       chỉ câu hỏi, KHÔNG SQL/Cypher/MQL · cold-start (0 query): fallback schema-only
[B2] A ──LLM call 2 (prompt ADJ)──▶ implicit_edges [{from,to,via,kind,reason}]
       input = engine + entities đánh số + confirmed edges ("do NOT repeat") + surface từ A
       KHÔNG card, KHÔNG query · B1 ∥ B2 chạy song song, cache riêng (md5 theo input)

[B3] code hợp nhất: C = A.declared_relations ∪ implicit_edges
       cùng schema cạnh {from,to,via,kind,reason} + engine; nguồn đọc từ kind

[B4] embed 1 VIEW DUY NHẤT (rev 2026-06-10): text = render B (description + glossary)
       + render A (entities + fields + types + quan hệ khai báo) → 1 vector/DB (.npy + manifest)
       Cùng cơ chế 1-view như Sudarshan (họ embed raw DDL text) — chỉ thay NỘI DUNG text
       → E1 so sánh sạch (1 view vs 1 view). Multi-view = arm tùy chọn, không spec chính.
       C không embed · KHÔNG khối D

Guards sau build (code):
  - tên trong B/C phải khớp về A (closed vocab check)
  - density check C: graph gần full-connected = cờ đỏ bịa hàng loạt → audit
  - Neo4j kỳ vọng implicit_edges ≈ [] (relationship đã khai báo đủ)
```

Chi phí build: 2 LLM call/DB × 208 DB = 416 call (1 lần, cache) + embed.

QUERY-TIME (KHÔNG intent parse — Sudarshan không có bước này; enrich = arm ablation):
  câu hỏi GỐC → dense retrieval 1-view (text B+A) → pool 30 xếp hạng sẵn, package top-5 cosine
        (pool tiêu thụ ADAPTIVE: đáp án rõ → dừng ở 5; mơ hồ → nở trang 10 tới 30 — xem branch dưới)
        → LLM mapping per-candidate đọc surface = B(description+glossary) + A(entity list) + quan hệ KHAI BÁO
          (NA tự khai, KHÔNG dựng graph)
        → code chấm: Coverage e^(−n·x) × Connectivity BFS trên C
        → rank theo điểm (không cổng boolean); < θ → chấm tiếp trang pool (10/trang)
        → agent đọc finalists (B rút gọn + evidence), quyết định / từ chối; re-search chỉ khi pool cạn điểm 0
```

**Ma trận xuyên suốt (tầng nào đọc khối nào, chốt 2026-06-10):**

| Tầng | Nhiệm vụ | Đọc |
|---|---|---|
| Embedding | match | 1 view duy nhất: text B (description+glossary) + A (entities/fields/quan hệ khai báo). C không embed |
| Mapping LLM | ground | B.description + B.glossary + A đầy đủ **kèm quan hệ KHAI BÁO** (FK/rel-type/reference). KHÔNG thấy cạnh suy luận C |
| BFS connectivity | connect | C duy nhất (explicit ∪ llm_inferred) |
| Agent tie-break | judge | evidence đã chấm (score, grounded mappings, NA, missing) + 1 dòng nhận diện từ B. KHÔNG đọc lại schema |

Nguyên tắc khử overlap: quan hệ **khai báo** = sự kiện schema, thuộc A, mapping được thấy (Neo4j phrase gọi thẳng tên relationship — giấu đi là engine-bias); quan hệ **suy luận** = sản phẩm LLM build-time, chỉ sống ở C cho BFS — LLM không ăn lại output LLM. Không tầng nào đọc raw DDL riêng. Query mẫu vào card prompt là **điểm duy nhất** workload chạm semantic layer — không embed query thô, không enrich query-time. `value_profiles` ngoài scope — ablation riêng nếu cần.

Invariant: LLM chỉ extraction (không tự chấm điểm/confidence); scoring deterministic; engine-neutral; không double-count retrieval trong rerank.

## 3. Semantic card — cấu trúc CHỐT (2026-06-10, rev 2-field **query-informed**)

**Input:** TOÀN BỘ inventory A đã chuẩn hóa từ code parse — entities + fields + types + **PK/FK/relationship tường minh** (không phải raw DDL; Gao&Luo cũng dùng input cấu trúc) — **CỘNG ≤5 câu hỏi NL từ support split** (CHỈ câu hỏi, KHÔNG kèm SQL/Cypher/MQL — tránh leak cấu trúc ngôn ngữ truy vấn). Không label. Card được THẤY quan hệ để mô tả đúng, nhưng KHÔNG sinh quan hệ (đó là việc của adjacency §4).

**Vì sao query-informed (quyết định tác giả 2026-06-10, research `reports/research-260610-query-informed-card.md`):**
- Salience cần workload: Yang/Procopiuc/Srivastava VLDB 2009 [VERIFIED] xác định bảng quan trọng từ query set; Gao&Luo [VERIFIED]: ~30% cột không giải nghĩa được từ schema thuần; dbt/Cube [INDUSTRY]: human curation encode tri thức workload.
- Cơ chế cụ thể (query thật vào prompt sinh mô tả) = **thesis-original, ENGINEERING-ABLATE** — không paper routing nào làm; bắt buộc E3a trước khi claim gain. (E3c obfuscation schema-side ĐÃ BỎ 2026-06-20 — khó defend; robustness do query-side Spider-Syn/Realistic gánh, xem §Kết quả robustness query-side.)
- Prompt instruction: "nếu có câu hỏi mẫu, ưu tiên nhắc entity/term mà câu hỏi thực gọi tới trong domain_description, và gloss term mù mờ xuất hiện trong câu hỏi".
- **Cold-start (~20 PG DB, 0 support query): fallback card schema-only** — không trộn; headline báo đủ 208 DB, bảng phụ tách nhóm có-query; claim về query-informed phải scope rõ nhóm có-query.
- Rủi ro leakage đã biết: phrasing Spider/DocSpider nằm trong pretraining LLM (arXiv:2402.08100 [VERIFIED]). Control trước đây = obfuscation E3c (ĐÃ BỎ 2026-06-20). Thay bằng robustness query-side Spider-Syn/Realistic (external peer-reviewed) — bắt tín hiệu khớp-tên; **residual: leakage phrasing-pretraining KHÔNG khử trực tiếp được nữa → khai limitation scope rõ.**

**Output — 2 field (rev 2026-06-10, quyết định tác giả: gọt key_entities + distinguishing_terms):**

```json
{
  "domain_description": "100-150 từ: domain, scope, entity chính, use case; CHỈ nhắc liên kết có trong input, không suy diễn liên kết mới",
  "term_glossary":      {"TÊN_MÙ_MỜ": "giải nghĩa ≤20 từ; chỉ gloss khi ngữ cảnh schema đủ suy ra, không thì bỏ qua"}
}
```

| Field | Tiêu chí | Căn cứ | Tag |
|---|---|---|---|
| domain_description | DB nói về gì | Gao&Luo 2502.20657 + DBCopilot 2312.03463 + Rumiantsau 2604.25149 | ENGINEERING-ABLATE (claim retrieval) → E1 |
| term_glossary | tên mù mờ | Gao&Luo column descriptions ≤20 từ; Sudarshan: bỏ metadata = −5% R@1/−7% R@3 | ENGINEERING-ABLATE → E1 arm phụ |

**Field đã BỎ (lý do 1 dòng):**
- `key_entities` — thừa: A đã embed nguyên list entity (view entity), card không làm bản sao; mapping/agent không đọc field này → consumer = 0 (gọt 2026-06-10)
- `distinguishing_terms` — lỗi validity: prompt nhìn 1 DB/lần, không thấy sibling → LLM chỉ đoán contrast, không tính được; phần giải nghĩa term glossary đã gánh; contrast thật đòi build cross-DB + E2 = ngoài thời gian (gọt 2026-06-10)
- `domain_tags` — không citation, trùng domain_description, rủi ro engine-bias
- `answerable_intents` — không paper nào ủng hộ câu hỏi LLM-tưởng-tượng làm tài liệu index; pseudo-query thật thắng + đỡ leakage
- `engine_affordances` — từ vựng năng lực engine kéo nhầm theo engine-type, vi phạm engine-neutral
- `relation_summary` — chết hẳn (2026-06-10): từng định chuyển thành `relation_notes` trong adjacency, nhưng relation_notes cũng bị gọt (consumer = 0 — văn xuôi không vào BFS, mapping không đọc)

## 4. Adjacency graph — cấu trúc CHỐT (2026-06-10)

**Nguyên tắc:** fill-in, không from-scratch. Code làm phần chắc chắn; LLM chỉ bù cạnh ngầm trên **từ vựng đóng theo chỉ số** (PICARD 2109.05093, GCD 2305.13971, LLM-FK 2603.07278).

**Input prompt:**
```
Engine: postgresql|mongodb|neo4j
Entities (closed list, refer by index):  [0] ... [N]   ← code parse
Confirmed edges (do NOT repeat):  [i]→[j] (FK / relationship / reference)  ← code parse
Schema text: <surface>
```

**Output (rev 2026-06-10, quyết định tác giả — gọt relation_notes + source tag):**
```json
{
  "implicit_edges": [
    {"from": 2, "to": 0, "via": "Campus",
     "kind": "ref_field | name_type_match", "reason": "1 dòng"}
  ]
}
```

Schema cạnh thống nhất = `{from, to, via, kind, reason}` + `engine` cấp graph. KHÔNG thêm field khác:
- `relation_notes` BỎ (2026-06-10) — consumer = 0 (mapping không đọc, BFS không ăn văn xuôi); relation_summary cũ chết hẳn
- `source` tag BỎ — thừa: `kind` đã encode nguồn (kind khai báo `fk`/`relationship`/`reference` chỉ code sinh; kind ngầm `ref_field`/`name_type_match` chỉ LLM sinh — 2 tập không giao)
- KHÔNG có field confidence (vi phạm invariant LLM-không-tự-chấm) — lọc cạnh theo `kind` khách quan thay thế (E5)
- Graph lưu = explicit (code) ∪ llm_inferred, phân biệt qua `kind` → ablation tắt nhóm cạnh không cần build lại
- Per engine: PG explicit=FK khai báo, ngầm=naming convention; Mongo explicit=reference field/embedded path, cạnh vô hướng (co-residency); Neo4j explicit=toàn bộ relationship tường minh, LLM gần như không bù (kỳ vọng `[]`)
- Connectivity cho Mongo/Neo4j = thesis-original (chưa có prior work) — khai rõ trong methodology
- Density check sau build: graph gần full-connected = cờ đỏ bịa hàng loạt → audit

## 5. Ablation bắt buộc (trước khi claim)

| ID | So sánh | Chứng minh gì | Bắt buộc? |
|---|---|---|---|
| E1 | card 2-field vs raw DDL (cùng retriever/split) | semantic card có thật sự tăng recall routing — **claim chính thesis** | BẮT BUỘC |
| E3a | card schema-only vs card query-informed (cùng giữ phần còn lại) | query mẫu trong card prompt có gain thật không | BẮT BUỘC trước claim query-informed |
| E3c | ~~card query-informed: clean vs obfuscated entity names~~ | ~~gain sống sót obfuscation = enrichment thật~~ | **ĐÃ BỎ 2026-06-20** (khó defend; robustness query-side Spider-Syn/Realistic thay) |
| E4 | adjacency fill-in vs Sudarshan from-scratch | thiết kế closed-vocab tốt hơn (edge recall, cần annotate ~20-30 DB) | bắt buộc |
| E5 | BFS với all edges vs lọc bỏ nhóm `name_type_match` | nhóm cạnh dễ bịa có phá connectivity không | bắt buộc |
| E3b | card query-informed ± khối D (pseudo-query view) | D thô còn cộng thêm gì khi card đã ăn query | tùy chọn |
| E6 | ≤5 vs ≤2 câu hỏi trong card prompt | độ nhạy số lượng query | tùy chọn, chạy nếu E3a có gain |
| E7 | retrieval câu hỏi thô vs + enrich (expansion_terms) | enrich query-time có đáng thêm không (HyDE/CRUSH4SQL) | tùy chọn |

E2 (±distinguishing_terms) ĐÃ HỦY 2026-06-10 — field bị gọt khỏi card (lỗi validity: LLM không thấy sibling, không tính được contrast thật).

**Kết quả đo của các ablation trên → [`results/RESULTS.md`](results/RESULTS.md).** Bao gồm: representation card vs raw (E1), robustness query-side Spider-Syn / Spider-Realistic, retrieval OURS (card + triage) vs baseline raw-DDL, và final routing (triage → chấm điểm → tie-break). BENCHMARK.md chỉ định nghĩa *ablation nào phải chạy*; con số nằm ở RESULTS.md.

## 6. Trạng thái artifact

| Artifact | Trạng thái |
|---|---|
| Card v1 (7 field) | đã build 208 DB — giữ làm arm so sánh, không đụng |
| Card v2 (2 field query-informed, §3) | **chưa build** — cần prompt mới (A + ≤5 câu hỏi support, fallback schema-only) + 208 call + re-embed |
| Adjacency graph (§4) | **chưa build** — cần prompt + script build + BFS đọc cache |
| Khối D (pseudo-query view) | RÚT khỏi spec chính 2026-06-10 (query chỉ sống trong card prompt) — giữ code làm arm E3b |
| Intent parse (Q1) | RÚT khỏi spec chính 2026-06-10 (Sudarshan không có; retrieval = câu hỏi gốc) — enrich giữ làm arm E7; `llm_intent.py` + flag `--enrich` cần gỡ khỏi đường mặc định |
| Agent flow (rerank redesign) | đã wire trong `m3_pydantic_agent.py` (NA tự khai, bỏ string-veto, pool pagination, θ=0.5 tạm); còn chờ adjacency cache để BFS chuẩn + gỡ intent/enrich khỏi mặc định |
