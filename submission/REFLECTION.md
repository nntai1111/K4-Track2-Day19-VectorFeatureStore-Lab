# Reflection — Lab 19

**Tên:** Nguyễn Như Tài
**Mã học viên:** 2A202602976
**Cohort:** A20-K4
**Path đã chạy:** lite (fastembed `bge-small-en-v1.5` 384d + Qdrant in-memory + Feast SQLite)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 trung bình: hybrid 78.6% > BM25 77.8% > vector 73.2%.

- **exact** (n=15): BM25 = hybrid 96.7%, vector 88.7%. Query chứa đúng thuật ngữ nên khớp từ vựng là đủ.
- **mixed** (n=20): hybrid thắng 100% (vector 98.5%, BM25 97.0%). RRF cộng điểm cho doc đứng cao ở *cả hai* danh sách nên lọc được nhiễu riêng của từng retriever.
- **paraphrase** (n=15): cả ba đều yếu; BM25 33.3% ≈ hybrid 32.0% > vector 24.0%. Vector *không* thắng như kỳ vọng vì `bge-small-en` là model tiếng Anh, biểu diễn kém câu tiếng Việt diễn đạt lại → nút thắt nằm ở model embedding, không ở RRF (cần bge-m3/e5 đa ngữ).

Không dùng hybrid khi: tra mã/ID/SKU/log lỗi chính xác → BM25 thuần (rẻ, P99 ~3 ms so với ~17 ms); corpus đa ngữ hoặc query hội thoại không chung từ vựng với doc, và đã có embedding tốt → vector thuần; khi ngân sách latency quá chặt để chạy hai retriever.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid thắng trung bình chỉ +0.8pp so với BM25 — với embedding yếu cho tiếng Việt, chọn model quan trọng hơn chọn thuật toán fusion.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/` — `python bonus/demo.py` exit 0)
- [ ] Pair work với: _<tên đồng đội nếu có>_
