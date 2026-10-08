# Enterprise AI Assistant

Backend service kết hợp RAG tra cứu tài liệu nội bộ và AI Agent thao tác database tự động qua giao thức MCP.

## Kiến trúc

- **RAG (Tài liệu):** Nhận file PDF/TXT -> cắt chunk 500 token, overlap 50 -> lưu ChromaDB -> tìm kiếm ngữ nghĩa kết hợp Flashrank lọc top 3 đoạn liên quan -> Gemini tổng hợp câu trả lời kèm trích dẫn nguồn.
- **Agent (Nghiệp vụ):** Kết nối database Supabase qua FastMCP server với 3 tools: tra cứu kho hàng, tạo đơn (tự trừ tồn kho) và cập nhật thông tin đơn. Lưu phiên chat qua cookie.
- **Giám sát:** Middleware tự bấm giờ (ms), ước tính token sử dụng và trả về qua header HTTP (`X-Latency-Ms`, `X-Estimated-Tokens`).
- **Đánh giá:** Script `eval/eval_ragas.py` chấm tự động độ trung thực (Faithfulness) và độ chuẩn xác (Answer Relevancy).

## Cài đặt & Khởi chạy

Yêu cầu: Python 3.10+, file `.env` chứa `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.

### Cách 1: Docker
```bash
docker compose up --build
```

### Cách 2: Chạy trực tiếp
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Endpoints chính

- `POST /api/v1/upload` - Nạp file PDF hoặc TXT vào ChromaDB.
- `POST /api/v1/chat` - Hỏi đáp nội dung tài liệu (hỗ trợ rerank).
- `POST /api/v1/agent` - Tra cứu kho hoặc đặt hàng qua AI Agent.
- `GET /metrics` - Xem tổng request, độ trễ trung bình và lượng token.
- `GET /docs` - Swagger UI để test API trực tiếp trên trình duyệt.

## Đánh giá chất lượng RAG

Chạy bộ test 5 kịch bản thực tế:
```bash
python eval/eval_ragas.py
```
Kết quả xuất ra bảng tóm tắt và lưu chi tiết tại `eval/results_*.json`.
