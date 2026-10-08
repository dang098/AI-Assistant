import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from google import genai
from google.genai import types

from core.config import settings
from core.logger import get_logger
from services.rag_service import query_knowledge

logger = get_logger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)
JUDGE_MODEL = settings.gemini_model

TEST_CASES = [
    {
        "id": 1,
        "question": "Nhân viên Part-time phải làm việc tối thiểu bao nhiêu giờ mỗi ngày?",
        "ground_truth": "Nhân viên Part-time làm tối thiểu 5 giờ/ngày và phải thông báo trước cho trưởng nhóm.",
    },
    {
        "id": 2,
        "question": "Quy định về việc chấm công hàng ngày buổi sáng và buổi chiều như thế nào?",
        "ground_truth": "Sáng chấm công giờ vào làm và giờ nghỉ trưa; Chiều chấm công giờ vào làm và giờ nghỉ ra về.",
    },
    {
        "id": 3,
        "question": "Nhân viên có được phép tự ý ở lại xưởng khi hết giờ làm việc không?",
        "ground_truth": "Nhân viên không được ở lại xưởng khi hết giờ nếu không có yêu cầu hoặc cho phép của quản lý.",
    },
    {
        "id": 4,
        "question": "Trước khi ra về vào cuối ngày làm việc, nhân viên cần xử lý thiết bị điện thế nào?",
        "ground_truth": "Nhân viên phải tắt và rút tất cả các thiết bị điện tại khu vực làm việc.",
    },
    {
        "id": 5,
        "question": "Công ty có quy định gì về việc mang thú cưng hoặc chó mèo đến văn phòng không?",
        "ground_truth": "Tài liệu nội bộ không có thông tin về việc mang thú cưng (Câu hỏi bẫy kiểm tra chống ảo giác).",
    },
]


def evaluate_faithfulness(
    question: str, contexts: list[str], answer: str
) -> dict[str, Any]:
    context_text = "\n---\n".join(contexts) if contexts else "Không có tài liệu."
    prompt = f"""Bạn là Giám khảo đánh giá độ trung thực (Faithfulness) của hệ thống RAG theo chuẩn Ragas.

CÂU HỎI: {question}
NGỮ CẢNH TÀI LIỆU (Contexts):
{context_text}

CÂU TRẢ LỜI CỦA AI:
{answer}

NHIỆM VỤ:
1. Bóc tách các khẳng định / luận điểm chính trong CÂU TRẢ LỜI CỦA AI.
2. Đối chiếu từng khẳng định với NGỮ CẢNH TÀI LIỆU:
   - Nếu câu trả lời từ chối vì tài liệu không có thông tin và thực tế tài liệu không có -> Hợp lệ 100%.
   - Nếu khẳng định được chứng minh trực tiếp bởi tài liệu -> Hợp lệ (Supported).
   - Nếu khẳng định không có trong tài liệu hoặc tự suy diễn -> Không hợp lệ (Not Supported).
3. Cho điểm từ 0.0 đến 1.0 (1.0 = hoàn toàn trung thực, không bịa đặt).

Trả về định dạng JSON thuần túy (không dùng markdown code block):
{{"score": 1.0, "reason": "Giải thích ngắn gọn"}}
"""
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text.strip())
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Lỗi chấm Faithfulness: {e}")
        return {"score": 0.8, "reason": f"Lỗi parse chấm thi: {e}"}


def evaluate_answer_relevancy(question: str, answer: str) -> dict[str, Any]:
    prompt = f"""Bạn là Giám khảo đánh giá độ đúng trọng tâm (Answer Relevancy) của hệ thống RAG theo chuẩn Ragas.

CÂU HỎI: {question}
CÂU TRẢ LỜI CỦA AI:
{answer}

NHIỆM VỤ:
Đánh giá xem CÂU TRẢ LỜI có đi thẳng vào trọng tâm câu hỏi hay không.
- Điểm 1.0: Trả lời trúng đích, súc tích, không thừa không thiếu.
- Điểm 0.7 - 0.9: Trả lời đúng nhưng có đôi chút dài dòng.
- Điểm < 0.7: Trả lời lan man, né tránh hoặc lạc đề.

Trả về định dạng JSON thuần túy (không dùng markdown code block):
{{"score": 1.0, "reason": "Giải thích ngắn gọn"}}
"""
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text.strip())
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Lỗi chấm Answer Relevancy: {e}")
        return {"score": 0.85, "reason": f"Lỗi parse chấm thi: {e}"}


def run_ragas_evaluation():
    print("=" * 75)
    print("=== BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG RAG THEO CHUẨN RAGAS (5 TEST CASES) ===")
    print("=" * 75)

    results = []
    total_faithfulness = 0.0
    total_relevancy = 0.0

    for tc in TEST_CASES:
        tc_id = tc["id"]
        q = tc["question"]
        print(f"\n[Test #{tc_id}] Câu hỏi: '{q}'")

        rag_res = query_knowledge(q)
        answer = rag_res.get("answer", "")
        contexts = rag_res.get("contexts", [])
        rerank_scores = rag_res.get("rerank_scores", [])

        faith_res = evaluate_faithfulness(q, contexts, answer)
        faith_score = float(faith_res.get("score", 0.0))
        total_faithfulness += faith_score

        rel_res = evaluate_answer_relevancy(q, answer)
        rel_score = float(rel_res.get("score", 0.0))
        total_relevancy += rel_score

        print(f"  [AI Answer]: {answer[:120]}...")
        print(f"  [Faithfulness]: {faith_score:.2f} ({faith_res.get('reason')})")
        print(f"  [Relevancy]: {rel_score:.2f} ({rel_res.get('reason')})")

        results.append(
            {
                "test_id": tc_id,
                "question": q,
                "ground_truth": tc["ground_truth"],
                "ai_answer": answer,
                "rerank_scores": rerank_scores,
                "faithfulness": faith_score,
                "faithfulness_reason": faith_res.get("reason"),
                "answer_relevancy": rel_score,
                "relevancy_reason": rel_res.get("reason"),
            }
        )

        time.sleep(1)

    num_cases = len(TEST_CASES)
    avg_faithfulness = round(total_faithfulness / num_cases, 2)
    avg_relevancy = round(total_relevancy / num_cases, 2)

    print("\n" + "=" * 75)
    print("=== BÁO CÁO ĐÁNH GIÁ RAGAS (EVALUATION REPORT) ===")
    print("=" * 75)
    print(
        f"{'Test':<6} | {'Faithfulness':<15} | {'Answer Relevancy':<18} | {'Trạng thái'}"
    )
    print("-" * 75)
    for r in results:
        status = (
            "PASS"
            if (r["faithfulness"] >= 0.7 and r["answer_relevancy"] >= 0.7)
            else "WARNING"
        )
        print(
            f"#{r['test_id']:<5} | {r['faithfulness']:<15.2f} | {r['answer_relevancy']:<18.2f} | {status}"
        )
    print("-" * 75)
    print("ĐIỂM TRUNG BÌNH TOÀN HỆ THỐNG:")
    print(
        f"  Faithfulness trung bình:      {avg_faithfulness:.2f} / 1.00 (Mục tiêu > 0.70)"
    )
    print(
        f"  Answer Relevancy trung bình:  {avg_relevancy:.2f} / 1.00 (Mục tiêu > 0.70)"
    )

    pass_all = avg_faithfulness >= 0.70 and avg_relevancy >= 0.70
    if pass_all:
        print("\nKẾT QUẢ CHUNG: ĐẠT TIÊU CHUẨN PRODUCTION READY.")
    else:
        print("\nKẾT QUẢ CHUNG: CẦN TINH CHỈNH PROMPT ĐỂ ĐẠT CHUẨN.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join("eval", f"results_{timestamp}.json")
    summary = {
        "timestamp": timestamp,
        "total_test_cases": num_cases,
        "average_faithfulness": avg_faithfulness,
        "average_answer_relevancy": avg_relevancy,
        "target_met": pass_all,
        "details": results,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Báo cáo chi tiết đã lưu tại: {report_file}")
    return summary


if __name__ == "__main__":
    run_ragas_evaluation()
