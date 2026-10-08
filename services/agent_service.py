import uuid
from typing import Any

from google import genai
from google.genai import types

from core.config import settings
from core.logger import get_logger
from mcp_server.server import check_inventory, create_order, update_order_info

logger = get_logger(__name__)

SESSIONS: dict[str, Any] = {}
_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def get_or_create_chat_session(session_id: str):
    if session_id in SESSIONS:
        return SESSIONS[session_id]

    if len(SESSIONS) >= 500:
        SESSIONS.pop(next(iter(SESSIONS)))

    client = get_genai_client()
    tools_list = [check_inventory, create_order, update_order_info]

    system_instruction = """Bạn là trợ lý bán hàng chuyên nghiệp.
Bạn có quyền sử dụng các công cụ để tra cứu tồn kho, tạo đơn hàng và sửa thông tin đơn hàng trên hệ thống.
Nguyên tắc:
1. Chủ động sử dụng công cụ phù hợp khi khách hỏi về sản phẩm, giá bán, tồn kho hoặc muốn đặt mua.
2. Ghi nhớ ngữ cảnh các câu trao đổi trước để xử lý tiếp mà không hỏi lại thông tin đã biết.
3. Trả lời lịch sự, rõ ràng và ngắn gọn bằng tiếng Việt."""

    config = types.GenerateContentConfig(
        tools=tools_list,
        system_instruction=system_instruction,
        temperature=0.2,
    )

    chat = client.chats.create(model=settings.gemini_model, config=config)
    SESSIONS[session_id] = chat
    logger.info(f"Khởi tạo phiên trò chuyện: '{session_id}'")
    return chat


def run_agent_turn(message: str, session_id: str | None = None) -> dict:
    active_session_id = session_id if session_id else str(uuid.uuid4())
    chat = get_or_create_chat_session(active_session_id)
    logger.info(f"Phiên [{active_session_id}] nhận tin: '{message}'")

    response = chat.send_message(message)

    tools_executed = []
    if hasattr(chat, "get_history"):
        for turn in chat.get_history():
            for part in turn.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tools_executed.append(part.function_call.name)

    return {
        "reply": response.text.strip()
        if response.text
        else "Đã xử lý xong yêu cầu của bạn.",
        "session_id": active_session_id,
        "tools_called": list(set(tools_executed)),
    }
