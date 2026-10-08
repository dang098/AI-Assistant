from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.logger import get_logger
from services.rag_service import query_knowledge

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    rerank_scores: list[float] = []


@router.post("/chat", response_model=ChatResponse, summary="Hỏi đáp")
def chat_with_docs(request: ChatRequest):
    try:
        result = query_knowledge(question=request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            rerank_scores=result.get("rerank_scores", []),
        )
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))
