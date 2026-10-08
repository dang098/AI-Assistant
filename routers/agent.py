import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core.logger import get_logger
from services.agent_service import run_agent_turn

logger = get_logger(__name__)
router = APIRouter()


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    reply: str
    tools_called: list[str] = []


@router.post("/agent", response_model=AgentResponse, summary="Trợ lý AI")
def execute_agent(request: AgentRequest, req: Request, res: Response):
    try:
        session_id = req.cookies.get("chat_session_id")
        is_new_session = False

        if not session_id:
            session_id = str(uuid.uuid4())
            is_new_session = True

        result = run_agent_turn(message=request.message, session_id=session_id)

        if is_new_session:
            res.set_cookie(
                key="chat_session_id",
                value=session_id,
                httponly=True,
                samesite="lax",
            )

        return AgentResponse(
            reply=result["reply"],
            tools_called=result["tools_called"],
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))
