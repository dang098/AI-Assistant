from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import get_logger
from core.metrics_store import metrics_store
from core.middleware import MetricsMiddleware
from routers import agent, chat, upload

logger = get_logger(__name__)

app = FastAPI(
    title="AI Assistant",
    description="Hệ thống Trợ lý AI Doanh nghiệp",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1", tags=["Tài liệu"])
app.include_router(chat.router, prefix="/api/v1", tags=["Hỏi đáp"])
app.include_router(agent.router, prefix="/api/v1", tags=["Trợ lý AI"])


@app.get("/health", tags=["Hệ thống"], summary="Kiểm tra trạng thái")
def health_check():
    return {"status": "ok", "env": settings.app_env, "version": "1.0.0"}


@app.get("/metrics", tags=["Hệ thống"], summary="Chỉ số hiệu năng & chi phí")
def get_metrics():
    return metrics_store.get_summary()


@app.on_event("startup")
async def on_startup():
    logger.info(f"AI Assistant API khoi dong ({settings.app_env})")
    logger.info(
        "Docs: http://localhost:8000/docs | Metrics: http://localhost:8000/metrics"
    )
