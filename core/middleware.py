import contextlib
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.logger import get_logger
from core.metrics_store import metrics_store

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        content_length = request.headers.get("content-length")
        input_bytes = (
            int(content_length) if (content_length and content_length.isdigit()) else 0
        )
        input_tokens = max(0, input_bytes // 3)

        try:
            response = await call_next(request)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            metrics_store.record(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=latency_ms,
                estimated_tokens=input_tokens,
            )
            logger.error(f"Request sập {request.method} {request.url.path}: {e!s}")
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        output_bytes = 0
        with contextlib.suppress(Exception):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            full_body = b"".join(chunks)
            output_bytes = len(full_body)

            async def stream_chunks():
                for c in chunks:
                    yield c

            response.body_iterator = stream_chunks()

        output_tokens = max(1, output_bytes // 3) if output_bytes > 0 else 0
        total_tokens = input_tokens + output_tokens

        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        response.headers["X-Estimated-Tokens"] = str(total_tokens)

        path = request.url.path
        if not (path.startswith(("/docs", "/redoc")) or path == "/openapi.json"):
            metrics_store.record(
                endpoint=path,
                method=request.method,
                status_code=response.status_code,
                latency_ms=latency_ms,
                estimated_tokens=total_tokens,
            )
            logger.info(
                f"[METRICS] {request.method} {path} | "
                f"Status: {response.status_code} | "
                f"Latency: {latency_ms:.2f}ms | "
                f"Tokens: ~{total_tokens}"
            )

        return response
