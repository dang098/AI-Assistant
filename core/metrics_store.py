import threading
from collections import deque
from typing import Any


class MetricsStore:
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()

        self.total_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.total_tokens: int = 0
        self.status_counts: dict[str, int] = {}

    def record(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        estimated_tokens: int,
    ):
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.total_tokens += estimated_tokens

            status_key = str(status_code)
            self.status_counts[status_key] = self.status_counts.get(status_key, 0) + 1

            self.records.append(
                {
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                    "estimated_tokens": estimated_tokens,
                }
            )

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = (
                round(self.total_latency_ms / self.total_requests, 2)
                if self.total_requests > 0
                else 0.0
            )

            recent = list(self.records)[-10:]
            recent.reverse()

            return {
                "total_requests": self.total_requests,
                "avg_latency_ms": avg_latency,
                "total_estimated_tokens": self.total_tokens,
                "status_breakdown": self.status_counts,
                "recent_requests": recent,
            }

    def reset(self):
        with self._lock:
            self.records.clear()
            self.total_requests = 0
            self.total_latency_ms = 0.0
            self.total_tokens = 0
            self.status_counts.clear()


metrics_store = MetricsStore()
