from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("cosmo.telemetry")


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_data"):
            entry.update(record.request_data)  # type: ignore[arg-type]
        return json.dumps(entry, default=str)


def configure_telemetry_logging(level: str = "INFO") -> None:
    """Attach the JSON formatter to the telemetry logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


class TelemetryMiddleware:
    """ASGI middleware for request tracing and structured logging."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        method = scope.get("method", "")
        path = scope.get("path", "")
        t0 = time.perf_counter()

        response_status = 0
        model_used = ""

        async def send_wrapper(message: Message) -> None:
            nonlocal response_status, model_used

            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
                headers = message.get("headers", [])

                # Inject request-id header
                headers.append((b"x-request-id", request_id.encode()))

                # Extract model-used if downstream set it
                for name, value in headers:
                    if name == b"x-model-used":
                        model_used = value.decode()

                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            extra: dict[str, Any] = {
                "request_data": {
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status": response_status,
                    "latency_ms": round(latency_ms, 2),
                },
            }
            if model_used:
                extra["request_data"]["model_used"] = model_used

            record = logger.makeRecord(
                name=logger.name,
                level=logging.INFO,
                fn="",
                lno=0,
                msg=f"{method} {path} {response_status} {latency_ms:.1f}ms",
                args=(),
                exc_info=None,
            )
            record.request_data = extra["request_data"]  # type: ignore[attr-defined]
            logger.handle(record)
