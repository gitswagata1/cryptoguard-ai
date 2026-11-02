from __future__ import annotations

import time
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class _TokenBucket:
    """Per-client token bucket for rate limiting."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Attempt to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until the next token becomes available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class RateLimitMiddleware:
    """ASGI middleware enforcing per-IP token-bucket rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        cleanup_interval: float = 300.0,
    ) -> None:
        self.app = app
        self.rpm = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0
        self._buckets: dict[str, _TokenBucket] = {}
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = cleanup_interval

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP from the ASGI scope."""
        client = scope.get("client")
        if client:
            return client[0]
        # Fallback: check X-Forwarded-For via headers
        for header_name, header_val in scope.get("headers", []):
            if header_name == b"x-forwarded-for":
                return header_val.decode().split(",")[0].strip()
        return "unknown"

    def _cleanup_stale(self) -> None:
        """Remove buckets that have been idle for longer than the cleanup interval."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_cutoff = now - self._cleanup_interval
        stale_keys = [
            ip for ip, bucket in self._buckets.items()
            if bucket.last_refill < stale_cutoff
        ]
        for key in stale_keys:
            del self._buckets[key]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        self._cleanup_stale()

        client_ip = self._get_client_ip(scope)
        bucket = self._buckets.get(client_ip)
        if bucket is None:
            bucket = _TokenBucket(self.rpm, self.refill_rate)
            self._buckets[client_ip] = bucket

        if not bucket.consume():
            retry_after = bucket.retry_after
            response = JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
