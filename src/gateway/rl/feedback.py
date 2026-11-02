from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """Single inference event for RL feedback."""

    request_type: str
    input_hash: str
    model_used: str
    routing_decision: str
    response_time_ms: float
    output_hash: str
    timestamp: float = field(default_factory=time.time)
    reward: float | None = None


class FeedbackCollector:
    """Collects inference records for RL training feedback loops."""

    def __init__(self) -> None:
        self._buffer: list[InferenceRecord] = []

    @property
    def buffer_size(self) -> int:
        """Number of records in the in-memory buffer."""
        return len(self._buffer)

    def record(self, record: InferenceRecord) -> None:
        """Append an inference record to the in-memory buffer."""
        self._buffer.append(record)
        logger.debug(
            "recorded inference model=%s type=%s latency=%.1fms",
            record.model_used,
            record.request_type,
            record.response_time_ms,
        )

    async def flush_to_redis(self, redis_client: Any) -> int:
        """Push buffered records to Redis sorted set and clear the buffer."""
        if not self._buffer:
            return 0

        pipe = redis_client.pipeline()
        key = "cosmo:rl:feedback"
        flushed = 0

        for rec in self._buffer:
            payload = json.dumps(asdict(rec))
            pipe.zadd(key, {payload: rec.timestamp})
            flushed += 1

        await pipe.execute()
        self._buffer.clear()
        logger.info("flushed %d inference records to Redis", flushed)
        return flushed

    async def get_batch(
        self,
        redis_client: Any,
        batch_size: int = 64,
    ) -> list[InferenceRecord]:
        """Retrieve a batch of inference records from Redis for training."""
        key = "cosmo:rl:feedback"
        raw_items: list[tuple[bytes | str, float]] = await redis_client.zpopmin(
            key, count=batch_size,
        )
        records = []
        for payload, _score in raw_items:
            data = json.loads(payload if isinstance(payload, str) else payload.decode())
            records.append(InferenceRecord(**data))
        return records
