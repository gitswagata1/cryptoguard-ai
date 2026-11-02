"""Pipeline management endpoints — submit events, check status, query ledger."""
from __future__ import annotations

import json
import logging
import struct

from fastapi import APIRouter, HTTPException

from gateway.dependencies import get_redis
from gateway.models.requests import PipelineSubmitRequest
from gateway.models.responses import LedgerEntryResponse, PipelineStatusResponse

logger = logging.getLogger("cosmo.gateway.pipeline")
router = APIRouter(prefix="/api/pipeline")


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    redis = get_redis()
    if not redis:
        return PipelineStatusResponse(connected=False, streams={})

    try:
        streams = {}
        for stream_name in [
            "cosmo:raw_timestamps",
            "cosmo:extracted_entropy",
            "cosmo:conditioned_entropy",
        ]:
            try:
                length = await redis.xlen(stream_name)
                streams[stream_name] = length
            except Exception:
                streams[stream_name] = 0
        return PipelineStatusResponse(connected=True, streams=streams)
    except Exception:
        logger.exception("Pipeline status check failed")
        return PipelineStatusResponse(connected=False, streams={})


@router.post("/submit")
async def pipeline_submit(req: PipelineSubmitRequest):
    redis = get_redis()
    if not redis:
        raise HTTPException(503, "Pipeline not connected — set CG_REDIS_URL")

    payload = struct.pack(">Q", req.timestamp_ns) + req.sensor_id.encode("ascii")
    await redis.xadd(
        "cosmo:raw_timestamps",
        {
            "sensor_id": req.sensor_id.encode(),
            "timestamp_ns": str(req.timestamp_ns).encode(),
            "sequence": str(req.sequence).encode(),
            "payload": payload,
        },
        maxlen=1_000_000,
    )
    return {"accepted": True, "stream": "cosmo:raw_timestamps"}


@router.get("/ledger", response_model=list[LedgerEntryResponse])
async def pipeline_ledger():
    redis = get_redis()
    if not redis:
        raise HTTPException(503, "Pipeline not connected")

    try:
        entries = await redis.lrange("cosmo:ledger:entries", 0, 49)
        result = []
        for entry_bytes in entries:
            entry = json.loads(entry_bytes)
            result.append(LedgerEntryResponse(
                batch_id=entry["batch_id"],
                merkle_root=entry["merkle_root"],
                n_fingerprints=entry["n_fingerprints"],
                timestamp=entry["timestamp"],
                entry_hash=entry["entry_hash"],
            ))
        return result
    except Exception:
        logger.exception("Ledger query failed")
        return []
