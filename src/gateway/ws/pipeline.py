"""WebSocket endpoint for real-time entropy pipeline state — Patent Claim 6."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from gateway.dependencies import get_redis

logger = logging.getLogger("cosmo.gateway.ws")
router = APIRouter()

PIPELINE_STREAMS = [
    "cosmo:raw_timestamps",
    "cosmo:extracted_entropy",
    "cosmo:conditioned_entropy",
    "cosmo:anchored_batches",
]


@router.websocket("/ws/pipeline")
async def pipeline_ws(ws: WebSocket):
    redis = get_redis()
    if not redis:
        await ws.close(code=1013, reason="Pipeline not connected")
        return

    await ws.accept()
    logger.info("WebSocket client connected for pipeline events")

    last_ids = {stream: "$" for stream in PIPELINE_STREAMS}

    try:
        while True:
            streams_arg = {k: v for k, v in last_ids.items()}
            events = await redis.xread(streams_arg, count=20, block=1000)

            if events:
                for stream_bytes, messages in events:
                    stream = stream_bytes.decode() if isinstance(stream_bytes, bytes) else stream_bytes
                    for msg_id, fields in messages:
                        msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                        last_ids[stream] = msg_id_str
                        decoded = {}
                        for k, v in fields.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            val = v.decode() if isinstance(v, bytes) else v
                            decoded[key] = val

                        await ws.send_json({
                            "stream": stream,
                            "id": msg_id_str,
                            "data": decoded,
                        })

            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                if msg == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket pipeline error")
        try:
            await ws.close(code=1011, reason="Internal error")
        except Exception:
            pass
