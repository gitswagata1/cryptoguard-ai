"""CryptoGuard AI — Production API Gateway.

Unified API gateway bridging NVIDIA Nemotron AI inference, NIST SP 800-90B
entropy analysis, LHL extraction, and real-time pipeline monitoring via
WebSocket. Operates in cloud mode (Nebius API) or local mode (cosmo_core).

Patent Reference: Claims 1-7 of CryptoGuard AI IDF-B.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gateway.config import Settings
from gateway.dependencies import (
    close_redis,
    get_settings,
    init_ai_client,
    init_feedback_collector,
    init_redis,
)

logger = logging.getLogger("cosmo.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("CryptoGuard AI gateway starting — mode=%s",
                "cloud" if settings.is_cloud_mode else "local")

    await init_redis(settings)
    await init_ai_client(settings)
    await init_feedback_collector(settings)

    logger.info("Gateway ready (pipeline=%s, rl=%s)",
                settings.is_pipeline_mode, settings.rl_feedback_enabled)

    yield

    await close_redis()
    logger.info("Gateway shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CryptoGuard AI",
        description="AI-Driven Cryptographic Security Analysis with NIST SP 800-90B Entropy Pipeline",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Model-Used"],
    )

    from gateway.middleware.rate_limit import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)

    from gateway.middleware.telemetry import TelemetryMiddleware
    app.add_middleware(TelemetryMiddleware)

    from gateway.routers import chat, entropy, health, pipeline, scan
    app.include_router(health.router, tags=["health"])
    app.include_router(scan.router, tags=["scanner"])
    app.include_router(entropy.router, tags=["entropy"])
    app.include_router(chat.router, tags=["chat"])
    app.include_router(pipeline.router, tags=["pipeline"])

    from gateway.ws.pipeline import router as ws_router
    app.include_router(ws_router, tags=["websocket"])

    web_dir = Path(__file__).resolve().parent.parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway.main:app", host="0.0.0.0", port=8000, reload=True)
