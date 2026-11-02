"""Health check endpoint reporting system capabilities."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from gateway.config import Settings
from gateway.dependencies import get_ai_client, get_redis, get_settings
from gateway.models.responses import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)):
    ai = get_ai_client()
    redis = get_redis()

    capabilities = ["entropy_analysis", "code_scan_local", "taxonomy"]
    if ai and ai.available:
        capabilities.extend(["nemotron_ultra", "nemotron_nano", "ai_chat"])
    if redis:
        capabilities.append("pipeline")

    mode = "cloud" if settings.is_cloud_mode else "local"
    models = [settings.model_ultra, settings.model_nano] if settings.is_cloud_mode else []

    return HealthResponse(
        status="ok",
        mode=mode,
        ai_available=ai is not None and ai.available,
        pipeline_available=redis is not None,
        redis_connected=redis is not None,
        capabilities=capabilities,
        models=models,
    )
