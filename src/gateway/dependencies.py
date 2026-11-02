"""Dependency injection providers for FastAPI endpoints."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import AsyncGenerator

import redis.asyncio as aioredis

from gateway.config import Settings

logger = logging.getLogger("cosmo.gateway")

_redis_pool: aioredis.Redis | None = None
_ai_client = None
_feedback_collector = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def init_redis(settings: Settings) -> aioredis.Redis | None:
    global _redis_pool
    if not settings.redis_url:
        logger.info("Redis not configured — pipeline features disabled")
        return None
    _redis_pool = aioredis.from_url(settings.redis_url, decode_responses=False)
    await _redis_pool.ping()
    logger.info("Redis connected at %s", settings.redis_url)
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


def get_redis() -> aioredis.Redis | None:
    return _redis_pool


async def init_ai_client(settings: Settings):
    global _ai_client
    if settings.is_cloud_mode:
        from gateway.ai.nim_client import NIMClient
        _ai_client = NIMClient(settings)
        logger.info("AI client: NIM cloud mode (Nemotron via Nebius)")
    else:
        from gateway.ai.local_fallback import LocalFallbackClient
        _ai_client = LocalFallbackClient()
        logger.info("AI client: local fallback mode (cosmo_core only)")
    return _ai_client


def get_ai_client():
    return _ai_client


async def init_feedback_collector(settings: Settings):
    global _feedback_collector
    if settings.rl_feedback_enabled:
        from gateway.rl.feedback import FeedbackCollector
        _feedback_collector = FeedbackCollector()
        logger.info("RL feedback collection enabled")
    return _feedback_collector


def get_feedback_collector():
    return _feedback_collector
