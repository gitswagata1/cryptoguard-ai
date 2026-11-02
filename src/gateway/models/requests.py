from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """Inbound chat message."""

    message: str
    context: str | None = None


class ScanRequest(BaseModel):
    """Source code to scan for vulnerabilities."""

    code: str
    language: str = "auto"


class EntropyRequest(BaseModel):
    """Hex-encoded binary data for entropy analysis."""

    data: str

    @field_validator("data")
    @classmethod
    def _must_be_hex(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]+", v):
            raise ValueError("data must be a valid hexadecimal string")
        return v


class PipelineSubmitRequest(BaseModel):
    """Sensor reading submitted to the pipeline."""

    sensor_id: str
    timestamp_ns: int
    sequence: int
