from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class NeVAClient:
    """Client for the NeVA multimodal NIM endpoint."""

    def __init__(self, endpoint: str, api_key: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key

    @property
    def available(self) -> bool:
        """True when an endpoint is configured."""
        return bool(self._endpoint)

    async def analyze(self, image_b64: str, prompt: str) -> str:
        """Send an image + text prompt to NeVA and return the response."""
        payload: dict[str, Any] = {
            "model": "neva",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._endpoint}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            logger.warning("NeVA returned empty choices")
            return ""
        return choices[0].get("message", {}).get("content", "")
