from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from fastapi import HTTPException
from openai import AsyncOpenAI

from ..config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are CryptoGuard, an expert cryptographic security analyst. "
    "You specialize in evaluating cryptographic implementations, identifying "
    "vulnerabilities in code that uses encryption, hashing, key derivation, "
    "random number generation, and related primitives. You follow NIST, OWASP, "
    "and CWE taxonomies. Provide precise, actionable guidance. When analysing "
    "code, classify findings by severity (critical/high/medium/low/info) and "
    "reference specific CWE IDs where applicable."
)

_SCAN_PROMPT_TEMPLATE = (
    "Analyse the following {language} code for cryptographic vulnerabilities. "
    "Return ONLY a JSON object with keys: score (0-100 security score), "
    "findings (array of {{severity, title, description, line, fix, vuln_class}}), "
    "summary (one-paragraph summary). {taxonomy}\n\n```{language}\n{code}\n```"
)

_ENTROPY_PROMPT_TEMPLATE = (
    "You are a cryptographic entropy analyst. Given these entropy metrics, "
    "provide a concise natural-language assessment of the randomness quality.\n\n"
    "Metrics:\n{metrics}"
)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output, handling markdown fences."""
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    payload = fenced.group(1).strip() if fenced else text.strip()
    return json.loads(payload)


class NIMClient:
    """AI client backed by Nebius AI Cloud (OpenAI-compatible)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.nim_base_url,
            api_key=settings.nebius_api_key,
        )

    @property
    def available(self) -> bool:
        """True when a valid API key is configured."""
        return bool(self._settings.nebius_api_key)

    async def chat(
        self,
        message: str,
        system_prompt: str,
        context: str | None = None,
    ) -> tuple[str, str]:
        """Send a chat completion and return (response, model_used)."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood. I have the context."})
        messages.append({"role": "user", "content": message})

        try:
            t0 = time.perf_counter()
            resp = await self._client.chat.completions.create(
                model=self._settings.model_ultra,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            content = resp.choices[0].message.content or ""
            model_used = resp.model or self._settings.model_ultra
            logger.info("chat completed in %.1f ms model=%s", elapsed, model_used)
            return content, model_used
        except Exception:
            logger.exception("NIM chat request failed")
            raise HTTPException(status_code=502, detail="Upstream AI service error")

    async def scan_code(
        self,
        code: str,
        language: str,
        taxonomy_prompt: str,
    ) -> dict:
        """Scan source code for cryptographic vulnerabilities."""
        prompt = _SCAN_PROMPT_TEMPLATE.format(
            language=language,
            code=code,
            taxonomy=taxonomy_prompt,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.model_ultra,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
            )
            raw = resp.choices[0].message.content or "{}"
            return _extract_json(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse scan JSON from model output")
            raise HTTPException(status_code=502, detail="Model returned invalid JSON")
        except Exception:
            logger.exception("NIM scan request failed")
            raise HTTPException(status_code=502, detail="Upstream AI service error")

    async def analyze_entropy(self, metrics: dict) -> str:
        """Generate a natural-language entropy assessment."""
        prompt = _ENTROPY_PROMPT_TEMPLATE.format(
            metrics=json.dumps(metrics, indent=2),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.model_nano,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            logger.exception("NIM entropy analysis failed")
            raise HTTPException(status_code=502, detail="Upstream AI service error")

    async def analyze_image(self, image_b64: str, prompt: str) -> str:
        """Analyse an image via NeVA multimodal endpoint."""
        if not self._settings.neva_endpoint:
            return "NeVA not configured"

        from .neva_client import NeVAClient

        neva = NeVAClient(
            endpoint=self._settings.neva_endpoint,
            api_key=self._settings.nebius_api_key,
        )
        try:
            return await neva.analyze(image_b64, prompt)
        except Exception:
            logger.exception("NeVA image analysis failed")
            raise HTTPException(status_code=502, detail="NeVA service error")
