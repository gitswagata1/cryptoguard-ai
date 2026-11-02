from __future__ import annotations

from typing import Protocol


class AIClient(Protocol):
    """Common interface for all AI backends."""

    @property
    def available(self) -> bool: ...

    async def chat(
        self,
        message: str,
        system_prompt: str,
        context: str | None = None,
    ) -> tuple[str, str]: ...

    async def scan_code(
        self,
        code: str,
        language: str,
        taxonomy_prompt: str,
    ) -> dict: ...

    async def analyze_entropy(self, metrics: dict) -> str: ...

    async def analyze_image(self, image_b64: str, prompt: str) -> str: ...
