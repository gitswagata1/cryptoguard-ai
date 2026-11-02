from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class InputType(Enum):
    """Classification of incoming request type."""

    CODE_SCAN = "code_scan"
    ENTROPY_DATA = "entropy_data"
    CHAT_QUERY = "chat_query"
    IMAGE_ANALYSIS = "image_analysis"
    PIPELINE_COMMAND = "pipeline_command"


class Model(Enum):
    """Available model targets."""

    ULTRA = "ultra"
    NANO = "nano"
    NEVA = "neva"
    LOCAL = "local"


@dataclass(frozen=True)
class RoutingDecision:
    """Result of the task classifier."""

    model: Model
    reason: str
    estimated_complexity: float


COMPLEXITY_THRESHOLD = 0.4

_CRYPTO_KEYWORDS = frozenset({
    "aes", "rsa", "ecdsa", "sha256", "sha512", "hmac", "pbkdf2", "scrypt",
    "argon2", "chacha20", "poly1305", "x25519", "ed25519", "kdf", "nonce",
    "iv", "cbc", "gcm", "ecb", "pkcs", "tls", "ssl", "certificate",
    "private_key", "public_key", "decrypt", "encrypt", "signature", "verify",
    "digest", "hash", "random", "entropy", "seed", "drbg", "csprng",
    "quantum", "lattice", "kyber", "dilithium", "lhl", "aead",
})

_CODE_PATTERNS = re.compile(
    r"(def |class |function |import |#include|public |private |"
    r"const |let |var |return |if\s*\(|for\s*\(|while\s*\()",
)


class TaskClassifier:
    """Dynamic routing engine for AI model selection (Patent Claim 1)."""

    def __init__(self, ai_available: bool = True) -> None:
        self._ai_available = ai_available

    def classify(self, input_text: str, input_type: InputType) -> RoutingDecision:
        """Determine which model should handle a given request."""
        if not self._ai_available:
            return RoutingDecision(
                model=Model.LOCAL,
                reason="AI backend unavailable, using local analysis",
                estimated_complexity=0.0,
            )

        if input_type is InputType.ENTROPY_DATA:
            return RoutingDecision(
                model=Model.NANO,
                reason="Entropy analysis routed to lightweight model",
                estimated_complexity=0.2,
            )

        if input_type is InputType.IMAGE_ANALYSIS:
            return RoutingDecision(
                model=Model.NEVA,
                reason="Image analysis requires multimodal model",
                estimated_complexity=0.5,
            )

        complexity = self._score_complexity(input_text)

        if complexity >= COMPLEXITY_THRESHOLD:
            return RoutingDecision(
                model=Model.ULTRA,
                reason=f"High complexity ({complexity:.2f}) routed to Ultra",
                estimated_complexity=complexity,
            )

        return RoutingDecision(
            model=Model.NANO,
            reason=f"Low complexity ({complexity:.2f}) routed to Nano",
            estimated_complexity=complexity,
        )

    @staticmethod
    def _score_complexity(text: str) -> float:
        """Score input complexity on a 0.0-1.0 scale."""
        text_lower = text.lower()
        tokens_est = len(text) / 4.0

        # Token length component (0-0.3): saturates around 2000 tokens
        length_score = min(tokens_est / 2000.0, 1.0) * 0.3

        # Code detection component (0 or 0.25)
        code_matches = len(_CODE_PATTERNS.findall(text))
        code_score = min(code_matches / 5.0, 1.0) * 0.25

        # Crypto keyword density (0-0.25)
        words = set(re.findall(r"[a-z0-9_]+", text_lower))
        crypto_hits = len(words & _CRYPTO_KEYWORDS)
        crypto_score = min(crypto_hits / 5.0, 1.0) * 0.25

        # Nesting depth component (0-0.2)
        max_depth = 0
        depth = 0
        for ch in text:
            if ch in "({[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch in ")}]":
                depth = max(depth - 1, 0)
        nesting_score = min(max_depth / 6.0, 1.0) * 0.2

        return length_score + code_score + crypto_score + nesting_score
