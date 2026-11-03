"""Domain-specific tokenizer for cryptographic source code analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator


class TokenType(Enum):
    CRYPTO_PRIMITIVE = auto()
    DEPRECATED_CRYPTO = auto()
    KEY_SIZE = auto()
    MODE = auto()
    API_CALL = auto()
    KEYWORD = auto()
    IDENTIFIER = auto()
    LITERAL = auto()
    OPERATOR = auto()
    DELIMITER = auto()
    COMMENT = auto()
    WHITESPACE = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    position: int


CRYPTO_PRIMITIVES = frozenset({
    "AES", "RSA", "ECC", "ECDSA", "ECDH", "Ed25519", "X25519",
    "SHA256", "SHA384", "SHA512", "SHA3", "SHA3_256", "SHA3_512",
    "SHAKE128", "SHAKE256", "HMAC", "HKDF", "PBKDF2", "scrypt",
    "argon2", "bcrypt", "ChaCha20", "Poly1305", "XSalsa20",
    "Curve25519", "P256", "P384", "secp256k1",
})

DEPRECATED_CRYPTO = frozenset({
    "DES", "3DES", "TripleDES", "RC4", "RC2", "MD5", "MD4",
    "SHA1", "Blowfish", "IDEA", "CAST5",
})

KEY_SIZES = frozenset({
    "128", "192", "256", "384", "512",
    "1024", "2048", "3072", "4096",
})

MODES = frozenset({
    "CBC", "GCM", "CTR", "ECB", "CFB", "OFB", "CCM", "SIV",
    "XTS", "AEAD",
})

CRYPTO_APIS = frozenset({
    "encrypt", "decrypt", "sign", "verify", "hash", "digest",
    "derive", "generate", "exchange", "seal", "open",
    "update", "finalize", "mac", "wrap", "unwrap",
})

PYTHON_KEYWORDS = frozenset({
    "import", "from", "class", "def", "return", "if", "else",
    "elif", "for", "while", "try", "except", "finally", "with",
    "as", "raise", "pass", "break", "continue", "and", "or",
    "not", "in", "is", "None", "True", "False", "yield", "async",
    "await", "lambda",
})

_PATTERNS: list[tuple[str, TokenType]] = [
    (r"#[^\n]*", TokenType.COMMENT),
    (r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', TokenType.COMMENT),
    (r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'', TokenType.LITERAL),
    (r"0[xX][0-9a-fA-F]+|0[bB][01]+|\d+\.?\d*(?:[eE][+-]?\d+)?", TokenType.LITERAL),
    (r"[a-zA-Z_]\w*", TokenType.IDENTIFIER),
    (r"[+\-*/%&|^~<>=!@]+|->", TokenType.OPERATOR),
    (r"[(){}\[\],;:.\\]", TokenType.DELIMITER),
    (r"\s+", TokenType.WHITESPACE),
]

_COMBINED = re.compile("|".join(f"({p})" for p, _ in _PATTERNS))


def _classify_word(value: str) -> TokenType:
    upper = value.upper()
    if value in CRYPTO_PRIMITIVES or upper in {p.upper() for p in CRYPTO_PRIMITIVES}:
        return TokenType.CRYPTO_PRIMITIVE
    if value in DEPRECATED_CRYPTO or upper in {d.upper() for d in DEPRECATED_CRYPTO}:
        return TokenType.DEPRECATED_CRYPTO
    if value in KEY_SIZES:
        return TokenType.KEY_SIZE
    if upper in MODES:
        return TokenType.MODE
    if value.lower() in CRYPTO_APIS:
        return TokenType.API_CALL
    if value in PYTHON_KEYWORDS:
        return TokenType.KEYWORD
    return TokenType.IDENTIFIER


class CryptoCodeTokenizer:
    """Regex-driven tokenizer with crypto-domain classification."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._token_counts: dict[str, int] = {}

    @property
    def vocabulary(self) -> dict[str, int]:
        return dict(self._vocab)

    def tokenize(self, code: str) -> list[Token]:
        tokens: list[Token] = []
        for m in _COMBINED.finditer(code):
            value = m.group()
            pos = m.start()
            for i, (_, ttype) in enumerate(_PATTERNS):
                if m.group(i + 1) is not None:
                    if ttype == TokenType.IDENTIFIER:
                        ttype = _classify_word(value)
                    elif ttype == TokenType.LITERAL and value.isdigit() and value in KEY_SIZES:
                        ttype = TokenType.KEY_SIZE
                    tokens.append(Token(type=ttype, value=value, position=pos))
                    self._record(value)
                    break
        return tokens

    def tokenize_filtered(self, code: str) -> list[Token]:
        return [
            t for t in self.tokenize(code)
            if t.type not in (TokenType.WHITESPACE, TokenType.COMMENT)
        ]

    def extract_crypto_features(self, code: str) -> dict:
        tokens = self.tokenize(code)
        features: dict = {
            "primitives": [],
            "deprecated": [],
            "key_sizes": [],
            "modes": [],
            "api_calls": [],
            "has_deprecated": False,
            "primitive_count": 0,
            "unique_primitives": 0,
        }
        seen_prims: set[str] = set()
        for t in tokens:
            if t.type == TokenType.CRYPTO_PRIMITIVE:
                features["primitives"].append(t.value)
                seen_prims.add(t.value.upper())
            elif t.type == TokenType.DEPRECATED_CRYPTO:
                features["deprecated"].append(t.value)
            elif t.type == TokenType.KEY_SIZE:
                features["key_sizes"].append(int(t.value))
            elif t.type == TokenType.MODE:
                features["modes"].append(t.value.upper())
            elif t.type == TokenType.API_CALL:
                features["api_calls"].append(t.value.lower())

        features["has_deprecated"] = len(features["deprecated"]) > 0
        features["primitive_count"] = len(features["primitives"])
        features["unique_primitives"] = len(seen_prims)
        features["primitives"] = sorted(set(features["primitives"]))
        features["deprecated"] = sorted(set(features["deprecated"]))
        features["key_sizes"] = sorted(set(features["key_sizes"]))
        features["modes"] = sorted(set(features["modes"]))
        features["api_calls"] = sorted(set(features["api_calls"]))
        return features

    def encode(self, code: str) -> list[int]:
        tokens = self.tokenize_filtered(code)
        return [self._vocab.get(t.value, 0) for t in tokens]

    def _record(self, value: str) -> None:
        if value not in self._vocab:
            self._vocab[value] = len(self._vocab) + 1
        self._token_counts[value] = self._token_counts.get(value, 0) + 1
