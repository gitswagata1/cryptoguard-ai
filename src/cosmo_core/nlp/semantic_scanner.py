"""TF-IDF semantic vulnerability scanner — matches code against known-bad patterns."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from cosmo_core.nlp.tokenizer import CryptoCodeTokenizer


class VulnSeverity(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


@dataclass(frozen=True, slots=True)
class VulnerabilityPattern:
    pattern_id: str
    text: str
    description: str
    severity: VulnSeverity
    vulnerability_class: str
    fix_suggestion: str


@dataclass(slots=True)
class SemanticMatch:
    pattern_id: str
    similarity_score: float
    severity: VulnSeverity
    description: str
    fix_suggestion: str
    vulnerability_class: str


VULNERABILITY_CORPUS: list[VulnerabilityPattern] = [
    VulnerabilityPattern(
        "VULN-ECB-001", "ECB mode deterministic encryption no IV block cipher",
        "ECB mode encrypts identical plaintext blocks to identical ciphertext, leaking patterns.",
        VulnSeverity.CRITICAL, "weak-cipher-mode",
        "Switch to AES-GCM or AES-CTR with a unique nonce per message.",
    ),
    VulnerabilityPattern(
        "VULN-HARDCODED-002", "hardcoded key material string literal secret password embedded",
        "Cryptographic key embedded in source code is extractable by anyone with repo access.",
        VulnSeverity.CRITICAL, "hardcoded-secret",
        "Load keys from a secrets manager or environment variable; never commit key material.",
    ),
    VulnerabilityPattern(
        "VULN-WEAKHASH-003", "MD5 SHA1 weak hash collision vulnerable digest",
        "MD5/SHA-1 are collision-vulnerable and unsuitable for security-sensitive hashing.",
        VulnSeverity.HIGH, "weak-hash",
        "Migrate to SHA-256 or SHA3-256 for integrity checks; use Argon2 for password hashing.",
    ),
    VulnerabilityPattern(
        "VULN-NOAUTH-004", "no authentication encrypt without MAC tag unauthenticated ciphertext",
        "Encryption without authentication (no MAC/tag) permits undetected ciphertext tampering.",
        VulnSeverity.CRITICAL, "unauthenticated-encryption",
        "Use an AEAD cipher (AES-GCM, ChaCha20-Poly1305) that provides built-in authentication.",
    ),
    VulnerabilityPattern(
        "VULN-STATICIV-005", "static IV nonce reuse predictable initialization vector constant",
        "Reusing an IV/nonce with the same key breaks confidentiality in most cipher modes.",
        VulnSeverity.CRITICAL, "nonce-reuse",
        "Generate a fresh random nonce (os.urandom) for every encryption operation.",
    ),
    VulnerabilityPattern(
        "VULN-KEYSIZE-006", "insufficient key size short key weak 512 1024 RSA small",
        "Key too short for the algorithm; brute-force or factoring becomes feasible.",
        VulnSeverity.HIGH, "insufficient-key-size",
        "Use >= 2048-bit RSA, >= 256-bit ECC, or >= 128-bit symmetric keys.",
    ),
    VulnerabilityPattern(
        "VULN-TIMING-007", "timing side channel comparison constant time leak observable",
        "Non-constant-time comparison leaks secret length or value via timing measurements.",
        VulnSeverity.HIGH, "timing-side-channel",
        "Use hmac.compare_digest() or secrets.compare_digest() for constant-time comparison.",
    ),
    VulnerabilityPattern(
        "VULN-PRNG-008", "random pseudo random insecure math random predictable seed",
        "Non-cryptographic PRNG output is predictable and must never be used as key material.",
        VulnSeverity.CRITICAL, "insecure-random",
        "Use os.urandom() or secrets module for cryptographic randomness.",
    ),
    VulnerabilityPattern(
        "VULN-PKCS1-009", "PKCS1 v1.5 padding RSA encrypt sign padding oracle",
        "PKCS#1 v1.5 padding is vulnerable to Bleichenbacher-style padding oracle attacks.",
        VulnSeverity.HIGH, "weak-padding",
        "Use OAEP for encryption and PSS for signatures.",
    ),
    VulnerabilityPattern(
        "VULN-NOVALIDATE-010", "certificate verify disable SSL TLS hostname check false",
        "Disabling certificate validation permits machine-in-the-middle attacks.",
        VulnSeverity.CRITICAL, "certificate-validation",
        "Always validate certificates; pin the CA if needed, never set verify=False.",
    ),
    VulnerabilityPattern(
        "VULN-DEPRECATED-011", "DES 3DES TripleDES RC4 RC2 Blowfish obsolete legacy",
        "Deprecated cipher with known weaknesses; practical attacks exist.",
        VulnSeverity.HIGH, "deprecated-cipher",
        "Replace with AES-256-GCM or ChaCha20-Poly1305.",
    ),
    VulnerabilityPattern(
        "VULN-PBKDF-012", "low iteration count PBKDF2 rounds cost factor weak stretch",
        "Insufficient KDF iterations make brute-force password cracking feasible.",
        VulnSeverity.MEDIUM, "weak-kdf-params",
        "Use >= 600,000 PBKDF2 iterations, or migrate to Argon2id.",
    ),
    VulnerabilityPattern(
        "VULN-EXPORTKEY-013", "key export serialize dump write log print private secret",
        "Private key material written to logs or serialized insecurely is exfiltrable.",
        VulnSeverity.HIGH, "key-exposure",
        "Never log key material; use encrypted at-rest storage for serialized keys.",
    ),
    VulnerabilityPattern(
        "VULN-ENTROPY-014", "low entropy insufficient randomness seed deterministic pool",
        "Insufficient entropy in key generation makes keys predictable.",
        VulnSeverity.HIGH, "low-entropy",
        "Condition raw entropy through HKDF-SHA3-256 and validate with NIST SP 800-90B.",
    ),
    VulnerabilityPattern(
        "VULN-RENEGO-015", "renegotiation downgrade protocol version fallback TLS SSL",
        "Protocol downgrade or renegotiation attacks weaken the effective security level.",
        VulnSeverity.MEDIUM, "protocol-downgrade",
        "Enforce TLS 1.3 minimum; disable renegotiation and fallback cipher suites.",
    ),
]


class TfidfVectorizer:
    """From-scratch TF-IDF vectorizer operating on whitespace-tokenized text."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def fit(self, documents: list[str]) -> TfidfVectorizer:
        vocab_set: dict[str, int] = {}
        doc_freq: dict[str, int] = {}
        for doc in documents:
            seen: set[str] = set()
            for word in self._tokenize(doc):
                if word not in vocab_set:
                    vocab_set[word] = len(vocab_set)
                if word not in seen:
                    doc_freq[word] = doc_freq.get(word, 0) + 1
                    seen.add(word)

        self._vocab = vocab_set
        n = len(documents)
        idf = np.zeros(len(vocab_set), dtype=np.float64)
        for word, idx in vocab_set.items():
            df = doc_freq.get(word, 0)
            idf[idx] = math.log((1 + n) / (1 + df)) + 1.0
        self._idf = idf
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("Call fit() before transform().")
        matrix = np.zeros((len(documents), len(self._vocab)), dtype=np.float64)
        for i, doc in enumerate(documents):
            words = self._tokenize(doc)
            if not words:
                continue
            for word in words:
                idx = self._vocab.get(word)
                if idx is not None:
                    matrix[i, idx] += 1.0
            row_max = matrix[i].max()
            if row_max > 0:
                matrix[i] /= row_max
            matrix[i] *= self._idf
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]\w*|\d+", text.lower())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticVulnerabilityScanner:
    """Scans code for vulnerability patterns via TF-IDF cosine similarity."""

    def __init__(self, threshold: float = 0.15) -> None:
        self._threshold = threshold
        self._patterns = list(VULNERABILITY_CORPUS)
        self._vectorizer = TfidfVectorizer()
        self._tokenizer = CryptoCodeTokenizer()
        corpus_texts = [p.text for p in self._patterns]
        self._pattern_vectors = self._vectorizer.fit_transform(corpus_texts)

    def scan(self, code: str) -> list[SemanticMatch]:
        features = self._tokenizer.extract_crypto_features(code)
        query_terms = (
            features["primitives"]
            + features["deprecated"]
            + features["modes"]
            + features["api_calls"]
            + [str(k) for k in features["key_sizes"]]
        )
        words = re.findall(r"[a-zA-Z_]\w*|\d+", code.lower())
        query_text = " ".join(query_terms) + " " + " ".join(words)

        query_vec = self._vectorizer.transform([query_text])[0]

        matches: list[SemanticMatch] = []
        for i, pattern in enumerate(self._patterns):
            sim = cosine_similarity(query_vec, self._pattern_vectors[i])
            if sim >= self._threshold:
                matches.append(SemanticMatch(
                    pattern_id=pattern.pattern_id,
                    similarity_score=round(sim, 4),
                    severity=pattern.severity,
                    description=pattern.description,
                    fix_suggestion=pattern.fix_suggestion,
                    vulnerability_class=pattern.vulnerability_class,
                ))
        matches.sort(key=lambda m: (-m.severity.value, -m.similarity_score))
        return matches

    def add_pattern(self, pattern: VulnerabilityPattern) -> None:
        self._patterns.append(pattern)
        corpus_texts = [p.text for p in self._patterns]
        self._pattern_vectors = self._vectorizer.fit_transform(corpus_texts)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)
