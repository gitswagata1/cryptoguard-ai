from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_KNOWLEDGE_BASE: dict[str, str] = {
    "quantum": (
        "Post-quantum cryptography is essential for long-term security. NIST has "
        "standardised CRYSTALS-Kyber (ML-KEM) for key encapsulation and "
        "CRYSTALS-Dilithium (ML-DSA) for digital signatures. Transition plans "
        "should begin now: inventory all cryptographic assets, identify "
        "quantum-vulnerable algorithms (RSA, ECDSA, ECDH), and adopt hybrid "
        "schemes that combine classical and post-quantum primitives."
    ),
    "md5": (
        "MD5 is cryptographically broken. Collision attacks are practical and "
        "demonstrated (e.g. Xiaoyun Wang, 2004). MD5 MUST NOT be used for "
        "digital signatures, certificate validation, or integrity verification "
        "of security-critical data. Replace with SHA-256 or SHA-3. MD5 may only "
        "be acceptable for non-security checksums (file deduplication)."
    ),
    "sha1": (
        "SHA-1 is deprecated for all security-sensitive applications (NIST SP "
        "800-131A Rev 2). The SHAttered attack (2017) demonstrated practical "
        "collision generation. Migrate to SHA-256 or SHA-3. Certificate "
        "authorities and TLS implementations have already phased out SHA-1."
    ),
    "kdf": (
        "Key derivation functions must use memory-hard algorithms for password "
        "hashing: Argon2id is the current recommendation (RFC 9106), with "
        "scrypt and bcrypt as acceptable alternatives. PBKDF2 with HMAC-SHA-256 "
        "is permitted when regulatory compliance requires it (NIST SP 800-132) "
        "but requires high iteration counts (minimum 600,000 per OWASP 2023). "
        "Never derive keys from passwords using a plain hash."
    ),
    "lhl": (
        "The Leftover Hash Lemma (LHL) guarantees that applying a universal "
        "hash function to a high-entropy source produces output that is "
        "statistically close to uniform. For an (n, k)-source with min-entropy "
        "k, the output length must be at most k - 2*log(1/epsilon) bits for "
        "statistical distance epsilon. This is foundational for randomness "
        "extraction and privacy amplification in QKD protocols."
    ),
    "entropy": (
        "Cryptographic entropy assessment requires multiple measures: Shannon "
        "entropy (information content), min-entropy (worst-case guessability), "
        "chi-squared (distribution uniformity), and serial correlation "
        "(independence of consecutive values). NIST SP 800-90B defines entropy "
        "source validation. A quality CSPRNG should achieve > 7.9 bits/byte "
        "Shannon entropy and pass all NIST SP 800-22 statistical tests."
    ),
    "aead": (
        "Authenticated Encryption with Associated Data (AEAD) provides "
        "confidentiality, integrity, and authenticity in a single operation. "
        "Preferred constructions: AES-256-GCM (widely supported, hardware "
        "acceleration), ChaCha20-Poly1305 (constant-time, no AES-NI needed), "
        "and AES-256-GCM-SIV (nonce-misuse resistant). Always use unique "
        "nonces; GCM catastrophically fails on nonce reuse."
    ),
    "random": (
        "Cryptographic random number generation must use a CSPRNG seeded from "
        "the OS entropy pool (/dev/urandom on Linux, BCryptGenRandom on "
        "Windows). Never use math.random(), rand(), or similar PRNGs for "
        "security purposes. In Python, use secrets.token_bytes(); in Go, use "
        "crypto/rand.Read(); in Java, use SecureRandom. Verify entropy quality "
        "with NIST SP 800-22 tests before deployment."
    ),
    "tls": (
        "TLS 1.3 (RFC 8446) is the mandatory minimum for new deployments. It "
        "removes insecure cipher suites, compresses the handshake to 1-RTT "
        "(0-RTT with PSK), and mandates forward secrecy. Disable TLS 1.0/1.1 "
        "entirely. For TLS 1.2, restrict to AEAD cipher suites with ECDHE key "
        "exchange. Pin certificates or use Certificate Transparency monitoring."
    ),
    "rsa": (
        "RSA key sizes below 2048 bits are insecure. NIST recommends RSA-3072 "
        "for security through 2031 and RSA-4096 for beyond. Use OAEP padding "
        "for encryption (never PKCS#1 v1.5 for new systems) and PSS padding "
        "for signatures. RSA is not quantum-safe; plan migration to ML-KEM / "
        "ML-DSA hybrid schemes."
    ),
}

_DEFAULT_RESPONSE = (
    "I am running in local analysis mode without AI cloud connectivity. "
    "I can provide deterministic cryptographic analysis, code scanning via "
    "the CWE taxonomy engine, and entropy assessment. For advanced natural "
    "language interaction, configure a Nebius API key to enable cloud mode."
)


def _match_topic(message: str) -> str:
    """Find the best matching knowledge-base entry for a message."""
    msg_lower = message.lower()
    best_key = ""
    best_count = 0
    for key in _KNOWLEDGE_BASE:
        occurrences = msg_lower.count(key)
        if occurrences > best_count:
            best_count = occurrences
            best_key = key
    if best_key:
        return _KNOWLEDGE_BASE[best_key]
    return _DEFAULT_RESPONSE


class LocalFallbackClient:
    """Deterministic crypto analysis client for offline / local mode."""

    @property
    def available(self) -> bool:
        """Local mode is always available."""
        return True

    async def chat(
        self,
        message: str,
        system_prompt: str,
        context: str | None = None,
    ) -> tuple[str, str]:
        """Return structured crypto guidance from the local knowledge base."""
        response = _match_topic(message)
        return response, "local-knowledge-base"

    async def scan_code(
        self,
        code: str,
        language: str,
        taxonomy_prompt: str,
    ) -> dict:
        """Delegate code scanning to the cosmo_core taxonomy engine."""
        try:
            from cosmo_core.taxonomy import scan_code as core_scan

            core_findings = core_scan(code, language)
            findings = []
            for f in core_findings:
                findings.append({
                    "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                    "title": f.title,
                    "description": f.description,
                    "line": f.line,
                    "fix": f.fix,
                    "vuln_class": f.vuln_class.value if f.vuln_class else None,
                })
            return {"findings": findings}
        except ImportError:
            logger.warning("cosmo_core not installed, returning empty scan")
            return {"findings": []}
        except Exception:
            logger.exception("Local code scan failed")
            return {"findings": []}

    async def analyze_entropy(self, metrics: dict) -> str:
        """Generate a template-based entropy assessment from raw metrics."""
        shannon = metrics.get("shannon_entropy", 0.0)
        min_ent = metrics.get("min_entropy", 0.0)
        chi_sq = metrics.get("chi_squared", 0.0)
        serial = metrics.get("serial_correlation", 0.0)
        quality = metrics.get("quality_score", 0)

        if quality >= 80:
            grade = "excellent"
        elif quality >= 60:
            grade = "good"
        elif quality >= 40:
            grade = "marginal"
        else:
            grade = "poor"

        parts = [
            f"Entropy quality assessment: {grade} (score {quality}/100).",
            f"Shannon entropy is {shannon:.4f} bits/byte"
            f" ({'adequate' if shannon > 7.5 else 'below recommended threshold'}).",
            f"Min-entropy is {min_ent:.4f}"
            f" ({'acceptable' if min_ent > 6.0 else 'low, potential predictability'}).",
            f"Chi-squared statistic: {chi_sq:.2f}"
            f" ({'uniform distribution' if 200 < chi_sq < 300 else 'non-uniform distribution detected'}).",
            f"Serial correlation: {serial:.6f}"
            f" ({'negligible' if abs(serial) < 0.01 else 'sequential dependence detected'}).",
        ]

        if quality < 60:
            parts.append(
                "Recommendation: this source may not be suitable for "
                "cryptographic key generation. Investigate the entropy source "
                "and consider applying a conditioning component per NIST SP 800-90B."
            )

        return " ".join(parts)

    async def analyze_image(self, image_b64: str, prompt: str) -> str:
        """Image analysis is not available in local mode."""
        return "Image analysis requires cloud mode with NeVA"
