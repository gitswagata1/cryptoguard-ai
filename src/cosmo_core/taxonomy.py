"""10-class cryptographic vulnerability taxonomy with regex-based scanner."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VulnerabilityClass(str, Enum):
    HARDCODED_SECRETS = "HARDCODED_SECRETS"
    WEAK_HASH = "WEAK_HASH"
    INSECURE_RNG = "INSECURE_RNG"
    DEPRECATED_CIPHER = "DEPRECATED_CIPHER"
    UNSAFE_MODE = "UNSAFE_MODE"
    IMPROPER_PADDING = "IMPROPER_PADDING"
    TIMING_UNSAFE = "TIMING_UNSAFE"
    NONCE_REUSE = "NONCE_REUSE"
    INSUFFICIENT_KEY_LENGTH = "INSUFFICIENT_KEY_LENGTH"
    MISSING_AEAD = "MISSING_AEAD"


@dataclass(frozen=True, slots=True)
class VulnerabilityRule:
    """A detection rule for a vulnerability class."""
    vuln_class: VulnerabilityClass
    severity: Severity
    title: str
    description: str
    patterns: list[str]
    fix_template: str


@dataclass(frozen=True, slots=True)
class Finding:
    """A single vulnerability finding in source code."""
    severity: Severity
    title: str
    description: str
    line: int
    fix: str
    vuln_class: VulnerabilityClass


TAXONOMY: list[VulnerabilityRule] = [
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.HARDCODED_SECRETS,
        severity=Severity.CRITICAL,
        title="Hardcoded secret or credential",
        description="Secret keys, passwords, or tokens embedded directly in source code.",
        patterns=[
            r"""(?i)(?:api[_-]?key|secret[_-]?key|password|passwd|token|auth[_-]?token)\s*[:=]\s*["'][A-Za-z0-9+/=_\-]{8,}["']""",
            r"""(?i)(?:AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}""",
            r"""(?i)(?:private[_-]?key)\s*[:=]\s*["']-----BEGIN""",
        ],
        fix_template="Move secrets to environment variables or a secrets manager.",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.WEAK_HASH,
        severity=Severity.HIGH,
        title="Weak or broken hash function",
        description="Use of MD5, SHA-1, or other cryptographically broken hash algorithms.",
        patterns=[
            r"""\bhashlib\.(?:md5|sha1)\b""",
            r"""\bMessageDigest\.getInstance\(\s*["'](?:MD5|SHA-?1)["']\s*\)""",
            r"""\bcrypto\.createHash\(\s*["'](?:md5|sha1)["']\s*\)""",
            r"""\b(?:md5|sha1)\.(?:New|Sum)\b""",
        ],
        fix_template="Replace with SHA-256 or SHA-3. For password hashing use bcrypt/scrypt/Argon2.",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.INSECURE_RNG,
        severity=Severity.CRITICAL,
        title="Insecure random number generator",
        description="Use of non-cryptographic PRNG for security-sensitive operations.",
        patterns=[
            r"""\brandom\.(?:random|randint|choice|randrange|seed)\b""",
            r"""\bMath\.random\(\)""",
            r"""\bjava\.util\.Random\b""",
            r"""\bmath/rand\b""",
            r"""\brand\.(?:Intn|Int63|Float64|Seed)\b""",
        ],
        fix_template="Use os.urandom(), secrets module, crypto/rand (Go), or SecureRandom (Java).",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.DEPRECATED_CIPHER,
        severity=Severity.HIGH,
        title="Deprecated or broken cipher",
        description="Use of DES, 3DES, RC4, Blowfish, or other deprecated ciphers.",
        patterns=[
            r"""\b(?:DES|TripleDES|Triple_DES|DES3|RC4|ARC4|Blowfish)\b""",
            r"""Cipher\.getInstance\(\s*["'](?:DES|DESede|RC4|Blowfish)""",
            r"""\bcrypto\.createCipher(?:iv)?\(\s*["'](?:des|rc4|bf)""",
            r"""\bdes\.NewCipher\b""",
        ],
        fix_template="Use AES-256-GCM or ChaCha20-Poly1305.",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.UNSAFE_MODE,
        severity=Severity.HIGH,
        title="Unsafe block cipher mode",
        description="Use of ECB mode which leaks patterns, or CBC without HMAC.",
        patterns=[
            r"""\bmodes\.ECB\b""",
            r"""\b(?:AES|DES)\.MODE_ECB\b""",
            r"""Cipher\.getInstance\(\s*["']AES/ECB""",
            r"""\bcrypto\.createCipher(?:iv)?\(\s*["']aes-\d+-ecb""",
            r"""\bNewECBEncrypter\b""",
            r"""\bMODE_ECB\b""",
        ],
        fix_template="Use GCM, CCM, or SIV mode instead of ECB. If CBC is required, add HMAC (encrypt-then-MAC).",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.IMPROPER_PADDING,
        severity=Severity.MEDIUM,
        title="Improper or missing padding scheme",
        description="Custom or absent padding that may enable padding oracle attacks.",
        patterns=[
            r"""\bpkcs7\.pad\b.*\bpkcs7\.unpad\b""",
            r"""\bPKCS1v15\b""",
            r"""\bCipher\.getInstance\(\s*["']RSA/ECB/PKCS1Padding""",
            r"""\bpadding\.PKCS1v15\(\)""",
        ],
        fix_template="Use OAEP padding for RSA. For symmetric ciphers, prefer AEAD modes that handle padding internally.",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.TIMING_UNSAFE,
        severity=Severity.HIGH,
        title="Timing-unsafe comparison",
        description="Direct equality comparison of secrets or MACs enables timing attacks.",
        patterns=[
            r"""(?:mac|hash|digest|signature|token|secret|hmac)\s*==\s*""",
            r"""\.equals\(\s*(?:mac|hash|digest|signature|token|hmac)""",
            r"""\bif\s+.*(?:password|secret|key)\s*==\s*""",
        ],
        fix_template="Use hmac.compare_digest() (Python), crypto.timingSafeEqual() (Node.js), or subtle.ConstantTimeCompare() (Go).",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.NONCE_REUSE,
        severity=Severity.CRITICAL,
        title="Static or reused nonce/IV",
        description="Hardcoded or zero IV/nonce destroys confidentiality in stream ciphers and CTR/GCM.",
        patterns=[
            r"""\bnonce\s*[:=]\s*(?:b["']\\x00|bytes\(\d+\)|b["']\x00|new\s+byte\[\d+\])""",
            r"""\biv\s*[:=]\s*(?:b["']\\x00|bytes\(\d+\)|b["']\x00|new\s+byte\[\d+\])""",
            r"""\b(?:nonce|iv)\s*[:=]\s*["'][0]+["']""",
            r"""\b(?:nonce|iv)\s*[:=]\s*(?:b["'][^"']+["']|"[^"]+"|'[^']+')(?:\s*#.*static)""",
        ],
        fix_template="Generate a fresh random nonce/IV for every encryption operation using os.urandom().",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.INSUFFICIENT_KEY_LENGTH,
        severity=Severity.MEDIUM,
        title="Insufficient key length",
        description="Key sizes below recommended minimums (RSA < 2048, AES < 128, EC < 256).",
        patterns=[
            r"""\bgenerate_private_key\(\s*(?:public_exponent\s*=\s*\d+\s*,\s*)?key_size\s*=\s*(?:512|768|1024)\b""",
            r"""\bKeyPairGenerator\.getInstance.*\.initialize\(\s*(?:512|768|1024)\b""",
            r"""\brsa\.GenerateKey\(.*,\s*(?:512|768|1024)\)""",
            r"""\baes-(?:40|56|64)-""",
        ],
        fix_template="Use RSA >= 2048 bits, AES >= 128 bits, EC >= 256 bits. Prefer 256-bit symmetric keys.",
    ),
    VulnerabilityRule(
        vuln_class=VulnerabilityClass.MISSING_AEAD,
        severity=Severity.MEDIUM,
        title="Encryption without authentication",
        description="Symmetric encryption without integrity protection enables bit-flipping and oracle attacks.",
        patterns=[
            r"""\bmodes\.CBC\b(?!.*HMAC)""",
            r"""\bAES\.MODE_CBC\b""",
            r"""\bAES\.MODE_CTR\b(?!.*HMAC)""",
            r"""Cipher\.getInstance\(\s*["']AES/CBC/PKCS5Padding["']\s*\)""",
            r"""\bcrypto\.createCipher(?:iv)?\(\s*["']aes-\d+-cbc""",
        ],
        fix_template="Use an AEAD cipher (AES-GCM, ChaCha20-Poly1305) or add encrypt-then-MAC with HMAC-SHA256.",
    ),
]


def build_taxonomy_prompt() -> str:
    """Build a structured prompt describing all vulnerability classes for LLM-based scanning."""
    lines = ["Cryptographic Vulnerability Taxonomy (10 classes):\n"]
    for i, rule in enumerate(TAXONOMY, 1):
        lines.append(
            f"{i}. [{rule.severity.value}] {rule.vuln_class.value}: {rule.title}\n"
            f"   Description: {rule.description}\n"
            f"   Fix: {rule.fix_template}\n"
        )
    lines.append(
        "\nReturn findings as JSON: [{severity, title, description, line, fix, vuln_class}]"
    )
    return "\n".join(lines)


def scan_code(code: str, language: str = "generic") -> list[Finding]:
    """Scan source code against all taxonomy rules and return findings with line numbers."""
    findings: list[Finding] = []
    lines = code.split("\n")

    for rule in TAXONOMY:
        for pattern_str in rule.patterns:
            try:
                pattern = re.compile(pattern_str)
            except re.error:
                continue
            for line_num, line_text in enumerate(lines, start=1):
                if pattern.search(line_text):
                    finding = Finding(
                        severity=rule.severity,
                        title=rule.title,
                        description=rule.description,
                        line=line_num,
                        fix=rule.fix_template,
                        vuln_class=rule.vuln_class,
                    )
                    # Avoid duplicate findings on the same line for the same rule
                    if not any(
                        f.line == line_num and f.vuln_class == rule.vuln_class
                        for f in findings
                    ):
                        findings.append(finding)
    return findings
