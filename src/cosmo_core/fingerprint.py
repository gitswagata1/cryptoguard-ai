"""SHA3-512 fingerprinting."""
from __future__ import annotations

import hashlib


def sha3_512_fingerprint(data: bytes) -> bytes:
    """Return 64-byte SHA3-512 digest of data."""
    return hashlib.sha3_512(data).digest()
