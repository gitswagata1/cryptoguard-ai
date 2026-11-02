"""Entropy pool with XOR mixing and HKDF-SHA3-256 conditioning."""
from __future__ import annotations

import os

from cryptography.hazmat.primitives.hashes import SHA3_256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class EntropyPool:
    """Accumulates entropy via XOR mixing and conditions output with HKDF-SHA3-256."""

    def __init__(self, pool_size: int = 256) -> None:
        self._pool = bytearray(pool_size)
        self._contributions: int = 0

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def contributions(self) -> int:
        return self._contributions

    def mix(self, data: bytes) -> None:
        """XOR data into the pool, wrapping cyclically."""
        for i, b in enumerate(data):
            self._pool[i % len(self._pool)] ^= b
        self._contributions += 1

    def condition(
        self,
        length: int = 32,
        salt: bytes | None = None,
        info: bytes = b"cosmo_core.conditioning",
    ) -> bytes:
        """Derive conditioned output via HKDF-SHA3-256."""
        return hkdf_sha3_256_condition(
            pool_bytes=bytes(self._pool),
            salt=salt,
            info=info,
            length=length,
        )

    def snapshot(self) -> bytes:
        """Return a copy of the raw pool state."""
        return bytes(self._pool)

    def reset(self) -> None:
        """Zero the pool and reset contribution count."""
        self._pool = bytearray(len(self._pool))
        self._contributions = 0


def hkdf_sha3_256_condition(
    pool_bytes: bytes,
    salt: bytes | None = None,
    info: bytes = b"cosmo_core.conditioning",
    length: int = 32,
) -> bytes:
    """Standalone HKDF-SHA3-256 key derivation."""
    if salt is None:
        salt = os.urandom(16)
    hkdf = HKDF(
        algorithm=SHA3_256(),
        length=length,
        salt=salt,
        info=info,
    )
    return hkdf.derive(pool_bytes)
