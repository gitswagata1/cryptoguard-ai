"""Tests for HKDF-SHA3-256 conditioning — validates the patent-critical fix."""
import os

import pytest

from cosmo_core.conditioning import EntropyPool, hkdf_sha3_256_condition


class TestHKDFSHA3256:
    def test_produces_correct_length(self):
        result = hkdf_sha3_256_condition(os.urandom(64), length=32)
        assert len(result) == 32

    def test_custom_length(self):
        for length in [16, 48, 64]:
            result = hkdf_sha3_256_condition(os.urandom(64), length=length)
            assert len(result) == length

    def test_deterministic_with_same_salt(self):
        pool_bytes = os.urandom(64)
        salt = os.urandom(16)
        r1 = hkdf_sha3_256_condition(pool_bytes, salt=salt, info=b"test")
        r2 = hkdf_sha3_256_condition(pool_bytes, salt=salt, info=b"test")
        assert r1 == r2

    def test_different_info_different_output(self):
        pool_bytes = os.urandom(64)
        salt = os.urandom(16)
        r1 = hkdf_sha3_256_condition(pool_bytes, salt=salt, info=b"purpose_a")
        r2 = hkdf_sha3_256_condition(pool_bytes, salt=salt, info=b"purpose_b")
        assert r1 != r2

    def test_different_salt_different_output(self):
        pool_bytes = os.urandom(64)
        r1 = hkdf_sha3_256_condition(pool_bytes, salt=b"\x00" * 16)
        r2 = hkdf_sha3_256_condition(pool_bytes, salt=b"\x01" * 16)
        assert r1 != r2


class TestEntropyPool:
    def test_initial_state_is_zeroed(self):
        pool = EntropyPool(pool_size=32)
        assert pool.snapshot() == b"\x00" * 32
        assert pool.contributions == 0

    def test_mix_xors_data_into_pool(self):
        pool = EntropyPool(pool_size=4)
        pool.mix(b"\xff\xff\xff\xff")
        assert pool.snapshot() == b"\xff\xff\xff\xff"
        assert pool.contributions == 1

    def test_double_mix_cancels_via_xor(self):
        pool = EntropyPool(pool_size=4)
        data = b"\xab\xcd\xef\x12"
        pool.mix(data)
        pool.mix(data)
        assert pool.snapshot() == b"\x00\x00\x00\x00"
        assert pool.contributions == 2

    def test_mix_wraps_cyclically(self):
        pool = EntropyPool(pool_size=4)
        pool.mix(b"\x01\x02\x03\x04\x10\x20")
        snap = pool.snapshot()
        assert snap[0] == 0x01 ^ 0x10
        assert snap[1] == 0x02 ^ 0x20

    def test_condition_returns_bytes(self):
        pool = EntropyPool(pool_size=64)
        pool.mix(os.urandom(64))
        result = pool.condition(length=32, salt=os.urandom(16))
        assert len(result) == 32
        assert isinstance(result, bytes)

    def test_reset_clears_pool(self):
        pool = EntropyPool(pool_size=32)
        pool.mix(os.urandom(32))
        assert pool.contributions == 1
        pool.reset()
        assert pool.snapshot() == b"\x00" * 32
        assert pool.contributions == 0
