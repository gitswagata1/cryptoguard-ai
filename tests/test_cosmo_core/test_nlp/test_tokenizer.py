"""Tests for NLP crypto-aware code tokenizer."""
import pytest

from cosmo_core.nlp.tokenizer import CryptoCodeTokenizer, TokenType


class TestCryptoCodeTokenizer:
    def setup_method(self):
        self.tok = CryptoCodeTokenizer()

    def test_identifies_crypto_primitives(self):
        tokens = self.tok.tokenize("AES RSA SHA256 HMAC")
        crypto = [t for t in tokens if t.type == TokenType.CRYPTO_PRIMITIVE]
        assert len(crypto) >= 2

    def test_identifies_deprecated(self):
        tokens = self.tok.tokenize("MD5 DES RC4 SHA1")
        deprecated = [t for t in tokens if t.type == TokenType.DEPRECATED_CRYPTO]
        assert len(deprecated) >= 2

    def test_identifies_key_sizes(self):
        tokens = self.tok.tokenize("key_size = 256")
        key_sizes = [t for t in tokens if t.type == TokenType.KEY_SIZE]
        assert any(t.value == "256" for t in key_sizes)

    def test_identifies_modes(self):
        tokens = self.tok.tokenize("mode = ECB; alt = GCM")
        modes = [t for t in tokens if t.type == TokenType.MODE]
        assert len(modes) >= 1

    def test_empty_code(self):
        tokens = self.tok.tokenize("")
        assert isinstance(tokens, list)

    def test_extract_features_structure(self):
        feat = self.tok.extract_crypto_features("AES ECB MD5 256")
        assert "primitives" in feat
        assert "deprecated" in feat
        assert "modes" in feat
        assert "key_sizes" in feat

    def test_extract_features_deprecated_flag(self):
        feat = self.tok.extract_crypto_features("MD5 hash")
        assert feat["has_deprecated"] is True

    def test_extract_features_clean_code(self):
        feat = self.tok.extract_crypto_features("x = 1 + 2")
        assert feat["has_deprecated"] is False
