"""Tests for NLP TF-IDF semantic vulnerability scanner."""
import pytest

from cosmo_core.nlp.semantic_scanner import SemanticVulnerabilityScanner


class TestSemanticVulnerabilityScanner:
    def setup_method(self):
        self.scanner = SemanticVulnerabilityScanner()

    def test_finds_ecb_pattern(self):
        matches = self.scanner.scan("using ECB mode no IV deterministic encryption")
        assert len(matches) > 0

    def test_finds_weak_hash(self):
        matches = self.scanner.scan("MD5 SHA1 weak hash function collision vulnerable")
        assert len(matches) > 0

    def test_finds_hardcoded_key(self):
        matches = self.scanner.scan("hardcoded key secret literal embedded password")
        assert len(matches) > 0

    def test_clean_code_fewer_matches(self):
        vuln = self.scanner.scan("ECB mode static IV no authentication")
        clean = self.scanner.scan("def add(a, b): return a + b")
        assert len(vuln) >= len(clean)

    def test_match_has_fields(self):
        matches = self.scanner.scan("ECB mode deterministic")
        if matches:
            m = matches[0]
            assert hasattr(m, "pattern_id")
            assert hasattr(m, "similarity_score")
            assert hasattr(m, "severity")
            assert hasattr(m, "fix_suggestion")

    def test_similarity_score_range(self):
        matches = self.scanner.scan("weak key size small 512 bit RSA")
        for m in matches:
            assert 0.0 <= m.similarity_score <= 1.0

    def test_empty_code(self):
        matches = self.scanner.scan("")
        assert isinstance(matches, list)
