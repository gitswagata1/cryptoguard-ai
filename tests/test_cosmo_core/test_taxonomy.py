"""Tests for the 10-class vulnerability taxonomy scanner."""
import pytest

from cosmo_core.taxonomy import (
    Finding,
    Severity,
    VulnerabilityClass,
    scan_code,
)


VULNERABLE_PYTHON = '''
SECRET_KEY = "hardcoded_secret_123"

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def generate_token():
    return str(random.randint(0, 999999)).zfill(6)

def encrypt_data(data, key=SECRET_KEY):
    from Crypto.Cipher import DES
    cipher = DES.new(key[:8].encode(), DES.MODE_ECB)
    padded = data + ' ' * (8 - len(data) % 8)
    return cipher.encrypt(padded.encode())

def verify_signature(message, signature):
    expected = hashlib.sha1(message.encode()).hexdigest()
    return signature == expected
'''


class TestScanCode:
    def test_detects_hardcoded_secret(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        titles = [f.title.lower() for f in findings]
        assert any("hardcoded" in t or "secret" in t for t in titles)

    def test_detects_weak_hash(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        classes = [f.vuln_class for f in findings]
        assert VulnerabilityClass.WEAK_HASH in classes

    def test_detects_insecure_random(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        titles = [f.title.lower() for f in findings]
        assert any("random" in t for t in titles)

    def test_detects_des(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        titles = [f.title.lower() for f in findings]
        assert any("des" in t or "cipher" in t or "deprecated" in t for t in titles)

    def test_detects_unsafe_mode(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        classes = [f.vuln_class for f in findings]
        assert VulnerabilityClass.UNSAFE_MODE in classes

    def test_clean_code_no_findings(self):
        clean = """
import secrets
import hashlib

def generate_token():
    return secrets.token_hex(32)

def hash_data(data):
    return hashlib.sha3_256(data).hexdigest()
"""
        findings = scan_code(clean, "python")
        assert len(findings) == 0

    def test_finding_has_line_numbers(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        for f in findings:
            if f.line is not None:
                assert f.line > 0

    def test_findings_have_severity(self):
        findings = scan_code(VULNERABLE_PYTHON, "python")
        valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        for f in findings:
            assert f.severity in valid_severities

    def test_auto_detect_language(self):
        findings = scan_code(VULNERABLE_PYTHON, "auto")
        assert len(findings) > 0
