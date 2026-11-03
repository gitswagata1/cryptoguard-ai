"""Tests for NLP vulnerability report generator."""
import pytest

from cosmo_core.nlp.report_generator import (
    ReportFinding,
    Severity,
    VulnerabilityReportGenerator,
)


class TestVulnerabilityReportGenerator:
    def setup_method(self):
        self.gen = VulnerabilityReportGenerator()
        self.findings = [
            ReportFinding(
                title="ECB Mode",
                severity=Severity.HIGH,
                vulnerability_class="weak-cipher-mode",
                description="ECB mode leaks patterns in ciphertext",
                fix_suggestion="Use GCM or CTR mode",
                confidence=0.95,
            ),
            ReportFinding(
                title="MD5 Usage",
                severity=Severity.MEDIUM,
                vulnerability_class="weak-hash",
                description="MD5 is vulnerable to collision attacks",
                fix_suggestion="Use SHA-256 or SHA-3",
                confidence=0.88,
            ),
        ]

    def test_generates_text_report(self):
        report = self.gen.generate_report(self.findings, fmt="text")
        assert isinstance(report, str)
        assert len(report) > 100

    def test_generates_markdown_report(self):
        report = self.gen.generate_report(self.findings, fmt="markdown")
        assert isinstance(report, str)
        assert "#" in report or "**" in report

    def test_executive_summary(self):
        summary = self.gen.generate_executive_summary(self.findings)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_empty_findings(self):
        report = self.gen.generate_report([], fmt="text")
        assert isinstance(report, str)

    def test_single_finding(self):
        report = self.gen.generate_report([self.findings[0]], fmt="text")
        assert "ECB" in report or "HIGH" in report or "high" in report.lower()

    def test_risk_matrix(self):
        matrix = self.gen.generate_risk_matrix(self.findings)
        assert isinstance(matrix, str)
