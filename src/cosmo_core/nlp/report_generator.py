"""Template-based NLG for vulnerability reports with severity-aware language."""
from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


_SEVERITY_LANGUAGE: dict[Severity, dict[str, str]] = {
    Severity.CRITICAL: {
        "urgency": "IMMEDIATE ACTION REQUIRED",
        "descriptor": "critical",
        "risk": "active exploitation is feasible with minimal effort",
        "tone": "This vulnerability poses an imminent threat and must be remediated before deployment.",
    },
    Severity.HIGH: {
        "urgency": "High Priority",
        "descriptor": "high-severity",
        "risk": "exploitation is practical under realistic conditions",
        "tone": "This issue should be addressed in the current development cycle.",
    },
    Severity.MEDIUM: {
        "urgency": "Moderate Priority",
        "descriptor": "moderate",
        "risk": "exploitation requires specific conditions but remains plausible",
        "tone": "Plan remediation within the next sprint.",
    },
    Severity.LOW: {
        "urgency": "Low Priority",
        "descriptor": "low-severity",
        "risk": "exploitation is theoretically possible but unlikely in practice",
        "tone": "Address when convenient; monitor for escalation.",
    },
    Severity.INFO: {
        "urgency": "Informational",
        "descriptor": "informational",
        "risk": "no direct exploit path identified",
        "tone": "Noted for completeness; no action required.",
    },
}

_IMPACT_LABELS = {Severity.CRITICAL: "Very High", Severity.HIGH: "High",
                  Severity.MEDIUM: "Medium", Severity.LOW: "Low", Severity.INFO: "Minimal"}


@dataclass
class ReportFinding:
    title: str
    severity: Severity
    vulnerability_class: str
    description: str
    fix_suggestion: str
    confidence: float = 1.0
    file_path: str | None = None
    line_number: int | None = None


def _sev(raw: Any) -> Severity:
    if isinstance(raw, Severity):
        return raw
    if isinstance(raw, int):
        return Severity(raw)
    name = str(raw).upper().strip()
    return Severity[name] if name in Severity.__members__ else Severity.INFO


class VulnerabilityReportGenerator:
    """Generates structured vulnerability reports from heterogeneous findings."""

    def generate_report(self, findings: list[ReportFinding], fmt: str = "text") -> str:
        findings = sorted(findings, key=lambda f: (-f.severity.value, f.title))
        sections = [
            self.generate_executive_summary(findings),
            self._findings_detail(findings, fmt),
            self.generate_risk_matrix(findings),
            self._remediation_plan(findings, fmt),
        ]
        sep = "\n\n---\n\n" if fmt == "markdown" else "\n\n" + "=" * 72 + "\n\n"
        return sep.join(s for s in sections if s)

    def generate_executive_summary(self, findings: list[ReportFinding]) -> str:
        total = len(findings)
        if total == 0:
            return "No vulnerabilities detected. The scanned code appears to follow cryptographic best practices."

        counts = Counter(f.severity for f in findings)
        mean_conf = sum(f.confidence for f in findings) / total
        classes = Counter(f.vulnerability_class for f in findings)
        top_class = classes.most_common(1)[0][0] if classes else "N/A"

        lines = [
            "EXECUTIVE SUMMARY",
            f"Total findings: {total} | Mean confidence: {mean_conf:.0%}",
            f"  Critical: {counts.get(Severity.CRITICAL, 0)}  "
            f"High: {counts.get(Severity.HIGH, 0)}  "
            f"Medium: {counts.get(Severity.MEDIUM, 0)}  "
            f"Low: {counts.get(Severity.LOW, 0)}  "
            f"Info: {counts.get(Severity.INFO, 0)}",
            f"Most common class: {top_class} ({classes[top_class]})",
        ]

        if counts.get(Severity.CRITICAL, 0) > 0:
            lines.append(
                f"\n  ** {counts[Severity.CRITICAL]} CRITICAL finding(s) require immediate remediation. **"
            )
        return "\n".join(lines)

    def generate_risk_matrix(self, findings: list[ReportFinding]) -> str:
        if not findings:
            return ""

        likelihood_map = {
            Severity.CRITICAL: "Very Likely",
            Severity.HIGH: "Likely",
            Severity.MEDIUM: "Possible",
            Severity.LOW: "Unlikely",
            Severity.INFO: "Rare",
        }

        header = "RISK MATRIX (Likelihood x Impact)\n"
        rows = ["  {:12s} | {:12s} | {:10s} | {}".format(
            "Likelihood", "Impact", "Severity", "Count")]
        rows.append("  " + "-" * 54)
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
            count = sum(1 for f in findings if f.severity == sev)
            if count:
                rows.append("  {:12s} | {:12s} | {:10s} | {}".format(
                    likelihood_map[sev], _IMPACT_LABELS[sev], sev.name, count))
        return header + "\n".join(rows)

    def _findings_detail(self, findings: list[ReportFinding], fmt: str) -> str:
        if not findings:
            return ""
        parts = ["FINDINGS DETAIL\n"]
        for i, f in enumerate(findings, 1):
            lang = _SEVERITY_LANGUAGE[f.severity]
            loc = ""
            if f.file_path:
                loc = f"  Location: {f.file_path}"
                if f.line_number is not None:
                    loc += f":{f.line_number}"
                loc += "\n"
            steps = self._generate_remediation_steps(f)
            steps_text = "\n".join(f"    {j}. {s}" for j, s in enumerate(steps, 1))

            if fmt == "markdown":
                block = (
                    f"### {i}. [{f.severity.name}] {f.title}\n"
                    f"**Class:** {f.vulnerability_class} | "
                    f"**Confidence:** {f.confidence:.0%} | "
                    f"**Priority:** {lang['urgency']}\n\n"
                    f"{f.description}\n\n"
                    f"*Risk:* {lang['risk']}. {lang['tone']}\n"
                    f"{loc}"
                    f"\n**Remediation:**\n{steps_text}\n"
                )
            else:
                block = (
                    f"  [{i}] {f.severity.name}: {f.title}\n"
                    f"      Class: {f.vulnerability_class} | Confidence: {f.confidence:.0%}\n"
                    f"      Priority: {lang['urgency']}\n"
                    f"{loc}"
                    f"      {f.description}\n"
                    f"      Risk: {lang['risk']}. {lang['tone']}\n"
                    f"      Remediation:\n{steps_text}\n"
                )
            parts.append(block)
        return "\n".join(parts)

    def _remediation_plan(self, findings: list[ReportFinding], fmt: str) -> str:
        actionable = [f for f in findings if f.severity >= Severity.MEDIUM]
        if not actionable:
            return ""

        lines = ["REMEDIATION PLAN\n"]
        phase_map = {
            Severity.CRITICAL: "Phase 1 — Immediate (before next deploy)",
            Severity.HIGH: "Phase 2 — Short-term (this sprint)",
            Severity.MEDIUM: "Phase 3 — Planned (next sprint)",
        }

        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM):
            group = [f for f in actionable if f.severity == sev]
            if not group:
                continue
            lines.append(f"  {phase_map[sev]}")
            for f in group:
                loc = f" @ {f.file_path}:{f.line_number}" if f.file_path else ""
                lines.append(f"    - {f.title}{loc}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _generate_remediation_steps(finding: ReportFinding) -> list[str]:
        steps: list[str] = []
        if finding.file_path:
            steps.append(f"Open {finding.file_path}" +
                         (f" at line {finding.line_number}" if finding.line_number else ""))
        steps.append(finding.fix_suggestion)
        if finding.severity >= Severity.HIGH:
            steps.append("Add a regression test covering the vulnerable code path.")
        if finding.severity == Severity.CRITICAL:
            steps.append("Verify the fix in a staging environment before deploying to production.")
        steps.append("Re-run the scanner to confirm the finding is resolved.")
        return steps
