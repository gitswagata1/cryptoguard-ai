from __future__ import annotations


class SecurityRewardFunction:
    """Reward computation for RL-based routing optimisation."""

    @staticmethod
    def score_scan(
        findings: list[dict],
        ground_truth_count: int,
        expected_severity_distribution: dict[str, float],
    ) -> float:
        """Score a scan result using precision/recall against ground truth."""
        detected = len(findings)
        if ground_truth_count == 0 and detected == 0:
            return 1.0
        if ground_truth_count == 0:
            # All findings are false positives
            return max(0.0, 1.0 - detected * 0.1)

        recall = min(detected / ground_truth_count, 1.0)
        precision = (
            min(ground_truth_count / detected, 1.0) if detected > 0 else 0.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Severity distribution alignment bonus
        if expected_severity_distribution and findings:
            severity_counts: dict[str, int] = {}
            for f in findings:
                sev = f.get("severity", "info").lower()
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            total = sum(severity_counts.values()) or 1
            actual_dist = {k: v / total for k, v in severity_counts.items()}

            dist_error = 0.0
            all_keys = set(expected_severity_distribution) | set(actual_dist)
            for key in all_keys:
                dist_error += abs(
                    expected_severity_distribution.get(key, 0.0)
                    - actual_dist.get(key, 0.0)
                )
            alignment_bonus = max(0.0, 0.2 * (1.0 - dist_error / 2.0))
        else:
            alignment_bonus = 0.0

        return min(f1 + alignment_bonus, 1.0)

    @staticmethod
    def score_entropy_consistency(ai_assessment: str, actual_quality: int) -> float:
        """Score consistency between AI assessment language and actual quality."""
        text = ai_assessment.lower()

        positive_terms = {"excellent", "strong", "high-quality", "robust", "good"}
        negative_terms = {"poor", "weak", "inadequate", "low", "insufficient", "bad"}

        positive_hits = sum(1 for t in positive_terms if t in text)
        negative_hits = sum(1 for t in negative_terms if t in text)

        sentiment = 0.0
        if positive_hits > negative_hits:
            sentiment = 1.0
        elif negative_hits > positive_hits:
            sentiment = -1.0

        # Map quality score to expected sentiment
        if actual_quality >= 70:
            expected_sentiment = 1.0
        elif actual_quality <= 30:
            expected_sentiment = -1.0
        else:
            expected_sentiment = 0.0

        # Reward agreement, penalise contradiction
        if sentiment == expected_sentiment:
            return 1.0
        if sentiment == 0.0 or expected_sentiment == 0.0:
            return 0.5
        # Direct contradiction (e.g. says "excellent" but quality < 30)
        return -0.5

    @staticmethod
    def score_latency(response_ms: float, complexity: float) -> float:
        """Normalised latency reward: fast responses for simple tasks score higher."""
        # Expected latency scales with complexity
        expected_ms = 500.0 + complexity * 4500.0  # 500ms-5000ms range
        ratio = response_ms / expected_ms if expected_ms > 0 else 1.0

        if ratio <= 1.0:
            return 1.0
        if ratio <= 2.0:
            return 1.0 - (ratio - 1.0) * 0.5  # linear decay 1.0 -> 0.5
        return max(0.0, 0.5 - (ratio - 2.0) * 0.25)

    @staticmethod
    def composite(
        security: float,
        latency: float,
        satisfaction: float,
    ) -> float:
        """Weighted composite reward."""
        return 0.5 * security + 0.3 * latency + 0.2 * satisfaction
