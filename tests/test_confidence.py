from __future__ import annotations

import unittest
from pathlib import Path

from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.confidence import ConfidenceScorer
from docuocr.models import FieldCandidate

ROOT = Path(__file__).resolve().parents[1]


def candidate(value: str, sources: list[str], source: str = "rule") -> FieldCandidate:
    return FieldCandidate(
        path="data.baby.firstName",
        raw_value=value,
        normalized_value=value,
        evidence_ids=[f"ocr:{source}"],
        source=source,
        support_sources=sources,
        recognition_confidence=0.99,
        association_confidence=0.99,
    )


class ConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = DocumentBlueprint.from_yaml(
            ROOT / "config" / "blueprints" / "newborn_screening.yaml"
        )
        self.scorer = ConfidenceScorer(0.90)

    def test_single_source_is_capped_below_auto_accept(self) -> None:
        decision = self.scorer.decide(
            [candidate("Amina", ["paddleocr"])],
            self.blueprint,
            image_quality=0.99,
            attempt=0,
            max_retries=2,
        )["data.baby.firstName"]
        self.assertEqual(decision.score, 0.89)
        self.assertEqual(decision.disposition, "retry")
        self.assertIn("single_source_cap", decision.reasons)

    def test_independent_agreement_can_auto_accept(self) -> None:
        decision = self.scorer.decide(
            [
                candidate("Amina", ["paddleocr"], "rule"),
                candidate("Amina", ["paddleocr", "local_vlm_visual"], "vlm_visual"),
            ],
            self.blueprint,
            image_quality=0.99,
            attempt=1,
            max_retries=2,
        )["data.baby.firstName"]
        self.assertGreaterEqual(decision.score, 0.90)
        self.assertEqual(decision.disposition, "accepted")

    def test_conflict_is_capped(self) -> None:
        decision = self.scorer.decide(
            [
                candidate("Amina", ["paddleocr"], "first"),
                candidate("Amira", ["local_vlm_visual"], "second"),
            ],
            self.blueprint,
            image_quality=0.99,
            attempt=2,
            max_retries=2,
        )["data.baby.firstName"]
        self.assertLessEqual(decision.score, 0.59)
        self.assertEqual(decision.disposition, "review")


if __name__ == "__main__":
    unittest.main()
