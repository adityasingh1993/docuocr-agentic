from __future__ import annotations

import unittest
from pathlib import Path

from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.grounding import (
    GroundedProposal,
    GroundingError,
    GroundingVerifier,
)
from docuocr.models import BBox, EvidenceKind, EvidenceRecord, OCRSpan

ROOT = Path(__file__).resolve().parents[1]


class GroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = DocumentBlueprint.from_yaml(
            ROOT / "config" / "blueprints" / "newborn_screening.yaml"
        )
        self.span = OCRSpan(
            id="ocr:p1:0001",
            text="Amina",
            confidence=0.98,
            bbox=BBox(x1=10, y1=10, x2=100, y2=35),
        )
        self.evidence = EvidenceRecord(
            id=self.span.id,
            kind=EvidenceKind.OCR,
            bbox=self.span.bbox,
            confidence=0.98,
            source="paddleocr",
        )

    def test_supported_value_is_accepted(self) -> None:
        result = GroundingVerifier().verify(
            GroundedProposal(
                path="data.baby.firstName", rawValue="Amina", evidenceIds=[self.span.id]
            ),
            self.blueprint,
            [self.evidence],
            [self.span],
            [],
            [],
            attempt=0,
        )
        self.assertEqual(result.normalized_value, "Amina")

    def test_hallucinated_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(GroundingError, "not_supported"):
            GroundingVerifier().verify(
                GroundedProposal(
                    path="data.baby.firstName",
                    rawValue="Fatima",
                    evidenceIds=[self.span.id],
                ),
                self.blueprint,
                [self.evidence],
                [self.span],
                [],
                [],
                attempt=0,
            )


if __name__ == "__main__":
    unittest.main()
