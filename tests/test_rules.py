from __future__ import annotations

import unittest
from pathlib import Path

from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.rules import map_rule_candidates
from docuocr.models import BBox, ControlKind, ControlState, FormControl, OCRSpan

ROOT = Path(__file__).resolve().parents[1]


class RuleMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = DocumentBlueprint.from_yaml(
            ROOT / "config" / "blueprints" / "newborn_screening.yaml"
        )

    def test_label_geometry_and_checkbox_are_separate_paths(self) -> None:
        spans = [
            OCRSpan(
                id="ocr:p1:0000",
                text="First name",
                confidence=0.99,
                bbox=BBox(x1=10, y1=10, x2=110, y2=35),
            ),
            OCRSpan(
                id="ocr:p1:0001",
                text="Amina",
                confidence=0.98,
                bbox=BBox(x1=130, y1=10, x2=210, y2=35),
            ),
            OCRSpan(
                id="ocr:p1:0002",
                text="Female",
                confidence=0.99,
                bbox=BBox(x1=55, y1=55, x2=130, y2=80),
            ),
        ]
        controls = [
            FormControl(
                id="control:p1:0000",
                kind=ControlKind.CHECKBOX,
                state=ControlState.CHECKED,
                state_confidence=0.98,
                bbox=BBox(x1=10, y1=55, x2=35, y2=80),
            )
        ]
        candidates = map_rule_candidates(self.blueprint, spans, controls)
        by_path = {item.path: item for item in candidates}
        self.assertEqual(by_path["data.baby.firstName"].normalized_value, "Amina")
        self.assertIs(by_path["data.baby.female"].normalized_value, True)
        self.assertEqual(by_path["data.baby.female"].support_sources, ["opencv"])


if __name__ == "__main__":
    unittest.main()
