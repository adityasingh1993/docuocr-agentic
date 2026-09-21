from __future__ import annotations

import unittest
from pathlib import Path

from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.rules import map_rule_candidates
from docuocr.models import (
    BBox,
    ControlKind,
    ControlState,
    FieldCandidate,
    FormControl,
    OCRSpan,
)
from docuocr.workflow.nodes import _reconcile_visual_controls, _vlm_target_paths

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

    def test_portuguese_printed_label_maps_without_changing_entered_value(self) -> None:
        spans = [
            OCRSpan(
                id="ocr:p1:label",
                text="PESO",
                confidence=0.99,
                bbox=BBox(x1=10, y1=10, x2=80, y2=35),
            ),
            OCRSpan(
                id="ocr:p1:value",
                text="3735",
                confidence=0.98,
                bbox=BBox(x1=100, y1=10, x2=170, y2=35),
            ),
        ]
        candidates = map_rule_candidates(self.blueprint, spans, [])
        weight = next(
            item for item in candidates if item.path == "data.baby.birthWeight"
        )
        self.assertEqual(weight.raw_value, "3735")
        self.assertEqual(weight.normalized_value, "3.735 kg")

    def test_group_label_splits_full_name_into_configured_literal_parts(self) -> None:
        spans = [
            OCRSpan(
                id="ocr:p1:label",
                text="RN:",
                confidence=0.99,
                bbox=BBox(x1=10, y1=10, x2=55, y2=35),
            ),
            OCRSpan(
                id="ocr:p1:value",
                text="Maria Joao da Silva",
                confidence=0.97,
                bbox=BBox(x1=70, y1=10, x2=260, y2=35),
            ),
        ]

        candidates = map_rule_candidates(self.blueprint, spans, [])
        by_path = {
            item.path: item
            for item in candidates
            if item.source == "rule_group_right_of_label"
        }

        self.assertEqual(by_path["data.baby.firstName"].raw_value, "Maria")
        self.assertEqual(
            by_path["data.baby.lastName"].raw_value, "Joao da Silva"
        )
        self.assertEqual(
            by_path["data.baby.firstName"].evidence_ids,
            ["ocr:p1:label", "ocr:p1:value"],
        )

    def test_grounded_visual_control_overrides_conflicting_contour_guess(self) -> None:
        opencv = FieldCandidate(
            path="data.baby.male",
            raw_value=True,
            normalized_value=True,
            evidence_ids=["control:wrong"],
            source="opencv_control",
            support_sources=["opencv"],
            recognition_confidence=0.95,
            association_confidence=0.95,
        )
        visual = FieldCandidate(
            path="data.baby.male",
            raw_value=False,
            normalized_value=False,
            evidence_ids=["ocr:sex", "image:page"],
            source="vlm_visual",
            support_sources=["paddleocr", "local_vlm_visual"],
            recognition_confidence=0.98,
            association_confidence=0.92,
        )
        reconciled, warnings = _reconcile_visual_controls(
            [opencv, visual], self.blueprint.checkbox_paths
        )
        self.assertEqual(reconciled, [visual])
        self.assertEqual(warnings, ["visual_control_overrode_opencv:data.baby.male:1"])

    def test_vlm_rechecks_single_source_and_control_candidates(self) -> None:
        single_source = FieldCandidate(
            path="data.baby.birthWeight",
            raw_value="3735",
            normalized_value="3.735 kg",
            evidence_ids=["ocr:weight"],
            source="rule_right_of_label",
            support_sources=["paddleocr"],
        )
        independently_supported = FieldCandidate(
            path="data.baby.dateOfBirth",
            raw_value="06/08/2024",
            normalized_value="2024-08-06",
            evidence_ids=["ocr:dob", "layout:dob"],
            source="rule_right_of_label",
            support_sources=["paddleocr", "paddleocr-vl"],
        )
        paths = _vlm_target_paths(
            self.blueprint, [single_source, independently_supported]
        )
        self.assertIn("data.baby.birthWeight", paths)
        self.assertNotIn("data.baby.dateOfBirth", paths)
        self.assertIn("data.baby.male", paths)
        self.assertIn("data.baby.female", paths)


if __name__ == "__main__":
    unittest.main()
