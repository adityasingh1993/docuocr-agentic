from __future__ import annotations

import unittest
from pathlib import Path

from docuocr.extraction.blueprint import (
    DocumentBlueprint,
    FieldSpec,
    NormalizationSettings,
)
from docuocr.extraction.normalization import normalize_value

ROOT = Path(__file__).resolve().parents[1]


class BlueprintAndNormalizationTests(unittest.TestCase):
    def test_blueprint_paths_do_not_treat_sample_container_as_leaf(self) -> None:
        blueprint = DocumentBlueprint.from_yaml(
            ROOT / "config" / "blueprints" / "newborn_screening.yaml"
        )
        self.assertNotIn("data.baby.sample", blueprint.output_paths)
        self.assertIn("data.baby.sample.collection.date", blueprint.output_paths)
        self.assertEqual(len(blueprint.output_paths), len(set(blueprint.output_paths)))

    def test_dates_are_deterministic_from_blueprint_order(self) -> None:
        result = normalize_value(
            "12/08/2024",
            FieldSpec(type="date", aliases=["dob"]),
            NormalizationSettings(date_order="DMY"),
        )
        self.assertEqual(result.value, "2024-08-12")

    def test_ambiguous_date_is_not_guessed_without_policy(self) -> None:
        result = normalize_value(
            "12/08/2024",
            FieldSpec(type="date", aliases=["dob"]),
            NormalizationSettings(date_order="REJECT_AMBIGUOUS"),
        )
        self.assertIsNone(result.value)
        self.assertIn("ambiguous_date", result.codes)

    def test_weight_is_normalized_to_kg(self) -> None:
        result = normalize_value(
            "3250 g",
            FieldSpec(type="weight", aliases=["weight"]),
            NormalizationSettings(weight_unit="kg"),
        )
        self.assertEqual(result.value, "3.25 kg")


if __name__ == "__main__":
    unittest.main()
