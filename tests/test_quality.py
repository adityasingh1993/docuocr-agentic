from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from docuocr.cv.enhance import ImageEnhancer
from docuocr.cv.quality import (
    ImageQualityAssessor,
    assess_ocr_readiness,
    enrich_quality_report,
    evaluate_enhancement,
)
from docuocr.models import (
    BBox,
    DocumentUnderstanding,
    OCRSpan,
    QualityReport,
    RecoveryAction,
)


class ImageQualityTests(unittest.TestCase):
    def test_sparse_black_text_on_white_form_is_not_glare(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV deployment dependency is not installed")
        image = np.full((900, 1400, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "First name   Amina",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "form.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            report = ImageQualityAssessor().assess(path)
        self.assertEqual(report.glare, 1.0)
        self.assertGreater(report.contrast, 0.90)

    def test_enhancement_requires_ocr_gain_and_preserves_existing_text(self) -> None:
        baseline = [_span("First name Amina", 0.82)]
        improved = [_span("First name Amina", 0.96)]
        quality = _quality_report()
        evaluation = evaluate_enhancement(
            strategy="clahe",
            candidate_path="candidate.png",
            baseline_quality=quality,
            candidate_quality=quality,
            baseline_spans=baseline,
            candidate_spans=improved,
            baseline_control_count=2,
            candidate_control_count=2,
            min_ocr_gain=0.02,
            min_text_retention=0.80,
            min_control_retention=0.75,
            ocr_engine_enabled=True,
        )
        self.assertTrue(evaluation.eligible)
        self.assertGreater(evaluation.ocr_gain, 0.02)

        destructive = evaluate_enhancement(
            strategy="clahe",
            candidate_path="candidate.png",
            baseline_quality=quality,
            candidate_quality=quality,
            baseline_spans=baseline,
            candidate_spans=[_span("unrelated noise", 0.99)],
            baseline_control_count=2,
            candidate_control_count=1,
            min_ocr_gain=0.02,
            min_text_retention=0.80,
            min_control_retention=0.75,
            ocr_engine_enabled=True,
        )
        self.assertFalse(destructive.eligible)
        self.assertIn("baseline_text_not_preserved", destructive.reason)
        self.assertIn("controls_not_preserved", destructive.reason)

    def test_handwriting_legibility_reduces_extraction_readiness(self) -> None:
        quality = _quality_report()
        ocr = assess_ocr_readiness([_span("First name Amina", 0.96)])
        good = enrich_quality_report(
            quality,
            ocr=ocr,
            understanding=DocumentUnderstanding(
                schema_match="match",
                handwriting_present=True,
                handwriting_legibility="good",
            ),
        )
        poor = enrich_quality_report(
            quality,
            ocr=ocr,
            understanding=DocumentUnderstanding(
                schema_match="match",
                handwriting_present=True,
                handwriting_legibility="poor",
            ),
        )
        assert good.extraction_readiness is not None
        assert poor.extraction_readiness is not None
        self.assertLess(poor.extraction_readiness, good.extraction_readiness)
        self.assertEqual(poor.handwriting_quality, 0.42)

    def test_full_page_planner_never_uses_binarization(self) -> None:
        quality = _quality_report().model_copy(
            update={
                "recommendations": [
                    "adaptive_binarize",
                    "illumination_normalization",
                ]
            }
        )
        variants = ImageEnhancer().plan_variants(quality, max_variants=3)
        flattened = {action for variant in variants for action in variant}
        self.assertIn(RecoveryAction.ILLUMINATION_NORMALIZATION, flattened)
        self.assertNotIn(RecoveryAction.ADAPTIVE_BINARIZE, flattened)

    def test_deskew_uses_the_estimated_correction_direction(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV deployment dependency is not installed")
        image = np.full((800, 1200, 3), 255, dtype=np.uint8)
        for y in range(100, 750, 80):
            cv2.line(image, (100, y), (1100, y), (0, 0, 0), 3)
        rotated = cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D((600, 400), 5.0, 1.0),
            (1200, 800),
            borderValue=(255, 255, 255),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rotated.png"
            output = Path(directory) / "deskewed.png"
            self.assertTrue(cv2.imwrite(str(source), rotated))
            before = ImageQualityAssessor().assess(source)
            ImageEnhancer().apply(
                source,
                output,
                [RecoveryAction.DESKEW],
                attempt=1,
                skew_degrees=before.estimated_skew_degrees,
            )
            after = ImageQualityAssessor().assess(output)
        self.assertLess(
            abs(after.estimated_skew_degrees),
            abs(before.estimated_skew_degrees),
        )


def _span(text: str, confidence: float) -> OCRSpan:
    return OCRSpan(
        id=f"ocr:{confidence}",
        text=text,
        confidence=confidence,
        bbox=BBox(x1=1, y1=1, x2=100, y2=30),
    )


def _quality_report() -> QualityReport:
    return QualityReport(
        overall=0.92,
        resolution=0.95,
        sharpness=0.90,
        contrast=0.90,
        illumination=0.92,
        glare=1.0,
        skew=1.0,
        width=1200,
        height=900,
    )


if __name__ == "__main__":
    unittest.main()
