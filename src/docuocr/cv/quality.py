from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from docuocr.models import (
    DocumentUnderstanding,
    EnhancementEvaluation,
    HandwritingLegibility,
    OCRReadinessReport,
    OCRSpan,
    QualityReport,
    VisualIssueSeverity,
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on deployment extra
        raise RuntimeError(
            "OpenCV is required; install opencv-python-headless"
        ) from exc
    return cv2


class ImageQualityAssessor:
    """Deterministic form-image quality features, all normalized to [0, 1]."""

    def __init__(self, target_short_side: int = 1200) -> None:
        self.target_short_side = target_short_side

    def assess(self, image_path: str | Path) -> QualityReport:
        cv2 = _cv2()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resolution = min(1.0, min(width, height) / float(self.target_short_side))

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = 1.0 - math.exp(-laplacian_variance / 220.0)

        p05, p95 = np.percentile(gray, [5, 95])
        ink = gray[gray < p95 - 12]
        if ink.size >= gray.size * 0.0005:
            foreground = float(np.percentile(ink, 20))
            contrast_span = float(p95) - foreground
        else:
            contrast_span = float(p95 - p05)
        contrast = min(1.0, max(0.0, contrast_span / 170.0))

        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        illumination_spread = float(np.std(small.astype(np.float32)))
        illumination = max(0.0, 1.0 - max(0.0, illumination_spread - 42.0) / 85.0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        glare_mask = (hsv[:, :, 2] >= 248) & (hsv[:, :, 1] <= 28)
        glare_fraction = float(np.mean(glare_mask))
        dark_fraction = float(np.mean(gray < 235))
        # A white form background is not glare. Treat clipped, low-saturation pixels
        # as specular risk only when the page also contains a substantial darker area.
        if glare_fraction >= 0.40 and dark_fraction < 0.25:
            glare = 1.0
        else:
            glare = max(0.0, 1.0 - min(1.0, glare_fraction / 0.06))

        skew_degrees = self._estimate_skew(gray)
        skew = max(0.0, 1.0 - min(1.0, abs(skew_degrees) / 8.0))

        components = {
            "resolution": (resolution, 0.16),
            "sharpness": (sharpness, 0.24),
            "contrast": (contrast, 0.22),
            "illumination": (illumination, 0.14),
            "glare": (glare, 0.12),
            "skew": (skew, 0.12),
        }
        overall = math.exp(
            sum(
                weight * math.log(max(value, 1e-6))
                for value, weight in components.values()
            )
        )

        issues: list[str] = []
        recommendations: list[str] = []
        if resolution < 0.90:
            issues.append("small_text_risk")
            recommendations.append("upscale")
        if sharpness < 0.80:
            issues.append("blur")
            recommendations.append("denoise_sharpen")
        if contrast < 0.80:
            issues.append("low_contrast")
            recommendations.append("clahe")
        if illumination < 0.80:
            issues.append("uneven_illumination")
            recommendations.append("illumination_normalization")
        if glare < 0.75:
            issues.append("glare_or_saturation")
        if skew < 0.85:
            issues.append("skew")
            recommendations.append("deskew")

        return QualityReport(
            overall=float(np.clip(overall, 0.0, 1.0)),
            resolution=resolution,
            sharpness=sharpness,
            contrast=contrast,
            illumination=illumination,
            glare=glare,
            skew=skew,
            width=width,
            height=height,
            estimated_skew_degrees=skew_degrees,
            issues=issues,
            recommendations=list(dict.fromkeys(recommendations)),
        )

    @staticmethod
    def _estimate_skew(gray: np.ndarray) -> float:
        cv2 = _cv2()
        edges = cv2.Canny(gray, 60, 180, apertureSize=3)
        min_length = max(30, int(min(gray.shape[:2]) * 0.15))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=80,
            minLineLength=min_length,
            maxLineGap=20,
        )
        if lines is None:
            return 0.0
        angles: list[float] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            while angle <= -45:
                angle += 90
            while angle > 45:
                angle -= 90
            if abs(angle) <= 15:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0


def assess_ocr_readiness(spans: list[OCRSpan]) -> OCRReadinessReport:
    """Score observable OCR output without treating it as calibrated accuracy."""

    weighted_confidence = 0.0
    high_confidence_characters = 0
    character_count = 0
    usable_spans = 0
    for span in spans:
        count = sum(not character.isspace() for character in span.text.strip())
        if count == 0:
            continue
        usable_spans += 1
        character_count += count
        weighted_confidence += span.confidence * count
        if span.confidence >= 0.80:
            high_confidence_characters += count

    if character_count == 0:
        return OCRReadinessReport(
            score=0.0,
            span_count=0,
            character_count=0,
            mean_confidence=0.0,
            high_confidence_fraction=0.0,
        )

    mean_confidence = weighted_confidence / character_count
    high_fraction = high_confidence_characters / character_count
    coverage = 1.0 - math.exp(-character_count / 120.0)
    score = 0.65 * mean_confidence + 0.20 * high_fraction + 0.15 * coverage
    return OCRReadinessReport(
        score=float(np.clip(score, 0.0, 1.0)),
        span_count=usable_spans,
        character_count=character_count,
        mean_confidence=float(np.clip(mean_confidence, 0.0, 1.0)),
        high_confidence_fraction=float(np.clip(high_fraction, 0.0, 1.0)),
    )


def enrich_quality_report(
    report: QualityReport,
    *,
    ocr: OCRReadinessReport | None,
    understanding: DocumentUnderstanding | None,
) -> QualityReport:
    """Fuse technical, OCR, and semantic signals into an uncalibrated readiness score."""

    handwriting_score: float | None = None
    semantic_score: float | None = None
    if understanding is not None:
        handwriting_score = {
            HandwritingLegibility.NOT_PRESENT: 1.0,
            HandwritingLegibility.GOOD: 0.95,
            HandwritingLegibility.FAIR: 0.72,
            HandwritingLegibility.POOR: 0.42,
            HandwritingLegibility.UNREADABLE: 0.10,
        }[understanding.handwriting_legibility]
        issue_score = 1.0
        for issue in understanding.quality_issues:
            issue_score = min(
                issue_score,
                {
                    VisualIssueSeverity.MILD: 0.88,
                    VisualIssueSeverity.MODERATE: 0.65,
                    VisualIssueSeverity.SEVERE: 0.35,
                }[issue.severity],
            )
        semantic_score = min(issue_score, handwriting_score)

    components: list[tuple[float, float]] = [(report.overall, 0.55)]
    if ocr is not None:
        components.append((ocr.score, 0.30))
    if semantic_score is not None:
        components.append((semantic_score, 0.15))
    weight_total = sum(weight for _, weight in components)
    readiness = math.exp(
        sum(
            (weight / weight_total) * math.log(max(value, 1e-6))
            for value, weight in components
        )
    )
    return report.model_copy(
        update={
            "ocr_readiness": ocr.score if ocr is not None else None,
            "semantic_quality": semantic_score,
            "handwriting_quality": handwriting_score,
            "extraction_readiness": float(np.clip(readiness, 0.0, 1.0)),
            "calibrated": False,
        }
    )


def high_confidence_text_retention(
    baseline: list[OCRSpan], candidate: list[OCRSpan]
) -> float:
    """Fraction of stable baseline tokens still present after enhancement."""

    baseline_tokens = _tokens(
        span.text for span in baseline if span.confidence >= 0.80
    )
    if not baseline_tokens:
        return 1.0
    candidate_tokens = _tokens(span.text for span in candidate)
    return len(baseline_tokens & candidate_tokens) / len(baseline_tokens)


def control_count_retention(baseline_count: int, candidate_count: int) -> float:
    if baseline_count <= 0:
        return 1.0
    return min(1.0, candidate_count / baseline_count)


def evaluate_enhancement(
    *,
    strategy: str,
    candidate_path: str,
    baseline_quality: QualityReport,
    candidate_quality: QualityReport,
    baseline_spans: list[OCRSpan],
    candidate_spans: list[OCRSpan],
    baseline_control_count: int,
    candidate_control_count: int,
    min_ocr_gain: float,
    min_text_retention: float,
    min_control_retention: float,
    ocr_engine_enabled: bool,
) -> EnhancementEvaluation:
    baseline_ocr = assess_ocr_readiness(baseline_spans)
    candidate_ocr = assess_ocr_readiness(candidate_spans)
    ocr_gain = candidate_ocr.score - baseline_ocr.score
    text_retention = high_confidence_text_retention(
        baseline_spans, candidate_spans
    )
    control_retention = control_count_retention(
        baseline_control_count, candidate_control_count
    )
    rejection_codes: list[str] = []
    if not ocr_engine_enabled:
        rejection_codes.append("ocr_engine_disabled")
    if ocr_gain < min_ocr_gain:
        rejection_codes.append("insufficient_ocr_gain")
    if text_retention < min_text_retention:
        rejection_codes.append("baseline_text_not_preserved")
    if control_retention < min_control_retention:
        rejection_codes.append("controls_not_preserved")
    if candidate_quality.overall < baseline_quality.overall - 0.10:
        rejection_codes.append("technical_quality_regressed")
    eligible = not rejection_codes
    return EnhancementEvaluation(
        strategy=strategy,
        candidate_path=candidate_path,
        technical_before=baseline_quality.overall,
        technical_after=candidate_quality.overall,
        ocr_before=baseline_ocr,
        ocr_after=candidate_ocr,
        ocr_gain=ocr_gain,
        text_retention=text_retention,
        control_retention=control_retention,
        eligible=eligible,
        reason="eligible" if eligible else "+".join(rejection_codes),
    )


def _tokens(values: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(
            token.casefold()
            for token in re.findall(r"[^\W_]+", str(value), flags=re.UNICODE)
            if token
        )
    return result
