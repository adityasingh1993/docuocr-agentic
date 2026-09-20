from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from docuocr.models import QualityReport


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
            recommendations.append("adaptive_binarize")
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
