from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from docuocr.models import BBox, ControlKind, ControlState, FormControl, OCRSpan

from .quality import _cv2


class FormControlDetector:
    """Detect checkbox/radio candidates and classify their visual state."""

    def __init__(
        self,
        *,
        min_size_fraction: float = 0.006,
        max_size_fraction: float = 0.055,
        empty_ink_max: float = 0.035,
        checked_ink_min: float = 0.070,
    ) -> None:
        self.min_size_fraction = min_size_fraction
        self.max_size_fraction = max_size_fraction
        self.empty_ink_max = empty_ink_max
        self.checked_ink_min = checked_ink_min

    def detect(
        self, image_path: str | Path, *, page: int = 1, attempt: int = 0
    ) -> list[FormControl]:
        cv2 = _cv2()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )
        height, width = gray.shape
        scale = min(width, height)
        min_size = max(8, int(scale * self.min_size_fraction))
        max_size = max(min_size + 1, int(scale * self.max_size_fraction))

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Separate long horizontal/vertical strokes before contouring. This recovers
        # checked boxes whose tick touches the border and distorts the raw contour.
        line_length = max(5, round(min_size * 0.70))
        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)),
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)),
        )
        border_mask = cv2.bitwise_or(horizontal, vertical)
        border_contours, _ = cv2.findContours(
            border_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[tuple[BBox, ControlKind, float]] = []
        for contour, border_only in [
            *((item, False) for item in contours),
            *((item, True) for item in border_contours),
        ]:
            x, y, w, h = cv2.boundingRect(contour)
            if not (min_size <= w <= max_size and min_size <= h <= max_size):
                continue
            aspect = w / float(h)
            if not 0.72 <= aspect <= 1.28:
                continue
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            rectangularity = area / float(w * h)
            if border_only and rectangularity >= 0.40:
                kind = ControlKind.CHECKBOX
                shape_confidence = min(1.0, 0.90 + 0.10 * (1.0 - abs(1.0 - aspect)))
            elif len(polygon) == 4 and rectangularity >= 0.45:
                kind = ControlKind.CHECKBOX
                shape_confidence = min(1.0, 1.0 - abs(1.0 - aspect))
            elif circularity >= 0.62:
                kind = ControlKind.RADIO
                shape_confidence = min(1.0, circularity)
            else:
                continue
            candidates.append(
                (BBox(x1=x, y1=y, x2=x + w, y2=y + h), kind, shape_confidence)
            )

        deduped = self._deduplicate(candidates)
        controls: list[FormControl] = []
        for index, (bbox, kind, shape_confidence) in enumerate(deduped):
            state, state_confidence, ink_ratio = self._classify_state(
                binary, bbox, shape_confidence
            )
            controls.append(
                FormControl(
                    id=f"control:p{page}:{index:04d}",
                    kind=kind,
                    state=state,
                    state_confidence=state_confidence,
                    bbox=bbox,
                    page=page,
                    ink_ratio=ink_ratio,
                    attempt=attempt,
                )
            )
        return controls

    def associate_labels(
        self,
        controls: list[FormControl],
        spans: list[OCRSpan],
        *,
        max_distance_in_box_widths: float = 14.0,
    ) -> list[FormControl]:
        result: list[FormControl] = []
        for control in controls:
            best: tuple[float, OCRSpan] | None = None
            _, cy = control.bbox.center
            for span in spans:
                if span.page != control.page or not span.text.strip():
                    continue
                _, sy = span.bbox.center
                vertical_delta = abs(sy - cy)
                vertical_limit = max(control.bbox.height, span.bbox.height) * 0.85
                if vertical_delta > vertical_limit:
                    continue
                right_distance = span.bbox.x1 - control.bbox.x2
                left_distance = control.bbox.x1 - span.bbox.x2
                if right_distance >= -control.bbox.width * 0.25:
                    horizontal = max(0.0, right_distance)
                    direction_penalty = 0.0
                elif left_distance >= 0:
                    horizontal = left_distance
                    direction_penalty = control.bbox.width * 2.0
                else:
                    continue
                if horizontal > control.bbox.width * max_distance_in_box_widths:
                    continue
                cost = horizontal + vertical_delta * 2.0 + direction_penalty
                if best is None or cost < best[0]:
                    best = (cost, span)
            result.append(
                control.model_copy(
                    update={"associated_label_id": best[1].id if best else None}
                )
            )
        return result

    def _classify_state(
        self, binary: np.ndarray, bbox: BBox, shape_confidence: float
    ) -> tuple[ControlState, float, float]:
        margin_x = max(2, round(bbox.width * 0.24))
        margin_y = max(2, round(bbox.height * 0.24))
        inner = binary[
            bbox.y1 + margin_y : bbox.y2 - margin_y,
            bbox.x1 + margin_x : bbox.x2 - margin_x,
        ]
        if inner.size == 0:
            return ControlState.AMBIGUOUS, 0.0, 0.0
        ink_ratio = float(np.count_nonzero(inner)) / float(inner.size)
        if ink_ratio <= self.empty_ink_max:
            distance = (self.empty_ink_max - ink_ratio) / max(self.empty_ink_max, 1e-6)
            confidence = 0.55 + 0.45 * min(1.0, distance)
            state = ControlState.UNCHECKED
        elif ink_ratio >= self.checked_ink_min:
            distance = (ink_ratio - self.checked_ink_min) / max(
                0.20 - self.checked_ink_min, 1e-6
            )
            confidence = 0.60 + 0.40 * min(1.0, max(0.0, distance))
            state = ControlState.CHECKED
        else:
            midpoint = (self.empty_ink_max + self.checked_ink_min) / 2.0
            uncertainty = 1.0 - min(
                1.0,
                abs(ink_ratio - midpoint) / (self.checked_ink_min - self.empty_ink_max),
            )
            confidence = 0.50 + 0.20 * (1.0 - uncertainty)
            state = ControlState.AMBIGUOUS
        return state, float(np.clip(confidence * shape_confidence, 0.0, 1.0)), ink_ratio

    @staticmethod
    def _deduplicate(
        candidates: list[tuple[BBox, ControlKind, float]],
    ) -> list[tuple[BBox, ControlKind, float]]:
        result: list[tuple[BBox, ControlKind, float]] = []
        for item in sorted(candidates, key=lambda value: value[2], reverse=True):
            bbox = item[0]
            if any(FormControlDetector._iou(bbox, kept[0]) >= 0.65 for kept in result):
                continue
            result.append(item)
        return sorted(result, key=lambda value: (value[0].y1, value[0].x1))

    @staticmethod
    def _iou(left: BBox, right: BBox) -> float:
        x1 = max(left.x1, right.x1)
        y1 = max(left.y1, right.y1)
        x2 = min(left.x2, right.x2)
        y2 = min(left.y2, right.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if not intersection:
            return 0.0
        left_area = left.width * left.height
        right_area = right.width * right.height
        return intersection / float(left_area + right_area - intersection)
