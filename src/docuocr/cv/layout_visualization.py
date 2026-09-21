from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from docuocr.models import LayoutBlock, LayoutVisualizationRecord
from docuocr.trace import sha256_file

from .quality import _cv2


class LayoutVisualizer:
    """Render deterministic, privacy-conscious layout overlays for audit."""

    _PALETTE = (
        (180, 119, 31),
        (14, 127, 255),
        (44, 160, 44),
        (40, 39, 214),
        (189, 103, 148),
        (75, 86, 140),
        (194, 119, 227),
        (127, 127, 127),
        (34, 189, 188),
        (207, 190, 23),
    )

    def render(
        self,
        image_path: str | Path,
        blocks: list[LayoutBlock],
        output_path: str | Path,
        *,
        attempt: int = 0,
    ) -> LayoutVisualizationRecord:
        cv2 = _cv2()
        source = Path(image_path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {source}")

        annotated = image.copy()
        height, width = annotated.shape[:2]
        short_side = min(width, height)
        line_thickness = max(1, round(short_side / 500))
        font_scale = max(0.4, min(1.0, short_side / 1000.0))
        font_thickness = max(1, line_thickness)
        rendered_count = 0
        label_counts: Counter[str] = Counter()

        for block in sorted(
            blocks,
            key=lambda item: (
                item.page,
                item.order if item.order is not None else 1_000_000,
                item.bbox.y1,
                item.bbox.x1,
                item.id,
            ),
        ):
            label_counts[block.label] += 1
            x1 = min(max(block.bbox.x1, 0), width - 1)
            y1 = min(max(block.bbox.y1, 0), height - 1)
            x2 = min(max(block.bbox.x2, 0), width - 1)
            y2 = min(max(block.bbox.y2, 0), height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            color = self._color_for(block.label)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                line_thickness,
                cv2.LINE_AA,
            )
            label = self._safe_label(block.label, block.confidence)
            self._draw_label(
                annotated,
                label,
                x1=x1,
                y1=y1,
                color=color,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )
            rendered_count += 1

        if not blocks:
            self._draw_empty_notice(
                annotated,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f"Failed to write layout visualization: {destination}")

        return LayoutVisualizationRecord(
            source_path=str(source),
            output_path=str(destination),
            source_sha256=sha256_file(source),
            output_sha256=sha256_file(destination),
            block_count=len(blocks),
            rendered_block_count=rendered_count,
            label_counts=dict(sorted(label_counts.items())),
            attempt=attempt,
        )

    @classmethod
    def _color_for(cls, label: str) -> tuple[int, int, int]:
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        return cls._PALETTE[digest[0] % len(cls._PALETTE)]

    @staticmethod
    def _safe_label(label: str, confidence: float) -> str:
        printable = label.encode("ascii", errors="replace").decode("ascii")
        return f"{printable[:48]} {confidence:.2f}"

    @staticmethod
    def _draw_label(
        image: object,
        label: str,
        *,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        font_scale: float,
        font_thickness: int,
    ) -> None:
        cv2 = _cv2()
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        image_height, image_width = image.shape[:2]
        pad = max(2, font_thickness + 1)
        box_width = min(text_width + 2 * pad, image_width - x1)
        text_y = y1 - pad
        if text_y - text_height - baseline - pad < 0:
            text_y = min(image_height - baseline - pad, y1 + text_height + 2 * pad)
        box_top = max(0, text_y - text_height - pad)
        box_bottom = min(image_height - 1, text_y + baseline + pad)
        box_right = min(image_width - 1, x1 + box_width)
        cv2.rectangle(image, (x1, box_top), (box_right, box_bottom), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + pad, max(text_height, text_y)),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_empty_notice(
        image: object, *, font_scale: float, font_thickness: int
    ) -> None:
        cv2 = _cv2()
        label = "No layout blocks detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        pad = max(4, font_thickness * 2)
        right = min(image.shape[1] - 1, text_width + 2 * pad)
        bottom = min(image.shape[0] - 1, text_height + baseline + 2 * pad)
        cv2.rectangle(image, (0, 0), (right, bottom), (40, 39, 214), -1)
        cv2.putText(
            image,
            label,
            (pad, text_height + pad),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
