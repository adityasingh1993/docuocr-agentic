from __future__ import annotations

from pathlib import Path
from typing import Any

from docuocr.config import PaddleSettings
from docuocr.models import BBox, LayoutBlock, OCRSpan


class PaddleTextEngine:
    """Lazy adapter for PaddleOCR 3.x general OCR results."""

    model_id = "paddleocr-general"

    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError(
                    "Install the paddle extra and a compatible PaddlePaddle build"
                ) from exc
            kwargs: dict[str, Any] = {
                "lang": self.settings.language,
                "device": self.settings.device,
                "engine": self.settings.engine,
                "text_det_thresh": self.settings.text_detection_threshold,
                "text_det_box_thresh": self.settings.text_box_threshold,
                "text_rec_score_thresh": self.settings.text_recognition_threshold,
            }
            if self.settings.text_detection_model_dir:
                kwargs["text_detection_model_dir"] = (
                    self.settings.text_detection_model_dir
                )
            if self.settings.text_recognition_model_dir:
                kwargs["text_recognition_model_dir"] = (
                    self.settings.text_recognition_model_dir
                )
            self._pipeline = PaddleOCR(**kwargs)
        return self._pipeline

    def extract(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "ocr",
    ) -> list[OCRSpan]:
        results = self._get_pipeline().predict(str(image_path))
        spans: list[OCRSpan] = []
        for result in results:
            payload = _result_payload(result)
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            boxes = list(payload.get("rec_boxes") or payload.get("dt_polys") or [])
            for text, score, box in zip(texts, scores, boxes, strict=False):
                if not str(text).strip():
                    continue
                spans.append(
                    OCRSpan(
                        id=f"{id_prefix}:p{page}:{len(spans):04d}",
                        text=str(text),
                        confidence=float(score),
                        bbox=_bbox(box),
                        page=page,
                        source="paddleocr",
                        attempt=attempt,
                    )
                )
        return spans


class PaddleLayoutEngine:
    """Lazy adapter for PaddleOCR-VL/PaddleOCR 3.x document parsing results."""

    model_id = "paddleocr-vl-layout"

    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError(
                    "PaddleOCR-VL is unavailable; install paddleocr[doc-parser]"
                ) from exc
            kwargs: dict[str, Any] = {
                "device": self.settings.device,
                "engine": self.settings.engine,
                "layout_threshold": self.settings.layout_threshold,
            }
            if self.settings.layout_model_dir:
                kwargs["layout_detection_model_dir"] = self.settings.layout_model_dir
            if self.settings.vl_rec_model_dir:
                kwargs["vl_rec_model_dir"] = self.settings.vl_rec_model_dir
            self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def parse(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "layout",
    ) -> list[LayoutBlock]:
        results = self._get_pipeline().predict(str(image_path))
        blocks: list[LayoutBlock] = []
        for result in results:
            payload = _result_payload(result)
            for item in payload.get("parsing_res_list") or []:
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif not isinstance(item, dict):
                    item = vars(item)
                box = item.get("block_bbox") or item.get("bbox")
                if not box:
                    continue
                blocks.append(
                    LayoutBlock(
                        id=f"{id_prefix}:p{page}:{len(blocks):04d}",
                        label=str(item.get("block_label") or "unknown"),
                        content=str(item.get("block_content") or ""),
                        confidence=float(
                            item.get("score") or item.get("confidence") or 0.75
                        ),
                        bbox=_bbox(box),
                        page=page,
                        order=_optional_int(item.get("block_order")),
                        source="paddleocr-vl",
                    )
                )
        return blocks


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported PaddleOCR result: {type(payload)!r}")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _bbox(value: Any) -> BBox:
    if hasattr(value, "tolist"):
        value = value.tolist()
    points = list(value)
    if len(points) == 4 and all(not isinstance(item, (list, tuple)) for item in points):
        x1, y1, x2, y2 = (round(float(item)) for item in points)
    else:
        flattened: list[tuple[float, float]] = []
        for point in points:
            flattened.append((float(point[0]), float(point[1])))
        x1 = round(min(item[0] for item in flattened))
        y1 = round(min(item[1] for item in flattened))
        x2 = round(max(item[0] for item in flattened))
        y2 = round(max(item[1] for item in flattened))
    return BBox(x1=max(0, x1), y1=max(0, y1), x2=max(x1 + 1, x2), y2=max(y1 + 1, y2))


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
