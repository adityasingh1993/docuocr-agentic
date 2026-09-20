from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from docuocr.config import PaddleSettings
from docuocr.models import BBox, LayoutBlock, OCRSpan


_ENGINE_INSTALL_HINTS = {
    "transformers": 'Install the Transformers backend with `python -m pip install -e ".[paddle-transformers]"`.',
    "onnxruntime": 'Install the ONNX Runtime backend with `python -m pip install -e ".[paddle-onnx]"`.',
    "paddle": 'Install the PaddlePaddle backend with `python -m pip install -e ".[paddle]"`.',
    "paddle_static": 'Install the PaddlePaddle backend with `python -m pip install -e ".[paddle]"`.',
    "paddle_dynamic": 'Install the PaddlePaddle backend with `python -m pip install -e ".[paddle]"`.',
}


def _load_paddle_symbol(name: str, engine: str) -> Any:
    try:
        module = importlib.import_module("paddleocr")
        return getattr(module, name)
    except Exception as exc:  # pragma: no cover - depends on deployment binaries
        hint = _ENGINE_INSTALL_HINTS[engine]
        detail = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"Unable to load paddleocr.{name} for engine {engine!r}. "
            f"Original error: {detail}. {hint} Use a clean environment containing "
            "only one inference backend; on Apple Silicon, verify that "
            "`platform.machine()` returns `arm64`."
        ) from exc


def _raise_model_format_error(
    exc: ValueError,
    *,
    component: str,
    engine: str,
    model_directories: list[str | None],
) -> None:
    if "No valid model files were found" not in str(exc):
        return
    configured = ", ".join(path for path in model_directories if path) or "automatic"
    raise RuntimeError(
        f"{component} model files are incompatible with engine {engine!r}. "
        f"Configured model directories: {configured}. Use a Paddle inference model "
        "with engine 'paddle', a Hugging Face/safetensors model with engine "
        "'transformers', or omit local directories in online mode to let PaddleOCR "
        "resolve compatible models."
    ) from exc


def _raise_if_legacy_engine_argument(exc: TypeError) -> None:
    if "engine" in str(exc) and "unexpected keyword" in str(exc):
        raise RuntimeError(
            "This configuration requires PaddleOCR >=3.5 because it uses the "
            "`engine` argument. Reinstall the selected project extra."
        ) from exc


class PaddleTextEngine:
    """Lazy adapter for PaddleOCR 3.x general OCR results."""

    model_id = "paddleocr-general"

    def __init__(self, settings: PaddleSettings) -> None:
        self.settings = settings
        self.engine = settings.resolved_text_engine
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            PaddleOCR = _load_paddle_symbol("PaddleOCR", self.engine)
            kwargs: dict[str, Any] = {
                "lang": self.settings.language,
                "device": self.settings.device,
                "engine": self.engine,
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
            try:
                self._pipeline = PaddleOCR(**kwargs)
            except TypeError as exc:
                _raise_if_legacy_engine_argument(exc)
                raise
            except ValueError as exc:
                _raise_model_format_error(
                    exc,
                    component="General OCR",
                    engine=self.engine,
                    model_directories=[
                        self.settings.text_detection_model_dir,
                        self.settings.text_recognition_model_dir,
                    ],
                )
                raise
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
        self.engine = settings.resolved_layout_engine
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            PaddleOCRVL = _load_paddle_symbol("PaddleOCRVL", self.engine)
            kwargs: dict[str, Any] = {
                "device": self.settings.device,
                "engine": self.engine,
                "layout_threshold": self.settings.layout_threshold,
            }
            if self.settings.layout_model_dir:
                kwargs["layout_detection_model_dir"] = self.settings.layout_model_dir
            if self.settings.vl_rec_model_dir:
                kwargs["vl_rec_model_dir"] = self.settings.vl_rec_model_dir
            try:
                self._pipeline = PaddleOCRVL(**kwargs)
            except TypeError as exc:
                _raise_if_legacy_engine_argument(exc)
                raise
            except ValueError as exc:
                _raise_model_format_error(
                    exc,
                    component="Layout OCR",
                    engine=self.engine,
                    model_directories=[
                        self.settings.layout_model_dir,
                        self.settings.vl_rec_model_dir,
                    ],
                )
                raise
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
