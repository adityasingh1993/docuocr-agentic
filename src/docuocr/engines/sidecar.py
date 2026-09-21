from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docuocr.models import FormControl, LayoutBlock, OCRSpan


class SidecarBundle:
    """Deterministic evidence adapter for tests, integration, and engine isolation."""

    model_id = "sidecar-evidence-v1"
    supports_region_ocr = False

    def __init__(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as stream:
            self.payload: dict[str, Any] = json.load(stream)

    def fork(self) -> SidecarBundle:
        return self

    def extract(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "ocr",
    ) -> list[OCRSpan]:
        result: list[OCRSpan] = []
        for index, payload in enumerate(self.payload.get("ocrSpans", [])):
            item = OCRSpan.model_validate(payload)
            result.append(
                item.model_copy(
                    update={
                        "id": f"{id_prefix}:p{page}:{index:04d}",
                        "attempt": attempt,
                    }
                )
            )
        return result

    def parse(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "layout",
    ) -> list[LayoutBlock]:
        result: list[LayoutBlock] = []
        for index, payload in enumerate(self.payload.get("layoutBlocks", [])):
            item = LayoutBlock.model_validate(payload)
            result.append(
                item.model_copy(update={"id": f"{id_prefix}:p{page}:{index:04d}"})
            )
        return result

    def detect(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "control",
    ) -> list[FormControl]:
        result: list[FormControl] = []
        for index, payload in enumerate(self.payload.get("controls", [])):
            item = FormControl.model_validate(payload)
            result.append(
                item.model_copy(
                    update={
                        "id": f"{id_prefix}:p{page}:{index:04d}",
                        "attempt": attempt,
                    }
                )
            )
        return result


class NullTextEngine:
    model_id = "disabled"
    supports_region_ocr = False

    def fork(self) -> NullTextEngine:
        return self

    def extract(self, image_path: str | Path, **_: Any) -> list[OCRSpan]:
        return []


class NullLayoutEngine:
    model_id = "disabled"

    def parse(self, image_path: str | Path, **_: Any) -> list[LayoutBlock]:
        return []
