from __future__ import annotations

from pathlib import Path
from typing import Protocol

from docuocr.models import FormControl, LayoutBlock, OCRSpan


class TextEngine(Protocol):
    model_id: str

    def extract(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "ocr",
    ) -> list[OCRSpan]: ...


class LayoutEngine(Protocol):
    model_id: str

    def parse(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "layout",
    ) -> list[LayoutBlock]: ...


class ControlEngine(Protocol):
    model_id: str

    def detect(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "control",
    ) -> list[FormControl]: ...
