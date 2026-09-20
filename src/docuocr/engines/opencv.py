from __future__ import annotations

from pathlib import Path

from docuocr.cv.controls import FormControlDetector
from docuocr.models import FormControl


class OpenCVControlEngine:
    model_id = "opencv-contour-fill-v1"

    def __init__(self, detector: FormControlDetector | None = None) -> None:
        self.detector = detector or FormControlDetector()

    def detect(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "control",
    ) -> list[FormControl]:
        controls = self.detector.detect(image_path, page=page, attempt=attempt)
        return [
            item.model_copy(update={"id": f"{id_prefix}:p{page}:{index:04d}"})
            for index, item in enumerate(controls)
        ]
