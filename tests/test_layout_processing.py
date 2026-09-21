from __future__ import annotations

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from docuocr.config import AssociationSettings
from docuocr.extraction.layout_processing import LayoutBlockProcessor
from docuocr.models import BBox, LayoutBlock, OCRSpan


class _ParallelRetryTextEngine:
    model_id = "parallel-test-ocr"
    supports_region_ocr = True

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.active = 0
        self.max_active = 0

    def fork(self) -> _ParallelRetryTextEngine:
        return self

    def extract(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "ocr",
    ) -> list[OCRSpan]:
        key = Path(image_path).name
        with self.lock:
            self.calls[key] = self.calls.get(key, 0) + 1
            call = self.calls[key]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            if key.endswith("000.png") and call == 1:
                raise RuntimeError("transient failure")
            return [
                OCRSpan(
                    id=f"{id_prefix}:p{page}:0000",
                    text="value",
                    confidence=0.98,
                    bbox=BBox(x1=5, y1=7, x2=45, y2=27),
                    page=page,
                    source="fake-ocr",
                    attempt=attempt,
                )
            ]
        finally:
            with self.lock:
                self.active -= 1


class LayoutBlockProcessingTests(unittest.TestCase):
    def test_parallel_region_ocr_retries_and_restores_global_coordinates(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "form.png"
            image = np.full((220, 420, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            blocks = [
                LayoutBlock(
                    id="layout:p1:0000",
                    label="text",
                    confidence=0.9,
                    bbox=BBox(x1=20, y1=30, x2=180, y2=100),
                ),
                LayoutBlock(
                    id="layout:p1:0001",
                    label="text",
                    confidence=0.9,
                    bbox=BBox(x1=220, y1=120, x2=390, y2=200),
                ),
            ]
            engine = _ParallelRetryTextEngine()
            settings = AssociationSettings(
                layout_parallel_workers=2,
                layout_block_retries=1,
                crop_padding_pixels=0,
            )

            processed, warnings = LayoutBlockProcessor(engine).process(
                image_path=image_path,
                run_dir=root / "artifacts",
                blocks=blocks,
                settings=settings,
                attempt=0,
            )

            self.assertEqual(warnings, [])
            self.assertGreaterEqual(engine.max_active, 2)
            self.assertEqual(
                [item.record.recognition_attempts for item in processed],  # type: ignore[union-attr]
                [2, 1],
            )
            self.assertEqual(
                [item.record.status for item in processed],  # type: ignore[union-attr]
                ["succeeded", "succeeded"],
            )
            first = processed[0].recognized_spans[0]
            second = processed[1].recognized_spans[0]
            self.assertEqual(first.bbox, BBox(x1=25, y1=37, x2=65, y2=57))
            self.assertEqual(second.bbox, BBox(x1=225, y1=127, x2=265, y2=147))
            self.assertEqual(first.source, "fake-ocr:layout_crop")
            self.assertTrue(all(item.crop_path.is_file() for item in processed))  # type: ignore[union-attr]

    def test_only_explicit_layouts_receive_supplemental_ocr(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "form.png"
            image = np.full((220, 420, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            blocks = [
                LayoutBlock(
                    id=f"layout:p1:{index:04d}",
                    label="text",
                    confidence=0.9,
                    bbox=BBox(
                        x1=20 + index * 200,
                        y1=30,
                        x2=180 + index * 200,
                        y2=100,
                    ),
                )
                for index in range(2)
            ]
            engine = _ParallelRetryTextEngine()

            processed, warnings = LayoutBlockProcessor(engine).process(
                image_path=image_path,
                run_dir=root / "artifacts",
                blocks=blocks,
                settings=AssociationSettings(
                    layout_block_retries=1,
                    crop_padding_pixels=0,
                ),
                attempt=0,
                ocr_block_ids={blocks[0].id},
            )

            self.assertEqual(len(engine.calls), 1)
            self.assertIsNotNone(processed[0].record)
            self.assertIsNone(processed[1].record)
            self.assertIn("layout_block_ocr_budget_applied:1/2", warnings)


if __name__ == "__main__":
    unittest.main()
