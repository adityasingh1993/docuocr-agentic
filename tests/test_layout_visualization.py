from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from docuocr.cv.layout_visualization import LayoutVisualizer
from docuocr.models import BBox, LayoutBlock
from docuocr.trace import sha256_file


class LayoutVisualizationTests(unittest.TestCase):
    def test_saves_overlay_without_modifying_source(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            destination = root / "images" / "layout-detected-0.png"
            image = np.full((240, 360, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))
            source_hash = sha256_file(source)
            blocks = [
                LayoutBlock(
                    id="layout:p1:0000",
                    label="text",
                    content="Sensitive value is not rendered",
                    confidence=0.93,
                    bbox=BBox(x1=30, y1=45, x2=300, y2=110),
                ),
                LayoutBlock(
                    id="layout:p1:0001",
                    label="table",
                    confidence=0.81,
                    bbox=BBox(x1=25, y1=135, x2=335, y2=220),
                ),
            ]

            record = LayoutVisualizer().render(
                source, blocks, destination, attempt=0
            )

            self.assertTrue(destination.is_file())
            self.assertEqual(sha256_file(source), source_hash)
            self.assertEqual(record.source_sha256, source_hash)
            self.assertEqual(record.output_sha256, sha256_file(destination))
            self.assertEqual(record.block_count, 2)
            self.assertEqual(record.rendered_block_count, 2)
            self.assertEqual(record.label_counts, {"table": 1, "text": 1})
            rendered = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertFalse(np.array_equal(image, rendered))

    def test_saves_notice_when_no_blocks_are_detected(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            destination = root / "layout-detected-0.png"
            image = np.full((120, 180, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))

            record = LayoutVisualizer().render(source, [], destination)

            self.assertTrue(destination.is_file())
            self.assertEqual(record.block_count, 0)
            self.assertEqual(record.rendered_block_count, 0)
            rendered = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertFalse(np.array_equal(image, rendered))


if __name__ == "__main__":
    unittest.main()
