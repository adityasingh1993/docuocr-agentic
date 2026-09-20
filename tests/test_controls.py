from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from docuocr.cv.controls import FormControlDetector
from docuocr.models import ControlState


class ControlDetectionTests(unittest.TestCase):
    def test_checked_and_unchecked_boxes_are_distinguished(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV deployment dependency is not installed")
        image = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 100), (85, 135), (0, 0, 0), 3)
        cv2.line(image, (57, 116), (68, 128), (0, 0, 0), 4)
        cv2.line(image, (68, 128), (80, 106), (0, 0, 0), 4)
        cv2.rectangle(image, (50, 180), (85, 215), (0, 0, 0), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            controls = FormControlDetector().detect(path)
        states = [item.state for item in controls]
        self.assertIn(ControlState.CHECKED, states)
        self.assertIn(ControlState.UNCHECKED, states)


if __name__ == "__main__":
    unittest.main()
