from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from docuocr.cv.quality import ImageQualityAssessor


class ImageQualityTests(unittest.TestCase):
    def test_sparse_black_text_on_white_form_is_not_glare(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV deployment dependency is not installed")
        image = np.full((900, 1400, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "First name   Amina",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "form.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            report = ImageQualityAssessor().assess(path)
        self.assertEqual(report.glare, 1.0)
        self.assertGreater(report.contrast, 0.90)


if __name__ == "__main__":
    unittest.main()
