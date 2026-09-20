from __future__ import annotations

import unittest
from unittest.mock import patch

from docuocr.config import PaddleSettings
from docuocr.engines.paddle import PaddleTextEngine


class PaddleAdapterTests(unittest.TestCase):
    def test_import_error_names_backend_and_original_cause(self) -> None:
        settings = PaddleSettings(engine="transformers")
        engine = PaddleTextEngine(settings)
        with patch(
            "docuocr.engines.paddle.importlib.import_module",
            side_effect=ImportError("No module named 'transformers'"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "engine 'transformers'.*No module named 'transformers'.*"
                "paddle-transformers",
            ):
                engine._get_pipeline()

    def test_model_format_error_is_actionable(self) -> None:
        class InvalidModelPipeline:
            def __init__(self, **_: object) -> None:
                raise ValueError(
                    "No valid model files were found for engine 'transformers'."
                )

        settings = PaddleSettings(
            engine="transformers",
            text_detection_model_dir="/models/detection",
            text_recognition_model_dir="/models/recognition",
        )
        engine = PaddleTextEngine(settings)
        with patch(
            "docuocr.engines.paddle._load_paddle_symbol",
            return_value=InvalidModelPipeline,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "General OCR model files are incompatible.*models/detection",
            ):
                engine._get_pipeline()

    def test_old_paddleocr_engine_argument_is_reported(self) -> None:
        class LegacyPipeline:
            def __init__(self, **_: object) -> None:
                raise TypeError("unexpected keyword argument 'engine'")

        engine = PaddleTextEngine(PaddleSettings())
        with patch(
            "docuocr.engines.paddle._load_paddle_symbol",
            return_value=LegacyPipeline,
        ):
            with self.assertRaisesRegex(RuntimeError, "PaddleOCR >=3.5"):
                engine._get_pipeline()


if __name__ == "__main__":
    unittest.main()
