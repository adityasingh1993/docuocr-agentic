from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docuocr.config import AppSettings, PaddleSettings, VLMSettings
from docuocr.trace import TraceWriter


class ConfigAndTraceTests(unittest.TestCase):
    def test_external_vlm_hostname_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VLMSettings(enabled=True, base_url="https://example.com/v1")

    def test_loopback_vlm_is_allowed(self) -> None:
        settings = VLMSettings(enabled=True, base_url="http://127.0.0.1:8081/v1")
        self.assertTrue(settings.enabled)

    def test_offline_paddle_requires_local_model_directories(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit paths"):
            AppSettings.model_validate({"offline": True, "paddle": {"enabled": True}})

    def test_text_only_offline_paddle_does_not_require_layout_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection = root / "detection"
            recognition = root / "recognition"
            detection.mkdir()
            recognition.mkdir()
            settings = AppSettings.model_validate(
                {
                    "offline": True,
                    "paddle": {
                        "enabled": True,
                        "text_enabled": True,
                        "layout_enabled": False,
                        "text_detection_model_dir": str(detection),
                        "text_recognition_model_dir": str(recognition),
                    },
                }
            )
            self.assertFalse(settings.paddle.layout_enabled)

    def test_component_engines_can_be_selected_independently(self) -> None:
        settings = PaddleSettings(
            engine="paddle", text_engine="paddle_static", layout_engine="transformers"
        )
        self.assertEqual(settings.resolved_text_engine, "paddle_static")
        self.assertEqual(settings.resolved_layout_engine, "transformers")

    def test_enabled_paddle_requires_at_least_one_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "text_enabled or layout_enabled"):
            PaddleSettings(enabled=True, text_enabled=False, layout_enabled=False)

    def test_layout_images_can_be_disabled(self) -> None:
        settings = AppSettings.model_validate(
            {"trace": {"save_layout_images": False}}
        )
        self.assertFalse(settings.trace.save_layout_images)

    def test_layout_association_parallelism_and_retries_are_configurable(self) -> None:
        settings = AppSettings.model_validate(
            {
                "association": {
                    "layout_parallel_workers": 3,
                    "layout_block_retries": 2,
                    "max_evidence_retries": 2,
                }
            }
        )
        self.assertEqual(settings.association.layout_parallel_workers, 3)
        self.assertEqual(settings.association.layout_block_retries, 2)
        self.assertEqual(settings.association.max_evidence_retries, 2)

    def test_trace_append_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            writer = TraceWriter(path)
            kwargs = {
                "run_id": "run-1",
                "node": "ocr",
                "attempt": 0,
                "status": "succeeded",
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:01+00:00",
                "duration_ms": 1000,
                "input_sha256": "abc",
            }
            writer.append(**kwargs)
            writer.append(**kwargs)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertNotIn("values", json.loads(lines[0]))


if __name__ == "__main__":
    unittest.main()
