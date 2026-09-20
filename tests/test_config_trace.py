from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docuocr.config import AppSettings, VLMSettings
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
