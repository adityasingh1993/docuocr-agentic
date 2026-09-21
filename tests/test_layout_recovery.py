from __future__ import annotations

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from docuocr.config import AppSettings
from docuocr.engines.sidecar import NullLayoutEngine
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.models import BBox, OCRSpan, QualityReport, RecoveryAction, RecoveryPlan
from docuocr.trace import sha256_file
from docuocr.workflow.nodes import WorkflowNodes


class _LayoutRecoveryTextEngine:
    model_id = "layout-recovery-test-ocr"
    supports_region_ocr = True

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def fork(self) -> _LayoutRecoveryTextEngine:
        return self

    def extract(
        self,
        image_path: str | Path,
        *,
        page: int = 1,
        attempt: int = 0,
        id_prefix: str = "ocr",
    ) -> list[OCRSpan]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            if Path(image_path).name.endswith("000.png"):
                values = ("First name", "Amina")
            else:
                values = ("Register number", "REG-22")
            return [
                OCRSpan(
                    id=f"{id_prefix}:p{page}:0000",
                    text=values[0],
                    confidence=0.98,
                    bbox=BBox(x1=5, y1=5, x2=135, y2=25),
                    page=page,
                    source="fake-ocr",
                    attempt=attempt,
                ),
                OCRSpan(
                    id=f"{id_prefix}:p{page}:0001",
                    text=values[1],
                    confidence=0.98,
                    bbox=BBox(x1=150, y1=5, x2=245, y2=25),
                    page=page,
                    source="fake-ocr",
                    attempt=attempt,
                ),
            ]
        finally:
            with self.lock:
                self.active -= 1


class _NoControls:
    model_id = "no-controls"

    def detect(self, image_path: str | Path, **_: object) -> list:
        return []


class LayoutRecoveryTests(unittest.TestCase):
    def test_unresolved_fields_recover_from_matching_layouts_in_parallel(self) -> None:
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "form.png"
            image = np.full((300, 500, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            run_dir = root / "artifacts" / "layout-recovery"
            run_dir.mkdir(parents=True)
            settings = AppSettings.model_validate(
                {
                    "artifact_root": str(root / "artifacts"),
                    "association": {
                        "layout_parallel_workers": 2,
                        "layout_block_retries": 0,
                        "whole_page_recovery_fallback": False,
                    },
                }
            )
            blueprint = DocumentBlueprint.model_validate(
                {
                    "id": "layout-recovery-v1",
                    "version": "1",
                    "document_type": "test",
                    "fields": {
                        "data.baby.firstName": {
                            "aliases": ["first name"],
                            "strategies": ["right_of_label"],
                        },
                        "data.baby.registerNumber": {
                            "aliases": ["register number"],
                            "strategies": ["right_of_label"],
                        },
                        "data.baby.lastName": {
                            "aliases": ["last name"],
                            "strategies": ["right_of_label"],
                        },
                    },
                }
            )
            engine = _LayoutRecoveryTextEngine()
            nodes = WorkflowNodes(
                settings=settings,
                blueprint=blueprint,
                text_engine=engine,
                layout_engine=NullLayoutEngine(),
                control_engine=_NoControls(),
                vlm=None,
            )
            quality = QualityReport(
                overall=0.8,
                resolution=0.8,
                sharpness=0.8,
                contrast=0.8,
                illumination=0.8,
                glare=0.8,
                skew=0.8,
                width=500,
                height=300,
            )
            plans = [
                RecoveryPlan(
                    field_path="data.baby.firstName",
                    actions=[RecoveryAction.UPSCALE, RecoveryAction.CLAHE],
                    attempt=1,
                ),
                RecoveryPlan(
                    field_path="data.baby.registerNumber",
                    actions=[RecoveryAction.UPSCALE, RecoveryAction.CLAHE],
                    attempt=1,
                ),
                RecoveryPlan(
                    field_path="data.baby.lastName",
                    actions=[RecoveryAction.UPSCALE, RecoveryAction.CLAHE],
                    attempt=1,
                ),
            ]
            state = {
                "job_id": "layout-recovery",
                "source_path": str(image_path),
                "original_sha256": sha256_file(image_path),
                "active_image_path": str(image_path),
                "run_dir": str(run_dir),
                "field_attempt": 0,
                "quality": quality.model_dump(mode="json"),
                "layout_blocks": [
                    {
                        "id": "layout:p1:name",
                        "label": "text",
                        "content": "First name",
                        "confidence": 0.95,
                        "bbox": {"x1": 20, "y1": 20, "x2": 470, "y2": 120},
                    },
                    {
                        "id": "layout:p1:register",
                        "label": "text",
                        "content": "Register number",
                        "confidence": 0.95,
                        "bbox": {"x1": 20, "y1": 160, "x2": 470, "y2": 270},
                    },
                ],
                "ocr_spans": [
                    _span("label:name", "First name", 30, 35, 145, 60),
                    _span(
                        "label:register", "Register number", 30, 175, 180, 200
                    ),
                ],
                "recovery_plans": [item.model_dump(mode="json") for item in plans],
                "candidates": [],
            }

            update = nodes.recover(state)  # type: ignore[arg-type]

            self.assertGreaterEqual(engine.max_active, 2)
            self.assertEqual(update["active_image_path"], str(image_path))
            self.assertFalse((run_dir / "images" / "recovery-page-1.png").exists())
            recovery_dir = run_dir / "images" / "layout-recovery"
            self.assertTrue((recovery_dir / "layout-recovery-1-000.png").is_file())
            self.assertTrue((recovery_dir / "layout-recovery-1-001.png").is_file())

            records = update["layout_extractions"]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["ocr_spans"] for record in records))
            register_record = next(
                record
                for record in records
                if record["block_id"] == "layout:p1:register"
            )
            self.assertGreaterEqual(
                register_record["ocr_spans"][0]["bbox"]["y1"], 148
            )
            self.assertTrue(
                all(
                    "data.baby.lastName" in record["unresolved_target_paths"]
                    for record in records
                )
            )
            self.assertTrue(all(record["status"] == "partial" for record in records))
            self.assertEqual(
                {candidate["path"] for record in records for candidate in record["candidates"]},
                {"data.baby.firstName", "data.baby.registerNumber"},
            )
            values = {
                item["path"]: item["normalized_value"]
                for item in update["candidates"]
            }
            self.assertEqual(values["data.baby.firstName"], "Amina")
            self.assertEqual(values["data.baby.registerNumber"], "REG-22")


def _span(
    identifier: str, text: str, x1: int, y1: int, x2: int, y2: int
) -> dict:
    return {
        "id": identifier,
        "text": text,
        "confidence": 0.99,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source": "baseline-ocr",
    }


if __name__ == "__main__":
    unittest.main()
