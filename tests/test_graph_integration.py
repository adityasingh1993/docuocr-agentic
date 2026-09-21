from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from docuocr.config import AppSettings
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.pipeline import DocumentPipeline


class GraphIntegrationTests(unittest.TestCase):
    def test_parallel_evidence_join_and_contract_assembly(self) -> None:
        if importlib.util.find_spec("langgraph") is None:
            self.skipTest("LangGraph is not installed")
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            image_path = root / "form.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            sidecar_path = root / "evidence.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "ocrSpans": [
                            _span("l1", "First name", 50, 70, 190, 105, "ocr-engine-a"),
                            _span("v1", "Amina", 220, 70, 330, 105, "ocr-engine-a"),
                            _span(
                                "l2", "First name", 50, 170, 190, 205, "ocr-engine-b"
                            ),
                            _span("v2", "Amina", 220, 170, 330, 205, "ocr-engine-b"),
                        ],
                        "layoutBlocks": [
                            {
                                "id": "layout-raw",
                                "label": "text",
                                "confidence": 0.95,
                                "bbox": {
                                    "x1": 40,
                                    "y1": 40,
                                    "x2": 350,
                                    "y2": 120,
                                },
                            }
                        ],
                        "controls": [],
                    }
                ),
                encoding="utf-8",
            )
            settings = AppSettings.model_validate(
                {
                    "artifact_root": str(root / "artifacts"),
                    "policy": {
                        "accept_threshold": 0.90,
                        "document_quality_threshold": 0.90,
                        "max_document_enhancements": 1,
                        "max_field_retries": 0,
                    },
                }
            )
            blueprint = DocumentBlueprint.model_validate(
                {
                    "id": "integration-v1",
                    "version": "1",
                    "document_type": "test",
                    "fields": {
                        "data.baby.firstName": {
                            "type": "string",
                            "aliases": ["first name"],
                            "strategies": ["right_of_label"],
                            "critical": True,
                        }
                    },
                }
            )
            with DocumentPipeline(
                settings=settings, blueprint=blueprint, sidecar_path=sidecar_path
            ) as pipeline:
                result = pipeline.extract(image_path, job_id="integration")
            self.assertEqual(result["data"]["baby"]["firstName"], "Amina")
            field = result["meta"]["processors"]["fields"]["data.baby.firstName"]
            self.assertEqual(field["disposition"], "accepted")
            self.assertGreaterEqual(field["confidence"], 0.90)
            self.assertTrue(
                (root / "artifacts" / "integration" / "trace.jsonl").is_file()
            )
            evidence = json.loads(
                (root / "artifacts" / "integration" / "evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(evidence["ocrSpans"]), 4)
            self.assertEqual(len(evidence["layoutBlocks"]), 1)
            self.assertTrue(evidence["records"])
            self.assertEqual(evidence["originalImagePath"], str(image_path.resolve()))
            self.assertEqual(evidence["activeImagePath"], str(image_path.resolve()))
            self.assertIn("qualityHistory", evidence)
            self.assertIn("enhancementEvaluations", evidence)
            self.assertEqual(evidence["layoutBlockOCR"], [])
            self.assertEqual(len(evidence["layoutExtractions"]), 1)
            layout_record = evidence["layoutExtractions"][0]
            self.assertEqual(layout_record["block_id"], "layout:p1:0000")
            self.assertEqual(
                layout_record["candidates"][0]["normalized_value"], "Amina"
            )
            self.assertTrue(layout_record["ocr_spans"])
            self.assertEqual(layout_record["unresolved_target_paths"], [])
            self.assertEqual(evidence["evidenceReverifications"], [])
            self.assertEqual(len(evidence["layoutVisualizations"]), 1)
            visualization = evidence["layoutVisualizations"][0]
            self.assertEqual(visualization["block_count"], 1)
            self.assertEqual(visualization["rendered_block_count"], 1)
            self.assertTrue(Path(visualization["output_path"]).is_file())
            self.assertTrue(
                (
                    root
                    / "artifacts"
                    / "integration"
                    / "images"
                    / "layout-detected-0.png"
                ).is_file()
            )
            trace_nodes = {
                json.loads(line)["node"]
                for line in (
                    root / "artifacts" / "integration" / "trace.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            }
            self.assertIn("document_understand", trace_nodes)
            self.assertIn("layout_block_map", trace_nodes)
            layout_manifest_path = Path(
                result["meta"]["processors"]["layoutManifest"]
            )
            self.assertTrue(layout_manifest_path.is_file())
            layout_manifest = json.loads(
                layout_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(layout_manifest["layouts"]), 1)
            self.assertEqual(
                layout_manifest["combined"]["data"]["baby"]["firstName"],
                "Amina",
            )
            self.assertEqual(
                layout_manifest["combined"]["acceptedPaths"],
                ["data.baby.firstName"],
            )

    def test_unresolved_result_gets_one_bounded_evidence_pass(self) -> None:
        if importlib.util.find_spec("langgraph") is None:
            self.skipTest("LangGraph is not installed")
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "blank.png"
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            sidecar_path = root / "evidence.json"
            sidecar_path.write_text(
                json.dumps(
                    {"ocrSpans": [], "layoutBlocks": [], "controls": []}
                ),
                encoding="utf-8",
            )
            settings = AppSettings.model_validate(
                {
                    "artifact_root": str(root / "artifacts"),
                    "policy": {
                        "max_document_enhancements": 0,
                        "max_field_retries": 0,
                    },
                    "association": {
                        "layout_blocks_enabled": False,
                        "max_evidence_retries": 1,
                        "reverify_with_vlm": False,
                    },
                }
            )
            blueprint = DocumentBlueprint.model_validate(
                {
                    "id": "missing-only-v1",
                    "version": "1",
                    "document_type": "test",
                    "fields": {
                        "data.baby.firstName": {
                            "type": "string",
                            "aliases": ["first name"],
                            "strategies": ["right_of_label"],
                            "critical": True,
                        }
                    },
                }
            )
            with DocumentPipeline(
                settings=settings,
                blueprint=blueprint,
                sidecar_path=sidecar_path,
            ) as pipeline:
                result = pipeline.extract(image_path, job_id="reverify-route")

            run_dir = root / "artifacts" / "reverify-route"
            evidence = json.loads(
                (run_dir / "evidence.json").read_text(encoding="utf-8")
            )
            trace = [
                json.loads(line)
                for line in (run_dir / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        field = result["meta"]["processors"]["fields"]["data.baby.firstName"]
        self.assertEqual(field["disposition"], "review")
        self.assertEqual(len(evidence["evidenceReverifications"]), 1)
        self.assertEqual(
            [
                item["node"]
                for item in trace
                if item["node"] in {"score", "evidence_reverify", "review"}
            ],
            ["score", "evidence_reverify", "score", "review"],
        )


def _span(
    identifier: str,
    text: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    source: str,
) -> dict:
    return {
        "id": identifier,
        "text": text,
        "confidence": 0.99,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source": source,
    }


if __name__ == "__main__":
    unittest.main()
