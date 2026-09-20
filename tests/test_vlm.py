from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from docuocr.config import AppSettings, VLMSettings
from docuocr.engines.vlm import (
    LocalVLMClient,
    VLMProposalResult,
    VLMResponseError,
    _decode_proposals,
)
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.grounding import GroundedProposal
from docuocr.models import BBox, EvidenceKind, EvidenceRecord, FieldCandidate, OCRSpan
from docuocr.workflow.nodes import WorkflowNodes

ROOT = Path(__file__).resolve().parents[1]


class VLMClientTests(unittest.TestCase):
    def test_truncated_batch_is_split_and_retried(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            prompt = body["messages"][1]["content"][0]["text"]
            ledger = json.loads(prompt.split("Evidence ledger:\n", 1)[1])
            paths = ledger["allowedPaths"]
            if len(paths) > 1:
                return _response('{"proposals":[', finish_reason="length")
            content = json.dumps(
                {
                    "proposals": [
                        {
                            "path": paths[0],
                            "rawValue": True,
                            "evidenceIds": ["image:test"],
                        }
                    ]
                }
            )
            return _response(content)

        settings = VLMSettings(
            enabled=True,
            max_paths_per_request=2,
            request_retries=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "form.png"
            image.write_bytes(b"not-decoded-by-mock-server")
            result = LocalVLMClient(
                settings, transport=httpx.MockTransport(handler)
            ).propose(
                image_path=image,
                target_paths=["data.baby.male", "data.baby.female"],
                spans=[],
                blocks=[],
                controls=[],
                image_evidence_id="image:test",
            )

        self.assertEqual(
            [item.path for item in result.proposals],
            ["data.baby.male", "data.baby.female"],
        )
        self.assertTrue(
            any(item.startswith("vlm_batch_split:2:") for item in result.warnings)
        )

    def test_malformed_single_path_becomes_warning(self) -> None:
        transport = httpx.MockTransport(
            lambda request: _response('{"proposals":[{"path":"broken')
        )
        settings = VLMSettings(enabled=True, request_retries=0)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "form.png"
            image.write_bytes(b"test")
            result = LocalVLMClient(settings, transport=transport).propose(
                image_path=image,
                target_paths=["data.baby.female"],
                spans=[],
                blocks=[],
                controls=[],
                image_evidence_id="image:test",
            )

        self.assertEqual(result.proposals, [])
        self.assertTrue(
            result.warnings[0].startswith(
                "vlm_path_response_failed:data.baby.female:VLMResponseError:"
            )
        )
        self.assertIn("invalid_json_response", result.warnings[0])

    def test_decoder_accepts_fenced_json_and_rejects_length_finish(self) -> None:
        proposals = _decode_proposals(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "```json\n"
                                '{"proposals":[{"path":"data.baby.female",'
                                '"rawValue":true,"evidenceIds":["image:test"]}]}'
                                "\n```"
                            )
                        },
                    }
                ]
            }
        )
        self.assertEqual(proposals[0].path, "data.baby.female")
        with self.assertRaisesRegex(VLMResponseError, "truncated_response"):
            _decode_proposals(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"proposals":['},
                        }
                    ]
                }
            )


class VLMWorkflowTests(unittest.TestCase):
    def test_visual_control_is_rechecked_even_when_opencv_has_a_candidate(self) -> None:
        class FakeVLM:
            model_id = "fake-visual-model"

            def __init__(self) -> None:
                self.target_paths: list[str] = []

            def propose(self, **kwargs: object) -> VLMProposalResult:
                self.target_paths = list(kwargs["target_paths"])  # type: ignore[arg-type]
                evidence_ids = ["ocr:sex", "image:vlm:p1:active"]
                return VLMProposalResult(
                    proposals=[
                        GroundedProposal(
                            path="data.baby.male",
                            rawValue=False,
                            evidenceIds=evidence_ids,
                        ),
                        GroundedProposal(
                            path="data.baby.female",
                            rawValue=True,
                            evidenceIds=evidence_ids,
                        ),
                    ]
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "form.png"
            image.write_bytes(b"actual-image-bytes")
            span = OCRSpan(
                id="ocr:sex",
                text="SEXO",
                confidence=0.99,
                bbox=BBox(x1=10, y1=10, x2=80, y2=35),
            )
            span_evidence = EvidenceRecord(
                id=span.id,
                kind=EvidenceKind.OCR,
                bbox=span.bbox,
                confidence=span.confidence,
                source=span.source,
            )
            wrong_male = FieldCandidate(
                path="data.baby.male",
                raw_value=True,
                normalized_value=True,
                evidence_ids=["control:wrong"],
                source="opencv_control",
                support_sources=["opencv"],
                recognition_confidence=0.95,
                association_confidence=0.95,
            )
            fake_vlm = FakeVLM()
            nodes = WorkflowNodes(
                settings=AppSettings(artifact_root=str(root / "artifacts")),
                blueprint=DocumentBlueprint.from_yaml(
                    ROOT / "config" / "blueprints" / "newborn_screening.yaml"
                ),
                text_engine=None,  # type: ignore[arg-type]
                layout_engine=None,  # type: ignore[arg-type]
                control_engine=None,  # type: ignore[arg-type]
                vlm=fake_vlm,  # type: ignore[arg-type]
            )
            update = nodes.vlm_map(
                {
                    "job_id": "visual-control-test",
                    "source_path": str(image),
                    "active_image_path": str(image),
                    "quality": _quality(),
                    "candidates": [wrong_male.model_dump(mode="json")],
                    "ocr_spans": [span.model_dump(mode="json")],
                    "layout_blocks": [],
                    "controls": [],
                    "evidence": [span_evidence.model_dump(mode="json")],
                }
            )

        candidates = [
            FieldCandidate.model_validate(item) for item in update["candidates"]
        ]
        by_path = {item.path: item for item in candidates}
        self.assertIs(by_path["data.baby.male"].normalized_value, False)
        self.assertIs(by_path["data.baby.female"].normalized_value, True)
        self.assertIn("data.baby.male", fake_vlm.target_paths)
        self.assertIn("data.baby.female", fake_vlm.target_paths)
        self.assertTrue(
            any(
                item.startswith("visual_control_overrode_opencv:data.baby.male")
                for item in update["warnings"]
            )
        )


def _response(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": content},
                }
            ]
        },
    )


def _quality() -> dict[str, object]:
    return {
        "overall": 0.98,
        "resolution": 0.98,
        "sharpness": 0.98,
        "contrast": 0.98,
        "illumination": 0.98,
        "glare": 0.98,
        "skew": 0.98,
        "width": 200,
        "height": 100,
    }


if __name__ == "__main__":
    unittest.main()
