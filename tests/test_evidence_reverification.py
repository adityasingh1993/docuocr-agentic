from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docuocr.config import AppSettings
from docuocr.engines.sidecar import NullLayoutEngine, NullTextEngine
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.models import BBox, EvidenceKind, EvidenceRecord, OCRSpan
from docuocr.workflow.graph import _score_route
from docuocr.workflow.nodes import WorkflowNodes


class EvidenceReverificationTests(unittest.TestCase):
    def test_missing_name_is_rebuilt_from_two_grounded_ocr_passes(self) -> None:
        blueprint = DocumentBlueprint.model_validate(
            {
                "id": "name-only",
                "version": "1",
                "document_type": "test",
                "fields": {
                    "data.baby.firstName": {
                        "type": "string",
                        "aliases": ["first name"],
                        "strategies": ["right_of_label"],
                        "group_aliases": ["rn"],
                        "group_value_part": "first_token",
                        "critical": True,
                    },
                    "data.baby.lastName": {
                        "type": "string",
                        "aliases": ["last name"],
                        "strategies": ["right_of_label"],
                        "group_aliases": ["rn"],
                        "group_value_part": "remaining_tokens",
                        "critical": True,
                    },
                },
            }
        )
        spans = [
            *_name_spans("ocr:page", "paddleocr"),
            *_name_spans("ocr:block", "paddleocr:layout_crop"),
        ]
        evidence = [
            EvidenceRecord(
                id=item.id,
                kind=EvidenceKind.OCR,
                bbox=item.bbox,
                confidence=item.confidence,
                source=item.source,
            ).model_dump(mode="json")
            for item in spans
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "artifacts" / "job"
            run_dir.mkdir(parents=True)
            image = root / "form.png"
            image.write_bytes(b"test-image")
            settings = AppSettings.model_validate(
                {
                    "artifact_root": str(root / "artifacts"),
                    "policy": {"max_field_retries": 0},
                    "association": {"reverify_with_vlm": False},
                }
            )
            nodes = WorkflowNodes(
                settings=settings,
                blueprint=blueprint,
                text_engine=NullTextEngine(),
                layout_engine=NullLayoutEngine(),
                control_engine=None,  # type: ignore[arg-type]
                vlm=None,
            )
            state = {
                "job_id": "job",
                "source_path": str(image),
                "original_image_path": str(image),
                "active_image_path": str(image),
                "run_dir": str(run_dir),
                "original_sha256": "test",
                "field_attempt": 0,
                "evidence_reverification_attempt": 0,
                "quality": _quality(),
                "decisions": {
                    "data.baby.firstName": {"disposition": "missing"},
                    "data.baby.lastName": {"disposition": "missing"},
                },
                "candidates": [],
                "ocr_spans": [item.model_dump(mode="json") for item in spans],
                "layout_blocks": [],
                "controls": [],
                "evidence": evidence,
            }

            verification = nodes.evidence_reverify(state)  # type: ignore[arg-type]
            rescored = nodes.score({**state, **verification})  # type: ignore[arg-type]
            assembled = nodes.assemble(
                {
                    **state,
                    **verification,
                    **rescored,
                    "processor_steps": [
                        *verification["processor_steps"],
                        *rescored["processor_steps"],
                    ],
                    "timings": {
                        **verification["timings"],
                        **rescored["timings"],
                    },
                }
            )

        self.assertEqual(verification["evidence_reverification_attempt"], 1)
        self.assertEqual(
            set(verification["evidence_reverifications"][0]["candidate_paths"]),
            {"data.baby.firstName", "data.baby.lastName"},
        )
        self.assertEqual(
            rescored["decisions"]["data.baby.firstName"]["disposition"],
            "accepted",
        )
        self.assertEqual(
            assembled["result"]["data"]["baby"]["firstName"], "Maria"
        )
        self.assertEqual(
            assembled["result"]["data"]["baby"]["lastName"], "Joao da Silva"
        )

    def test_score_route_allows_only_configured_reverification_attempts(self) -> None:
        settings = AppSettings.model_validate(
            {"association": {"max_evidence_retries": 1}}
        )

        class _Nodes:
            pass

        nodes = _Nodes()
        nodes.settings = settings  # type: ignore[attr-defined]
        state = {
            "decisions": {"data.name": {"disposition": "missing"}},
            "evidence_reverification_attempt": 0,
        }
        self.assertEqual(
            _score_route(state, nodes),  # type: ignore[arg-type]
            "reverify",
        )
        state["evidence_reverification_attempt"] = 1
        self.assertEqual(
            _score_route(state, nodes),  # type: ignore[arg-type]
            "review",
        )

    def test_score_route_does_not_reverify_a_schema_mismatch(self) -> None:
        settings = AppSettings.model_validate(
            {"association": {"max_evidence_retries": 1}}
        )

        class _Nodes:
            pass

        nodes = _Nodes()
        nodes.settings = settings  # type: ignore[attr-defined]
        state = {
            "decisions": {"data.name": {"disposition": "review"}},
            "document_understanding": {"schema_match": "mismatch"},
            "evidence_reverification_attempt": 0,
        }
        self.assertEqual(
            _score_route(state, nodes),  # type: ignore[arg-type]
            "review",
        )


def _name_spans(prefix: str, source: str) -> list[OCRSpan]:
    return [
        OCRSpan(
            id=f"{prefix}:label",
            text="RN:",
            confidence=0.99,
            bbox=BBox(x1=10, y1=10, x2=50, y2=35),
            source=source,
        ),
        OCRSpan(
            id=f"{prefix}:value",
            text="Maria Joao da Silva",
            confidence=0.98,
            bbox=BBox(x1=70, y1=10, x2=260, y2=35),
            source=source,
        ),
    ]


def _quality() -> dict[str, object]:
    return {
        "overall": 0.99,
        "resolution": 0.99,
        "sharpness": 0.99,
        "contrast": 0.99,
        "illumination": 0.99,
        "glare": 0.99,
        "skew": 0.99,
        "width": 400,
        "height": 200,
    }


if __name__ == "__main__":
    unittest.main()
