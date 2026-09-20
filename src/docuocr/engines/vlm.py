from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from docuocr.config import VLMSettings
from docuocr.extraction.grounding import GroundedProposal
from docuocr.models import FormControl, LayoutBlock, OCRSpan


class ProposalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposals: list[GroundedProposal] = Field(default_factory=list)


class LocalVLMClient:
    """OpenAI-compatible client restricted by validated local endpoint settings."""

    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings
        self.model_id = settings.model

    def propose(
        self,
        *,
        image_path: str | Path,
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None = None,
    ) -> list[GroundedProposal]:
        prompt = self._prompt(target_paths, spans, blocks, controls, image_evidence_id)
        media_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        schema = ProposalBatch.model_json_schema(by_alias=True)
        body = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You map form evidence to an allowed schema. Never invent a value or evidence ID. "
                        "Return no proposal when the image/evidence is insufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_form_proposals",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        text = str(content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return ProposalBatch.model_validate(json.loads(text)).proposals

    @staticmethod
    def _prompt(
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None,
    ) -> str:
        ledger = {
            "allowedPaths": target_paths,
            "ocr": [
                {"id": item.id, "text": item.text, "bbox": item.bbox.as_list()}
                for item in spans
            ],
            "layout": [
                {
                    "id": item.id,
                    "label": item.label,
                    "content": item.content,
                    "bbox": item.bbox.as_list(),
                }
                for item in blocks
            ],
            "controls": [
                {
                    "id": item.id,
                    "state": item.state.value,
                    "labelEvidenceId": item.associated_label_id,
                    "bbox": item.bbox.as_list(),
                }
                for item in controls
            ],
            "imageEvidenceId": image_evidence_id,
        }
        return (
            "Propose values only for allowedPaths. Each proposal must contain path, rawValue, and "
            "evidenceIds. A string rawValue must be literally supported by cited OCR/layout text. "
            "A boolean must cite a non-ambiguous control. If imageEvidenceId is present and you "
            "independently read the pixels, cite it in addition to OCR/control evidence. Do not "
            "provide confidence. Evidence ledger:\n"
            + json.dumps(ledger, ensure_ascii=False, separators=(",", ":"))
        )
