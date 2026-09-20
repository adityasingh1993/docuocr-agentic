from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuocr.config import VLMSettings
from docuocr.extraction.grounding import GroundedProposal
from docuocr.extraction.rules import alias_score
from docuocr.models import FormControl, LayoutBlock, OCRSpan


class ProposalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposals: list[GroundedProposal] = Field(default_factory=list)


class VLMProposalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposals: list[GroundedProposal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VLMResponseError(RuntimeError):
    """The local endpoint returned an unusable model response."""


class LocalVLMClient:
    """OpenAI-compatible client restricted by validated local endpoint settings."""

    def __init__(
        self,
        settings: VLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.model_id = settings.model
        self._transport = transport

    def propose(
        self,
        *,
        image_path: str | Path,
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None = None,
        field_hints: dict[str, dict[str, Any]] | None = None,
    ) -> VLMProposalResult:
        paths = list(dict.fromkeys(target_paths))
        if not paths:
            return VLMProposalResult()

        resolved_image = Path(image_path)
        media_type = mimetypes.guess_type(str(resolved_image))[0] or "image/png"
        encoded = base64.b64encode(resolved_image.read_bytes()).decode("ascii")
        hints = field_hints or {}
        result = VLMProposalResult()
        with httpx.Client(
            timeout=self.settings.timeout_seconds,
            transport=self._transport,
        ) as client:
            for start in range(0, len(paths), self.settings.max_paths_per_request):
                batch_paths = paths[start : start + self.settings.max_paths_per_request]
                batch_result = self._propose_resilient(
                    client=client,
                    encoded_image=encoded,
                    media_type=media_type,
                    target_paths=batch_paths,
                    spans=spans,
                    blocks=blocks,
                    controls=controls,
                    image_evidence_id=image_evidence_id,
                    field_hints={
                        path: hints[path] for path in batch_paths if path in hints
                    },
                )
                result.proposals.extend(batch_result.proposals)
                result.warnings.extend(batch_result.warnings)
        return result

    def _propose_resilient(
        self,
        *,
        client: httpx.Client,
        encoded_image: str,
        media_type: str,
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None,
        field_hints: dict[str, dict[str, Any]],
    ) -> VLMProposalResult:
        try:
            proposals = self._request_batch(
                client=client,
                encoded_image=encoded_image,
                media_type=media_type,
                target_paths=target_paths,
                spans=spans,
                blocks=blocks,
                controls=controls,
                image_evidence_id=image_evidence_id,
                field_hints=field_hints,
            )
        except VLMResponseError as exc:
            if len(target_paths) > 1:
                midpoint = len(target_paths) // 2
                warning = (
                    f"vlm_batch_split:{len(target_paths)}:{_safe_error_message(exc)}"
                )
                combined = VLMProposalResult(warnings=[warning])
                for subset in (target_paths[:midpoint], target_paths[midpoint:]):
                    partial = self._propose_resilient(
                        client=client,
                        encoded_image=encoded_image,
                        media_type=media_type,
                        target_paths=subset,
                        spans=spans,
                        blocks=blocks,
                        controls=controls,
                        image_evidence_id=image_evidence_id,
                        field_hints={
                            path: field_hints[path]
                            for path in subset
                            if path in field_hints
                        },
                    )
                    combined.proposals.extend(partial.proposals)
                    combined.warnings.extend(partial.warnings)
                return combined
            return VLMProposalResult(
                warnings=[
                    (
                        f"vlm_path_response_failed:{target_paths[0]}:"
                        f"{_safe_error_message(exc)}"
                    )
                ]
            )
        except httpx.HTTPError as exc:
            return VLMProposalResult(
                warnings=[
                    (
                        f"vlm_request_failed:{','.join(target_paths)}:"
                        f"{_safe_error_message(exc)}"
                    )
                ]
            )

        allowed = set(target_paths)
        accepted = [item for item in proposals if item.path in allowed]
        warnings = [
            f"vlm_rejected_unknown_path:{item.path}"
            for item in proposals
            if item.path not in allowed
        ]
        return VLMProposalResult(proposals=accepted, warnings=warnings)

    def _request_batch(
        self,
        *,
        client: httpx.Client,
        encoded_image: str,
        media_type: str,
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None,
        field_hints: dict[str, dict[str, Any]],
    ) -> list[GroundedProposal]:
        selected_controls = self._select_controls(
            target_paths, field_hints, spans, controls
        )
        prompt = self._prompt(
            target_paths,
            field_hints,
            spans,
            blocks,
            selected_controls,
            image_evidence_id,
        )
        schema = ProposalBatch.model_json_schema(by_alias=True)
        body = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You map form evidence to an allowed schema. Never invent a value "
                        "or evidence ID. Return no proposal when the image/evidence is "
                        "insufficient. Return one compact JSON object and no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded_image}"
                            },
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
        response: httpx.Response | None = None
        for attempt in range(self.settings.request_retries + 1):
            try:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt >= self.settings.request_retries:
                    raise
        if response is None:  # pragma: no cover - defensive; loop always sets or raises
            raise VLMResponseError("missing_http_response")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise VLMResponseError("invalid_http_json") from exc
        return _decode_proposals(payload)

    def _select_controls(
        self,
        target_paths: list[str],
        field_hints: dict[str, dict[str, Any]],
        spans: list[OCRSpan],
        controls: list[FormControl],
    ) -> list[FormControl]:
        if self.settings.max_controls_per_request == 0:
            return []
        aliases = [
            alias
            for path in target_paths
            for alias in field_hints.get(path, {}).get("printedLabelAliases", [])
            if field_hints.get(path, {}).get("kind") == "control_option"
        ]
        if not aliases:
            return []
        spans_by_id = {item.id: item for item in spans}
        ranked: list[tuple[float, FormControl]] = []
        for control in controls:
            label = spans_by_id.get(control.associated_label_id or "")
            if label is None:
                continue
            match = alias_score(label.text, aliases)
            if match < 0.76:
                continue
            score = match * 0.75 + control.state_confidence * 0.25
            ranked.append((score, control))
        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].bbox.y1,
                item[1].bbox.x1,
            )
        )
        return [item for _, item in ranked[: self.settings.max_controls_per_request]]

    @staticmethod
    def _prompt(
        target_paths: list[str],
        field_hints: dict[str, dict[str, Any]],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        image_evidence_id: str | None,
    ) -> str:
        ledger = {
            "allowedPaths": target_paths,
            "fieldHints": field_hints,
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
            "Propose values only for allowedPaths. Each proposal must contain path, "
            "rawValue, and evidenceIds. Printed labels may be interpreted semantically "
            "across languages only to associate them with fieldHints. Never translate, "
            "rewrite, or correct handwritten/entered names, identifiers, dates, or "
            "free-text values; preserve their literal OCR text. A string rawValue must be "
            "literally supported by cited OCR/layout text. For a control_option, printed "
            "option text such as M or F is a label, not proof that it is selected. Judge "
            "selection from visible ink in the supplied image: circles, ticks, crosses, "
            "or fills count as marks; untouched options are false. Detector control states "
            "are hints and may be wrong. When an exclusive group is visually clear, return "
            "a boolean proposal for every allowed option in that group. Cite "
            "imageEvidenceId plus the option or group-label OCR evidence when available. "
            "Do not cite a detector control that conflicts with the pixels. Do not provide "
            "confidence. Return compact JSON only. Evidence ledger:\n"
            + json.dumps(ledger, ensure_ascii=False, separators=(",", ":"))
        )


def _decode_proposals(payload: dict[str, Any]) -> list[GroundedProposal]:
    try:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMResponseError("missing_response_content") from exc
    if finish_reason == "length":
        raise VLMResponseError("truncated_response:finish_reason=length")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    text = str(content).strip()
    if not text:
        raise VLMResponseError("empty_response_content")
    start = text.find("{")
    if start < 0:
        raise VLMResponseError("json_object_not_found")
    try:
        decoded, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise VLMResponseError(
            f"invalid_json_response:{exc.msg}:position={exc.pos}"
        ) from exc
    try:
        return ProposalBatch.model_validate(decoded).proposals
    except ValidationError as exc:
        raise VLMResponseError(
            f"invalid_response_schema:{exc.error_count()}_errors"
        ) from exc


def _safe_error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}:{message[:240]}"
