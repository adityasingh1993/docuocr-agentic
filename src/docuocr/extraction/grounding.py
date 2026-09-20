from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from docuocr.models import (
    ControlState,
    EvidenceKind,
    EvidenceRecord,
    FieldCandidate,
    FormControl,
    LayoutBlock,
    OCRSpan,
)

from .blueprint import DocumentBlueprint, FieldSpec
from .normalization import normalize_value
from .rules import canonical_text


class GroundedProposal(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda value: (
            value.split("_")[0]
            + "".join(part.capitalize() for part in value.split("_")[1:])
        ),
        populate_by_name=True,
        extra="forbid",
    )
    path: str
    raw_value: str | bool | None = None
    evidence_ids: list[str] = Field(min_length=1)


class GroundingError(ValueError):
    pass


class GroundingVerifier:
    def verify(
        self,
        proposal: GroundedProposal,
        blueprint: DocumentBlueprint,
        evidence: list[EvidenceRecord],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        *,
        attempt: int,
        visual_verification: bool = False,
    ) -> FieldCandidate:
        if proposal.path not in blueprint.output_paths:
            raise GroundingError("unknown_output_path")
        records = {item.id: item for item in evidence}
        unknown = [item for item in proposal.evidence_ids if item not in records]
        if unknown:
            raise GroundingError(f"unknown_evidence_ids:{','.join(unknown)}")

        span_text = {item.id: item.text for item in spans}
        block_text = {item.id: item.content for item in blocks}
        controls_by_id = {item.id: item for item in controls}
        cited_text = " ".join(
            span_text.get(item, block_text.get(item, ""))
            for item in proposal.evidence_ids
        )
        field_spec = blueprint.fields.get(proposal.path)
        if field_spec is None:
            field_spec = FieldSpec(
                type="boolean", aliases=[proposal.path.rsplit(".", 1)[-1]]
            )
        if field_spec.type == "boolean":
            cited_controls = [
                controls_by_id[item]
                for item in proposal.evidence_ids
                if item in controls_by_id
            ]
            if not cited_controls or proposal.raw_value is None:
                raise GroundingError("boolean_requires_control_evidence")
            expected = any(
                item.state == ControlState.CHECKED for item in cited_controls
            )
            if any(item.state == ControlState.AMBIGUOUS for item in cited_controls):
                raise GroundingError("ambiguous_control_evidence")
            if bool(proposal.raw_value) != expected:
                raise GroundingError("boolean_conflicts_with_control")
        elif proposal.raw_value is None or not _text_supports_value(
            str(proposal.raw_value), cited_text
        ):
            raise GroundingError("value_not_supported_by_cited_text")

        normalized = normalize_value(
            proposal.raw_value, field_spec, blueprint.normalization
        )
        cited_records = [records[item] for item in proposal.evidence_ids]
        support_sources = list(dict.fromkeys(record.source for record in cited_records))
        if visual_verification and any(
            record.kind == EvidenceKind.CROP for record in cited_records
        ):
            support_sources.append("local_vlm_visual")
        recognition = min((record.confidence for record in cited_records), default=0.0)
        return FieldCandidate(
            path=proposal.path,
            raw_value=proposal.raw_value,
            normalized_value=normalized.value,
            evidence_ids=proposal.evidence_ids,
            source="vlm_visual" if visual_verification else "vlm_grounded",
            support_sources=list(dict.fromkeys(support_sources)),
            recognition_confidence=recognition,
            association_confidence=0.92,
            attempt=attempt,
            validation_codes=normalized.codes,
        )


def _text_supports_value(value: str, evidence_text: str) -> bool:
    wanted = canonical_text(value)
    available = canonical_text(evidence_text)
    if not wanted:
        return False
    if wanted in available:
        return True
    tokens = [token for token in wanted.split() if len(token) > 1]
    return bool(tokens) and all(token in available.split() for token in tokens)
