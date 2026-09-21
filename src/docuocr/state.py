from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class DocumentState(TypedDict, total=False):
    job_id: str
    thread_id: str
    source_path: str
    original_image_path: str
    active_image_path: str
    run_dir: str
    original_sha256: str
    blueprint_path: str
    blueprint_id: str
    started_at: str
    document_attempt: int
    field_attempt: int
    evidence_reverification_attempt: int
    quality: dict[str, Any]
    quality_history: Annotated[list[dict[str, Any]], operator.add]
    enhancements: Annotated[list[dict[str, Any]], operator.add]
    enhancement_evaluations: Annotated[list[dict[str, Any]], operator.add]
    enhancement_selected: bool
    document_understanding: dict[str, Any]
    document_understanding_attempted: bool
    document_understanding_image_sha256: str
    layout_blocks: list[dict[str, Any]]
    layout_visualizations: Annotated[list[dict[str, Any]], operator.add]
    layout_block_ocr: Annotated[list[dict[str, Any]], operator.add]
    layout_extractions: Annotated[list[dict[str, Any]], operator.add]
    ocr_spans: list[dict[str, Any]]
    controls: list[dict[str, Any]]
    layout_ledger: Annotated[list[dict[str, Any]], operator.add]
    ocr_ledger: Annotated[list[dict[str, Any]], operator.add]
    control_ledger: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    candidates: list[dict[str, Any]]
    decisions: dict[str, dict[str, Any]]
    unresolved_paths: list[str]
    recovery_plans: list[dict[str, Any]]
    evidence_reverifications: Annotated[list[dict[str, Any]], operator.add]
    human_corrections: dict[str, Any]
    review_required: bool
    processor_steps: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], merge_dicts]
    result: dict[str, Any]


class GraphInput(TypedDict):
    source_path: str
    blueprint_path: str
    job_id: str


class GraphOutput(TypedDict, total=False):
    job_id: str
    review_required: bool
    result: dict[str, Any]
    errors: list[str]
