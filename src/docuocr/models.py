from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BBox(StrictModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_extents(self) -> BBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def expanded(self, pad_x: int, pad_y: int, width: int, height: int) -> BBox:
        return BBox(
            x1=max(0, self.x1 - pad_x),
            y1=max(0, self.y1 - pad_y),
            x2=min(width, self.x2 + pad_x),
            y2=min(height, self.y2 + pad_y),
        )

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class OCRSpan(StrictModel):
    id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    page: int = Field(default=1, ge=1)
    script: str | None = None
    source: str = "paddleocr"
    attempt: int = Field(default=0, ge=0)


class LayoutBlock(StrictModel):
    id: str
    label: str
    content: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: BBox
    page: int = Field(default=1, ge=1)
    order: int | None = None
    source: str = "paddleocr-vl"


class ControlKind(StrEnum):
    CHECKBOX = "checkbox"
    RADIO = "radio"


class ControlState(StrEnum):
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    AMBIGUOUS = "ambiguous"


class FormControl(StrictModel):
    id: str
    kind: ControlKind
    state: ControlState
    state_confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    page: int = Field(default=1, ge=1)
    ink_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    associated_label_id: str | None = None
    source: str = "opencv"
    attempt: int = Field(default=0, ge=0)


class EvidenceKind(StrEnum):
    OCR = "ocr"
    LAYOUT = "layout"
    CONTROL = "control"
    CROP = "crop"


class EvidenceRecord(StrictModel):
    id: str
    kind: EvidenceKind
    page: int = Field(default=1, ge=1)
    bbox: BBox | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    text_sha256: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    transform_chain: list[str] = Field(default_factory=list)


class QualityReport(StrictModel):
    overall: float = Field(ge=0.0, le=1.0)
    resolution: float = Field(ge=0.0, le=1.0)
    sharpness: float = Field(ge=0.0, le=1.0)
    contrast: float = Field(ge=0.0, le=1.0)
    illumination: float = Field(ge=0.0, le=1.0)
    glare: float = Field(ge=0.0, le=1.0)
    skew: float = Field(ge=0.0, le=1.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    estimated_skew_degrees: float = 0.0
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class EnhancementRecord(StrictModel):
    strategy: str
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    target_bbox: BBox | None = None
    attempt: int = Field(ge=1)


class FieldCandidate(StrictModel):
    path: str
    raw_value: str | bool | int | float | None = None
    normalized_value: str | bool | int | float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source: str
    support_sources: list[str] = Field(default_factory=list)
    recognition_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    association_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    attempt: int = Field(default=0, ge=0)
    validation_codes: list[str] = Field(default_factory=list)

    def value_fingerprint(self) -> str:
        payload = json.dumps(self.normalized_value, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConfidenceFeatures(StrictModel):
    image_quality: float = Field(ge=0.0, le=1.0)
    recognition: float = Field(ge=0.0, le=1.0)
    association: float = Field(ge=0.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    validation: float = Field(ge=0.0, le=1.0)
    grounded: bool
    independent_sources: int = Field(default=1, ge=0)
    conflict: bool = False


class FieldDecision(StrictModel):
    path: str
    candidate: FieldCandidate | None = None
    score: float = Field(ge=0.0, le=1.0)
    calibrated: bool = False
    disposition: Literal["accepted", "retry", "review", "missing"]
    features: ConfidenceFeatures | None = None
    reasons: list[str] = Field(default_factory=list)


class RecoveryAction(StrEnum):
    UPSCALE = "upscale"
    CLAHE = "clahe"
    DESKEW = "deskew"
    DENOISE_SHARPEN = "denoise_sharpen"
    ADAPTIVE_BINARIZE = "adaptive_binarize"
    CHECKBOX_FOCUS = "checkbox_focus"
    REVIEW = "review"


class RecoveryPlan(StrictModel):
    field_path: str
    actions: list[RecoveryAction]
    bbox: BBox | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    attempt: int = Field(ge=1)
    reason_codes: list[str] = Field(default_factory=list)
