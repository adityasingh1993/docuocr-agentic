from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part[:1].upper() + part[1:] for part in rest)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="forbid",
    )


class SampleCollection(CamelModel):
    date: str | None = None
    time: str | None = None


class BabySample(CamelModel):
    collection: SampleCollection = Field(default_factory=SampleCollection)


class BabyData(CamelModel):
    birth_time: str | None = None
    birth_weight: str | None = None
    date_of_birth: str | None = None
    female: bool | None = None
    first_name: str | None = None
    last_name: str | None = None
    male: bool | None = None
    medication: str | None = None
    nationality: str | None = None
    parenteral: str | bool | None = None
    register_number: str | None = None
    sample: BabySample = Field(default_factory=BabySample)

    @field_validator("first_name", "last_name", "nationality", mode="before")
    @classmethod
    def do_not_silently_blank_text(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PayerData(CamelModel):
    health_insurance: str | bool | None = None


class ExtractionData(CamelModel):
    baby: BabyData = Field(default_factory=BabyData)
    baby_consentfor_all_and_ngs_tracking: bool | None = None
    baby_first_time_screening: bool | None = None
    baby_hearing_screen_performed: bool | None = None
    mother: dict[str, Any] = Field(default_factory=dict)
    payer: PayerData = Field(default_factory=PayerData)
    serial_number: str | None = None


class PreSampleCollection(CamelModel):
    date: str | None = None
    time: str | None = None


class PreSample(CamelModel):
    collection: PreSampleCollection = Field(default_factory=PreSampleCollection)


class PreBaby(CamelModel):
    birth_time: str | None = None
    date_of_birth: str | None = None
    sample: PreSample = Field(default_factory=PreSample)


class PreExtraction(CamelModel):
    baby: PreBaby = Field(default_factory=PreBaby)


class Identifiers(CamelModel):
    serial_number: str | None = None


class Job(CamelModel):
    id: str


class ProcessorStep(CamelModel):
    name: str
    status: Literal["succeeded", "failed", "review", "skipped"]
    attempt: int = 0
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False
    started_at: str
    ended_at: str
    duration_ms: float = Field(ge=0.0)
    input_sha256: str | None = None
    output_sha256: str | None = None
    model_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FieldDecisionMeta(CamelModel):
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated: bool = False
    disposition: Literal["accepted", "retry", "review", "missing"]
    attempts: int = Field(default=0, ge=0)
    evidence_refs: list[str] = Field(default_factory=list)
    validation_codes: list[str] = Field(default_factory=list)
    source: str | None = None


class ProcessorBundle(CamelModel):
    steps: list[ProcessorStep] = Field(default_factory=list)
    fields: dict[str, FieldDecisionMeta] = Field(default_factory=dict)
    evidence_manifest: str | None = None
    trace_manifest: str | None = None


class Timing(CamelModel):
    create_document_blueprint: float = Field(default=0.0, ge=0.0)
    extraction: float = Field(default=0.0, ge=0.0)
    get_extraction_prompt: float = Field(default=0.0, ge=0.0)
    total: float = Field(default=0.0, ge=0.0)
    steps: dict[str, float] = Field(default_factory=dict)


class ExtractionMeta(CamelModel):
    identifiers: Identifiers = Field(default_factory=Identifiers)
    job: Job
    pre: PreExtraction = Field(default_factory=PreExtraction)
    processors: ProcessorBundle = Field(default_factory=ProcessorBundle)
    timing: Timing = Field(default_factory=Timing)


class ExtractionEnvelope(CamelModel):
    data: ExtractionData = Field(default_factory=ExtractionData)
    meta: ExtractionMeta

    def as_public_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")
