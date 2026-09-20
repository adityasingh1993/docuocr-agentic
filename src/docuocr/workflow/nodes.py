from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docuocr.config import AppSettings
from docuocr.contract import (
    ExtractionEnvelope,
    FieldDecisionMeta,
    ProcessorBundle,
    ProcessorStep,
)
from docuocr.cv.enhance import ImageEnhancer
from docuocr.cv.quality import ImageQualityAssessor
from docuocr.engines.base import ControlEngine, LayoutEngine, TextEngine
from docuocr.engines.vlm import LocalVLMClient
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.confidence import ConfidenceScorer
from docuocr.extraction.grounding import GroundingError, GroundingVerifier
from docuocr.extraction.rules import associate_controls, map_rule_candidates
from docuocr.extraction.validation import validate_candidates
from docuocr.models import (
    BBox,
    EvidenceKind,
    EvidenceRecord,
    FieldCandidate,
    FieldDecision,
    FormControl,
    LayoutBlock,
    OCRSpan,
    QualityReport,
    RecoveryAction,
    RecoveryPlan,
)
from docuocr.state import DocumentState
from docuocr.trace import TraceWriter, sha256_file, sha256_json, utc_now


class WorkflowNodes:
    def __init__(
        self,
        *,
        settings: AppSettings,
        blueprint: DocumentBlueprint,
        text_engine: TextEngine,
        layout_engine: LayoutEngine,
        control_engine: ControlEngine,
        vlm: LocalVLMClient | None,
    ) -> None:
        self.settings = settings
        self.blueprint = blueprint
        self.text_engine = text_engine
        self.layout_engine = layout_engine
        self.control_engine = control_engine
        self.vlm = vlm
        self.quality_assessor = ImageQualityAssessor()
        self.enhancer = ImageEnhancer()
        self.scorer = ConfidenceScorer(settings.policy.accept_threshold)
        self.grounding = GroundingVerifier()

    def ingest(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            source = Path(state["source_path"]).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            job_id = state["job_id"]
            run_dir = (self.settings.artifact_path() / job_id).resolve()
            if self.settings.artifact_path() not in run_dir.parents:
                raise ValueError("job_id escapes the artifact root")
            (run_dir / "images").mkdir(parents=True, exist_ok=True)
            original_sha = sha256_file(source)
            reference = {"sourcePath": str(source), "sha256": original_sha}
            (run_dir / "images" / "original-reference.json").write_text(
                json.dumps(reference, indent=2), encoding="utf-8"
            )
            return {
                "source_path": str(source),
                "active_image_path": str(source),
                "run_dir": str(run_dir),
                "original_sha256": original_sha,
                "blueprint_id": self.blueprint.id,
                "started_at": utc_now(),
                "document_attempt": 0,
                "field_attempt": 0,
                "errors": [],
                "warnings": [],
                "timings": {},
                "_stage_confidence": 1.0,
            }

        return self._execute("ingest", state, 0, work)

    def assess_quality(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            report = self.quality_assessor.assess(state["active_image_path"])
            warnings = []
            if report.overall < self.settings.policy.document_quality_threshold:
                warnings.append("document_quality_below_policy")
            route = (
                "enhance"
                if report.overall < self.settings.policy.document_quality_threshold
                and attempt < self.settings.policy.max_document_enhancements
                else "extract"
            )
            return {
                "quality": report.model_dump(mode="json"),
                "quality_history": [report.model_dump(mode="json")],
                "warnings": warnings,
                "_stage_confidence": report.overall,
                "_decision": route,
            }

        return self._execute(
            "assess_quality", state, attempt, work, model_id="opencv-quality-v1"
        )

    def enhance_document(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0) + 1

        def work() -> dict[str, Any]:
            quality = QualityReport.model_validate(state["quality"])
            output = (
                Path(state["run_dir"]) / "images" / f"document-enhanced-{attempt}.png"
            )
            record = self.enhancer.apply(
                state["active_image_path"],
                output,
                self.enhancer.plan_for_quality(quality),
                attempt=attempt,
                skew_degrees=quality.estimated_skew_degrees,
            )
            return {
                "active_image_path": str(output),
                "document_attempt": attempt,
                "enhancements": [record.model_dump(mode="json")],
                "_stage_confidence": 1.0,
            }

        return self._execute(
            "enhance_document", state, attempt, work, model_id="opencv-enhance-v1"
        )

    def extraction_start(self, state: DocumentState) -> dict[str, Any]:
        return self._execute(
            "extraction_start", state, 0, lambda: {"_stage_confidence": 1.0}
        )

    def layout(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            blocks = self.layout_engine.parse(state["active_image_path"])
            evidence = [_layout_evidence(item) for item in blocks]
            return {
                "layout_blocks": [item.model_dump(mode="json") for item in blocks],
                "layout_ledger": [item.model_dump(mode="json") for item in blocks],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "_stage_confidence": _mean([item.confidence for item in blocks]),
            }

        return self._execute(
            "layout", state, 0, work, model_id=self.layout_engine.model_id
        )

    def ocr(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            spans = self.text_engine.extract(state["active_image_path"])
            evidence = [_ocr_evidence(item) for item in spans]
            return {
                "ocr_spans": [item.model_dump(mode="json") for item in spans],
                "ocr_ledger": [item.model_dump(mode="json") for item in spans],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "_stage_confidence": _mean([item.confidence for item in spans]),
            }

        return self._execute("ocr", state, 0, work, model_id=self.text_engine.model_id)

    def controls(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            controls = self.control_engine.detect(state["active_image_path"])
            evidence = [_control_evidence(item) for item in controls]
            return {
                "controls": [item.model_dump(mode="json") for item in controls],
                "control_ledger": [item.model_dump(mode="json") for item in controls],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "_stage_confidence": _mean(
                    [item.state_confidence for item in controls]
                ),
            }

        return self._execute(
            "controls", state, 0, work, model_id=self.control_engine.model_id
        )

    def map_rules(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            spans = _models(OCRSpan, state.get("ocr_spans", []))
            controls = associate_controls(
                _models(FormControl, state.get("controls", [])), spans
            )
            candidates = map_rule_candidates(self.blueprint, spans, controls)
            covered = {
                item.path for item in candidates if item.normalized_value is not None
            }
            return {
                "controls": [item.model_dump(mode="json") for item in controls],
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "_stage_confidence": len(covered)
                / max(1, len(self.blueprint.output_paths)),
            }

        return self._execute(
            "map_rules", state, 0, work, model_id=f"blueprint:{self.blueprint.id}"
        )

    def vlm_map(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            candidates = _models(FieldCandidate, state.get("candidates", []))
            if self.vlm is None:
                return {
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "_decision": "disabled",
                }
            present = {
                item.path for item in candidates if item.normalized_value is not None
            }
            target_paths = [
                path for path in self.blueprint.output_paths if path not in present
            ]
            if not target_paths:
                return {
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "_stage_confidence": 1.0,
                    "_decision": "no_unresolved_mapping",
                }
            additions, warnings = self._vlm_candidates(
                image_path=state["active_image_path"],
                target_paths=target_paths,
                spans=_models(OCRSpan, state.get("ocr_spans", [])),
                blocks=_models(LayoutBlock, state.get("layout_blocks", [])),
                controls=_models(FormControl, state.get("controls", [])),
                evidence=_models(EvidenceRecord, state.get("evidence", [])),
                attempt=0,
                visual_verification=False,
            )
            return {
                "candidates": [
                    item.model_dump(mode="json") for item in candidates + additions
                ],
                "warnings": warnings,
                "_stage_confidence": len({item.path for item in additions})
                / max(1, len(target_paths)),
            }

        return self._execute(
            "vlm_map",
            state,
            0,
            work,
            model_id=self.vlm.model_id if self.vlm else "disabled",
        )

    def score(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("field_attempt", 0)

        def work() -> dict[str, Any]:
            candidates = validate_candidates(
                _models(FieldCandidate, state.get("candidates", [])), self.blueprint
            )
            quality = QualityReport.model_validate(state["quality"])
            decisions = self.scorer.decide(
                candidates,
                self.blueprint,
                image_quality=quality.overall,
                attempt=attempt,
                max_retries=self.settings.policy.max_field_retries,
            )
            evidence = _models(EvidenceRecord, state.get("evidence", []))
            plans = self.scorer.recovery_plans(decisions, evidence, attempt=attempt + 1)
            unresolved = [
                path
                for path, decision in decisions.items()
                if decision.disposition != "accepted"
            ]
            route = (
                "recover"
                if any(item.disposition == "retry" for item in decisions.values())
                else (
                    "review"
                    if any(
                        item.disposition != "accepted" for item in decisions.values()
                    )
                    else "complete"
                )
            )
            return {
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "decisions": {
                    path: decision.model_dump(mode="json")
                    for path, decision in decisions.items()
                },
                "unresolved_paths": unresolved,
                "recovery_plans": [item.model_dump(mode="json") for item in plans],
                "_stage_confidence": min(
                    (item.score for item in decisions.values()), default=0.0
                ),
                "_decision": route,
            }

        return self._execute(
            "score", state, attempt, work, model_id="confidence-policy-v1"
        )

    def recover(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("field_attempt", 0) + 1

        def work() -> dict[str, Any]:
            plans = _models(RecoveryPlan, state.get("recovery_plans", []))
            existing = _models(FieldCandidate, state.get("candidates", []))
            if not plans:
                return {
                    "field_attempt": attempt,
                    "candidates": state.get("candidates", []),
                }
            if attempt == 1:
                recovered = self._recover_page(state, plans, attempt)
            else:
                recovered = self._recover_crops(state, plans, attempt)
            return {
                "field_attempt": attempt,
                "active_image_path": recovered["active_image_path"],
                "quality": recovered["quality"],
                "quality_history": [recovered["quality"]],
                "enhancements": recovered["enhancements"],
                "evidence": recovered["evidence"],
                "ocr_ledger": recovered["ocr_ledger"],
                "control_ledger": recovered["control_ledger"],
                "candidates": [
                    item.model_dump(mode="json")
                    for item in existing + recovered["candidates"]
                ],
                "warnings": recovered["warnings"],
                "_decision": "rescore",
            }

        return self._execute(
            "recover", state, attempt, work, model_id="opencv+paddle+local-vlm"
        )

    def review(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("field_attempt", 0)

        def work() -> dict[str, Any]:
            if self.settings.policy.review_mode == "interrupt":
                from langgraph.types import interrupt

                response = interrupt(
                    {
                        "jobId": state["job_id"],
                        "unresolvedPaths": state.get("unresolved_paths", []),
                        "decisions": state.get("decisions", {}),
                    }
                )
                corrections = (
                    response.get("corrections", {})
                    if isinstance(response, dict)
                    else {}
                )
                corrections = {
                    path: value
                    for path, value in corrections.items()
                    if path in self.blueprint.output_paths
                }
                return {"human_corrections": corrections, "review_required": False}
            return {
                "review_required": True,
                "_stage_confidence": 0.0,
                "_decision": "queued",
            }

        return self._execute("review", state, attempt, work, decision="human_review")

    def assemble(self, state: DocumentState) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            decisions = {
                path: FieldDecision.model_validate(value)
                for path, value in state.get("decisions", {}).items()
            }
            data: dict[str, Any] = {}
            for path, decision in decisions.items():
                if (
                    decision.disposition == "accepted"
                    and decision.candidate is not None
                ):
                    _set_path(
                        data,
                        path.removeprefix("data."),
                        decision.candidate.normalized_value,
                    )
            for path, value in state.get("human_corrections", {}).items():
                _set_path(data, path.removeprefix("data."), value)

            raw_values = {
                path: decision.candidate.raw_value
                for path, decision in decisions.items()
                if decision.candidate is not None
            }
            pre: dict[str, Any] = {}
            pre_mapping = {
                "data.baby.birthTime": "baby.birthTime",
                "data.baby.dateOfBirth": "baby.dateOfBirth",
                "data.baby.sample.collection.date": "baby.sample.collection.date",
                "data.baby.sample.collection.time": "baby.sample.collection.time",
            }
            for source, destination in pre_mapping.items():
                if source in raw_values:
                    _set_path(pre, destination, raw_values[source])

            field_meta = {
                path: FieldDecisionMeta(
                    confidence=decision.score,
                    calibrated=decision.calibrated,
                    disposition=decision.disposition,
                    attempts=state.get("field_attempt", 0),
                    evidence_refs=(
                        decision.candidate.evidence_ids if decision.candidate else []
                    ),
                    validation_codes=(
                        decision.candidate.validation_codes
                        if decision.candidate
                        else []
                    ),
                    source=decision.candidate.source if decision.candidate else None,
                )
                for path, decision in decisions.items()
            }
            run_dir = Path(state["run_dir"])
            evidence_path = run_dir / "evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "originalSha256": state.get("original_sha256"),
                        "records": state.get("evidence", []),
                        "ocrSpans": state.get("ocr_ledger", []),
                        "layoutBlocks": state.get("layout_ledger", []),
                        "controls": state.get("control_ledger", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            processor_steps = [
                ProcessorStep.model_validate(item)
                for item in state.get("processor_steps", [])
            ]
            timings = state.get("timings", {})
            envelope = ExtractionEnvelope.model_validate(
                {
                    "data": data,
                    "meta": {
                        "identifiers": {
                            "serialNumber": _get_path(data, "serialNumber")
                        },
                        "job": {"id": state["job_id"]},
                        "pre": pre,
                        "processors": ProcessorBundle(
                            steps=processor_steps,
                            fields=field_meta,
                            evidence_manifest=str(evidence_path),
                            trace_manifest=str(run_dir / "trace.jsonl"),
                        ).model_dump(mode="json"),
                        "timing": {
                            "createDocumentBlueprint": timings.get("map_rules", 0.0),
                            "extraction": sum(timings.values()),
                            "getExtractionPrompt": timings.get("vlm_map", 0.0),
                            "total": sum(timings.values()),
                            "steps": timings,
                        },
                    },
                }
            )
            return {"result": envelope.as_public_dict(), "_stage_confidence": 1.0}

        return self._execute("assemble", state, state.get("field_attempt", 0), work)

    def _recover_page(
        self, state: DocumentState, plans: list[RecoveryPlan], attempt: int
    ) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        output = run_dir / "images" / f"recovery-page-{attempt}.png"
        record = self.enhancer.apply(
            state["active_image_path"],
            output,
            [RecoveryAction.CLAHE, RecoveryAction.DENOISE_SHARPEN],
            attempt=attempt,
        )
        quality = self.quality_assessor.assess(output)
        target_paths = [item.field_path for item in plans]
        prefix = f"retry{attempt}"
        spans = self.text_engine.extract(
            output, attempt=attempt, id_prefix=f"ocr:{prefix}"
        )
        controls = self.control_engine.detect(
            output, attempt=attempt, id_prefix=f"control:{prefix}"
        )
        controls = associate_controls(controls, spans)
        candidates = map_rule_candidates(
            self.blueprint,
            spans,
            controls,
            attempt=attempt,
            only_paths=set(target_paths),
        )
        image_evidence = EvidenceRecord(
            id=f"crop:{prefix}:page",
            kind=EvidenceKind.CROP,
            bbox=BBox(x1=0, y1=0, x2=quality.width, y2=quality.height),
            confidence=quality.overall,
            source="opencv_enhancement",
            artifact_path=str(output),
            artifact_sha256=record.output_sha256,
            transform_chain=[record.strategy],
        )
        evidence = [
            *[_ocr_evidence(item) for item in spans],
            *[_control_evidence(item) for item in controls],
            image_evidence,
        ]
        warnings: list[str] = []
        if self.vlm is not None:
            additions, vlm_warnings = self._vlm_candidates(
                image_path=output,
                target_paths=target_paths,
                spans=spans,
                blocks=[],
                controls=controls,
                evidence=evidence,
                attempt=attempt,
                visual_verification=True,
                image_evidence_id=image_evidence.id,
            )
            candidates.extend(additions)
            warnings.extend(vlm_warnings)
        return {
            "active_image_path": str(output),
            "quality": quality.model_dump(mode="json"),
            "enhancements": [record.model_dump(mode="json")],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "ocr_ledger": [item.model_dump(mode="json") for item in spans],
            "control_ledger": [item.model_dump(mode="json") for item in controls],
            "candidates": candidates,
            "warnings": warnings,
        }

    def _recover_crops(
        self, state: DocumentState, plans: list[RecoveryPlan], attempt: int
    ) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        all_candidates: list[FieldCandidate] = []
        all_evidence: list[EvidenceRecord] = []
        all_records: list[dict[str, Any]] = []
        ocr_ledger: list[dict[str, Any]] = []
        control_ledger: list[dict[str, Any]] = []
        warnings: list[str] = []
        for index, plan in enumerate(plans):
            output = run_dir / "images" / f"recovery-{attempt}-{index:03d}.png"
            record = self.enhancer.apply(
                state["active_image_path"],
                output,
                plan.actions,
                attempt=attempt,
                bbox=plan.bbox,
            )
            prefix = f"retry{attempt}:{index:03d}"
            spans = self.text_engine.extract(
                output, attempt=attempt, id_prefix=f"ocr:{prefix}"
            )
            controls = self.control_engine.detect(
                output, attempt=attempt, id_prefix=f"control:{prefix}"
            )
            controls = associate_controls(controls, spans)
            candidates = map_rule_candidates(
                self.blueprint,
                spans,
                controls,
                attempt=attempt,
                only_paths={plan.field_path},
            )
            crop_evidence = EvidenceRecord(
                id=f"crop:{prefix}",
                kind=EvidenceKind.CROP,
                bbox=plan.bbox,
                confidence=QualityReport.model_validate(state["quality"]).overall,
                source="opencv_enhancement",
                artifact_path=str(output),
                artifact_sha256=record.output_sha256,
                transform_chain=[record.strategy],
            )
            evidence = [
                *[_ocr_evidence(item) for item in spans],
                *[_control_evidence(item) for item in controls],
                crop_evidence,
            ]
            if self.vlm is not None:
                additions, vlm_warnings = self._vlm_candidates(
                    image_path=output,
                    target_paths=[plan.field_path],
                    spans=spans,
                    blocks=[],
                    controls=controls,
                    evidence=evidence,
                    attempt=attempt,
                    visual_verification=True,
                    image_evidence_id=crop_evidence.id,
                )
                candidates.extend(additions)
                warnings.extend(vlm_warnings)
            all_candidates.extend(candidates)
            all_evidence.extend(evidence)
            ocr_ledger.extend(item.model_dump(mode="json") for item in spans)
            control_ledger.extend(item.model_dump(mode="json") for item in controls)
            all_records.append(record.model_dump(mode="json"))
        return {
            "active_image_path": state["active_image_path"],
            "quality": state["quality"],
            "enhancements": all_records,
            "evidence": [item.model_dump(mode="json") for item in all_evidence],
            "ocr_ledger": ocr_ledger,
            "control_ledger": control_ledger,
            "candidates": all_candidates,
            "warnings": warnings,
        }

    def _vlm_candidates(
        self,
        *,
        image_path: str | Path,
        target_paths: list[str],
        spans: list[OCRSpan],
        blocks: list[LayoutBlock],
        controls: list[FormControl],
        evidence: list[EvidenceRecord],
        attempt: int,
        visual_verification: bool,
        image_evidence_id: str | None = None,
    ) -> tuple[list[FieldCandidate], list[str]]:
        if self.vlm is None:
            return [], []
        proposals = self.vlm.propose(
            image_path=image_path,
            target_paths=target_paths,
            spans=spans,
            blocks=blocks,
            controls=controls,
            image_evidence_id=image_evidence_id,
        )
        candidates: list[FieldCandidate] = []
        warnings: list[str] = []
        for proposal in proposals:
            try:
                candidates.append(
                    self.grounding.verify(
                        proposal,
                        self.blueprint,
                        evidence,
                        spans,
                        blocks,
                        controls,
                        attempt=attempt,
                        visual_verification=visual_verification,
                    )
                )
            except GroundingError as exc:
                warnings.append(f"rejected_vlm_proposal:{proposal.path}:{exc}")
        return candidates, warnings

    def _execute(
        self,
        name: str,
        state: DocumentState,
        attempt: int,
        work: Callable[[], dict[str, Any]],
        *,
        model_id: str | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now()
        started = time.perf_counter()
        input_hash = state.get("original_sha256")
        if (
            input_hash is None
            and state.get("source_path")
            and Path(state["source_path"]).is_file()
        ):
            input_hash = sha256_file(state["source_path"])
        try:
            update = work()
            stage_confidence = update.pop("_stage_confidence", None)
            node_decision = update.pop("_decision", decision)
            if model_id == "disabled":
                status = "skipped"
            else:
                status = "review" if name == "review" else "succeeded"
        except Exception as exc:
            ended_at = utc_now()
            duration = (time.perf_counter() - started) * 1000.0
            run_dir = state.get("run_dir")
            if run_dir:
                TraceWriter(
                    Path(run_dir) / "trace.jsonl", self.settings.trace.include_values
                ).append(
                    run_id=state["job_id"],
                    node=name,
                    attempt=attempt,
                    status="failed",
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=duration,
                    input_sha256=input_hash,
                    model_id=model_id,
                    warnings=[f"{type(exc).__name__}:{exc}"],
                )
            raise
        ended_at = utc_now()
        duration = (time.perf_counter() - started) * 1000.0
        run_dir = update.get("run_dir") or state.get("run_dir")
        evidence_refs = [
            item.get("id")
            for item in update.get("evidence", [])
            if isinstance(item, dict) and item.get("id")
        ]
        confidence = stage_confidence
        if isinstance(update.get("quality"), dict):
            confidence = update["quality"].get("overall")
        step = ProcessorStep(
            name=name,
            status=status,
            attempt=attempt,
            confidence=confidence,
            calibrated=False,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration,
            input_sha256=input_hash,
            output_sha256=sha256_json(update),
            model_id=model_id,
            evidence_refs=evidence_refs,
            decisions=[node_decision] if node_decision else [],
            warnings=update.get("warnings", []),
        )
        update["processor_steps"] = [step.model_dump(mode="json")]
        update["timings"] = {name: round(duration, 3)}
        if run_dir:
            TraceWriter(
                Path(run_dir) / "trace.jsonl", self.settings.trace.include_values
            ).append(
                run_id=state["job_id"],
                node=name,
                attempt=attempt,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration,
                input_sha256=input_hash,
                output=update,
                confidence=confidence,
                decision=node_decision,
                model_id=model_id,
                evidence_refs=evidence_refs,
                warnings=update.get("warnings", []),
            )
        if "result" in update:
            result = update["result"]
            result["meta"]["processors"]["steps"].append(
                step.model_dump(by_alias=True, mode="json")
            )
            result["meta"]["timing"]["steps"][name] = round(duration, 3)
            total = sum(result["meta"]["timing"]["steps"].values())
            result["meta"]["timing"]["total"] = round(total, 3)
            result["meta"]["timing"]["extraction"] = round(total, 3)
            Path(run_dir, "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return update


def _ocr_evidence(item: OCRSpan) -> EvidenceRecord:
    return EvidenceRecord(
        id=item.id,
        kind=EvidenceKind.OCR,
        page=item.page,
        bbox=item.bbox,
        confidence=item.confidence,
        source=item.source,
        text_sha256=hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
    )


def _layout_evidence(item: LayoutBlock) -> EvidenceRecord:
    return EvidenceRecord(
        id=item.id,
        kind=EvidenceKind.LAYOUT,
        page=item.page,
        bbox=item.bbox,
        confidence=item.confidence,
        source=item.source,
        text_sha256=hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
    )


def _control_evidence(item: FormControl) -> EvidenceRecord:
    return EvidenceRecord(
        id=item.id,
        kind=EvidenceKind.CONTROL,
        page=item.page,
        bbox=item.bbox,
        confidence=item.state_confidence,
        source=item.source,
    )


def _models(model: Any, values: list[dict[str, Any]]) -> list[Any]:
    return [model.model_validate(value) for value in values]


def _set_path(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = root
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _get_path(root: dict[str, Any], dotted: str) -> Any:
    current: Any = root
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
