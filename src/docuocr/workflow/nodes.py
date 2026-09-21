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
from docuocr.cv.layout_visualization import LayoutVisualizer
from docuocr.cv.quality import (
    ImageQualityAssessor,
    assess_ocr_readiness,
    enrich_quality_report,
    evaluate_enhancement,
)
from docuocr.engines.base import ControlEngine, LayoutEngine, TextEngine
from docuocr.engines.vlm import LocalVLMClient
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.extraction.confidence import ConfidenceScorer
from docuocr.extraction.grounding import GroundingError, GroundingVerifier
from docuocr.extraction.rules import associate_controls, map_rule_candidates
from docuocr.extraction.validation import validate_candidates
from docuocr.models import (
    BBox,
    DocumentUnderstanding,
    EnhancementEvaluation,
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
    SchemaMatch,
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
        self.layout_visualizer = LayoutVisualizer()
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
                "original_image_path": str(source),
                "active_image_path": str(source),
                "run_dir": str(run_dir),
                "original_sha256": original_sha,
                "blueprint_id": self.blueprint.id,
                "started_at": utc_now(),
                "document_attempt": 0,
                "document_understanding_attempted": False,
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
            return {
                "quality": report.model_dump(mode="json"),
                "quality_history": [report.model_dump(mode="json")],
                "warnings": warnings,
                "_stage_confidence": report.overall,
                "_decision": "acquire_baseline_evidence",
            }

        return self._execute(
            "assess_quality", state, attempt, work, model_id="opencv-quality-v1"
        )

    def enhance_document(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0) + 1

        def work() -> dict[str, Any]:
            baseline_quality = QualityReport.model_validate(state["quality"])
            baseline_spans = _models(OCRSpan, state.get("ocr_spans", []))
            baseline_controls = _models(FormControl, state.get("controls", []))
            understanding = _document_understanding(state)
            profiles = (
                understanding.recommended_profiles if understanding is not None else []
            )
            variants = self.enhancer.plan_variants(
                baseline_quality,
                profiles,
                max_variants=self.settings.policy.max_enhancement_variants,
            )
            warnings: list[str] = []
            if not variants:
                warnings.append("document_enhancement_no_safe_profile")
                return {
                    "active_image_path": state["active_image_path"],
                    "document_attempt": attempt,
                    "quality": baseline_quality.model_dump(mode="json"),
                    "enhancements": [],
                    "enhancement_evaluations": [],
                    "enhancement_selected": False,
                    "warnings": warnings,
                    "_stage_confidence": _effective_quality(baseline_quality),
                    "_decision": "original_retained",
                }

            evaluations: list[EnhancementEvaluation] = []
            records: list[dict[str, Any]] = []
            candidates: list[tuple[Path, QualityReport, float, int]] = []
            run_images = Path(state["run_dir"]) / "images"
            for index, actions in enumerate(variants):
                output = run_images / f"document-candidate-{attempt}-{index:02d}.png"
                strategy = "+".join(item.value for item in actions)
                try:
                    record = self.enhancer.apply(
                        state["active_image_path"],
                        output,
                        actions,
                        attempt=attempt,
                        skew_degrees=baseline_quality.estimated_skew_degrees,
                    )
                    records.append(record.model_dump(mode="json"))
                    candidate_quality = self.quality_assessor.assess(output)
                    candidate_spans = self.text_engine.extract(
                        output,
                        attempt=attempt,
                        id_prefix=f"ocr:probe{attempt}:{index:02d}",
                    )
                    candidate_controls = self.control_engine.detect(
                        output,
                        attempt=attempt,
                        id_prefix=f"control:probe{attempt}:{index:02d}",
                    )
                except Exception as exc:
                    warnings.append(
                        f"document_enhancement_probe_failed:{strategy}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    continue

                candidate_ocr = assess_ocr_readiness(candidate_spans)
                enriched_quality = enrich_quality_report(
                    candidate_quality,
                    ocr=candidate_ocr,
                    understanding=understanding,
                )
                evaluation = evaluate_enhancement(
                    strategy=strategy,
                    candidate_path=str(output),
                    baseline_quality=baseline_quality,
                    candidate_quality=candidate_quality,
                    baseline_spans=baseline_spans,
                    candidate_spans=candidate_spans,
                    baseline_control_count=len(baseline_controls),
                    candidate_control_count=len(candidate_controls),
                    min_ocr_gain=self.settings.policy.enhancement_min_ocr_gain,
                    min_text_retention=(
                        self.settings.policy.enhancement_min_text_retention
                    ),
                    min_control_retention=(
                        self.settings.policy.enhancement_min_control_retention
                    ),
                    ocr_engine_enabled=self.text_engine.model_id != "disabled",
                )
                evaluations.append(evaluation)
                candidates.append(
                    (
                        output,
                        enriched_quality,
                        evaluation.ocr_after.score,
                        len(evaluations) - 1,
                    )
                )

            eligible_candidates = [
                item for item in candidates if evaluations[item[3]].eligible
            ]
            selected = max(
                eligible_candidates,
                key=lambda item: (
                    item[2],
                    _effective_quality(item[1]),
                    item[1].overall,
                ),
                default=None,
            )
            if selected is None:
                active_image = state["active_image_path"]
                selected_quality = baseline_quality
                decision = "original_retained"
                warnings.append("document_enhancement_rejected_no_measured_gain")
                enhancement_selected = False
            else:
                active_image = str(selected[0])
                selected_quality = selected[1]
                selected_index = selected[3]
                evaluations[selected_index] = evaluations[selected_index].model_copy(
                    update={"selected": True, "reason": "selected_best_ocr_gain"}
                )
                decision = f"enhanced_selected:{evaluations[selected_index].strategy}"
                enhancement_selected = True
            return {
                "active_image_path": active_image,
                "document_attempt": attempt,
                "quality": selected_quality.model_dump(mode="json"),
                "quality_history": [
                    item[1].model_dump(mode="json") for item in candidates
                ],
                "enhancements": records,
                "enhancement_evaluations": [
                    item.model_dump(mode="json") for item in evaluations
                ],
                "enhancement_selected": enhancement_selected,
                "warnings": warnings,
                "_stage_confidence": _effective_quality(selected_quality),
                "_decision": decision,
            }

        return self._execute(
            "enhance_document",
            state,
            attempt,
            work,
            model_id="opencv+paddle-quality-selection-v1",
        )

    def extraction_start(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)
        return self._execute(
            "extraction_start",
            state,
            attempt,
            lambda: {"_stage_confidence": 1.0},
        )

    def layout(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            prefix = "layout" if attempt == 0 else f"layout:doc{attempt}"
            blocks = self.layout_engine.parse(
                state["active_image_path"], attempt=attempt, id_prefix=prefix
            )
            evidence = [_layout_evidence(item) for item in blocks]
            visualizations: list[dict[str, Any]] = []
            warnings: list[str] = []
            if (
                self.settings.trace.save_layout_images
                and self.layout_engine.model_id != "disabled"
            ):
                output_path = (
                    Path(state["run_dir"])
                    / "images"
                    / f"layout-detected-{attempt}.png"
                )
                try:
                    record = self.layout_visualizer.render(
                        state["active_image_path"],
                        blocks,
                        output_path,
                        attempt=attempt,
                    )
                    visualizations.append(record.model_dump(mode="json"))
                except Exception as exc:
                    warnings.append(
                        "layout_visualization_failed:"
                        f"{type(exc).__name__}:{exc}"
                    )
            return {
                "layout_blocks": [item.model_dump(mode="json") for item in blocks],
                "layout_ledger": [item.model_dump(mode="json") for item in blocks],
                "layout_visualizations": visualizations,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "warnings": warnings,
                "_stage_confidence": _mean([item.confidence for item in blocks]),
            }

        return self._execute(
            "layout", state, attempt, work, model_id=self.layout_engine.model_id
        )

    def ocr(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            prefix = "ocr" if attempt == 0 else f"ocr:doc{attempt}"
            spans = self.text_engine.extract(
                state["active_image_path"], attempt=attempt, id_prefix=prefix
            )
            evidence = [_ocr_evidence(item) for item in spans]
            return {
                "ocr_spans": [item.model_dump(mode="json") for item in spans],
                "ocr_ledger": [item.model_dump(mode="json") for item in spans],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "_stage_confidence": _mean([item.confidence for item in spans]),
            }

        return self._execute(
            "ocr", state, attempt, work, model_id=self.text_engine.model_id
        )

    def controls(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            prefix = "control" if attempt == 0 else f"control:doc{attempt}"
            controls = self.control_engine.detect(
                state["active_image_path"], attempt=attempt, id_prefix=prefix
            )
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
            "controls", state, attempt, work, model_id=self.control_engine.model_id
        )

    def document_understand(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            technical = QualityReport.model_validate(state["quality"])
            spans = _models(OCRSpan, state.get("ocr_spans", []))
            blocks = _models(LayoutBlock, state.get("layout_blocks", []))
            controls = _models(FormControl, state.get("controls", []))
            ocr_report = (
                None
                if self.text_engine.model_id == "disabled"
                else assess_ocr_readiness(spans)
            )
            understanding = _document_understanding(state)
            warnings: list[str] = []
            decision = "cached"
            analysis_sha = state.get("document_understanding_image_sha256")
            if not state.get("document_understanding_attempted", False):
                if (
                    self.vlm is not None
                    and self.settings.vlm.document_understanding_enabled
                ):
                    active_image = Path(state["active_image_path"])
                    result = self.vlm.understand(
                        image_path=active_image,
                        spans=spans,
                        blocks=blocks,
                        controls=controls,
                        quality=technical,
                        expected_document_type=self.blueprint.document_type,
                        anchors=self.blueprint.anchors,
                        field_hints=self.blueprint.vlm_hints(
                            self.blueprint.output_paths
                        ),
                    )
                    understanding = result.understanding
                    warnings.extend(result.warnings)
                    analysis_sha = sha256_file(active_image)
                    decision = "analyzed" if understanding is not None else "failed"
                    if understanding is not None:
                        analysis_path = (
                            Path(state["run_dir"]) / "document-understanding.json"
                        )
                        analysis_path.write_text(
                            json.dumps(
                                {
                                    "imageSha256": analysis_sha,
                                    "modelId": self.vlm.model_id,
                                    "analysis": understanding.model_dump(mode="json"),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                else:
                    decision = "disabled"

            enriched = enrich_quality_report(
                technical,
                ocr=ocr_report,
                understanding=understanding,
            )
            if (
                enriched.extraction_readiness is not None
                and enriched.extraction_readiness
                < self.settings.policy.document_quality_threshold
            ):
                warnings.append("extraction_readiness_below_policy")
            if (
                understanding is not None
                and understanding.schema_match == SchemaMatch.MISMATCH
            ):
                warnings.append("configured_blueprint_mismatch")
            update: dict[str, Any] = {
                "quality": enriched.model_dump(mode="json"),
                "quality_history": [enriched.model_dump(mode="json")],
                "document_understanding_attempted": True,
                "warnings": warnings,
                "_stage_confidence": _effective_quality(enriched),
                "_decision": decision,
            }
            if understanding is not None:
                update["document_understanding"] = understanding.model_dump(
                    mode="json"
                )
            if analysis_sha:
                update["document_understanding_image_sha256"] = analysis_sha
            return update

        return self._execute(
            "document_understand",
            state,
            attempt,
            work,
            model_id=(
                self.vlm.model_id
                if self.vlm is not None
                and self.settings.vlm.document_understanding_enabled
                else "deterministic-quality-fusion-v1"
            ),
        )

    def map_rules(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            spans = _models(OCRSpan, state.get("ocr_spans", []))
            controls = associate_controls(
                _models(FormControl, state.get("controls", [])), spans
            )
            candidates = map_rule_candidates(
                self.blueprint, spans, controls, attempt=attempt
            )
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
            "map_rules",
            state,
            attempt,
            work,
            model_id=f"blueprint:{self.blueprint.id}",
        )

    def vlm_map(self, state: DocumentState) -> dict[str, Any]:
        document_attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            candidates = _models(FieldCandidate, state.get("candidates", []))
            if self.vlm is None:
                return {
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "_decision": "disabled",
                }
            checkbox_paths = self.blueprint.checkbox_paths
            target_paths = _vlm_target_paths(self.blueprint, candidates)
            if not target_paths:
                return {
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "_stage_confidence": 1.0,
                    "_decision": "no_unresolved_mapping",
                }
            quality = QualityReport.model_validate(state["quality"])
            active_image = Path(state["active_image_path"])
            image_evidence = EvidenceRecord(
                id="image:vlm:p1:active",
                kind=EvidenceKind.IMAGE,
                bbox=BBox(x1=0, y1=0, x2=quality.width, y2=quality.height),
                confidence=_effective_quality(quality),
                source="document_image",
                artifact_path=str(active_image),
                artifact_sha256=sha256_file(active_image),
                transform_chain=["active_document"],
            )
            evidence = _models(EvidenceRecord, state.get("evidence", []))
            evidence.append(image_evidence)
            additions, warnings = self._vlm_candidates(
                image_path=active_image,
                target_paths=target_paths,
                spans=_models(OCRSpan, state.get("ocr_spans", [])),
                blocks=_models(LayoutBlock, state.get("layout_blocks", [])),
                controls=_models(FormControl, state.get("controls", [])),
                evidence=evidence,
                attempt=document_attempt,
                visual_verification=True,
                image_evidence_id=image_evidence.id,
                document_context=_document_understanding(state),
            )
            merged, reconciliation_warnings = _reconcile_visual_controls(
                candidates + additions, checkbox_paths
            )
            return {
                "candidates": [item.model_dump(mode="json") for item in merged],
                "evidence": [image_evidence.model_dump(mode="json")],
                "warnings": warnings + reconciliation_warnings,
                "_stage_confidence": len({item.path for item in additions})
                / max(1, len(target_paths)),
            }

        return self._execute(
            "vlm_map",
            state,
            document_attempt,
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
                image_quality=_effective_quality(quality),
                attempt=attempt,
                max_retries=self.settings.policy.max_field_retries,
            )
            understanding = _document_understanding(state)
            if (
                understanding is not None
                and understanding.schema_match == SchemaMatch.MISMATCH
            ):
                decisions = {
                    path: decision.model_copy(
                        update={
                            "score": min(decision.score, 0.49),
                            "disposition": "review",
                            "reasons": list(
                                dict.fromkeys(
                                    [*decision.reasons, "document_schema_mismatch"]
                                )
                            ),
                        }
                    )
                    for path, decision in decisions.items()
                }
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
            merged, reconciliation_warnings = _reconcile_visual_controls(
                existing + recovered["candidates"], self.blueprint.checkbox_paths
            )
            return {
                "field_attempt": attempt,
                "active_image_path": recovered["active_image_path"],
                "quality": recovered["quality"],
                "quality_history": [recovered["quality"]],
                "enhancements": recovered["enhancements"],
                "evidence": recovered["evidence"],
                "ocr_ledger": recovered["ocr_ledger"],
                "control_ledger": recovered["control_ledger"],
                "candidates": [item.model_dump(mode="json") for item in merged],
                "warnings": recovered["warnings"] + reconciliation_warnings,
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
                        "originalImagePath": state.get("original_image_path"),
                        "activeImagePath": state.get("active_image_path"),
                        "quality": state.get("quality"),
                        "qualityHistory": state.get("quality_history", []),
                        "documentUnderstanding": state.get(
                            "document_understanding"
                        ),
                        "documentUnderstandingImageSha256": state.get(
                            "document_understanding_image_sha256"
                        ),
                        "enhancements": state.get("enhancements", []),
                        "enhancementEvaluations": state.get(
                            "enhancement_evaluations", []
                        ),
                        "records": state.get("evidence", []),
                        "ocrSpans": state.get("ocr_ledger", []),
                        "layoutBlocks": state.get("layout_ledger", []),
                        "layoutVisualizations": state.get(
                            "layout_visualizations", []
                        ),
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
        technical_quality = self.quality_assessor.assess(output)
        target_paths = [item.field_path for item in plans]
        prefix = f"retry{attempt}"
        spans = self.text_engine.extract(
            output, attempt=attempt, id_prefix=f"ocr:{prefix}"
        )
        controls = self.control_engine.detect(
            output, attempt=attempt, id_prefix=f"control:{prefix}"
        )
        controls = associate_controls(controls, spans)
        quality = enrich_quality_report(
            technical_quality,
            ocr=assess_ocr_readiness(spans),
            understanding=_document_understanding(state),
        )
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
            confidence=_effective_quality(quality),
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
                document_context=_document_understanding(state),
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
                    document_context=_document_understanding(state),
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
        document_context: DocumentUnderstanding | None = None,
    ) -> tuple[list[FieldCandidate], list[str]]:
        if self.vlm is None:
            return [], []
        result = self.vlm.propose(
            image_path=image_path,
            target_paths=target_paths,
            spans=spans,
            blocks=blocks,
            controls=controls,
            image_evidence_id=image_evidence_id,
            field_hints=self.blueprint.vlm_hints(target_paths),
            document_context=document_context,
        )
        candidates: list[FieldCandidate] = []
        warnings = list(result.warnings)
        for proposal in result.proposals:
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
            readiness = update["quality"].get("extraction_readiness")
            confidence = (
                readiness
                if readiness is not None
                else update["quality"].get("overall")
            )
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


def _reconcile_visual_controls(
    candidates: list[FieldCandidate], checkbox_paths: set[str]
) -> tuple[list[FieldCandidate], list[str]]:
    """Prefer a uniquely grounded visual reading over conflicting contour guesses."""

    visual_by_path: dict[str, list[FieldCandidate]] = {}
    for candidate in candidates:
        sources = {item.casefold() for item in candidate.support_sources}
        if (
            candidate.path in checkbox_paths
            and candidate.source == "vlm_visual"
            and "local_vlm_visual" in sources
            and len(sources) >= 2
            and candidate.normalized_value is not None
        ):
            visual_by_path.setdefault(candidate.path, []).append(candidate)

    authoritative: dict[str, str] = {}
    for path, items in visual_by_path.items():
        fingerprints = {item.value_fingerprint() for item in items}
        if len(fingerprints) == 1:
            authoritative[path] = next(iter(fingerprints))

    filtered: list[FieldCandidate] = []
    removed: dict[str, int] = {}
    for candidate in candidates:
        expected = authoritative.get(candidate.path)
        if (
            expected is not None
            and candidate.source == "opencv_control"
            and candidate.normalized_value is not None
            and candidate.value_fingerprint() != expected
        ):
            removed[candidate.path] = removed.get(candidate.path, 0) + 1
            continue
        filtered.append(candidate)
    warnings = [
        f"visual_control_overrode_opencv:{path}:{count}"
        for path, count in sorted(removed.items())
    ]
    return filtered, warnings


def _vlm_target_paths(
    blueprint: DocumentBlueprint, candidates: list[FieldCandidate]
) -> list[str]:
    present = {item.path for item in candidates if item.normalized_value is not None}
    sources_by_path: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.normalized_value is None:
            continue
        sources_by_path.setdefault(candidate.path, set()).update(
            source.casefold() for source in candidate.support_sources if source
        )
    return [
        path
        for path in blueprint.output_paths
        if (
            path not in present
            or path in blueprint.checkbox_paths
            or len(sources_by_path.get(path, set())) < 2
        )
    ]


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


def _document_understanding(state: DocumentState) -> DocumentUnderstanding | None:
    payload = state.get("document_understanding")
    return DocumentUnderstanding.model_validate(payload) if payload else None


def _effective_quality(report: QualityReport) -> float:
    return (
        report.extraction_readiness
        if report.extraction_readiness is not None
        else report.overall
    )


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
