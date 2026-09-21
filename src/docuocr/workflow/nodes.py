from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docuocr.config import AppSettings, AssociationSettings
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
from docuocr.extraction.layout_processing import (
    LayoutBlockProcessor,
    LayoutRecoveryTask,
    controls_in_region,
    spans_in_region,
)
from docuocr.extraction.rules import (
    alias_score,
    associate_controls,
    canonical_text,
    map_rule_candidates,
)
from docuocr.extraction.validation import validate_candidates
from docuocr.models import (
    BBox,
    DocumentUnderstanding,
    EnhancementEvaluation,
    EnhancementRecord,
    EvidenceKind,
    EvidenceRecord,
    EvidenceReverificationRecord,
    FieldCandidate,
    FieldDecision,
    FormControl,
    LayoutBlock,
    LayoutExtractionRecord,
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
        self.layout_block_processor = LayoutBlockProcessor(text_engine)
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
                "evidence_reverification_attempt": 0,
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

    def layout_block_map(self, state: DocumentState) -> dict[str, Any]:
        attempt = state.get("document_attempt", 0)

        def work() -> dict[str, Any]:
            existing = _models(FieldCandidate, state.get("candidates", []))
            if not self.settings.association.layout_blocks_enabled:
                return {
                    "candidates": [item.model_dump(mode="json") for item in existing],
                    "_decision": "disabled",
                }
            blocks = _models(LayoutBlock, state.get("layout_blocks", []))
            if not blocks:
                return {
                    "candidates": [item.model_dump(mode="json") for item in existing],
                    "_stage_confidence": 0.0,
                    "_decision": "no_layout_blocks",
                }

            baseline_spans = _models(OCRSpan, state.get("ocr_spans", []))
            controls = _models(FormControl, state.get("controls", []))
            resolved_paths = {
                item.path for item in existing if item.normalized_value is not None
            }
            ocr_block_ids = _layout_mapping_ocr_block_ids(
                blocks=blocks,
                spans=baseline_spans,
                blueprint=self.blueprint,
                unresolved_paths=[
                    path
                    for path in self.blueprint.output_paths
                    if path not in resolved_paths
                ],
                settings=self.settings.association,
            )
            processed, warnings = self.layout_block_processor.process(
                image_path=state["active_image_path"],
                run_dir=state["run_dir"],
                blocks=blocks,
                settings=self.settings.association,
                attempt=attempt,
                ocr_block_ids=ocr_block_ids,
            )
            recognized_spans = [
                span for item in processed for span in item.recognized_spans
            ]
            local_candidates: list[FieldCandidate] = []
            extraction_records: list[LayoutExtractionRecord] = []
            for item in processed:
                block_spans = _unique_models_by_id(
                    [
                        *spans_in_region(baseline_spans, item.block),
                        *item.recognized_spans,
                    ]
                )
                block_controls = controls_in_region(controls, item.block)
                block_candidates = map_rule_candidates(
                    self.blueprint,
                    block_spans,
                    block_controls,
                    attempt=attempt,
                )
                local_candidates.extend(block_candidates)
                candidate_paths = {
                    candidate.path
                    for candidate in block_candidates
                    if candidate.normalized_value is not None
                }
                block_warnings = [
                    warning
                    for warning in warnings
                    if item.block.id in warning
                    or warning.startswith(
                        (
                            "layout_block_ocr_unsupported:",
                            "layout_parallelism_downgraded:",
                        )
                    )
                ]
                ocr_failed = (
                    item.record is not None and item.record.status == "failed"
                )
                hard_failure = ocr_failed or any(
                    warning.startswith(
                        (
                            "layout_block_crop_failed:",
                            "layout_block_ocr_failed:",
                            "layout_block_ocr_unsupported:",
                        )
                    )
                    for warning in block_warnings
                )
                if hard_failure:
                    status = "partial" if block_candidates else "failed"
                elif block_warnings:
                    status = "partial"
                else:
                    status = "succeeded"
                extraction_records.append(
                    LayoutExtractionRecord(
                        id=f"layout-extraction:mapping:{attempt}:{item.block.id}",
                        stage="mapping",
                        block_id=item.block.id,
                        block_label=item.block.label,
                        block_source=item.block.source,
                        page=item.block.page,
                        bbox=item.crop_bbox,
                        crop_path=(
                            str(item.crop_path) if item.crop_path is not None else None
                        ),
                        target_paths=self.blueprint.output_paths,
                        unresolved_target_paths=[
                            path
                            for path in self.blueprint.output_paths
                            if path not in candidate_paths
                        ],
                        ocr_span_ids=[span.id for span in block_spans],
                        ocr_spans=block_spans,
                        control_ids=[control.id for control in block_controls],
                        controls=block_controls,
                        candidates=block_candidates,
                        recognition_attempts=(
                            item.record.recognition_attempts if item.record else 0
                        ),
                        status=status,
                        warnings=block_warnings,
                        model_id=self.text_engine.model_id,
                        attempt=attempt,
                    )
                )

            combined_spans = _unique_models_by_id(
                [*baseline_spans, *recognized_spans]
            )
            merged = _merge_candidates([*existing, *local_candidates])
            records = [
                item.record.model_dump(mode="json")
                for item in processed
                if item.record is not None
            ]
            evidence = [_ocr_evidence(item) for item in recognized_spans]
            covered = {
                item.path for item in merged if item.normalized_value is not None
            }
            return {
                "ocr_spans": [
                    item.model_dump(mode="json") for item in combined_spans
                ],
                "ocr_ledger": [
                    item.model_dump(mode="json") for item in recognized_spans
                ],
                "layout_block_ocr": records,
                "layout_extractions": [
                    item.model_dump(mode="json") for item in extraction_records
                ],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "candidates": [item.model_dump(mode="json") for item in merged],
                "warnings": warnings,
                "_stage_confidence": len(covered)
                / max(1, len(self.blueprint.output_paths)),
                "_decision": f"processed:{len(processed)}",
            }

        return self._execute(
            "layout_block_map",
            state,
            attempt,
            work,
            model_id=f"{self.text_engine.model_id}+layout-association-v1",
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
            additions, warnings, _ = self._vlm_candidates(
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

    def evidence_reverify(self, state: DocumentState) -> dict[str, Any]:
        verification_attempt = state.get("evidence_reverification_attempt", 0) + 1

        def work() -> dict[str, Any]:
            target_paths = [
                path
                for path, payload in state.get("decisions", {}).items()
                if payload.get("disposition") != "accepted"
            ]
            existing = _models(FieldCandidate, state.get("candidates", []))
            if not target_paths:
                return {
                    "evidence_reverification_attempt": verification_attempt,
                    "field_attempt": state.get("field_attempt", 0) + 1,
                    "candidates": [
                        item.model_dump(mode="json") for item in existing
                    ],
                    "_stage_confidence": 1.0,
                    "_decision": "nothing_to_reverify",
                }

            spans = _models(OCRSpan, state.get("ocr_spans", []))
            controls = associate_controls(
                _models(FormControl, state.get("controls", [])), spans
            )
            blocks = _models(LayoutBlock, state.get("layout_blocks", []))
            rule_additions = map_rule_candidates(
                self.blueprint,
                spans,
                controls,
                attempt=state.get("field_attempt", 0) + 1,
                only_paths=set(target_paths),
            )
            evidence = _models(EvidenceRecord, state.get("evidence", []))
            new_evidence: list[EvidenceRecord] = []
            vlm_additions: list[FieldCandidate] = []
            warnings: list[str] = []
            if (
                self.vlm is not None
                and self.settings.association.reverify_with_vlm
            ):
                quality = QualityReport.model_validate(state["quality"])
                active_image = Path(state["active_image_path"])
                image_evidence = EvidenceRecord(
                    id=f"image:reverify{verification_attempt}:p1:active",
                    kind=EvidenceKind.IMAGE,
                    bbox=BBox(x1=0, y1=0, x2=quality.width, y2=quality.height),
                    confidence=_effective_quality(quality),
                    source="document_image",
                    artifact_path=str(active_image),
                    artifact_sha256=sha256_file(active_image),
                    transform_chain=["evidence_reverification"],
                )
                evidence.append(image_evidence)
                new_evidence.append(image_evidence)
                vlm_additions, vlm_warnings, _ = self._vlm_candidates(
                    image_path=active_image,
                    target_paths=target_paths,
                    spans=spans,
                    blocks=blocks,
                    controls=controls,
                    evidence=evidence,
                    attempt=state.get("field_attempt", 0) + 1,
                    visual_verification=True,
                    image_evidence_id=image_evidence.id,
                    document_context=_document_understanding(state),
                )
                warnings.extend(vlm_warnings)

            merged, reconciliation_warnings = _reconcile_visual_controls(
                _merge_candidates(
                    [*existing, *rule_additions, *vlm_additions]
                ),
                self.blueprint.checkbox_paths,
            )
            warnings.extend(reconciliation_warnings)
            added_paths = sorted(
                {
                    item.path
                    for item in [*rule_additions, *vlm_additions]
                    if item.normalized_value is not None
                }
            )
            record = EvidenceReverificationRecord(
                attempt=verification_attempt,
                target_paths=target_paths,
                rule_candidate_count=len(rule_additions),
                vlm_candidate_count=len(vlm_additions),
                candidate_paths=added_paths,
            )
            return {
                "field_attempt": state.get("field_attempt", 0) + 1,
                "evidence_reverification_attempt": verification_attempt,
                "evidence_reverifications": [record.model_dump(mode="json")],
                "controls": [item.model_dump(mode="json") for item in controls],
                "candidates": [item.model_dump(mode="json") for item in merged],
                "evidence": [
                    item.model_dump(mode="json") for item in new_evidence
                ],
                "warnings": warnings,
                "_stage_confidence": len(added_paths) / max(1, len(target_paths)),
                "_decision": "rescore_evidence",
            }

        model_id = "evidence-rules-v1"
        if self.vlm is not None and self.settings.association.reverify_with_vlm:
            model_id = f"evidence-rules+{self.vlm.model_id}"
        return self._execute(
            "evidence_reverify",
            state,
            verification_attempt,
            work,
            model_id=model_id,
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
                    "_decision": "no_recovery_plans",
                }

            if (
                self.settings.association.layout_recovery_enabled
                and attempt
                <= self.settings.association.max_layout_recovery_attempts
            ):
                recovered, unmatched = self._recover_layouts(
                    state, plans, attempt
                )
            else:
                recovered = _empty_recovery_result(state)
                warning = (
                    "layout_recovery_disabled"
                    if not self.settings.association.layout_recovery_enabled
                    else "layout_recovery_attempt_limit_reached"
                )
                recovered["warnings"].append(warning)
                unmatched = plans

            if (
                unmatched
                and self.settings.association.whole_page_recovery_fallback
            ):
                page_recovery = self._recover_page(state, unmatched, attempt)
                recovered = _combine_recovery_results(recovered, page_recovery)
                recovery_mode = (
                    "layout_then_whole_page_fallback"
                    if len(unmatched) < len(plans)
                    else "whole_page_fallback"
                )
            else:
                recovery_mode = "layout"
                recovered["warnings"].extend(
                    f"layout_recovery_no_matching_block:{plan.field_path}"
                    for plan in unmatched
                )

            merged, reconciliation_warnings = _reconcile_visual_controls(
                _merge_candidates([*existing, *recovered["candidates"]]),
                self.blueprint.checkbox_paths,
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
                "layout_extractions": recovered["layout_extractions"],
                "candidates": [item.model_dump(mode="json") for item in merged],
                "warnings": recovered["warnings"] + reconciliation_warnings,
                "_decision": f"rescore:{recovery_mode}",
            }

        return self._execute(
            "recover",
            state,
            attempt,
            work,
            model_id="layout-opencv+paddle+local-vlm",
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
            layout_manifest_path = run_dir / "layout-extractions.json"
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
                        "layoutBlockOCR": state.get("layout_block_ocr", []),
                        "layoutExtractions": state.get(
                            "layout_extractions", []
                        ),
                        "layoutExtractionManifest": str(layout_manifest_path),
                        "evidenceReverifications": state.get(
                            "evidence_reverifications", []
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
                            layout_manifest=str(layout_manifest_path),
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
            result = envelope.as_public_dict()
            corrected_paths = set(state.get("human_corrections", {}))
            accepted_paths = sorted(
                {
                    path
                    for path, decision in decisions.items()
                    if decision.disposition == "accepted"
                }
                | corrected_paths
            )
            unresolved_paths = sorted(
                path
                for path, decision in decisions.items()
                if decision.disposition != "accepted" and path not in corrected_paths
            )
            layout_manifest_path.write_text(
                json.dumps(
                    {
                        "layouts": state.get("layout_extractions", []),
                        "combined": {
                            "data": result["data"],
                            "acceptedPaths": accepted_paths,
                            "unresolvedPaths": unresolved_paths,
                            "fieldDecisions": {
                                path: decision.model_dump(mode="json")
                                for path, decision in decisions.items()
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"result": result, "_stage_confidence": 1.0}

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
            additions, vlm_warnings, _ = self._vlm_candidates(
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
            "layout_extractions": [],
            "candidates": candidates,
            "warnings": warnings,
        }

    def _recover_layouts(
        self, state: DocumentState, plans: list[RecoveryPlan], attempt: int
    ) -> tuple[dict[str, Any], list[RecoveryPlan]]:
        blocks = _models(LayoutBlock, state.get("layout_blocks", []))
        spans = _models(OCRSpan, state.get("ocr_spans", []))
        tasks, unmatched = _layout_recovery_tasks(
            plans=plans,
            blocks=blocks,
            spans=spans,
            blueprint=self.blueprint,
            settings=self.settings.association,
        )
        if not tasks:
            return _empty_recovery_result(state), unmatched

        recovered_blocks, warnings = self.layout_block_processor.recover(
            image_path=state["active_image_path"],
            run_dir=state["run_dir"],
            tasks=tasks,
            settings=self.settings.association,
            attempt=attempt,
            enhancer=self.enhancer,
        )
        all_candidates: list[FieldCandidate] = []
        all_evidence: list[EvidenceRecord] = []
        all_records: list[dict[str, Any]] = []
        ocr_ledger: list[dict[str, Any]] = []
        control_ledger: list[dict[str, Any]] = []
        extraction_records: list[dict[str, Any]] = []
        quality = QualityReport.model_validate(state["quality"])
        layout_vlm_blocks = 0
        for index, item in enumerate(recovered_blocks):
            task = item.task
            target_paths = list(task.target_paths)
            block_warnings = [
                warning
                for warning in warnings
                if task.block.id in warning
                or warning.startswith("layout_parallelism_downgraded:")
            ]
            local_spans = list(item.recognized_spans)
            local_controls: list[FormControl] = []
            candidates: list[FieldCandidate] = []
            evidence: list[EvidenceRecord] = []
            if item.crop_path is not None and item.enhancement is not None:
                try:
                    local_controls = self.control_engine.detect(
                        item.crop_path,
                        page=task.block.page,
                        attempt=attempt,
                        id_prefix=f"control:layout-recovery{attempt}:{index:03d}",
                    )
                except Exception as exc:  # noqa: BLE001 - isolate one layout failure
                    warning = (
                        "layout_recovery_controls_failed:"
                        f"{task.block.id}:{type(exc).__name__}:{exc}"
                    )
                    warnings.append(warning)
                    block_warnings.append(warning)
                local_controls = associate_controls(
                    local_controls, local_spans
                )
                candidates = map_rule_candidates(
                    self.blueprint,
                    local_spans,
                    local_controls,
                    attempt=attempt,
                    only_paths=set(target_paths),
                )
                page_spans = [
                    _recovery_span_to_page(
                        span, item.crop_bbox, item.enhancement
                    )
                    for span in local_spans
                ]
                page_controls = [
                    _recovery_control_to_page(
                        control, item.crop_bbox, item.enhancement
                    )
                    for control in local_controls
                ]
                crop_evidence = EvidenceRecord(
                    id=f"crop:layout-recovery{attempt}:{index:03d}",
                    kind=EvidenceKind.CROP,
                    page=task.block.page,
                    bbox=item.crop_bbox,
                    confidence=_effective_quality(quality),
                    source="opencv_layout_recovery",
                    artifact_path=str(item.crop_path),
                    artifact_sha256=item.enhancement.output_sha256,
                    transform_chain=[item.enhancement.strategy],
                )
                evidence = [
                    *[_ocr_evidence(span) for span in page_spans],
                    *[_control_evidence(control) for control in page_controls],
                    crop_evidence,
                ]
                vlm_target_paths = _layout_vlm_target_paths(
                    task, candidates, local_spans, self.blueprint
                )
                vlm_attempted = False
                vlm_request_count = 0
                if (
                    self.vlm is not None
                    and self.settings.association.layout_recovery_vlm_enabled
                    and vlm_target_paths
                    and layout_vlm_blocks
                    < self.settings.association.max_layout_vlm_blocks
                ):
                    request_paths = vlm_target_paths[
                        : self.settings.vlm.max_paths_per_request
                    ]
                    if len(request_paths) < len(vlm_target_paths):
                        warning = (
                            "layout_recovery_vlm_path_budget_exhausted:"
                            f"{task.block.id}"
                        )
                        warnings.append(warning)
                        block_warnings.append(warning)
                    additions, vlm_warnings, vlm_request_count = self._vlm_candidates(
                        image_path=item.crop_path,
                        target_paths=request_paths,
                        spans=local_spans,
                        blocks=[],
                        controls=local_controls,
                        evidence=evidence,
                        attempt=attempt,
                        visual_verification=True,
                        image_evidence_id=crop_evidence.id,
                        document_context=_document_understanding(state),
                    )
                    vlm_attempted = True
                    layout_vlm_blocks += 1
                    candidates.extend(additions)
                    warnings.extend(vlm_warnings)
                    block_warnings.extend(vlm_warnings)
                elif (
                    self.vlm is not None
                    and vlm_target_paths
                    and layout_vlm_blocks
                    >= self.settings.association.max_layout_vlm_blocks
                ):
                    warning = (
                        "layout_recovery_vlm_block_budget_exhausted:"
                        f"{task.block.id}"
                    )
                    warnings.append(warning)
                    block_warnings.append(warning)
            else:
                page_spans = []
                page_controls = []
                vlm_target_paths = []
                vlm_attempted = False
                vlm_request_count = 0

            candidate_paths = {
                candidate.path
                for candidate in candidates
                if candidate.normalized_value is not None
            }
            unresolved_target_paths = [
                path for path in target_paths if path not in candidate_paths
            ]
            if item.error:
                status = "partial" if candidates else "failed"
            elif block_warnings or unresolved_target_paths:
                status = "partial"
            else:
                status = "succeeded"
            extraction = LayoutExtractionRecord(
                id=(
                    f"layout-extraction:recovery:{attempt}:"
                    f"{task.block.id}"
                ),
                stage="recovery",
                block_id=task.block.id,
                block_label=task.block.label,
                block_source=task.block.source,
                page=task.block.page,
                bbox=item.crop_bbox,
                crop_path=(str(item.crop_path) if item.crop_path else None),
                target_paths=target_paths,
                unresolved_target_paths=unresolved_target_paths,
                ocr_span_ids=[span.id for span in page_spans],
                ocr_spans=page_spans,
                control_ids=[control.id for control in page_controls],
                controls=page_controls,
                candidates=candidates,
                vlm_attempted=vlm_attempted,
                vlm_target_paths=vlm_target_paths,
                vlm_request_count=vlm_request_count,
                recognition_attempts=item.recognition_attempts,
                status=status,
                warnings=list(dict.fromkeys(block_warnings)),
                model_id=self.text_engine.model_id,
                attempt=attempt,
            )
            extraction_records.append(extraction.model_dump(mode="json"))
            all_candidates.extend(candidates)
            all_evidence.extend(evidence)
            ocr_ledger.extend(
                span.model_dump(mode="json") for span in page_spans
            )
            control_ledger.extend(
                control.model_dump(mode="json") for control in page_controls
            )
            if item.enhancement is not None:
                all_records.append(item.enhancement.model_dump(mode="json"))

        return {
            "active_image_path": state["active_image_path"],
            "quality": state["quality"],
            "enhancements": all_records,
            "evidence": [item.model_dump(mode="json") for item in all_evidence],
            "ocr_ledger": ocr_ledger,
            "control_ledger": control_ledger,
            "layout_extractions": extraction_records,
            "candidates": all_candidates,
            "warnings": list(dict.fromkeys(warnings)),
        }, unmatched

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
    ) -> tuple[list[FieldCandidate], list[str], int]:
        if self.vlm is None:
            return [], [], 0
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
        return candidates, warnings, result.request_count

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


def _layout_recovery_tasks(
    *,
    plans: list[RecoveryPlan],
    blocks: list[LayoutBlock],
    spans: list[OCRSpan],
    blueprint: DocumentBlueprint,
    settings: AssociationSettings,
) -> tuple[list[LayoutRecoveryTask], list[RecoveryPlan]]:
    """Assign each unresolved field to bounded layout crops, never implicitly a page."""

    excluded = {
        item.casefold().strip().replace(" ", "_")
        for item in settings.excluded_layout_labels
    }
    eligible = [
        block
        for block in sorted(blocks, key=_layout_sort_key)
        if block.label.casefold().strip().replace(" ", "_") not in excluded
    ][: settings.max_layout_blocks]
    assignments: dict[str, list[tuple[RecoveryPlan, str]]] = {
        block.id: [] for block in eligible
    }
    search_plans: list[RecoveryPlan] = []
    unmatched: list[RecoveryPlan] = []

    for plan in plans:
        selected = _block_for_evidence_bbox(plan.bbox, eligible)
        selection_reason = "evidence"
        if selected is None:
            aliases = _path_label_aliases(blueprint, plan.field_path)
            selected = _block_for_printed_label(aliases, eligible, spans)
            selection_reason = "label"

        if selected is not None:
            assignments[selected.id].append((plan, selection_reason))
            continue
        if settings.layout_recovery_search_all_blocks and eligible:
            search_plans.append(plan)
            continue
        unmatched.append(plan)

    # Search fallback is a global crop budget, not a per-field budget. Every
    # unresolved path shares the same ranked crops so OCR runs only once per crop.
    for block in _rank_layout_search_blocks(
        search_plans,
        blueprint,
        eligible,
        spans,
        limit=settings.max_layout_recovery_search_blocks,
    ):
        assignments[block.id].extend((plan, "search") for plan in search_plans)

    tasks: list[LayoutRecoveryTask] = []
    for block in eligible:
        assigned = assignments[block.id]
        if not assigned:
            continue
        target_paths = tuple(
            dict.fromkeys(plan.field_path for plan, _ in assigned)
        )
        reason_by_path = {
            plan.field_path: reason for plan, reason in assigned
        }
        assigned_plans = [plan for plan, _ in assigned]
        tasks.append(
            LayoutRecoveryTask(
                block=block,
                target_paths=target_paths,
                actions=_layout_recovery_actions(
                    assigned_plans, target_paths, blueprint.checkbox_paths
                ),
                selection_reasons=tuple(
                    (path, reason_by_path[path]) for path in target_paths
                ),
            )
        )
    return tasks, unmatched


def _layout_mapping_ocr_block_ids(
    *,
    blocks: list[LayoutBlock],
    spans: list[OCRSpan],
    blueprint: DocumentBlueprint,
    unresolved_paths: list[str],
    settings: AssociationSettings,
) -> set[str]:
    """Choose a small, document-wide budget of layouts for supplemental OCR."""

    if settings.max_layout_ocr_blocks == 0 or not unresolved_paths:
        return set()
    excluded = {
        item.casefold().strip().replace(" ", "_")
        for item in settings.excluded_layout_labels
    }
    eligible = [
        block
        for block in sorted(blocks, key=_layout_sort_key)
        if block.label.casefold().strip().replace(" ", "_") not in excluded
    ][: settings.max_layout_blocks]
    alias_groups = [
        _path_label_aliases(blueprint, path) for path in unresolved_paths
    ]
    ranked: list[tuple[float, int, float, int, LayoutBlock]] = []
    for index, block in enumerate(eligible):
        block_spans = spans_in_region(spans, block)
        relevance = max(
            (
                _block_printed_label_score(aliases, block, spans)
                for aliases in alias_groups
            ),
            default=0.0,
        )
        ranked.append(
            (
                relevance,
                int(not block_spans),
                block.confidence,
                -index,
                block,
            )
        )
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return {
        item[4].id
        for item in ranked[: settings.max_layout_ocr_blocks]
    }


def _layout_sort_key(block: LayoutBlock) -> tuple[int, int, int, int, str]:
    return (
        block.page,
        block.order if block.order is not None else 1_000_000,
        block.bbox.y1,
        block.bbox.x1,
        block.id,
    )


def _block_for_evidence_bbox(
    evidence_bbox: BBox | None, blocks: list[LayoutBlock]
) -> LayoutBlock | None:
    if evidence_bbox is None:
        return None
    evidence_area = evidence_bbox.width * evidence_bbox.height
    center_x, center_y = evidence_bbox.center
    matches: list[tuple[float, int, LayoutBlock]] = []
    for block in blocks:
        box = block.bbox
        intersection = _intersection_area(evidence_bbox, box)
        if intersection <= 0:
            continue
        coverage = intersection / max(1, evidence_area)
        center_inside = box.x1 <= center_x <= box.x2 and box.y1 <= center_y <= box.y2
        block_area = box.width * box.height
        if coverage < 0.35 and not (
            center_inside and evidence_area <= block_area * 4
        ):
            continue
        score = coverage + (0.25 if center_inside else 0.0)
        matches.append((score, -block_area, block))
    return max(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def _block_for_printed_label(
    aliases: list[str], blocks: list[LayoutBlock], spans: list[OCRSpan]
) -> LayoutBlock | None:
    if not aliases:
        return None
    matches: list[tuple[float, int, LayoutBlock]] = []
    for block in blocks:
        score = _block_printed_label_score(aliases, block, spans)
        if score < 0.78:
            continue
        area = block.bbox.width * block.bbox.height
        matches.append((score, -area, block))
    return max(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def _rank_layout_search_blocks(
    plans: list[RecoveryPlan],
    blueprint: DocumentBlueprint,
    blocks: list[LayoutBlock],
    spans: list[OCRSpan],
    *,
    limit: int,
) -> list[LayoutBlock]:
    if not plans:
        return []
    alias_groups = [
        _path_label_aliases(blueprint, plan.field_path) for plan in plans
    ]
    ranked = [
        (
            max(
                (
                    _block_printed_label_score(aliases, block, spans)
                    for aliases in alias_groups
                ),
                default=0.0,
            ),
            block.confidence,
            -index,
            block,
        )
        for index, block in enumerate(blocks)
    ]
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked[:limit]]


def _block_printed_label_score(
    aliases: list[str], block: LayoutBlock, spans: list[OCRSpan]
) -> float:
    if not aliases:
        return 0.0
    block_spans = sorted(
        spans_in_region(spans, block),
        key=lambda item: (item.bbox.y1, item.bbox.x1, item.id),
    )
    fragments = [block.content, *(span.text for span in block_spans)]
    for start in range(len(block_spans)):
        for length in range(2, min(4, len(block_spans) - start) + 1):
            fragments.append(
                " ".join(
                    item.text for item in block_spans[start : start + length]
                )
            )
    return max(
        (_printed_label_score(fragment, aliases) for fragment in fragments),
        default=0.0,
    )


def _printed_label_score(text: str, aliases: list[str]) -> float:
    score = alias_score(text, aliases)
    normalized = canonical_text(text)
    for alias in aliases:
        target = canonical_text(alias)
        if target and target in normalized:
            score = max(score, 1.0 if target == normalized else 0.96)
    return score


def _path_label_aliases(
    blueprint: DocumentBlueprint, field_path: str
) -> list[str]:
    spec = blueprint.fields.get(field_path)
    if spec is not None:
        return list(dict.fromkeys([*spec.aliases, *spec.group_aliases]))
    aliases: list[str] = []
    for group in blueprint.checkbox_groups.values():
        for option in group.options.values():
            if option.output_path != field_path:
                continue
            aliases.extend(group.aliases)
            aliases.extend(
                alias
                for alias in option.aliases
                if len(canonical_text(alias)) > 1
            )
    return list(dict.fromkeys(aliases))


def _layout_vlm_target_paths(
    task: LayoutRecoveryTask,
    candidates: list[FieldCandidate],
    spans: list[OCRSpan],
    blueprint: DocumentBlueprint,
) -> list[str]:
    """Use crop VLM only where local OCR gives a reason to inspect the crop."""

    resolved = {
        candidate.path
        for candidate in candidates
        if candidate.normalized_value is not None
    }
    selection_reasons = dict(task.selection_reasons)
    result: list[str] = []
    for path in task.target_paths:
        if path in resolved:
            continue
        reason = selection_reasons.get(path, "search")
        aliases = _path_label_aliases(blueprint, path)
        if reason in {"evidence", "label"} or _spans_match_aliases(spans, aliases):
            result.append(path)
    return result


def _spans_match_aliases(spans: list[OCRSpan], aliases: list[str]) -> bool:
    if not spans or not aliases:
        return False
    ordered = sorted(spans, key=lambda item: (item.bbox.y1, item.bbox.x1, item.id))
    fragments = [item.text for item in ordered]
    for start in range(len(ordered)):
        for length in range(2, min(4, len(ordered) - start) + 1):
            fragments.append(
                " ".join(item.text for item in ordered[start : start + length])
            )
    return any(_printed_label_score(text, aliases) >= 0.78 for text in fragments)


def _layout_recovery_actions(
    plans: list[RecoveryPlan],
    target_paths: tuple[str, ...],
    checkbox_paths: set[str],
) -> tuple[RecoveryAction, ...]:
    requested = list(
        dict.fromkeys(action for plan in plans for action in plan.actions)
    )
    checkbox_only = bool(target_paths) and all(
        path in checkbox_paths for path in target_paths
    )
    if checkbox_only:
        allowed = [
            action for action in requested if action != RecoveryAction.REVIEW
        ]
        if RecoveryAction.CHECKBOX_FOCUS not in allowed:
            allowed.insert(0, RecoveryAction.CHECKBOX_FOCUS)
    else:
        destructive = {
            RecoveryAction.ADAPTIVE_BINARIZE,
            RecoveryAction.CHECKBOX_FOCUS,
            RecoveryAction.REVIEW,
        }
        allowed = [action for action in requested if action not in destructive]
        if RecoveryAction.CLAHE not in allowed:
            allowed.append(RecoveryAction.CLAHE)
    if RecoveryAction.UPSCALE not in allowed:
        allowed.insert(0, RecoveryAction.UPSCALE)
    return tuple(dict.fromkeys(allowed))


def _intersection_area(left: BBox, right: BBox) -> int:
    return max(0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )


def _recovery_span_to_page(
    span: OCRSpan, crop_bbox: BBox, enhancement: EnhancementRecord
) -> OCRSpan:
    return span.model_copy(
        update={"bbox": _recovery_bbox_to_page(span.bbox, crop_bbox, enhancement)}
    )


def _recovery_control_to_page(
    control: FormControl, crop_bbox: BBox, enhancement: EnhancementRecord
) -> FormControl:
    return control.model_copy(
        update={"bbox": _recovery_bbox_to_page(control.bbox, crop_bbox, enhancement)}
    )


def _recovery_bbox_to_page(
    local: BBox, crop_bbox: BBox, enhancement: EnhancementRecord
) -> BBox:
    raw_scale = enhancement.parameters.get("upscale", 1.0)
    scale = (
        float(raw_scale)
        if isinstance(raw_scale, int | float) and raw_scale > 0
        else 1.0
    )
    local_x1 = int(round(local.x1 / scale))
    local_y1 = int(round(local.y1 / scale))
    local_x2 = int(round(local.x2 / scale))
    local_y2 = int(round(local.y2 / scale))
    x1 = min(max(crop_bbox.x1 + local_x1, crop_bbox.x1), crop_bbox.x2 - 1)
    y1 = min(max(crop_bbox.y1 + local_y1, crop_bbox.y1), crop_bbox.y2 - 1)
    x2 = min(max(crop_bbox.x1 + local_x2, x1 + 1), crop_bbox.x2)
    y2 = min(max(crop_bbox.y1 + local_y2, y1 + 1), crop_bbox.y2)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _empty_recovery_result(state: DocumentState) -> dict[str, Any]:
    return {
        "active_image_path": state["active_image_path"],
        "quality": state["quality"],
        "enhancements": [],
        "evidence": [],
        "ocr_ledger": [],
        "control_ledger": [],
        "layout_extractions": [],
        "candidates": [],
        "warnings": [],
    }


def _combine_recovery_results(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    combined = {
        "active_image_path": second.get(
            "active_image_path", first["active_image_path"]
        ),
        "quality": second.get("quality", first["quality"]),
    }
    for key in (
        "enhancements",
        "evidence",
        "ocr_ledger",
        "control_ledger",
        "layout_extractions",
        "candidates",
    ):
        combined[key] = [*first.get(key, []), *second.get(key, [])]
    combined["warnings"] = list(
        dict.fromkeys([*first.get("warnings", []), *second.get("warnings", [])])
    )
    return combined


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


def _unique_models_by_id(values: list[Any]) -> list[Any]:
    result: dict[str, Any] = {}
    for item in values:
        result.setdefault(item.id, item)
    return list(result.values())


def _merge_candidates(candidates: list[FieldCandidate]) -> list[FieldCandidate]:
    result: dict[tuple[str, str, str, tuple[str, ...]], FieldCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.path,
            candidate.value_fingerprint(),
            candidate.source,
            tuple(candidate.evidence_ids),
        )
        previous = result.get(key)
        if previous is None or (
            candidate.recognition_confidence * candidate.association_confidence
            > previous.recognition_confidence * previous.association_confidence
        ):
            result[key] = candidate
    return list(result.values())


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
