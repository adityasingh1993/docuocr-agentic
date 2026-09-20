from __future__ import annotations

import math
from collections import defaultdict

from docuocr.models import (
    BBox,
    ConfidenceFeatures,
    EvidenceRecord,
    FieldCandidate,
    FieldDecision,
    RecoveryAction,
    RecoveryPlan,
)

from .blueprint import DocumentBlueprint

_FATAL_VALIDATION_PREFIXES = (
    "invalid_",
    "ambiguous_",
    "normalization_failed",
    "pattern_mismatch",
    "birth_weight_out_of_range",
    "collection_before_birth",
    "exclusive_group_conflict",
)


class ConfidenceScorer:
    """Conservative policy score. It is intentionally uncalibrated."""

    def __init__(self, accept_threshold: float = 0.90) -> None:
        self.accept_threshold = accept_threshold
        self.weights = {
            "image_quality": 0.14,
            "recognition": 0.24,
            "association": 0.20,
            "agreement": 0.16,
            "validation": 0.16,
            "grounding": 0.10,
        }

    def decide(
        self,
        candidates: list[FieldCandidate],
        blueprint: DocumentBlueprint,
        *,
        image_quality: float,
        attempt: int,
        max_retries: int,
    ) -> dict[str, FieldDecision]:
        grouped: dict[str, list[FieldCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.path].append(candidate)
        return {
            path: self._decide_one(
                path,
                grouped.get(path, []),
                image_quality=image_quality,
                attempt=attempt,
                max_retries=max_retries,
                critical=path in blueprint.critical_paths,
            )
            for path in blueprint.output_paths
        }

    def _decide_one(
        self,
        path: str,
        items: list[FieldCandidate],
        *,
        image_quality: float,
        attempt: int,
        max_retries: int,
        critical: bool,
    ) -> FieldDecision:
        usable = [item for item in items if item.normalized_value is not None]
        if not usable:
            disposition = (
                "retry"
                if attempt < max_retries
                else ("review" if critical else "missing")
            )
            return FieldDecision(
                path=path,
                score=0.0,
                disposition=disposition,
                reasons=["no_grounded_candidate"],
            )

        clusters: dict[str, list[FieldCandidate]] = defaultdict(list)
        for item in usable:
            clusters[item.value_fingerprint()].append(item)
        winning = max(
            clusters.values(),
            key=lambda group: (
                len({source for item in group for source in item.support_sources}),
                max(
                    item.recognition_confidence * item.association_confidence
                    for item in group
                ),
            ),
        )
        candidate = max(
            winning,
            key=lambda item: item.recognition_confidence * item.association_confidence,
        )
        conflict = len(clusters) > 1
        total_votes = sum(len(group) for group in clusters.values())
        agreement = len(winning) / max(total_votes, 1)
        sources = {
            source.casefold()
            for item in winning
            for source in item.support_sources
            if source
        }
        fatal_codes = [
            code
            for item in winning
            for code in item.validation_codes
            if code.startswith(_FATAL_VALIDATION_PREFIXES)
        ]
        validation = 0.25 if fatal_codes else 1.0
        grounded = all(bool(item.evidence_ids) for item in winning)
        features = ConfidenceFeatures(
            image_quality=max(0.01, image_quality),
            recognition=max(0.01, max(item.recognition_confidence for item in winning)),
            association=max(0.01, max(item.association_confidence for item in winning)),
            agreement=max(0.01, agreement),
            validation=validation,
            grounded=grounded,
            independent_sources=len(sources),
            conflict=conflict,
        )
        values = {
            "image_quality": features.image_quality,
            "recognition": features.recognition,
            "association": features.association,
            "agreement": features.agreement,
            "validation": features.validation,
            "grounding": 1.0 if grounded else 0.01,
        }
        score = math.exp(
            sum(
                self.weights[name] * math.log(max(value, 1e-6))
                for name, value in values.items()
            )
        )
        reasons: list[str] = []
        if not grounded:
            score = 0.0
            reasons.append("ungrounded")
        if fatal_codes:
            score = min(score, 0.49)
            reasons.extend(sorted(set(fatal_codes)))
        if conflict:
            score = min(score, 0.59)
            reasons.append("source_conflict")
        if len(sources) < 2:
            score = min(score, 0.89)
            reasons.append("single_source_cap")
        score = round(max(0.0, min(1.0, score)), 6)
        if score >= self.accept_threshold:
            disposition = "accepted"
        elif attempt < max_retries:
            disposition = "retry"
        else:
            disposition = (
                "review"
                if critical or candidate.normalized_value is not None
                else "missing"
            )
        return FieldDecision(
            path=path,
            candidate=candidate,
            score=score,
            calibrated=False,
            disposition=disposition,
            features=features,
            reasons=list(dict.fromkeys(reasons)),
        )

    def recovery_plans(
        self,
        decisions: dict[str, FieldDecision],
        evidence: list[EvidenceRecord],
        *,
        attempt: int,
    ) -> list[RecoveryPlan]:
        by_id = {item.id: item for item in evidence}
        plans: list[RecoveryPlan] = []
        for path, decision in decisions.items():
            if decision.disposition != "retry":
                continue
            refs = decision.candidate.evidence_ids if decision.candidate else []
            boxes = [
                by_id[item].bbox for item in refs if item in by_id and by_id[item].bbox
            ]
            bbox = _union_boxes([item for item in boxes if item is not None])
            actions = [RecoveryAction.UPSCALE, RecoveryAction.CLAHE]
            if "control" in " ".join(refs).casefold() or path.endswith(
                ("male", "female")
            ):
                actions = [RecoveryAction.CHECKBOX_FOCUS, RecoveryAction.UPSCALE]
            plans.append(
                RecoveryPlan(
                    field_path=path,
                    actions=actions,
                    bbox=bbox,
                    evidence_ids=refs,
                    attempt=attempt,
                    reason_codes=decision.reasons or ["below_accept_threshold"],
                )
            )
        return plans


def _union_boxes(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )
