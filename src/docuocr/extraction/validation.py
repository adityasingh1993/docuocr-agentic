from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from docuocr.models import FieldCandidate

from .blueprint import DocumentBlueprint


def validate_candidates(
    candidates: list[FieldCandidate], blueprint: DocumentBlueprint
) -> list[FieldCandidate]:
    validated: list[FieldCandidate] = []
    for candidate in candidates:
        codes = list(candidate.validation_codes)
        spec = blueprint.fields.get(candidate.path)
        if candidate.normalized_value is None and "missing_value" not in codes:
            codes.append("normalization_failed")
        if (
            spec
            and spec.pattern
            and candidate.normalized_value is not None
            and re.fullmatch(spec.pattern, str(candidate.normalized_value)) is None
        ):
            codes.append("pattern_mismatch")
        if candidate.path == "data.baby.birthWeight" and candidate.normalized_value:
            match = re.fullmatch(r"(\d+(?:\.\d+)?) kg", str(candidate.normalized_value))
            if not match:
                codes.append("invalid_weight")
            else:
                low, high = blueprint.validation.birth_weight_kg
                if not low <= float(match.group(1)) <= high:
                    codes.append("birth_weight_out_of_range")
        validated.append(
            candidate.model_copy(
                update={"validation_codes": list(dict.fromkeys(codes))}
            )
        )
    return _apply_cross_field_rules(validated, blueprint)


def _apply_cross_field_rules(
    candidates: list[FieldCandidate], blueprint: DocumentBlueprint
) -> list[FieldCandidate]:
    additions: dict[int, list[str]] = defaultdict(list)
    best = _best_by_path(candidates)
    if blueprint.validation.sample_collection_not_before_birth:
        birth = best.get("data.baby.dateOfBirth")
        collected = best.get("data.baby.sample.collection.date")
        if (
            birth
            and collected
            and birth.normalized_value
            and collected.normalized_value
        ):
            try:
                if date.fromisoformat(
                    str(collected.normalized_value)
                ) < date.fromisoformat(str(birth.normalized_value)):
                    additions[id(collected)].append("collection_before_birth")
            except ValueError:
                pass
    if blueprint.validation.sex_mutually_exclusive:
        male = best.get("data.baby.male")
        female = best.get("data.baby.female")
        if (
            male
            and female
            and male.normalized_value is True
            and female.normalized_value is True
        ):
            additions[id(male)].append("exclusive_group_conflict")
            additions[id(female)].append("exclusive_group_conflict")
    return [
        candidate.model_copy(
            update={
                "validation_codes": list(
                    dict.fromkeys(
                        candidate.validation_codes + additions.get(id(candidate), [])
                    )
                )
            }
        )
        for candidate in candidates
    ]


def _best_by_path(candidates: list[FieldCandidate]) -> dict[str, FieldCandidate]:
    result: dict[str, FieldCandidate] = {}
    for candidate in candidates:
        previous = result.get(candidate.path)
        strength = candidate.recognition_confidence * candidate.association_confidence
        previous_strength = (
            previous.recognition_confidence * previous.association_confidence
            if previous
            else -1.0
        )
        if strength > previous_strength:
            result[candidate.path] = candidate
    return result
