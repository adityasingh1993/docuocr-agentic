from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from docuocr.models import ControlState, FieldCandidate, FormControl, OCRSpan

from .blueprint import CheckboxOption, DocumentBlueprint, FieldSpec
from .normalization import normalize_value


def canonical_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).split())


def alias_score(text: str, aliases: Iterable[str]) -> float:
    normalized = canonical_text(text)
    best = 0.0
    for alias in aliases:
        target = canonical_text(alias)
        if not target:
            continue
        if normalized == target:
            score = 1.0
        elif target in normalized:
            score = min(0.98, len(target) / max(len(normalized), 1) + 0.35)
        else:
            score = SequenceMatcher(None, normalized, target).ratio()
        best = max(best, score)
    return best


def associate_controls(
    controls: list[FormControl], spans: list[OCRSpan]
) -> list[FormControl]:
    by_id = {span.id: span for span in spans}
    result: list[FormControl] = []
    for control in controls:
        if control.associated_label_id in by_id:
            result.append(control)
            continue
        _, cy = control.bbox.center
        choices: list[tuple[float, OCRSpan]] = []
        for span in spans:
            if span.page != control.page:
                continue
            _, sy = span.bbox.center
            if abs(cy - sy) > max(control.bbox.height, span.bbox.height):
                continue
            distance = span.bbox.x1 - control.bbox.x2
            if -control.bbox.width * 0.25 <= distance <= control.bbox.width * 16:
                choices.append((max(0.0, distance) + abs(cy - sy) * 2.0, span))
        chosen = min(choices, default=None, key=lambda item: item[0])
        result.append(
            control.model_copy(
                update={"associated_label_id": chosen[1].id if chosen else None}
            )
        )
    return result


def map_rule_candidates(
    blueprint: DocumentBlueprint,
    spans: list[OCRSpan],
    controls: list[FormControl],
    *,
    attempt: int = 0,
    only_paths: set[str] | None = None,
) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for path, spec in blueprint.fields.items():
        if only_paths is not None and path not in only_paths:
            continue
        candidates.extend(_text_candidates(path, spec, spans, blueprint, attempt))
    associated = associate_controls(controls, spans)
    by_id = {span.id: span for span in spans}
    for group in blueprint.checkbox_groups.values():
        for option in group.options.values():
            if only_paths is not None and option.output_path not in only_paths:
                continue
            candidates.extend(_control_candidates(option, associated, by_id, attempt))
    return candidates


def _text_candidates(
    path: str,
    spec: FieldSpec,
    spans: list[OCRSpan],
    blueprint: DocumentBlueprint,
    attempt: int,
) -> list[FieldCandidate]:
    found = _text_candidates_for_aliases(
        path,
        spec,
        spans,
        blueprint,
        attempt,
        aliases=spec.aliases,
        value_part="whole",
        match_threshold=0.78,
        source_prefix="",
    )
    if spec.group_aliases:
        found.extend(
            _text_candidates_for_aliases(
                path,
                spec,
                spans,
                blueprint,
                attempt,
                aliases=spec.group_aliases,
                value_part=spec.group_value_part,
                match_threshold=0.90,
                source_prefix="group_",
                group_aliases=True,
            )
        )
    deduped: dict[tuple[str, str, tuple[str, ...]], FieldCandidate] = {}
    for candidate in found:
        key = (
            candidate.source,
            candidate.value_fingerprint(),
            tuple(candidate.evidence_ids),
        )
        previous = deduped.get(key)
        if previous is None or (
            candidate.recognition_confidence * candidate.association_confidence
            > previous.recognition_confidence * previous.association_confidence
        ):
            deduped[key] = candidate
    return list(deduped.values())


def _text_candidates_for_aliases(
    path: str,
    spec: FieldSpec,
    spans: list[OCRSpan],
    blueprint: DocumentBlueprint,
    attempt: int,
    *,
    aliases: Iterable[str],
    value_part: str,
    match_threshold: float,
    source_prefix: str,
    group_aliases: bool = False,
) -> list[FieldCandidate]:
    aliases = list(aliases)
    found: list[FieldCandidate] = []
    for label in spans:
        match = (
            _group_alias_score(label.text, aliases)
            if group_aliases
            else alias_score(label.text, aliases)
        )
        if match < match_threshold:
            continue
        same_line = _same_line_suffix(label.text, aliases)
        if same_line and "same_line" in spec.strategies:
            selected = _select_value_part(same_line, value_part)
            if selected is not None:
                normalized = normalize_value(selected, spec, blueprint.normalization)
                found.append(
                    FieldCandidate(
                        path=path,
                        raw_value=selected,
                        normalized_value=normalized.value,
                        evidence_ids=[label.id],
                        source=f"rule_{source_prefix}same_line",
                        support_sources=[label.source],
                        recognition_confidence=label.confidence,
                        association_confidence=match,
                        attempt=attempt,
                        validation_codes=normalized.codes,
                    )
                )
        neighbours = _neighbours(label, spans, spec.strategies)
        # A strong nearest value is preferable to collecting unrelated fields farther below.
        # We keep multiple weak alternatives so the confidence layer can surface conflicts.
        if neighbours and neighbours[0][0] >= 0.78:
            neighbours = neighbours[:1]
        for geometry_score, value_span, strategy in neighbours[:2]:
            if _looks_like_any_label(value_span.text, blueprint):
                continue
            selected = _select_value_part(value_span.text, value_part)
            if selected is None:
                continue
            normalized = normalize_value(selected, spec, blueprint.normalization)
            found.append(
                FieldCandidate(
                    path=path,
                    raw_value=selected,
                    normalized_value=normalized.value,
                    evidence_ids=[label.id, value_span.id],
                    source=f"rule_{source_prefix}{strategy}",
                    support_sources=list(
                        dict.fromkeys([label.source, value_span.source])
                    ),
                    recognition_confidence=min(label.confidence, value_span.confidence),
                    association_confidence=max(0.0, min(1.0, match * geometry_score)),
                    attempt=attempt,
                    validation_codes=normalized.codes,
                )
            )
    return found


def _group_alias_score(text: str, aliases: Iterable[str]) -> float:
    normalized = canonical_text(text)
    best = alias_score(text, aliases)
    for alias in aliases:
        target = canonical_text(alias)
        if target and (
            normalized == target
            or normalized.startswith(f"{target} ")
            or normalized.startswith(f"{target}:")
        ):
            best = max(best, 1.0 if normalized == target else 0.96)
    return best


def _select_value_part(value: str, part: str) -> str | None:
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if part == "first_token":
        return cleaned.split(maxsplit=1)[0]
    if part == "remaining_tokens":
        pieces = cleaned.split(maxsplit=1)
        return pieces[1] if len(pieces) == 2 else None
    return cleaned


def _same_line_suffix(text: str, aliases: Iterable[str]) -> str | None:
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(re.escape(alias), text, flags=re.IGNORECASE)
        if not match:
            continue
        suffix = text[match.end() :].lstrip(" :-–—\t")
        if suffix:
            return suffix
    return None


def _neighbours(
    label: OCRSpan,
    spans: list[OCRSpan],
    strategies: Iterable[str],
) -> list[tuple[float, OCRSpan, str]]:
    choices: list[tuple[float, OCRSpan, str]] = []
    for span in spans:
        if span.id == label.id or span.page != label.page:
            continue
        _, label_y = label.bbox.center
        _, value_y = span.bbox.center
        same_row = (
            abs(label_y - value_y) <= max(label.bbox.height, span.bbox.height) * 0.80
        )
        right_gap = span.bbox.x1 - label.bbox.x2
        if (
            "right_of_label" in strategies
            and same_row
            and right_gap >= -label.bbox.width * 0.05
        ):
            distance = right_gap / max(label.bbox.height, 1)
            if distance <= 24:
                choices.append((1.0 / (1.0 + distance / 8.0), span, "right_of_label"))
        vertical_gap = span.bbox.y1 - label.bbox.y2
        horizontal_overlap = max(
            0,
            min(label.bbox.x2, span.bbox.x2) - max(label.bbox.x1, span.bbox.x1),
        )
        overlap_ratio = horizontal_overlap / max(
            1, min(label.bbox.width, span.bbox.width)
        )
        if (
            "below_label" in strategies
            and vertical_gap >= -label.bbox.height * 0.05
            and vertical_gap <= label.bbox.height * 4.0
            and overlap_ratio >= 0.20
        ):
            choices.append(
                (
                    0.86 / (1.0 + vertical_gap / max(label.bbox.height * 5, 1)),
                    span,
                    "below_label",
                )
            )
    return sorted(choices, key=lambda item: item[0], reverse=True)


def _looks_like_any_label(text: str, blueprint: DocumentBlueprint) -> bool:
    for spec in blueprint.fields.values():
        if alias_score(text, [*spec.aliases, *spec.group_aliases]) >= 0.93:
            return True
    return False


def _control_candidates(
    option: CheckboxOption,
    controls: list[FormControl],
    spans_by_id: dict[str, OCRSpan],
    attempt: int,
) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for control in controls:
        label = spans_by_id.get(control.associated_label_id or "")
        if label is None:
            continue
        match = alias_score(label.text, option.aliases)
        if match < 0.76:
            continue
        if control.state == ControlState.AMBIGUOUS:
            value: bool | None = None
            codes = ["ambiguous_control_state"]
        else:
            value = control.state == ControlState.CHECKED
            codes = []
        candidates.append(
            FieldCandidate(
                path=option.output_path,
                raw_value=value,
                normalized_value=value,
                evidence_ids=[control.id, label.id],
                source="opencv_control",
                support_sources=["opencv"],
                recognition_confidence=control.state_confidence,
                association_confidence=match,
                attempt=attempt,
                validation_codes=codes,
            )
        )
    return candidates


def choose_raw_by_path(candidates: list[FieldCandidate]) -> dict[str, Any]:
    grouped: dict[str, list[FieldCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.path].append(candidate)
    return {
        path: max(
            items,
            key=lambda item: item.recognition_confidence * item.association_confidence,
        ).raw_value
        for path, items in grouped.items()
    }
