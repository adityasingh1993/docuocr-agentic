from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .blueprint import FieldSpec, NormalizationSettings


class NormalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | bool | int | float | None = None
    codes: list[str] = Field(default_factory=list)


_TRUE = {"true", "yes", "y", "checked", "1", "x"}
_FALSE = {"false", "no", "n", "unchecked", "0"}


def normalize_value(
    raw: Any,
    spec: FieldSpec,
    settings: NormalizationSettings,
) -> NormalizationResult:
    if raw is None:
        return NormalizationResult(value=None, codes=["missing_value"])
    if spec.type == "boolean":
        if isinstance(raw, bool):
            return NormalizationResult(value=raw)
        token = str(raw).strip().casefold()
        if token in _TRUE:
            return NormalizationResult(value=True)
        if token in _FALSE:
            return NormalizationResult(value=False)
        return NormalizationResult(value=None, codes=["invalid_boolean"])

    text = " ".join(str(raw).strip().split())
    if not text:
        return NormalizationResult(value=None, codes=["missing_value"])
    if spec.type == "date":
        return _normalize_date(text, settings.date_order)
    if spec.type == "time":
        return _normalize_time(text)
    if spec.type == "weight":
        return _normalize_weight(text, settings.weight_unit)
    return NormalizationResult(value=text)


def _normalize_date(text: str, order: str) -> NormalizationResult:
    normalized = text.replace(".", "/").replace("-", "/")
    iso_match = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", normalized)
    if iso_match:
        parts = tuple(int(item) for item in iso_match.groups())
        return _validated_date(*parts)

    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", normalized)
    if not match:
        for pattern in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"):
            try:
                parsed = datetime.strptime(text, pattern).replace(tzinfo=UTC)
                return NormalizationResult(value=parsed.date().isoformat())
            except ValueError:
                continue
        return NormalizationResult(value=None, codes=["invalid_date"])

    first, second, year = (int(item) for item in match.groups())
    year = year + 2000 if year < 100 else year
    if order == "REJECT_AMBIGUOUS" and first <= 12 and second <= 12:
        return NormalizationResult(value=None, codes=["ambiguous_date"])
    if order == "MDY":
        month, day = first, second
    else:
        day, month = first, second
    return _validated_date(year, month, day)


def _validated_date(year: int, month: int, day: int) -> NormalizationResult:
    try:
        value = date(year, month, day).isoformat()
    except ValueError:
        return NormalizationResult(value=None, codes=["invalid_date"])
    return NormalizationResult(value=value)


def _normalize_time(text: str) -> NormalizationResult:
    compact = text.casefold().replace(".", "").replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::?(\d{2}))?(am|pm)?", compact)
    if not match:
        return NormalizationResult(value=None, codes=["invalid_time"])
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3)
    if suffix:
        if not 1 <= hour <= 12:
            return NormalizationResult(value=None, codes=["invalid_time"])
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    if hour > 23 or minute > 59:
        return NormalizationResult(value=None, codes=["invalid_time"])
    return NormalizationResult(value=f"{hour:02d}:{minute:02d}")


def _normalize_weight(text: str, output_unit: str) -> NormalizationResult:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(kg|kgs|g|grams?)?", text.casefold())
    if not match:
        return NormalizationResult(value=None, codes=["invalid_weight"])
    amount = float(match.group(1).replace(",", "."))
    source_unit = match.group(2) or ("g" if amount > 20 else "kg")
    kilograms = amount / 1000.0 if source_unit.startswith("g") else amount
    if output_unit == "g":
        return NormalizationResult(value=f"{round(kilograms * 1000):d} g")
    rendered = f"{kilograms:.3f}".rstrip("0").rstrip(".")
    return NormalizationResult(value=f"{rendered} kg")
