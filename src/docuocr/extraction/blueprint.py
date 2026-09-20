from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlueprintModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldSpec(BlueprintModel):
    type: Literal["string", "date", "time", "weight", "boolean"] = "string"
    aliases: list[str] = Field(min_length=1)
    strategies: list[
        Literal["same_line", "right_of_label", "below_label", "vlm_grounded"]
    ] = Field(default_factory=lambda: ["right_of_label", "below_label"])
    critical: bool = False
    required: bool = False
    pattern: str | None = None


class CheckboxOption(BlueprintModel):
    output_path: str
    aliases: list[str] = Field(min_length=1)


class CheckboxGroup(BlueprintModel):
    exclusive: bool = False
    aliases: list[str] = Field(default_factory=list)
    options: dict[str, CheckboxOption] = Field(min_length=1)


class NormalizationSettings(BlueprintModel):
    date_order: Literal["DMY", "MDY", "YMD", "REJECT_AMBIGUOUS"] = "REJECT_AMBIGUOUS"
    weight_unit: Literal["kg", "g"] = "kg"


class ValidationSettings(BlueprintModel):
    birth_weight_kg: tuple[float, float] = (0.3, 8.0)
    sample_collection_not_before_birth: bool = True
    sex_mutually_exclusive: bool = True


class DocumentBlueprint(BlueprintModel):
    id: str
    version: str
    document_type: str
    anchors: list[str] = Field(default_factory=list)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    fields: dict[str, FieldSpec]
    checkbox_groups: dict[str, CheckboxGroup] = Field(default_factory=dict)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)

    @model_validator(mode="after")
    def validate_paths(self) -> DocumentBlueprint:
        paths = list(self.fields)
        for group in self.checkbox_groups.values():
            paths.extend(option.output_path for option in group.options.values())
        duplicates = {path for path in paths if paths.count(path) > 1}
        if duplicates:
            raise ValueError(f"Duplicate output paths: {sorted(duplicates)}")
        invalid = [path for path in paths if not path.startswith("data.")]
        if invalid:
            raise ValueError(f"Blueprint paths must start with data.: {invalid}")
        return self

    @property
    def output_paths(self) -> list[str]:
        paths = list(self.fields)
        for group in self.checkbox_groups.values():
            paths.extend(option.output_path for option in group.options.values())
        return list(dict.fromkeys(paths))

    @property
    def critical_paths(self) -> set[str]:
        return {
            path for path, spec in self.fields.items() if spec.critical or spec.required
        }

    @property
    def checkbox_paths(self) -> set[str]:
        return {
            option.output_path
            for group in self.checkbox_groups.values()
            for option in group.options.values()
        }

    def vlm_hints(self, paths: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Describe field semantics without changing raw OCR/value text."""

        requested = set(paths)
        hints: dict[str, dict[str, Any]] = {}
        for path, spec in self.fields.items():
            if path not in requested:
                continue
            hints[path] = {
                "kind": "printed_label_value",
                "valueType": spec.type,
                "printedLabelAliases": spec.aliases,
            }
        for group_name, group in self.checkbox_groups.items():
            for option_name, option in group.options.items():
                if option.output_path not in requested:
                    continue
                hints[option.output_path] = {
                    "kind": "control_option",
                    "group": group_name,
                    "groupLabelAliases": group.aliases,
                    "exclusive": group.exclusive,
                    "option": option_name,
                    "printedLabelAliases": option.aliases,
                }
        return hints

    @classmethod
    def from_yaml(cls, path: str | Path) -> DocumentBlueprint:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload: dict[str, Any] = yaml.safe_load(stream) or {}
        return cls.model_validate(payload)
