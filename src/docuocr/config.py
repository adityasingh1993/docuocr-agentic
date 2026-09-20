from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicySettings(SettingsModel):
    accept_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    document_quality_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    max_document_enhancements: int = Field(default=1, ge=0, le=5)
    max_field_retries: int = Field(default=2, ge=0, le=5)
    review_mode: Literal["queue", "interrupt"] = "queue"


class PaddleSettings(SettingsModel):
    enabled: bool = False
    engine: Literal[
        "paddle", "paddle_static", "paddle_dynamic", "transformers", "onnxruntime"
    ] = "transformers"
    device: str = "gpu:0"
    language: str = "en"
    layout_model_dir: str | None = None
    vl_rec_model_dir: str | None = None
    text_detection_model_dir: str | None = None
    text_recognition_model_dir: str | None = None
    layout_threshold: float = Field(default=0.30, gt=0.0, le=1.0)
    text_detection_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    text_box_threshold: float = Field(default=0.40, gt=0.0, le=1.0)
    text_recognition_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class VLMSettings(SettingsModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"
    api_key: str = "local-only"
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=900.0)
    max_tokens: int = Field(default=1200, ge=128, le=8192)
    allow_private_lan: bool = False

    @model_validator(mode="after")
    def enforce_local_endpoint(self) -> VLMSettings:
        if not self.enabled:
            return self
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("vlm.base_url must be an HTTP(S) URL")
        host = parsed.hostname.lower()
        if host in {"localhost", "::1"}:
            return self
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("VLM hostnames other than localhost are denied") from exc
        if address.is_loopback:
            return self
        if self.allow_private_lan and address.is_private:
            return self
        raise ValueError(
            "VLM endpoint must be loopback unless allow_private_lan is explicitly enabled"
        )


class TraceSettings(SettingsModel):
    include_values: bool = False
    save_crops: bool = True


class AppSettings(SettingsModel):
    schema_version: str = "1.0"
    artifact_root: str = "artifacts"
    offline: bool = True
    policy: PolicySettings = Field(default_factory=PolicySettings)
    paddle: PaddleSettings = Field(default_factory=PaddleSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)

    @model_validator(mode="after")
    def enforce_offline_model_paths(self) -> AppSettings:
        if not (self.offline and self.paddle.enabled):
            return self
        required = {
            "layout_model_dir": self.paddle.layout_model_dir,
            "vl_rec_model_dir": self.paddle.vl_rec_model_dir,
            "text_detection_model_dir": self.paddle.text_detection_model_dir,
            "text_recognition_model_dir": self.paddle.text_recognition_model_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Offline Paddle mode requires explicit paths: {missing}")
        absent = [
            name for name, value in required.items() if not Path(str(value)).is_dir()
        ]
        if absent:
            raise ValueError(f"Offline Paddle model directories do not exist: {absent}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppSettings:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        return cls.model_validate(payload)

    def artifact_path(self) -> Path:
        return Path(self.artifact_root).expanduser().resolve()
