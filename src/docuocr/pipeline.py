from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Self

from docuocr.config import AppSettings
from docuocr.engines.opencv import OpenCVControlEngine
from docuocr.engines.paddle import PaddleLayoutEngine, PaddleTextEngine
from docuocr.engines.sidecar import NullLayoutEngine, NullTextEngine, SidecarBundle
from docuocr.engines.vlm import LocalVLMClient
from docuocr.extraction.blueprint import DocumentBlueprint
from docuocr.workflow.graph import build_graph
from docuocr.workflow.nodes import WorkflowNodes


class DocumentPipeline:
    def __init__(
        self,
        *,
        settings: AppSettings,
        blueprint: DocumentBlueprint,
        sidecar_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        if sidecar_path:
            sidecar = SidecarBundle(sidecar_path)
            text_engine = sidecar
            layout_engine = sidecar
            control_engine = sidecar
        else:
            text_engine = (
                PaddleTextEngine(settings.paddle)
                if settings.paddle.enabled
                else NullTextEngine()
            )
            layout_engine = (
                PaddleLayoutEngine(settings.paddle)
                if settings.paddle.enabled
                else NullLayoutEngine()
            )
            control_engine = OpenCVControlEngine()
        vlm = LocalVLMClient(settings.vlm) if settings.vlm.enabled else None
        nodes = WorkflowNodes(
            settings=settings,
            blueprint=blueprint,
            text_engine=text_engine,
            layout_engine=layout_engine,
            control_engine=control_engine,
            vlm=vlm,
        )
        settings.artifact_path().mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            settings.artifact_path() / "checkpoints.sqlite", check_same_thread=False
        )
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.graph = build_graph(nodes, checkpointer=SqliteSaver(self._connection))

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        blueprint_path: str | Path,
        *,
        sidecar_path: str | Path | None = None,
    ) -> DocumentPipeline:
        return cls(
            settings=AppSettings.from_yaml(config_path),
            blueprint=DocumentBlueprint.from_yaml(blueprint_path),
            sidecar_path=sidecar_path,
        )

    def extract(
        self, source_path: str | Path, *, job_id: str | None = None
    ) -> dict[str, Any]:
        job_id = job_id or uuid.uuid4().hex
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_id) is None:
            raise ValueError(
                "job_id must contain only letters, digits, dot, underscore, or hyphen"
            )
        result = self.graph.invoke(
            {
                "source_path": str(source_path),
                "blueprint_path": "loaded",
                "job_id": job_id,
                "thread_id": job_id,
            },
            config={"configurable": {"thread_id": job_id}},
        )
        return result["result"]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
