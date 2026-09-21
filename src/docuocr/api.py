from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile

from docuocr.pipeline import DocumentPipeline


def create_app(config_path: str, blueprint_path: str) -> FastAPI:
    app = FastAPI(title="DocuOCR Agentic", version="0.3.1")
    pipeline = DocumentPipeline.from_files(config_path, blueprint_path)

    @app.on_event("shutdown")
    def shutdown() -> None:
        pipeline.close()

    @app.post("/v1/extractions")
    async def extract(
        document: Annotated[UploadFile, File()],
        job_id: Annotated[str | None, Form()] = None,
    ) -> dict:
        suffix = Path(document.filename or "document.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
            temp_path = Path(stream.name)
            while chunk := await document.read(1024 * 1024):
                stream.write(chunk)
        try:
            return pipeline.extract(temp_path, job_id=job_id)
        finally:
            temp_path.unlink(missing_ok=True)

    return app
