from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class TraceWriter:
    """Append-only, idempotent JSONL trace with values redacted by default."""

    def __init__(self, path: str | Path, include_values: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_values = include_values
        self._lock = threading.Lock()
        self._event_ids = self._load_event_ids()

    def _load_event_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_id := event.get("eventId"):
                    ids.add(str(event_id))
        return ids

    @staticmethod
    def event_id(run_id: str, node: str, attempt: int, input_sha256: str | None) -> str:
        value = f"{run_id}|{node}|{attempt}|{input_sha256 or '-'}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    def append(
        self,
        *,
        run_id: str,
        node: str,
        attempt: int,
        status: str,
        started_at: str,
        ended_at: str,
        duration_ms: float,
        input_sha256: str | None = None,
        output: Any = None,
        confidence: float | None = None,
        decision: str | None = None,
        model_id: str | None = None,
        evidence_refs: list[str] | None = None,
        warnings: list[str] | None = None,
        values: Any = None,
    ) -> dict[str, Any]:
        event_id = self.event_id(run_id, node, attempt, input_sha256)
        event: dict[str, Any] = {
            "eventId": event_id,
            "runId": run_id,
            "node": node,
            "attempt": attempt,
            "status": status,
            "startedAt": started_at,
            "endedAt": ended_at,
            "durationMs": round(duration_ms, 3),
            "inputSha256": input_sha256,
            "outputSha256": sha256_json(output) if output is not None else None,
            "confidence": confidence,
            "decision": decision,
            "modelId": model_id,
            "evidenceRefs": evidence_refs or [],
            "warnings": warnings or [],
        }
        if self.include_values and values is not None:
            event["values"] = values
        with self._lock:
            if event_id in self._event_ids:
                return event
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
            self._event_ids.add(event_id)
        return event
