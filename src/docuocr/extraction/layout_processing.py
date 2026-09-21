from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from docuocr.config import AssociationSettings
from docuocr.cv.enhance import ImageEnhancer
from docuocr.cv.quality import _cv2
from docuocr.engines.base import TextEngine
from docuocr.models import (
    BBox,
    EnhancementRecord,
    FormControl,
    LayoutBlock,
    LayoutBlockOCRRecord,
    OCRSpan,
    RecoveryAction,
)
from docuocr.trace import sha256_file


@dataclass(frozen=True)
class ProcessedLayoutBlock:
    block: LayoutBlock
    crop_bbox: BBox
    crop_path: Path | None
    recognized_spans: tuple[OCRSpan, ...] = ()
    record: LayoutBlockOCRRecord | None = None


@dataclass(frozen=True)
class LayoutRecoveryTask:
    block: LayoutBlock
    target_paths: tuple[str, ...]
    actions: tuple[RecoveryAction, ...]


@dataclass(frozen=True)
class RecoveredLayoutBlock:
    task: LayoutRecoveryTask
    crop_bbox: BBox
    crop_path: Path | None
    enhancement: EnhancementRecord | None
    recognized_spans: tuple[OCRSpan, ...] = ()
    recognition_attempts: int = 0
    error: str | None = None


@dataclass(frozen=True)
class _PreparedCrop:
    index: int
    block: LayoutBlock
    crop_bbox: BBox
    crop_path: Path


@dataclass(frozen=True)
class _RecognitionOutcome:
    spans: tuple[OCRSpan, ...]
    attempts: int
    error: str | None = None


class LayoutBlockProcessor:
    """Crop and OCR layout regions with bounded, isolated parallel workers."""

    def __init__(self, text_engine: TextEngine) -> None:
        self.text_engine = text_engine
        self._worker_local = threading.local()

    def process(
        self,
        *,
        image_path: str | Path,
        run_dir: str | Path,
        blocks: list[LayoutBlock],
        settings: AssociationSettings,
        attempt: int,
    ) -> tuple[list[ProcessedLayoutBlock], list[str]]:
        cv2 = _cv2()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")
        height, width = image.shape[:2]
        selected = self._select_blocks(blocks, width, height, settings)
        base_results = [
            ProcessedLayoutBlock(block=block, crop_bbox=bbox, crop_path=None)
            for block, bbox in selected
        ]
        if not settings.layout_block_ocr_enabled or not selected:
            return base_results, []
        if not getattr(self.text_engine, "supports_region_ocr", False):
            return base_results, [
                f"layout_block_ocr_unsupported:{self.text_engine.model_id}"
            ]

        crop_dir = Path(run_dir) / "images" / "layout-blocks"
        crop_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[_PreparedCrop] = []
        warnings: list[str] = []
        for index, (block, bbox) in enumerate(selected):
            crop_path = crop_dir / f"layout-block-{attempt}-{index:03d}.png"
            crop = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2].copy()
            if crop.size == 0 or not cv2.imwrite(str(crop_path), crop):
                warnings.append(f"layout_block_crop_failed:{block.id}")
                continue
            prepared.append(
                _PreparedCrop(
                    index=index,
                    block=block,
                    crop_bbox=bbox,
                    crop_path=crop_path,
                )
            )

        outcomes, recognition_warnings = self._recognize_many(
            prepared,
            attempt=attempt,
            workers=settings.layout_parallel_workers,
            retries=settings.layout_block_retries,
            id_scope="layout",
            source_suffix="layout_crop",
            translate_to_page=True,
        )
        warnings.extend(recognition_warnings)

        processed_by_index = {item.block.id: item for item in base_results}
        for task, outcome in zip(prepared, outcomes, strict=True):
            if outcome.error:
                warnings.append(
                    f"layout_block_ocr_failed:{task.block.id}:{outcome.error}"
                )
            record = LayoutBlockOCRRecord(
                block_id=task.block.id,
                label=task.block.label,
                bbox=task.crop_bbox,
                crop_path=str(task.crop_path),
                crop_sha256=sha256_file(task.crop_path),
                span_ids=[item.id for item in outcome.spans],
                recognition_attempts=outcome.attempts,
                status="failed" if outcome.error else "succeeded",
                error=outcome.error,
                model_id=self.text_engine.model_id,
                attempt=attempt,
            )
            processed_by_index[task.block.id] = ProcessedLayoutBlock(
                block=task.block,
                crop_bbox=task.crop_bbox,
                crop_path=task.crop_path,
                recognized_spans=outcome.spans,
                record=record,
            )
        return [processed_by_index[item.id] for item, _ in selected], warnings

    def recover(
        self,
        *,
        image_path: str | Path,
        run_dir: str | Path,
        tasks: list[LayoutRecoveryTask],
        settings: AssociationSettings,
        attempt: int,
        enhancer: ImageEnhancer,
    ) -> tuple[list[RecoveredLayoutBlock], list[str]]:
        """Enhance and OCR only the selected layout blocks."""

        if not tasks:
            return [], []
        if not getattr(self.text_engine, "supports_region_ocr", False):
            warning = f"layout_recovery_ocr_unsupported:{self.text_engine.model_id}"
            return [
                RecoveredLayoutBlock(
                    task=task,
                    crop_bbox=task.block.bbox,
                    crop_path=None,
                    enhancement=None,
                    error=warning,
                )
                for task in tasks
            ], [warning]

        crop_dir = Path(run_dir) / "images" / "layout-recovery"
        crop_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[_PreparedCrop] = []
        enhancements: dict[int, EnhancementRecord] = {}
        results: dict[int, RecoveredLayoutBlock] = {}
        warnings: list[str] = []
        for index, task in enumerate(tasks):
            crop_path = crop_dir / f"layout-recovery-{attempt}-{index:03d}.png"
            try:
                record = enhancer.apply(
                    image_path,
                    crop_path,
                    task.actions,
                    attempt=attempt,
                    bbox=task.block.bbox,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one layout failure
                error = _safe_error(exc)
                warnings.append(
                    f"layout_recovery_enhancement_failed:{task.block.id}:{error}"
                )
                results[index] = RecoveredLayoutBlock(
                    task=task,
                    crop_bbox=task.block.bbox,
                    crop_path=None,
                    enhancement=None,
                    error=error,
                )
                continue
            crop_bbox = record.target_bbox or task.block.bbox
            enhancements[index] = record
            prepared.append(
                _PreparedCrop(
                    index=index,
                    block=task.block,
                    crop_bbox=crop_bbox,
                    crop_path=crop_path,
                )
            )

        outcomes, recognition_warnings = self._recognize_many(
            prepared,
            attempt=attempt,
            workers=settings.layout_parallel_workers,
            retries=settings.layout_block_retries,
            id_scope="layout-recovery",
            source_suffix=f"layout_recovery:{attempt}",
            translate_to_page=False,
        )
        warnings.extend(recognition_warnings)
        tasks_by_index = {index: task for index, task in enumerate(tasks)}
        for prepared_crop, outcome in zip(prepared, outcomes, strict=True):
            task = tasks_by_index[prepared_crop.index]
            if outcome.error:
                warnings.append(
                    "layout_recovery_ocr_failed:"
                    f"{task.block.id}:{outcome.error}"
                )
            results[prepared_crop.index] = RecoveredLayoutBlock(
                task=task,
                crop_bbox=prepared_crop.crop_bbox,
                crop_path=prepared_crop.crop_path,
                enhancement=enhancements[prepared_crop.index],
                recognized_spans=outcome.spans,
                recognition_attempts=outcome.attempts,
                error=outcome.error,
            )
        return [results[index] for index in range(len(tasks))], warnings

    def _recognize_many(
        self,
        prepared: list[_PreparedCrop],
        *,
        attempt: int,
        workers: int,
        retries: int,
        id_scope: str,
        source_suffix: str,
        translate_to_page: bool,
    ) -> tuple[list[_RecognitionOutcome], list[str]]:
        if not prepared:
            return [], []
        requested_workers = min(workers, len(prepared))
        fork = getattr(self.text_engine, "fork", None)
        actual_workers = requested_workers
        warnings: list[str] = []
        if actual_workers > 1 and not callable(fork):
            actual_workers = 1
            warnings.append(
                f"layout_parallelism_downgraded:{self.text_engine.model_id}"
            )

        if actual_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_workers, thread_name_prefix="layout-ocr"
            ) as executor:
                outcomes = list(
                    executor.map(
                        lambda task: self._recognize(
                            task,
                            attempt=attempt,
                            retries=retries,
                            isolated=True,
                            id_scope=id_scope,
                            source_suffix=source_suffix,
                            translate_to_page=translate_to_page,
                        ),
                        prepared,
                    )
                )
        else:
            outcomes = [
                self._recognize(
                    task,
                    attempt=attempt,
                    retries=retries,
                    isolated=False,
                    id_scope=id_scope,
                    source_suffix=source_suffix,
                    translate_to_page=translate_to_page,
                )
                for task in prepared
            ]
        return outcomes, warnings

    def _recognize(
        self,
        task: _PreparedCrop,
        *,
        attempt: int,
        retries: int,
        isolated: bool,
        id_scope: str,
        source_suffix: str,
        translate_to_page: bool,
    ) -> _RecognitionOutcome:
        engine = self._engine_for_worker(isolated)
        last_error: Exception | None = None
        for recognition_attempt in range(1, retries + 2):
            try:
                local_spans = engine.extract(
                    task.crop_path,
                    page=task.block.page,
                    attempt=attempt,
                    id_prefix=f"ocr:{id_scope}{attempt}:{task.index:03d}",
                )
                if translate_to_page:
                    recognized = tuple(
                        _translate_span(item, task.crop_bbox, source_suffix)
                        for item in local_spans
                    )
                else:
                    recognized = tuple(
                        item.model_copy(
                            update={"source": f"{item.source}:{source_suffix}"}
                        )
                        for item in local_spans
                    )
                return _RecognitionOutcome(
                    spans=recognized, attempts=recognition_attempt
                )
            except Exception as exc:  # isolate one region from the document run
                last_error = exc
        assert last_error is not None
        return _RecognitionOutcome(
            spans=(),
            attempts=retries + 1,
            error=_safe_error(last_error),
        )

    def _engine_for_worker(self, isolated: bool) -> TextEngine:
        if not isolated:
            return self.text_engine
        engine = getattr(self._worker_local, "engine", None)
        if engine is None:
            engine = self.text_engine.fork()
            self._worker_local.engine = engine
        return engine

    @staticmethod
    def _select_blocks(
        blocks: list[LayoutBlock],
        width: int,
        height: int,
        settings: AssociationSettings,
    ) -> list[tuple[LayoutBlock, BBox]]:
        excluded = {
            item.casefold().strip().replace(" ", "_")
            for item in settings.excluded_layout_labels
        }
        page_area = max(1, width * height)
        selected: list[tuple[LayoutBlock, BBox]] = []
        for block in sorted(
            blocks,
            key=lambda item: (
                item.page,
                item.order if item.order is not None else 1_000_000,
                item.bbox.y1,
                item.bbox.x1,
                item.id,
            ),
        ):
            label = block.label.casefold().strip().replace(" ", "_")
            if label in excluded:
                continue
            bbox = _clamped_expanded_bbox(
                block.bbox,
                settings.crop_padding_pixels,
                width,
                height,
            )
            if bbox is None:
                continue
            if (bbox.width * bbox.height) / page_area < settings.min_block_area_fraction:
                continue
            selected.append((block, bbox))
            if len(selected) >= settings.max_layout_blocks:
                break
        return selected


def spans_in_region(spans: list[OCRSpan], block: LayoutBlock) -> list[OCRSpan]:
    return [
        item
        for item in spans
        if item.page == block.page and _belongs_to_region(item.bbox, block.bbox)
    ]


def controls_in_region(
    controls: list[FormControl], block: LayoutBlock
) -> list[FormControl]:
    return [
        item
        for item in controls
        if item.page == block.page and _belongs_to_region(item.bbox, block.bbox)
    ]


def _belongs_to_region(item: BBox, region: BBox) -> bool:
    center_x, center_y = item.center
    if (
        region.x1 <= center_x <= region.x2
        and region.y1 <= center_y <= region.y2
    ):
        return True
    x1 = max(item.x1, region.x1)
    y1 = max(item.y1, region.y1)
    x2 = min(item.x2, region.x2)
    y2 = min(item.y2, region.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, item.width * item.height) >= 0.35


def _translate_span(span: OCRSpan, crop_bbox: BBox, source_suffix: str) -> OCRSpan:
    width = crop_bbox.width
    height = crop_bbox.height
    x1 = min(max(span.bbox.x1, 0), width - 1)
    y1 = min(max(span.bbox.y1, 0), height - 1)
    x2 = min(max(span.bbox.x2, x1 + 1), width)
    y2 = min(max(span.bbox.y2, y1 + 1), height)
    return span.model_copy(
        update={
            "bbox": BBox(
                x1=crop_bbox.x1 + x1,
                y1=crop_bbox.y1 + y1,
                x2=crop_bbox.x1 + x2,
                y2=crop_bbox.y1 + y2,
            ),
            "source": f"{span.source}:{source_suffix}",
        }
    )


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())[:240]
    return f"{type(exc).__name__}:{message}"


def _clamped_expanded_bbox(
    bbox: BBox, padding: int, width: int, height: int
) -> BBox | None:
    x1 = max(0, min(width - 1, bbox.x1 - padding))
    y1 = max(0, min(height - 1, bbox.y1 - padding))
    x2 = max(0, min(width, bbox.x2 + padding))
    y2 = max(0, min(height, bbox.y2 + padding))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)
