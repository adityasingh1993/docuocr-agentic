from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from docuocr.models import (
    BBox,
    DocumentEnhancementProfile,
    EnhancementRecord,
    QualityReport,
    RecoveryAction,
)
from docuocr.trace import sha256_file

from .quality import _cv2


class ImageEnhancer:
    """Approved deterministic image transformations with provenance."""

    _PROFILE_ACTIONS = {
        DocumentEnhancementProfile.DESKEW: RecoveryAction.DESKEW,
        DocumentEnhancementProfile.LOCAL_CONTRAST: RecoveryAction.CLAHE,
        DocumentEnhancementProfile.ILLUMINATION_NORMALIZATION: (
            RecoveryAction.ILLUMINATION_NORMALIZATION
        ),
        DocumentEnhancementProfile.MILD_DENOISE_SHARPEN: (
            RecoveryAction.DENOISE_SHARPEN
        ),
        DocumentEnhancementProfile.UPSCALE: RecoveryAction.UPSCALE,
    }

    def plan_for_quality(self, report: QualityReport) -> list[RecoveryAction]:
        actions: list[RecoveryAction] = []
        mapping = {
            "upscale": RecoveryAction.UPSCALE,
            "denoise_sharpen": RecoveryAction.DENOISE_SHARPEN,
            "clahe": RecoveryAction.CLAHE,
            "illumination_normalization": RecoveryAction.ILLUMINATION_NORMALIZATION,
            "deskew": RecoveryAction.DESKEW,
        }
        for item in report.recommendations:
            if action := mapping.get(item):
                actions.append(action)
        return actions[:3]

    def plan_variants(
        self,
        report: QualityReport,
        suggested_profiles: Iterable[DocumentEnhancementProfile] = (),
        *,
        max_variants: int = 3,
    ) -> list[list[RecoveryAction]]:
        """Build bounded, non-destructive full-page enhancement candidates."""

        if max_variants <= 0:
            return []
        actions: list[RecoveryAction] = []
        for profile in suggested_profiles:
            action = self._PROFILE_ACTIONS.get(profile)
            if action is not None:
                actions.append(action)
        actions.extend(self.plan_for_quality(report))
        actions = list(dict.fromkeys(actions))

        # Binarization and checkbox-focused transforms can erase handwriting and
        # are therefore restricted to targeted recovery crops.
        unsafe_full_page = {
            RecoveryAction.ADAPTIVE_BINARIZE,
            RecoveryAction.CHECKBOX_FOCUS,
            RecoveryAction.REVIEW,
        }
        actions = [item for item in actions if item not in unsafe_full_page]
        order = {
            RecoveryAction.DESKEW: 0,
            RecoveryAction.ILLUMINATION_NORMALIZATION: 1,
            RecoveryAction.CLAHE: 2,
            RecoveryAction.DENOISE_SHARPEN: 3,
            RecoveryAction.UPSCALE: 4,
        }
        actions.sort(key=lambda item: order[item])
        if not actions:
            return []

        variants: list[list[RecoveryAction]] = []
        for action in actions:
            if action == RecoveryAction.ILLUMINATION_NORMALIZATION:
                variants.append([action, RecoveryAction.CLAHE])
            else:
                variants.append([action])
        if len(actions) > 1:
            variants.insert(0, actions[:3])

        unique: list[list[RecoveryAction]] = []
        seen: set[tuple[RecoveryAction, ...]] = set()
        for variant in variants:
            key = tuple(dict.fromkeys(variant))
            if key in seen:
                continue
            seen.add(key)
            unique.append(list(key))
            if len(unique) >= max_variants:
                break
        return unique

    def apply(
        self,
        image_path: str | Path,
        output_path: str | Path,
        actions: Iterable[RecoveryAction],
        *,
        attempt: int,
        bbox: BBox | None = None,
        skew_degrees: float = 0.0,
    ) -> EnhancementRecord:
        cv2 = _cv2()
        source = Path(image_path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")

        height, width = image.shape[:2]
        target = bbox.expanded(12, 12, width, height) if bbox else None
        if target:
            working = image[target.y1 : target.y2, target.x1 : target.x2].copy()
        else:
            working = image.copy()

        action_list = list(dict.fromkeys(actions))
        parameters: dict[str, object] = {}
        for action in action_list:
            if action == RecoveryAction.UPSCALE:
                scale = 3.0 if min(working.shape[:2]) < 500 else 2.0
                working = cv2.resize(
                    working,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LANCZOS4,
                )
                parameters["upscale"] = scale
            elif action == RecoveryAction.CLAHE:
                lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
                light, channel_a, channel_b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                light = clahe.apply(light)
                working = cv2.cvtColor(
                    cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR
                )
                parameters["clahe"] = {"clip_limit": 2.0, "grid": [8, 8]}
            elif action == RecoveryAction.DENOISE_SHARPEN:
                denoised = cv2.fastNlMeansDenoisingColored(working, None, 5, 5, 7, 21)
                blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
                working = cv2.addWeighted(denoised, 1.7, blurred, -0.7, 0)
                parameters["denoise_sharpen"] = {"h": 5, "sigma": 1.2, "amount": 0.7}
            elif action == RecoveryAction.ILLUMINATION_NORMALIZATION:
                lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
                light, channel_a, channel_b = cv2.split(lab)
                short_side = min(light.shape[:2])
                kernel = max(31, int(short_side * 0.08))
                if kernel % 2 == 0:
                    kernel += 1
                background = cv2.GaussianBlur(light, (kernel, kernel), 0)
                normalized = cv2.divide(light, background, scale=245)
                working = cv2.cvtColor(
                    cv2.merge((normalized, channel_a, channel_b)), cv2.COLOR_LAB2BGR
                )
                parameters["illumination_normalization"] = {
                    "method": "lab_background_division",
                    "kernel": kernel,
                }
            elif action == RecoveryAction.ADAPTIVE_BINARIZE:
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    11,
                )
                working = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                parameters["adaptive_binarize"] = {"block_size": 31, "c": 11}
            elif action == RecoveryAction.DESKEW and abs(skew_degrees) >= 0.25:
                h, w = working.shape[:2]
                correction = skew_degrees
                matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), correction, 1.0)
                working = cv2.warpAffine(
                    working,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                parameters["deskew"] = {
                    "estimated_degrees": skew_degrees,
                    "applied_degrees": correction,
                }
            elif action == RecoveryAction.CHECKBOX_FOCUS:
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                working = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                parameters["checkbox_focus"] = {"threshold": "otsu"}

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), working):
            raise OSError(f"Failed to write enhanced image: {destination}")

        return EnhancementRecord(
            strategy="+".join(action.value for action in action_list),
            input_path=str(source),
            output_path=str(destination),
            input_sha256=sha256_file(source),
            output_sha256=sha256_file(destination),
            parameters=parameters,
            target_bbox=target,
            attempt=attempt,
        )
