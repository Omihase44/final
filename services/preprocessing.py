from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

from services.enhancement import ImageEnhancementService
from utils.image_processing import (
    decode_image_bytes,
    ensure_three_channel,
    preprocess_classifier_image,
    preprocess_segmentation_image,
)


@dataclass(frozen=True)
class PreparedAnalysisInputs:
    original_image: np.ndarray
    enhanced_image: np.ndarray
    classifier_input: np.ndarray
    segmentation_input: np.ndarray
    enhancement_backend: str
    enhancement_steps: List[str]


class NeuroImagePreprocessingService:
    """Decode and preprocess scans once for downstream inference services."""

    def __init__(self):
        self._enhancement_service = ImageEnhancementService()

    def prepare(
        self,
        image_bytes: Optional[bytes] = None,
        image: Optional[np.ndarray] = None,
    ) -> PreparedAnalysisInputs:
        if image is None:
            if not image_bytes:
                raise ValueError("Image bytes are required for preprocessing.")
            image = decode_image_bytes(image_bytes)

        original_image = ensure_three_channel(image)
        enhancement_result = self._enhancement_service.enhance(original_image)
        enhanced_image = enhancement_result["enhanced_image"]

        return PreparedAnalysisInputs(
            original_image=original_image,
            enhanced_image=enhanced_image,
            classifier_input=preprocess_classifier_image(enhanced_image),
            segmentation_input=preprocess_segmentation_image(enhanced_image),
            enhancement_backend=str(enhancement_result.get("backend") or "opencv"),
            enhancement_steps=list(enhancement_result.get("steps") or []),
        )


@lru_cache(maxsize=1)
def get_preprocessing_service() -> NeuroImagePreprocessingService:
    return NeuroImagePreprocessingService()
