import json
import logging
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from models.alzheimer_model import AlzheimerStagingModel
from models.tumor_model import TumorGradingModel


LOGGER = logging.getLogger(__name__)
ALZHEIMER_STAGE_ALIASES = {
    "Early": "Early Stage",
    "Moderate": "Moderate Stage",
    "Severe": "Severe Stage",
    "Early Stage": "Early Stage",
    "Moderate Stage": "Moderate Stage",
    "Severe Stage": "Severe Stage",
}


def normalize_alzheimer_stage(stage: Optional[str]) -> Optional[str]:
    if stage is None:
        return None
    return ALZHEIMER_STAGE_ALIASES.get(stage, stage)


def safe_dict(obj) -> Dict[str, object]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_confidence(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("%"):
            normalized = normalized[:-1]
        try:
            value = float(normalized)
        except (TypeError, ValueError):
            return 0.0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric_value > 1:
        numeric_value /= 100.0
    return round(numeric_value, 2)


class ClinicalClassificationService:
    """Coordinate tumor grading and Alzheimer staging models."""

    def __init__(self):
        self.tumor_model = TumorGradingModel()
        self.alzheimer_model = AlzheimerStagingModel()

    def classify(self, image: np.ndarray) -> Dict[str, Dict[str, object]]:
        tumor_result = safe_dict(self.tumor_model.predict(image))
        alzheimer_result = safe_dict(self.alzheimer_model.predict(image))
        stage = normalize_alzheimer_stage(alzheimer_result.get("stage"))

        tumor_payload = {
            "detected": bool(tumor_result.get("detected", False)),
            "classification": tumor_result.get("classification", "Tumor Detected" if tumor_result.get("detected") else "No Tumor"),
            "grade": tumor_result.get("grade"),
            "confidence": normalize_confidence(tumor_result.get("confidence", 0)),
            "backend": tumor_result.get("backend", self.tumor_model.backend),
            "tumor_detected": bool(tumor_result.get("detected", False)),
            "tumor_grade": tumor_result.get("grade"),
        }
        alzheimer_payload = {
            "detected": bool(alzheimer_result.get("detected", False)),
            "stage": stage,
            "confidence": normalize_confidence(alzheimer_result.get("confidence", 0)),
            "backend": alzheimer_result.get("backend", self.alzheimer_model.backend),
            "alzheimers_detected": bool(alzheimer_result.get("detected", False)),
            "alz_detected": bool(alzheimer_result.get("detected", False)),
            "alz_stage": stage,
        }

        LOGGER.info(
            "Clinical prediction generated",
            extra={
                "tumor_detected": tumor_payload["detected"],
                "tumor_grade": tumor_payload["grade"],
                "alz_detected": alzheimer_payload["detected"],
                "alz_stage": alzheimer_payload["stage"],
            },
        )

        return {
            "tumor": tumor_payload,
            "alzheimers": alzheimer_payload,
        }


@lru_cache(maxsize=1)
def get_classification_service() -> ClinicalClassificationService:
    return ClinicalClassificationService()
