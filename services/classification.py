import json
import logging
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from models.alzheimer_model import normalize_alzheimer_stage_label
from services.model_metrics import get_model_metrics
from services.model_registry import get_model_registry


LOGGER = logging.getLogger(__name__)


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
        model_registry = get_model_registry()
        self.tumor_model = model_registry.get_tumor_model()
        self.alzheimer_model = model_registry.get_alzheimer_model()

    def classify(
        self,
        image: np.ndarray,
        detection_type: str = "combined",
        prepared_inputs=None,
    ) -> Dict[str, Dict[str, object]]:
        detection_type = str(detection_type or "combined").strip().lower()
        include_tumor = detection_type in {"combined", "brain"}
        include_alzheimer = detection_type in {"combined", "alz"}
        classifier_input = getattr(prepared_inputs, "classifier_input", None)

        tumor_result = (
            safe_dict(self.tumor_model.predict(image, prepared_image=classifier_input))
            if include_tumor
            else {}
        )
        alzheimer_result = (
            safe_dict(self.alzheimer_model.predict(image, prepared_image=classifier_input))
            if include_alzheimer
            else {}
        )
        stage = normalize_alzheimer_stage_label(alzheimer_result.get("stage"))
        tumor_metrics = get_model_metrics("brain_classifier")
        alzheimer_metrics = get_model_metrics("alzheimer_classifier")

        tumor_detected = bool(tumor_result.get("detected", False))
        tumor_payload = {
            "detected": tumor_detected,
            "classification": tumor_result.get("classification", "Tumor Detected" if tumor_detected else "No Tumor"),
            "tumor_type": tumor_result.get("tumor_type") or tumor_result.get("classification", "Tumor Detected" if tumor_detected else "No Tumor"),
            "grade": tumor_result.get("grade"),
            "tumor_stage": tumor_result.get("tumor_stage") or tumor_result.get("grade"),
            "confidence": normalize_confidence(tumor_result.get("confidence", 0)),
            "backend": tumor_result.get("backend", self.tumor_model.backend if include_tumor else "skipped"),
            "tumor_detected": tumor_detected,
            "tumor_grade": tumor_result.get("grade"),
            "type_confidence": normalize_confidence(tumor_result.get("type_confidence", tumor_result.get("confidence", 0))),
            "stage_confidence": normalize_confidence(tumor_result.get("stage_confidence", tumor_result.get("confidence", 0))),
            "model_metrics": tumor_metrics,
        }
        alzheimer_detected = bool(alzheimer_result.get("detected", False))
        alzheimer_payload = {
            "detected": alzheimer_detected,
            "stage": stage,
            "confidence": normalize_confidence(alzheimer_result.get("confidence", 0)),
            "backend": alzheimer_result.get("backend", self.alzheimer_model.backend if include_alzheimer else "skipped"),
            "alzheimers_detected": alzheimer_detected,
            "alz_detected": alzheimer_detected,
            "alz_stage": stage,
            "model_metrics": alzheimer_metrics,
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
