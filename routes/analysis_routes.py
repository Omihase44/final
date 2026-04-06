import json
import logging
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from services.classification import get_classification_service
from services.enhancement import save_analysis_assets
from services.model_metrics import get_model_metrics
from services.preprocessing import get_preprocessing_service
from services.segmentation import get_segmentation_service
from services.volume_calc import get_volume_calculation_service
from utils.image_processing import (
    decode_base64_image,
    encode_image_to_base64,
    normalize_voxel_metadata,
)
from utils.tensorflow_compat import ModelUnavailableError


analysis_bp = Blueprint("analysis", __name__)
VALID_ANALYSIS_TYPES = {"combined", "brain", "alz"}
LOGGER = logging.getLogger(__name__)


def _format_confidence(confidence: float) -> str:
    confidence_percent = round(float(confidence) * 100, 2)
    formatted = f"{confidence_percent:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def safe_dict(obj) -> Dict[str, object]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped:
            try:
                obj = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return obj if isinstance(obj, dict) else {}
    return {}


def _normalize_confidence_value(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("%"):
            normalized = normalized[:-1]
        try:
            value = float(normalized)
        except ValueError:
            return 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric_value > 1:
        numeric_value /= 100.0
    return numeric_value


def _normalize_tumor_prediction(value: object) -> Dict[str, object]:
    result = safe_dict(value)
    detected = bool(result.get("detected", False))
    return {
        "detected": detected,
        "classification": result.get("classification") or ("Tumor Detected" if detected else "No Tumor"),
        "tumor_type": result.get("tumor_type") or result.get("classification") or ("Tumor Detected" if detected else "No Tumor"),
        "grade": result.get("grade"),
        "tumor_stage": result.get("tumor_stage") or result.get("grade"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
        "type_confidence": _normalize_confidence_value(result.get("type_confidence", result.get("confidence", 0))),
        "stage_confidence": _normalize_confidence_value(result.get("stage_confidence", result.get("confidence", 0))),
        "model_metrics": safe_dict(result.get("model_metrics")),
    }


def _normalize_alzheimers_prediction(value: object) -> Dict[str, object]:
    result = safe_dict(value)
    detected = bool(result.get("detected", False))
    return {
        "detected": detected,
        "stage": result.get("stage") or ("Moderate" if detected else "NonDemented"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
        "model_metrics": safe_dict(result.get("model_metrics")),
    }


def predict_tumor(image: np.ndarray) -> Dict[str, object]:
    classification_service = get_classification_service()
    classification_result = safe_dict(classification_service.classify(image, detection_type="brain"))
    result = _normalize_tumor_prediction(classification_result.get("tumor"))
    return {
        "detected": result["detected"],
        "classification": result["classification"],
        "tumor_type": result["tumor_type"],
        "grade": result["grade"],
        "tumor_stage": result["tumor_stage"],
        "confidence": result["confidence"],
        "model_metrics": result["model_metrics"] or get_model_metrics("brain_classifier"),
    }


def analyze_medical_image(
    image_bytes: bytes,
    detection_type: str = "combined",
    voxel_metadata: Optional[Dict[str, object]] = None,
    include_encoded_images: bool = True,
) -> Dict[str, object]:
    """Run the unified neurodiagnostic pipeline for a single MRI image."""
    preprocessing_service = get_preprocessing_service()
    classification_service = get_classification_service()
    segmentation_service = get_segmentation_service()
    volume_service = get_volume_calculation_service()

    prepared_inputs = preprocessing_service.prepare(image_bytes=image_bytes)
    original_image = prepared_inputs.original_image
    enhanced_image = prepared_inputs.enhanced_image
    classification_result = safe_dict(
        classification_service.classify(
            enhanced_image,
            detection_type=detection_type,
            prepared_inputs=prepared_inputs,
        )
    )
    tumor_result = _normalize_tumor_prediction(classification_result.get("tumor"))
    alzheimer_result = _normalize_alzheimers_prediction(classification_result.get("alzheimers"))
    segmentation_result = segmentation_service.segment(
        original_image=original_image,
        working_image=enhanced_image,
        tumor_detected=tumor_result["detected"],
        prepared_input=prepared_inputs.segmentation_input,
    )
    mask = segmentation_result["mask"]

    voxel_config = normalize_voxel_metadata(voxel_metadata)
    volume_result = volume_service.calculate(
        mask,
        pixel_spacing_x=voxel_config["pixel_spacing_x"],
        pixel_spacing_y=voxel_config["pixel_spacing_y"],
        slice_thickness=voxel_config["slice_thickness"],
    )

    if not tumor_result["detected"]:
        volume_result["tumor_volume_mm3"] = 0.0
        volume_result["white_pixel_count"] = 0
        mask[:, :] = 0

    overlay = segmentation_result["overlay"]
    assets = save_analysis_assets(
        {
            "original_image": original_image,
            "enhanced_image": enhanced_image,
            "overlay_image": overlay,
            "mask_image": mask,
        }
    )

    original_base64 = encode_image_to_base64(original_image, ".png") if include_encoded_images else None
    enhanced_base64 = encode_image_to_base64(enhanced_image, ".png") if include_encoded_images else None
    overlay_base64 = encode_image_to_base64(overlay, ".png") if include_encoded_images else None
    mask_base64 = encode_image_to_base64(mask, ".png") if include_encoded_images else None
    tumor_volume_mm3 = round(volume_result["tumor_volume_mm3"], 2)
    detection_type = (detection_type or "combined").lower()
    tumor_metrics = tumor_result.get("model_metrics") or get_model_metrics("brain_classifier")
    alzheimer_metrics = alzheimer_result.get("model_metrics") or get_model_metrics("alzheimer_classifier")

    tumor_payload = {
        **tumor_result,
        "tumor_type": tumor_result["tumor_type"],
        "tumor_stage": tumor_result["tumor_stage"],
        "confidence_score": round(float(tumor_result["confidence"]), 4),
        "confidence": _format_confidence(tumor_result["confidence"]),
        "type_confidence_score": round(float(tumor_result["type_confidence"]), 4),
        "type_confidence": _format_confidence(tumor_result["type_confidence"]),
        "stage_confidence_score": round(float(tumor_result["stage_confidence"]), 4),
        "stage_confidence": _format_confidence(tumor_result["stage_confidence"]),
        "model_metrics": tumor_metrics,
        "volume_mm3": tumor_volume_mm3,
        "tumor_volume_mm3": tumor_volume_mm3,
    }
    alzheimer_payload = {
        **alzheimer_result,
        "confidence_score": round(float(alzheimer_result["confidence"]), 4),
        "confidence": _format_confidence(alzheimer_result["confidence"]),
        "model_metrics": alzheimer_metrics,
    }
    mask_quality = "clear" if volume_result["white_pixel_count"] > 0 and segmentation_result["contour_count"] <= 3 else "review"
    tumor_accuracy_label = tumor_metrics.get("accuracy_label") or "Unavailable"
    tumor_accuracy_score = tumor_metrics.get("accuracy")
    alzheimer_accuracy_label = alzheimer_metrics.get("accuracy_label") or "Unavailable"

    response = {
        "analysis_type": detection_type,
        "tumor": tumor_payload,
        "alzheimers": alzheimer_payload,
        "model_metrics": {
            "tumor": tumor_metrics,
            "alzheimers": alzheimer_metrics,
        },
        "enhancement": {
            "backend": prepared_inputs.enhancement_backend,
            "steps": prepared_inputs.enhancement_steps,
            "original_image": assets["files"]["original_image"],
            "enhanced_image": assets["files"]["enhanced_image"],
            "original_image_base64": original_base64,
            "enhanced_image_base64": enhanced_base64,
        },
        "segmentation": {
            "available": bool(tumor_result["detected"]),
            "backend": segmentation_result["backend"],
            "white_pixel_count": volume_result["white_pixel_count"],
            "voxel_volume_mm3": volume_result["voxel_volume_mm3"],
            "mask_image": assets["files"]["mask_image"],
            "overlay_image": assets["files"]["overlay_image"],
            "segmentation_overlay": assets["files"]["overlay_image"],
            "mask_image_base64": mask_base64,
            "overlay_image_base64": overlay_base64,
            "segmentation_overlay_base64": overlay_base64,
            "bounding_box": segmentation_result["bounding_box"],
            "contour_count": segmentation_result["contour_count"],
            "mask_quality": mask_quality,
        },
        "images": {
            "original": {"path": assets["files"]["original_image"], "base64": original_base64},
            "enhanced": {"path": assets["files"]["enhanced_image"], "base64": enhanced_base64},
            "overlay": {"path": assets["files"]["overlay_image"], "base64": overlay_base64},
            "mask": {"path": assets["files"]["mask_image"], "base64": mask_base64},
        },
        "study_id": assets["study_id"],
        "original_image": assets["files"]["original_image"],
        "enhanced_image": assets["files"]["enhanced_image"],
        "overlay_image": assets["files"]["overlay_image"],
        "mask_image": assets["files"]["mask_image"],
        "original_image_base64": original_base64,
        "enhanced_image_base64": enhanced_base64,
        "segmentation_image": overlay_base64,
        "segmentation_mask": mask_base64,
        "segmentation_overlay": overlay_base64,
        "voxel_metadata": voxel_config,
        "white_pixel_count": volume_result["white_pixel_count"],
        "tumor_detected": tumor_result["detected"],
        "tumor_type": tumor_payload["tumor_type"],
        "tumor_grade": tumor_result["grade"],
        "tumor_stage": tumor_payload["tumor_stage"],
        "tumor_confidence": tumor_payload["confidence"],
        "tumor_confidence_score": round(float(tumor_result["confidence"]), 4),
        "model_accuracy": tumor_accuracy_label,
        "model_accuracy_score": tumor_accuracy_score,
        "tumor_volume_mm3": tumor_volume_mm3,
        "alzheimers_detected": alzheimer_result["detected"],
        "alz_detected": alzheimer_result["detected"],
        "alzheimer_stage": alzheimer_result["stage"],
        "alz_stage": alzheimer_result["stage"],
        "alzheimer_confidence": alzheimer_payload["confidence"],
        "alzheimer_confidence_score": round(float(alzheimer_result["confidence"]), 4),
        "ai_clinical_insights": {
            "tumor_type": tumor_payload["tumor_type"],
            "tumor_stage": tumor_payload["tumor_stage"],
            "tumor_confidence": tumor_payload["confidence"],
            "tumor_volume_mm3": tumor_volume_mm3,
            "model_accuracy": tumor_accuracy_label,
            "model_accuracy_score": tumor_accuracy_score,
            "alzheimer_stage": alzheimer_result["stage"],
            "alzheimer_model_accuracy": alzheimer_accuracy_label,
            "segmentation_boundary": "Detected" if segmentation_result["contour_count"] else "Not detected",
            "mask_quality": mask_quality,
        },
        "confidence": _format_confidence(
            max(float(tumor_result["confidence"]), float(alzheimer_result["confidence"]))
        ),
    }

    if detection_type == "brain":
        response["primary_result"] = {
            "type": "tumor",
            "detected": tumor_payload["detected"],
            "classification": tumor_payload["classification"],
            "tumor_type": tumor_payload["tumor_type"],
            "grade": tumor_payload["grade"],
            "tumor_stage": tumor_payload["tumor_stage"],
            "confidence": tumor_payload["confidence"],
        }
    elif detection_type == "alz":
        response["primary_result"] = {
            "type": "alzheimers",
            "detected": alzheimer_payload["detected"],
            "stage": alzheimer_payload["stage"],
            "confidence": alzheimer_payload["confidence"],
        }
    else:
        response["primary_result"] = {
            "type": "combined",
            "tumor_detected": tumor_payload["detected"],
            "alzheimers_detected": alzheimer_payload["detected"],
        }

    return response


def _extract_request_image_bytes() -> bytes:
    for field_name in ("file", "image", "scan"):
        if field_name in request.files:
            uploaded_file = request.files[field_name]
            image_bytes = uploaded_file.read()
            if not image_bytes:
                raise ValueError("Uploaded image file is empty.")
            return image_bytes

    payload = _extract_json_payload()
    image_base64 = payload.get("image") or payload.get("image_base64")
    if image_base64:
        image = decode_base64_image(image_base64)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Unable to decode the submitted base64 image.")
        return encoded.tobytes()

    raise ValueError("No image provided. Use multipart file upload or base64 JSON.")


def _extract_voxel_payload() -> Dict[str, object]:
    if request.is_json:
        payload = _extract_json_payload()
        voxel_payload = _coerce_mapping(payload.get("voxel_metadata"), allow_raw_image=False)
        return {
            "pixel_spacing_x": payload.get("pixel_spacing_x", voxel_payload.get("pixel_spacing_x")),
            "pixel_spacing_y": payload.get("pixel_spacing_y", voxel_payload.get("pixel_spacing_y")),
            "slice_thickness": payload.get("slice_thickness", voxel_payload.get("slice_thickness")),
        }

    return {
        "pixel_spacing_x": request.form.get("pixel_spacing_x"),
        "pixel_spacing_y": request.form.get("pixel_spacing_y"),
        "slice_thickness": request.form.get("slice_thickness"),
    }


def _coerce_mapping(value, allow_raw_image: bool = False) -> Dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"image": stripped} if allow_raw_image else {}
        if isinstance(parsed, dict):
            return parsed
        if allow_raw_image and isinstance(parsed, str):
            return {"image": parsed}
    return {}


def _extract_json_payload() -> Dict[str, object]:
    return safe_dict(_coerce_mapping(request.get_json(silent=True), allow_raw_image=True))


@analysis_bp.route("/analyze", methods=["POST"])
@analysis_bp.route("/api/analyze", methods=["POST"])
def analyze_route():
    try:
        configured_api_keys = current_app.config.get("API_KEYS") or set()
        if configured_api_keys and "user_id" not in request.headers and "user_id" not in request.form:
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if api_key not in configured_api_keys:
                return jsonify({"success": False, "error": "Unauthorized API request."}), 401

        image_bytes = _extract_request_image_bytes()
        analysis = analyze_medical_image(
            image_bytes=image_bytes,
            detection_type="brain",
            voxel_metadata=_extract_voxel_payload(),
            include_encoded_images=False,
        )
        tumor_payload = safe_dict(analysis.get("tumor"))

        return jsonify(
            {
                "success": True,
                "tumor": {
                    "detected": tumor_payload.get("detected", False),
                    "classification": tumor_payload.get("classification"),
                    "tumor_type": tumor_payload.get("tumor_type"),
                    "grade": tumor_payload.get("grade"),
                    "tumor_stage": tumor_payload.get("tumor_stage"),
                    "confidence": tumor_payload.get("confidence"),
                    "model_metrics": tumor_payload.get("model_metrics"),
                    "model_accuracy": (tumor_payload.get("model_metrics") or {}).get("accuracy_label"),
                },
                "segmentation": safe_dict(analysis.get("segmentation")),
                "ai_clinical_insights": safe_dict(analysis.get("ai_clinical_insights")),
            }
        )
    except ModelUnavailableError as exc:
        LOGGER.error("Model unavailable for analysis request: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Unexpected failure during /analyze request")
        return jsonify({"success": False, "error": str(exc)}), 500
