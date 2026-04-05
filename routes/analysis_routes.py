import json
from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from services.classification import get_classification_service
from services.enhancement import ImageEnhancementService, save_analysis_assets
from services.segmentation import get_segmentation_service
from services.volume_calc import get_volume_calculation_service
from utils.image_processing import (
    decode_base64_image,
    decode_image_bytes,
    encode_image_to_base64,
    normalize_voxel_metadata,
)


analysis_bp = Blueprint("analysis", __name__)
VALID_ANALYSIS_TYPES = {"combined", "brain", "alz"}


@lru_cache(maxsize=1)
def _get_enhancement_service():
    return ImageEnhancementService()


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
        "grade": result.get("grade"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
    }


def _normalize_alzheimers_prediction(value: object) -> Dict[str, object]:
    result = safe_dict(value)
    detected = bool(result.get("detected", False))
    return {
        "detected": detected,
        "stage": result.get("stage") or ("Positive" if detected else "NonDementia"),
        "confidence": _normalize_confidence_value(result.get("confidence", 0)),
    }


def predict_tumor(image: np.ndarray) -> Dict[str, object]:
    classification_service = get_classification_service()
    classification_result = safe_dict(classification_service.classify(image))
    result = _normalize_tumor_prediction(classification_result.get("tumor"))
    return {
        "detected": result["detected"],
        "grade": result["grade"],
        "confidence": result["confidence"],
    }


def analyze_medical_image(
    image_bytes: bytes,
    detection_type: str = "combined",
    voxel_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run the unified neurodiagnostic pipeline for a single MRI image."""
    original_image = decode_image_bytes(image_bytes)
    enhancement_service = _get_enhancement_service()
    classification_service = get_classification_service()
    segmentation_service = get_segmentation_service()
    volume_service = get_volume_calculation_service()

    enhancement_result = enhancement_service.enhance(original_image)
    enhanced_image = enhancement_result["enhanced_image"]
    classification_result = safe_dict(classification_service.classify(enhanced_image))
    tumor_result = _normalize_tumor_prediction(classification_result.get("tumor"))
    alzheimer_result = _normalize_alzheimers_prediction(classification_result.get("alzheimers"))
    segmentation_result = segmentation_service.segment(
        original_image=original_image,
        working_image=enhanced_image,
        tumor_detected=tumor_result["detected"],
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

    original_base64 = encode_image_to_base64(original_image, ".png")
    enhanced_base64 = encode_image_to_base64(enhanced_image, ".png")
    overlay_base64 = encode_image_to_base64(overlay, ".png")
    mask_base64 = encode_image_to_base64(mask, ".png")
    tumor_volume_mm3 = round(volume_result["tumor_volume_mm3"], 2)
    detection_type = (detection_type or "combined").lower()

    tumor_payload = {
        **tumor_result,
        "confidence": _format_confidence(tumor_result["confidence"]),
        "volume_mm3": tumor_volume_mm3,
        "tumor_volume_mm3": tumor_volume_mm3,
    }
    alzheimer_payload = {
        **alzheimer_result,
        "confidence": _format_confidence(alzheimer_result["confidence"]),
    }

    response = {
        "analysis_type": detection_type,
        "tumor": tumor_payload,
        "alzheimers": alzheimer_payload,
        "enhancement": {
            "backend": enhancement_result["backend"],
            "steps": enhancement_result["steps"],
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
            "mask_image_base64": mask_base64,
            "overlay_image_base64": overlay_base64,
            "bounding_box": segmentation_result["bounding_box"],
            "contour_count": segmentation_result["contour_count"],
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
        "voxel_metadata": voxel_config,
        "white_pixel_count": volume_result["white_pixel_count"],
        "tumor_detected": tumor_result["detected"],
        "tumor_grade": tumor_result["grade"],
        "tumor_confidence": tumor_payload["confidence"],
        "tumor_volume_mm3": tumor_volume_mm3,
        "alzheimers_detected": alzheimer_result["detected"],
        "alz_detected": alzheimer_result["detected"],
        "alzheimer_stage": alzheimer_result["stage"],
        "alz_stage": alzheimer_result["stage"],
        "alzheimer_confidence": alzheimer_payload["confidence"],
        "confidence": _format_confidence(
            max(float(tumor_result["confidence"]), float(alzheimer_result["confidence"]))
        ),
    }

    if detection_type == "brain":
        response["primary_result"] = {
            "type": "tumor",
            "detected": tumor_payload["detected"],
            "classification": tumor_payload["classification"],
            "grade": tumor_payload["grade"],
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
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    try:
        result = safe_dict(predict_tumor(image))
        confidence = _normalize_confidence_value(result.get("confidence", 0))
        confidence_percent = str(round(confidence * 100, 2)) + "%"

        return jsonify(
            {
                "success": True,
                "tumor": {
                    "detected": result.get("detected", False),
                    "grade": result.get("grade"),
                    "confidence": confidence_percent,
                },
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
