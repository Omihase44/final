import base64
import io
import json
from typing import Dict, Optional

import cv2
import numpy as np

try:
    import pydicom

    PYDICOM_AVAILABLE = True
except Exception:
    pydicom = None
    PYDICOM_AVAILABLE = False


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image -= float(np.min(image))
    peak = float(np.max(image))
    if peak > 0:
        image /= peak
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _decode_dicom_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    if not PYDICOM_AVAILABLE:
        return None

    try:
        dataset = pydicom.dcmread(io.BytesIO(image_bytes), force=True)
        pixel_array = dataset.pixel_array
    except Exception:
        return None

    image = _normalize_to_uint8(pixel_array)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = cv2.bitwise_not(image)
    return ensure_three_channel(image)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR OpenCV image."""
    if not image_bytes:
        raise ValueError("Empty image payload received.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        image = _decode_dicom_bytes(image_bytes)
    if image is None:
        raise ValueError("Invalid image file. Please upload a valid MRI scan.")
    return image


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded image or data URI into a BGR image."""
    if not image_base64:
        raise ValueError("Missing base64 image data.")

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    return decode_image_bytes(image_bytes)


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    """Convert grayscale MRI slices to 3-channel BGR images when needed."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def encode_image_to_base64(image: np.ndarray, file_extension: str = ".png") -> str:
    """Encode an OpenCV image to base64 for API responses."""
    success, encoded = cv2.imencode(file_extension, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def preprocess_classifier_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Prepare a 3-channel image for classification models."""
    image = ensure_three_channel(image)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.expand_dims(normalized, axis=0)


def preprocess_segmentation_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Prepare a grayscale image for U-Net style segmentation models."""
    image = ensure_three_channel(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, target_size, interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.expand_dims(normalized[..., np.newaxis], axis=0)


def extract_radiology_features(image: np.ndarray) -> Dict[str, float]:
    """Extract lightweight MRI texture features for fallback clinical inference."""
    image = ensure_three_channel(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = grayscale.astype(np.float32) / 255.0

    edges = cv2.Canny(grayscale, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    bright_threshold = float(max(0.65, np.quantile(normalized, 0.9)))
    bright_ratio = float(np.mean(normalized >= bright_threshold))

    laplacian = cv2.Laplacian(normalized, cv2.CV_32F)
    texture_score = float(np.mean(np.abs(laplacian)))

    height, width = grayscale.shape
    y0, y1 = height // 4, (height * 3) // 4
    x0, x1 = width // 4, (width * 3) // 4
    center_patch = normalized[y0:y1, x0:x1]
    center_focus = float(np.mean(center_patch) / (np.mean(normalized) + 1e-6))

    left_half = normalized[:, : width // 2]
    right_half = normalized[:, width // 2 :]
    asymmetry = float(abs(np.mean(left_half) - np.mean(right_half)))

    return {
        "mean_intensity": float(np.mean(normalized)),
        "std_intensity": float(np.std(normalized)),
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
        "texture_score": texture_score,
        "center_focus": center_focus,
        "asymmetry": asymmetry,
    }


def build_segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a red mask overlay on the original MRI image."""
    image = ensure_three_channel(image)
    overlay = image.copy()

    if np.count_nonzero(mask) == 0:
        return overlay

    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)


def sanitize_positive_float(value: Optional[object], default: float) -> float:
    """Return a positive float from user input or a safe default."""
    try:
        numeric_value = float(value)
        if numeric_value > 0:
            return numeric_value
    except (TypeError, ValueError):
        pass
    return float(default)


def normalize_voxel_metadata(payload: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Normalize voxel spacing metadata for tumor volume calculation."""
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except Exception:
                payload = {}
            else:
                payload = parsed if isinstance(parsed, dict) else {}
        else:
            payload = {}
    elif not isinstance(payload, dict):
        payload = {}
    else:
        payload = payload or {}

    return {
        "pixel_spacing_x": sanitize_positive_float(payload.get("pixel_spacing_x"), 1.0),
        "pixel_spacing_y": sanitize_positive_float(payload.get("pixel_spacing_y"), 1.0),
        "slice_thickness": sanitize_positive_float(payload.get("slice_thickness"), 1.0),
    }
