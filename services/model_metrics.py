import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional

from utils.tensorflow_compat import DEFAULT_MODEL_MANIFEST_PATH


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_MODEL_ACCURACY_PATH = os.path.join(BASE_DIR, "models", "model_accuracy.json")


def _resolve_path(path_value: Optional[str], default_path: str) -> str:
    normalized = str(path_value or "").strip()
    if not normalized:
        normalized = default_path
    expanded = os.path.expandvars(normalized)
    return os.path.abspath(expanded if os.path.isabs(expanded) else os.path.join(BASE_DIR, expanded))


def _read_json_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_metric_value(value) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("%"):
            normalized = normalized[:-1]
        try:
            value = float(normalized)
        except ValueError:
            return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value > 1:
        numeric_value /= 100.0
    return max(0.0, min(1.0, numeric_value))


def _format_metric_label(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    percentage = round(float(value) * 100.0, 2)
    formatted = f"{percentage:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


@lru_cache(maxsize=1)
def _load_manifest() -> Dict[str, dict]:
    manifest_path = _resolve_path(os.environ.get("MODEL_MANIFEST_PATH"), DEFAULT_MODEL_MANIFEST_PATH)
    return _read_json_file(manifest_path)


@lru_cache(maxsize=1)
def _load_accuracy_registry() -> Dict[str, dict]:
    accuracy_path = _resolve_path(os.environ.get("MODEL_ACCURACY_PATH"), DEFAULT_MODEL_ACCURACY_PATH)
    return _read_json_file(accuracy_path)


def clear_model_metrics_cache() -> None:
    _load_manifest.cache_clear()
    _load_accuracy_registry.cache_clear()


def get_model_metrics(model_key: str, registry_key: Optional[str] = None) -> Dict[str, object]:
    manifest = _load_manifest()
    manifest_entry = manifest.get(model_key) if isinstance(manifest, dict) else {}
    metrics_payload = {}
    source = "unavailable"

    metadata_path = None
    if isinstance(manifest_entry, dict):
        metadata_path = manifest_entry.get("metadata_path")
    if metadata_path:
        metadata_payload = _read_json_file(_resolve_path(metadata_path, DEFAULT_MODEL_MANIFEST_PATH))
        raw_metrics = metadata_payload.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics_payload = raw_metrics
            source = "metadata"

    if not metrics_payload:
        registry = _load_accuracy_registry()
        registry_payload = registry.get(registry_key or model_key)
        if isinstance(registry_payload, dict):
            metrics_payload = registry_payload
            source = "accuracy_registry"

    accuracy = _normalize_metric_value(metrics_payload.get("accuracy") or metrics_payload.get("categorical_accuracy"))
    precision = _normalize_metric_value(metrics_payload.get("precision"))
    recall = _normalize_metric_value(metrics_payload.get("recall"))
    f1_score = _normalize_metric_value(metrics_payload.get("f1_score") or metrics_payload.get("f1"))
    available = any(metric is not None for metric in (accuracy, precision, recall, f1_score))

    return {
        "model_key": model_key,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy_label": _format_metric_label(accuracy),
        "precision_label": _format_metric_label(precision),
        "recall_label": _format_metric_label(recall),
        "f1_score_label": _format_metric_label(f1_score),
        "available": available,
        "source": source if available else "unavailable",
    }


def write_model_accuracy_registry_entry(
    model_key: str,
    metrics: Dict[str, object],
    output_path: Optional[str] = None,
) -> str:
    resolved_output_path = _resolve_path(output_path or os.environ.get("MODEL_ACCURACY_PATH"), DEFAULT_MODEL_ACCURACY_PATH)
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

    payload = _read_json_file(resolved_output_path)
    payload[model_key] = {
        "accuracy": _normalize_metric_value(metrics.get("accuracy") or metrics.get("categorical_accuracy")),
        "precision": _normalize_metric_value(metrics.get("precision")),
        "recall": _normalize_metric_value(metrics.get("recall")),
        "f1_score": _normalize_metric_value(metrics.get("f1_score") or metrics.get("f1")),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(resolved_output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    clear_model_metrics_cache()
    return resolved_output_path
