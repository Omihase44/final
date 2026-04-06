import contextlib
import importlib
import io
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_MODEL_MANIFEST_PATH = os.path.join(BASE_DIR, "models", "model_manifest.json")


class ModelUnavailableError(RuntimeError):
    """Raised when a required ML model is unavailable for inference."""


def _resolve_relative_path(path_value: str) -> str:
    expanded = os.path.expandvars(str(path_value or "").strip())
    if not expanded:
        return expanded
    return expanded if os.path.isabs(expanded) else os.path.join(BASE_DIR, expanded)


def require_strict_model_loading() -> bool:
    return os.environ.get("STRICT_MODEL_LOADING", "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_model_manifest() -> Dict[str, Dict[str, Any]]:
    manifest_path = os.environ.get("MODEL_MANIFEST_PATH", "").strip() or DEFAULT_MODEL_MANIFEST_PATH
    resolved_path = _resolve_relative_path(manifest_path)
    if not resolved_path or not os.path.exists(resolved_path):
        return {}

    try:
        with open(resolved_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def get_model_manifest_entry(manifest_key: Optional[str]) -> Dict[str, Any]:
    if not manifest_key:
        return {}
    manifest = _load_model_manifest()
    entry = manifest.get(manifest_key)
    return entry if isinstance(entry, dict) else {}


def resolve_model_metadata(manifest_key: Optional[str]) -> Dict[str, Any]:
    manifest_entry = get_model_manifest_entry(manifest_key)
    metadata_path = str(manifest_entry.get("metadata_path") or "").strip()
    if not metadata_path:
        return {}

    resolved_path = _resolve_relative_path(metadata_path)
    if not resolved_path or not os.path.exists(resolved_path):
        return {}

    try:
        with open(resolved_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def resolve_model_path(default_path: str, env_key: Optional[str] = None, manifest_key: Optional[str] = None) -> str:
    specific_override = os.environ.get(env_key or "")
    if specific_override:
        return os.path.abspath(_resolve_relative_path(specific_override))

    model_path = os.environ.get("MODEL_PATH", "").strip()
    if model_path:
        resolved_model_path = _resolve_relative_path(model_path)
        if os.path.isdir(resolved_model_path):
            nested_candidate = os.path.join(resolved_model_path, default_path)
            if os.path.exists(nested_candidate):
                return os.path.abspath(nested_candidate)
            basename_candidate = os.path.join(resolved_model_path, os.path.basename(default_path))
            return os.path.abspath(basename_candidate)
        return os.path.abspath(resolved_model_path)

    if manifest_key:
        manifest_entry = _load_model_manifest().get(manifest_key)
        if isinstance(manifest_entry, dict):
            manifest_relative_path = manifest_entry.get("path")
            if manifest_relative_path:
                return os.path.abspath(_resolve_relative_path(str(manifest_relative_path)))

    return os.path.abspath(_resolve_relative_path(default_path))


@lru_cache(maxsize=1)
def get_tensorflow_keras() -> Dict[str, Any]:
    """Load TensorFlow Keras lazily and quietly when it is available."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            layers_module = importlib.import_module("tensorflow.keras.layers")
            models_module = importlib.import_module("tensorflow.keras.models")
    except Exception:
        return {
            "available": False,
            "layers": None,
            "models": None,
        }

    return {
        "available": True,
        "layers": layers_module,
        "models": models_module,
    }


@lru_cache(maxsize=6)
def _safe_load_keras_model_cached(resolved_model_path: str):
    """Load a Keras model when TensorFlow is usable, otherwise return None."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    if not resolved_model_path or not os.path.exists(resolved_model_path):
        return None

    load_model = getattr(tensorflow_keras["models"], "load_model", None)
    if load_model is None:
        return None

    try:
        return load_model(resolved_model_path, compile=False)
    except Exception:
        return None


def safe_load_keras_model(model_path: str):
    resolved_model_path = os.path.abspath(_resolve_relative_path(model_path))
    return _safe_load_keras_model_cached(resolved_model_path)


def clear_model_cache() -> None:
    _load_model_manifest.cache_clear()
    _safe_load_keras_model_cached.cache_clear()
    get_tensorflow_keras.cache_clear()
