import contextlib
import importlib
import io
import os
from functools import lru_cache
from typing import Any, Dict


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


def safe_load_keras_model(model_path: str):
    """Load a Keras model when TensorFlow is usable, otherwise return None."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    load_model = getattr(tensorflow_keras["models"], "load_model", None)
    if load_model is None:
        return None

    try:
        return load_model(model_path, compile=False)
    except Exception:
        return None
