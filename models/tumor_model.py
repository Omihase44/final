import importlib
from typing import Any, Dict, Optional

import numpy as np

from utils.image_processing import extract_radiology_features, preprocess_classifier_image
from utils.tensorflow_compat import (
    ModelUnavailableError,
    get_tensorflow_keras,
    require_strict_model_loading,
    resolve_model_metadata,
    resolve_model_path,
    safe_load_keras_model,
)


TUMOR_TYPE_CLASSES = ["glioma tumor", "meningioma tumor", "no tumor", "pituitary tumor"]
WHO_GRADE_CLASSES = ["Grade I", "Grade II", "Grade III", "Grade IV"]
TUMOR_STAGE_BY_TYPE = {
    "meningioma tumor": "Grade II",
    "glioma tumor": "Grade III",
    "pituitary tumor": "Grade IV",
}


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def build_tumor_grading_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    variant: str = "cnn",
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    fine_tune_layers: int = 0,
) -> Optional[Any]:
    """Build a modified CNN or transfer-learning classifier for tumor staging."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    layers = tensorflow_keras["layers"]
    models = tensorflow_keras["models"]
    BatchNormalization = layers.BatchNormalization
    Conv2D = layers.Conv2D
    Dense = layers.Dense
    Dropout = layers.Dropout
    Flatten = layers.Flatten
    Input = layers.Input
    MaxPooling2D = layers.MaxPooling2D
    Model = models.Model

    inputs = Input(shape=input_shape)
    normalized_variant = str(variant or "cnn").strip().lower()

    if normalized_variant == "cnn":
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
    else:
        applications = importlib.import_module("tensorflow.keras.applications")
        preprocessing_layers = {
            "mobilenetv2": importlib.import_module("tensorflow.keras.applications.mobilenet_v2"),
            "resnet50": importlib.import_module("tensorflow.keras.applications.resnet50"),
            "vgg16": importlib.import_module("tensorflow.keras.applications.vgg16"),
        }
        if normalized_variant == "mobilenetv2":
            backbone_class = applications.MobileNetV2
            preprocessing = preprocessing_layers["mobilenetv2"].preprocess_input
        elif normalized_variant == "resnet50":
            backbone_class = applications.ResNet50
            preprocessing = preprocessing_layers["resnet50"].preprocess_input
        elif normalized_variant == "vgg16":
            backbone_class = applications.VGG16
            preprocessing = preprocessing_layers["vgg16"].preprocess_input
        else:
            raise ValueError(f"Unsupported tumor model variant: {variant}")

        rescaled_inputs = layers.Rescaling(255.0, name=f"{normalized_variant}_rescale")(inputs)
        processed_inputs = layers.Lambda(preprocessing, name=f"{normalized_variant}_preprocess")(rescaled_inputs)
        base_model = backbone_class(
            include_top=False,
            weights=None if str(weights).lower() == "none" else weights,
            input_shape=input_shape,
        )
        if fine_tune_layers > 0:
            base_model.trainable = True
            frozen_boundary = max(len(base_model.layers) - int(fine_tune_layers), 0)
            for layer in base_model.layers[:frozen_boundary]:
                layer.trainable = False
            for layer in base_model.layers[frozen_boundary:]:
                layer.trainable = True
        else:
            base_model.trainable = False
        x = base_model(processed_inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax", name="tumor_grade")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"tumor_grading_model_{normalized_variant}")


class TumorGradingModel:
    """Tumor classifier that supports real models when available and heuristics otherwise."""

    def __init__(self, model_path: str = "brain_model_new.h5"):
        self.model_path = resolve_model_path(
            model_path,
            "BRAIN_MODEL_PATH",
            manifest_key="brain_classifier",
        )
        self.metadata = resolve_model_metadata("brain_classifier")
        self.class_names = self._resolve_class_names()
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"
        self.strict_loading = require_strict_model_loading()

    def _load_model(self):
        return safe_load_keras_model(self.model_path)

    def _resolve_class_names(self):
        metadata_classes = self.metadata.get("class_names")
        if isinstance(metadata_classes, list) and metadata_classes:
            return [str(name) for name in metadata_classes]
        return list(TUMOR_TYPE_CLASSES)

    def predict(self, image: np.ndarray, prepared_image: Optional[np.ndarray] = None) -> Dict[str, object]:
        features = extract_radiology_features(image)
        model_output = self._predict_with_model(image, prepared_image=prepared_image)

        if model_output:
            tumor_type = model_output["classification"]
            type_confidence = model_output["confidence"]
            backend = self.backend
        elif self.strict_loading:
            raise ModelUnavailableError("Tumor classification model is unavailable.")
        else:
            tumor_type, type_confidence = self._heuristic_tumor_type(features)
            backend = "heuristic"

        tumor_detected = tumor_type != "no tumor"
        tumor_grade, grade_confidence = self._estimate_grade(features, tumor_type, tumor_detected)
        confidence = type_confidence if not tumor_detected else max(type_confidence, grade_confidence)

        return {
            "detected": tumor_detected,
            "classification": tumor_type,
            "tumor_type": tumor_type,
            "grade": tumor_grade,
            "tumor_stage": tumor_grade,
            "confidence": round(confidence, 2),
            "type_confidence": round(type_confidence, 2),
            "stage_confidence": round(grade_confidence, 2),
            "backend": backend,
        }

    def _predict_with_model(
        self,
        image: np.ndarray,
        prepared_image: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, object]]:
        if self.model is None:
            return None

        try:
            prepared_image = prepared_image if prepared_image is not None else preprocess_classifier_image(image)
            prediction = self.model.predict(prepared_image, verbose=0)
            scores = np.asarray(prediction).squeeze()
            if scores.ndim != 1 or scores.shape[0] < len(self.class_names):
                return None

            class_index = int(np.argmax(scores[: len(self.class_names)]))
            return {
                "classification": self.class_names[class_index],
                "confidence": clamp(scores[class_index]),
            }
        except Exception:
            return None

    def _heuristic_tumor_type(self, features: Dict[str, float]) -> tuple[str, float]:
        tumor_score = clamp(
            (features["bright_ratio"] * 1.8)
            + (features["edge_density"] * 2.2)
            + (features["texture_score"] * 2.0)
            + (features["asymmetry"] * 3.5)
        )

        if tumor_score < 0.22:
            return "no tumor", round(clamp(0.82 - tumor_score * 0.8, 0.1, 0.93), 2)

        if features["center_focus"] > 1.12 and features["bright_ratio"] > 0.08:
            tumor_type = "pituitary tumor"
        elif features["asymmetry"] > 0.05 and features["edge_density"] > 0.04:
            tumor_type = "glioma tumor"
        elif features["bright_ratio"] > 0.10:
            tumor_type = "meningioma tumor"
        else:
            tumor_type = "glioma tumor"

        return tumor_type, round(clamp(0.55 + tumor_score * 0.35, 0.55, 0.97), 2)

    def _estimate_grade(
        self,
        features: Dict[str, float],
        tumor_type: str,
        tumor_detected: bool,
    ) -> tuple[Optional[str], float]:
        if not tumor_detected:
            return None, 0.1

        mapped_grade = TUMOR_STAGE_BY_TYPE.get(str(tumor_type).strip().lower())
        if mapped_grade:
            confidence = clamp(
                0.74
                + (features["texture_score"] * 0.18)
                + (features["edge_density"] * 0.12)
            )
            return mapped_grade, round(confidence, 2)

        severity = clamp(
            (features["bright_ratio"] * 2.1)
            + (features["edge_density"] * 1.8)
            + (features["texture_score"] * 1.5)
            + (features["asymmetry"] * 2.2)
        )
        if severity < 0.3:
            return "Grade I", round(clamp(0.62 + severity * 0.25, 0.62, 0.9), 2)
        if severity < 0.52:
            return "Grade II", round(clamp(0.66 + severity * 0.23, 0.66, 0.92), 2)
        if severity < 0.72:
            return "Grade III", round(clamp(0.7 + severity * 0.2, 0.7, 0.94), 2)
        return "Grade IV", round(clamp(0.75 + severity * 0.18, 0.75, 0.96), 2)
