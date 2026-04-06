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


ALZHEIMER_STAGE_CLASSES = ["NonDemented", "Very Mild", "Mild", "Moderate"]
LEGACY_ALZHEIMER_CLASSES = ["MildDementia", "ModerateDementia", "NonDementia", "VeryMildDementia"]
ALZHEIMER_STAGE_ALIASES = {
    "nondemented": "NonDemented",
    "non_demented": "NonDemented",
    "non dementia": "NonDemented",
    "nondementia": "NonDemented",
    "verymilddementia": "Very Mild",
    "very_mild": "Very Mild",
    "very mild": "Very Mild",
    "mci": "Very Mild",
    "milddementia": "Mild",
    "mild": "Mild",
    "earlystage": "Mild",
    "early stage": "Mild",
    "moderatedementia": "Moderate",
    "moderate": "Moderate",
    "moderatestage": "Moderate",
    "moderate stage": "Moderate",
    "severe": "Moderate",
    "severe stage": "Moderate",
}


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def normalize_alzheimer_stage_label(stage: Optional[str]) -> Optional[str]:
    if stage is None:
        return None
    normalized = str(stage).strip()
    if not normalized:
        return None
    alias_key = normalized.lower().replace("-", " ").replace("_", " ")
    alias_key = " ".join(alias_key.split())
    alias_compact = alias_key.replace(" ", "")
    return ALZHEIMER_STAGE_ALIASES.get(alias_compact) or ALZHEIMER_STAGE_ALIASES.get(alias_key) or normalized


def build_alzheimer_staging_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    variant: str = "cnn",
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    fine_tune_layers: int = 0,
) -> Optional[Any]:
    """Build a modified CNN or transfer-learning model for Alzheimer staging."""
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
            raise ValueError(f"Unsupported Alzheimer model variant: {variant}")

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
    outputs = Dense(num_classes, activation="softmax", name="cdr_stage")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"alzheimer_staging_model_{normalized_variant}")


class AlzheimerStagingModel:
    """Alzheimer staging model with support for legacy weights and safe fallbacks."""

    def __init__(self, model_path: str = "alz_model_new.h5"):
        self.model_path = resolve_model_path(
            model_path,
            "ALZ_MODEL_PATH",
            manifest_key="alzheimer_classifier",
        )
        self.metadata = resolve_model_metadata("alzheimer_classifier")
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
        return list(LEGACY_ALZHEIMER_CLASSES)

    def predict(self, image: np.ndarray, prepared_image: Optional[np.ndarray] = None) -> Dict[str, object]:
        features = extract_radiology_features(image)
        model_output = self._predict_with_model(image, features, prepared_image=prepared_image)

        if model_output:
            return model_output
        if self.strict_loading:
            raise ModelUnavailableError("Alzheimer staging model is unavailable.")

        disease_score = clamp(
            ((1.2 - features["center_focus"]) * 0.75)
            + (features["texture_score"] * 1.7)
            + (features["std_intensity"] * 0.9)
            + (features["asymmetry"] * 1.8)
            + ((1.0 - features["mean_intensity"]) * 0.3)
        )

        if disease_score < 0.22:
            return {
                "detected": False,
                "stage": "NonDemented",
                "confidence": round(clamp(0.1 + disease_score * 0.4, 0.1, 0.4), 2),
                "backend": "heuristic",
            }

        if disease_score < 0.38:
            stage = "Very Mild"
        elif disease_score < 0.6:
            stage = "Mild"
        else:
            stage = "Moderate"

        return {
            "detected": True,
            "stage": stage,
            "confidence": round(clamp(0.55 + disease_score * 0.35, 0.55, 0.96), 2),
            "backend": "heuristic",
        }

    def _predict_with_model(
        self,
        image: np.ndarray,
        features: Dict[str, float],
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
            confidence = clamp(scores[class_index])
            label = normalize_alzheimer_stage_label(self.class_names[class_index])

            if label == "NonDemented":
                return {
                    "detected": False,
                    "stage": "NonDemented",
                    "confidence": round(confidence, 2),
                    "backend": self.backend,
                }

            return {
                "detected": True,
                "stage": label or "Moderate",
                "confidence": round(confidence, 2),
                "backend": self.backend,
            }
        except Exception:
            return None
