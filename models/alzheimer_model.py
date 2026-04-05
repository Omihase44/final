import os
from typing import Any, Dict, Optional

import numpy as np

from utils.image_processing import extract_radiology_features, preprocess_classifier_image
from utils.tensorflow_compat import get_tensorflow_keras, safe_load_keras_model


ALZHEIMER_STAGE_CLASSES = ["MCI", "Early Stage", "Moderate Stage", "Severe Stage"]
LEGACY_ALZHEIMER_CLASSES = ["MildDementia", "ModerateDementia", "NonDementia", "VeryMildDementia"]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def build_alzheimer_staging_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
) -> Optional[Any]:
    """Build a modified CNN with BatchNorm, Dropout, and softmax staging output."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    layers = tensorflow_keras["layers"]
    models = tensorflow_keras["models"]
    BatchNormalization = layers.BatchNormalization
    Conv2D = layers.Conv2D
    Dense = layers.Dense
    Dropout = layers.Dropout
    GlobalAveragePooling2D = layers.GlobalAveragePooling2D
    Input = layers.Input
    MaxPooling2D = layers.MaxPooling2D
    Model = models.Model

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="cdr_stage")(x)
    return Model(inputs=inputs, outputs=outputs, name="alzheimer_staging_model")


class AlzheimerStagingModel:
    """Alzheimer staging model with support for legacy weights and safe fallbacks."""

    def __init__(self, model_path: str = "alz_model_new.h5"):
        self.model_path = os.path.abspath(model_path)
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"

    def _load_model(self):
        return safe_load_keras_model(self.model_path)

    def predict(self, image: np.ndarray) -> Dict[str, object]:
        features = extract_radiology_features(image)
        model_output = self._predict_with_model(image, features)

        if model_output:
            return model_output

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
                "stage": None,
                "confidence": round(clamp(0.1 + disease_score * 0.4, 0.1, 0.4), 2),
                "backend": "heuristic",
            }

        if disease_score < 0.38:
            stage = "MCI"
        elif disease_score < 0.56:
            stage = "Early Stage"
        elif disease_score < 0.76:
            stage = "Moderate Stage"
        else:
            stage = "Severe Stage"

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
    ) -> Optional[Dict[str, object]]:
        if self.model is None:
            return None

        try:
            prepared_image = preprocess_classifier_image(image)
            prediction = self.model.predict(prepared_image, verbose=0)
            scores = np.asarray(prediction).squeeze()
            if scores.ndim != 1 or scores.shape[0] < len(LEGACY_ALZHEIMER_CLASSES):
                return None

            class_index = int(np.argmax(scores[: len(LEGACY_ALZHEIMER_CLASSES)]))
            confidence = clamp(scores[class_index])
            label = LEGACY_ALZHEIMER_CLASSES[class_index]

            if label == "NonDementia":
                return {
                    "detected": False,
                    "stage": None,
                    "confidence": round(confidence, 2),
                    "backend": self.backend,
                }

            stage_mapping = {
                "VeryMildDementia": "MCI",
                "MildDementia": "Early Stage",
                "ModerateDementia": "Moderate Stage",
            }
            stage = stage_mapping.get(label, "Moderate Stage")

            severe_proxy = clamp(
                (features["texture_score"] * 1.8)
                + (features["asymmetry"] * 1.8)
                + ((1.2 - features["center_focus"]) * 0.8)
            )
            if stage == "Moderate Stage" and severe_proxy > 0.72:
                stage = "Severe Stage"

            return {
                "detected": True,
                "stage": stage,
                "confidence": round(confidence, 2),
                "backend": self.backend,
            }
        except Exception:
            return None
