import os
from typing import Any, Dict, Optional

import numpy as np

from utils.image_processing import extract_radiology_features, preprocess_classifier_image
from utils.tensorflow_compat import get_tensorflow_keras, safe_load_keras_model


TUMOR_TYPE_CLASSES = ["glioma tumor", "meningioma tumor", "no tumor", "pituitary tumor"]
WHO_GRADE_CLASSES = ["Grade I", "Grade II", "Grade III", "Grade IV"]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def build_tumor_grading_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
) -> Optional[Any]:
    """Build a modified CNN with BatchNorm, Dropout, and softmax grading output."""
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
    outputs = Dense(num_classes, activation="softmax", name="tumor_grade")(x)
    return Model(inputs=inputs, outputs=outputs, name="tumor_grading_model")


class TumorGradingModel:
    """Tumor classifier that supports real models when available and heuristics otherwise."""

    def __init__(self, model_path: str = "brain_model_new.h5"):
        self.model_path = os.path.abspath(model_path)
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"

    def _load_model(self):
        return safe_load_keras_model(self.model_path)

    def predict(self, image: np.ndarray) -> Dict[str, object]:
        features = extract_radiology_features(image)
        model_output = self._predict_with_model(image)

        if model_output:
            tumor_type = model_output["classification"]
            type_confidence = model_output["confidence"]
            backend = self.backend
        else:
            tumor_type, type_confidence = self._heuristic_tumor_type(features)
            backend = "heuristic"

        tumor_detected = tumor_type != "no tumor"
        tumor_grade, grade_confidence = self._estimate_grade(features, tumor_type, tumor_detected)
        confidence = type_confidence if not tumor_detected else max(type_confidence, grade_confidence)

        return {
            "detected": tumor_detected,
            "classification": tumor_type,
            "grade": tumor_grade,
            "confidence": round(confidence, 2),
            "backend": backend,
        }

    def _predict_with_model(self, image: np.ndarray) -> Optional[Dict[str, object]]:
        if self.model is None:
            return None

        try:
            prepared_image = preprocess_classifier_image(image)
            prediction = self.model.predict(prepared_image, verbose=0)
            scores = np.asarray(prediction).squeeze()
            if scores.ndim != 1 or scores.shape[0] < len(TUMOR_TYPE_CLASSES):
                return None

            class_index = int(np.argmax(scores[: len(TUMOR_TYPE_CLASSES)]))
            return {
                "classification": TUMOR_TYPE_CLASSES[class_index],
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

        type_bias = {
            "glioma tumor": 0.22,
            "meningioma tumor": 0.08,
            "pituitary tumor": 0.04,
        }.get(tumor_type, 0.0)

        severity = clamp(
            (features["bright_ratio"] * 2.1)
            + (features["edge_density"] * 1.8)
            + (features["texture_score"] * 1.5)
            + (features["asymmetry"] * 2.2)
            + type_bias
        )

        if severity < 0.32:
            grade = "Grade I"
        elif severity < 0.52:
            grade = "Grade II"
        elif severity < 0.72:
            grade = "Grade III"
        else:
            grade = "Grade IV"

        return grade, round(clamp(0.6 + severity * 0.3, 0.6, 0.95), 2)
