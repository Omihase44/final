import os
from typing import Any, Optional

import cv2
import numpy as np

from utils.image_processing import preprocess_segmentation_image
from utils.tensorflow_compat import get_tensorflow_keras, safe_load_keras_model


def build_unet_model(
    input_shape: tuple[int, int, int] = (128, 128, 1),
) -> Optional[Any]:
    """Build a U-Net style segmentation network with encoder-decoder skip connections."""
    tensorflow_keras = get_tensorflow_keras()
    if not tensorflow_keras["available"]:
        return None

    layers = tensorflow_keras["layers"]
    models = tensorflow_keras["models"]
    BatchNormalization = layers.BatchNormalization
    Concatenate = layers.Concatenate
    Conv2D = layers.Conv2D
    Dropout = layers.Dropout
    Input = layers.Input
    MaxPooling2D = layers.MaxPooling2D
    UpSampling2D = layers.UpSampling2D
    Model = models.Model

    inputs = Input(shape=input_shape)

    down_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    down_1 = BatchNormalization()(down_1)
    down_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(down_1)
    down_1 = BatchNormalization()(down_1)
    pool_1 = MaxPooling2D((2, 2))(down_1)

    down_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool_1)
    down_2 = BatchNormalization()(down_2)
    down_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(down_2)
    down_2 = BatchNormalization()(down_2)
    pool_2 = MaxPooling2D((2, 2))(down_2)

    bridge = Conv2D(128, (3, 3), activation="relu", padding="same")(pool_2)
    bridge = BatchNormalization()(bridge)
    bridge = Conv2D(128, (3, 3), activation="relu", padding="same")(bridge)
    bridge = BatchNormalization()(bridge)
    bridge = Dropout(0.3)(bridge)

    up_1 = UpSampling2D((2, 2))(bridge)
    up_1 = Conv2D(64, (2, 2), activation="relu", padding="same")(up_1)
    up_1 = Concatenate()([up_1, down_2])
    up_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_1)
    up_1 = BatchNormalization()(up_1)
    up_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_1)
    up_1 = BatchNormalization()(up_1)

    up_2 = UpSampling2D((2, 2))(up_1)
    up_2 = Conv2D(32, (2, 2), activation="relu", padding="same")(up_2)
    up_2 = Concatenate()([up_2, down_1])
    up_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(up_2)
    up_2 = BatchNormalization()(up_2)
    up_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(up_2)
    up_2 = BatchNormalization()(up_2)

    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="tumor_mask")(up_2)
    return Model(inputs=inputs, outputs=outputs, name="tumor_unet")


class TumorSegmentationModel:
    """Lesion segmentation model with heuristic fallback when weights are unavailable."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: tuple[int, int] = (128, 128),
    ):
        default_model_path = os.environ.get("SEGMENTATION_MODEL_PATH") or os.path.join(
            "models",
            "tumor_segmentation_unet.h5",
        )
        resolved_model_path = model_path or default_model_path
        self.model_path = os.path.abspath(resolved_model_path) if resolved_model_path else None
        self.input_size = input_size
        self.model = self._load_model()
        self.backend = "keras" if self.model is not None else "heuristic"

    def _load_model(self):
        if not self.model_path:
            return None
        return safe_load_keras_model(self.model_path)

    def predict_mask(self, image: np.ndarray, tumor_detected: bool = True) -> np.ndarray:
        if not tumor_detected:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        model_input = preprocess_segmentation_image(image, self.input_size)

        if self.model is not None:
            mask = self._predict_with_model(model_input)
        else:
            mask = self._heuristic_mask(model_input[0, :, :, 0])

        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return self._postprocess_mask(resized_mask)

    def _predict_with_model(self, model_input: np.ndarray) -> np.ndarray:
        try:
            prediction = self.model.predict(model_input, verbose=0)[0, :, :, 0]
            return np.where(prediction >= 0.5, 255, 0).astype(np.uint8)
        except Exception:
            return self._heuristic_mask(model_input[0, :, :, 0])

    def _heuristic_mask(self, normalized_image: np.ndarray) -> np.ndarray:
        image_u8 = np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(image_u8, (5, 5), 0)

        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        percentile_value = int(np.percentile(blurred, 90))
        _, high_intensity_mask = cv2.threshold(blurred, percentile_value, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(otsu_mask, high_intensity_mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        largest_component = self._largest_component(mask)
        if np.count_nonzero(largest_component) == 0:
            percentile_value = int(np.percentile(blurred, 95))
            _, fallback_mask = cv2.threshold(blurred, percentile_value, 255, cv2.THRESH_BINARY)
            largest_component = self._largest_component(fallback_mask)

        return largest_component

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return np.zeros_like(mask, dtype=np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        minimum_area = max(25, int(mask.shape[0] * mask.shape[1] * 0.002))

        if int(stats[largest_label, cv2.CC_STAT_AREA]) < minimum_area:
            return np.zeros_like(mask, dtype=np.uint8)

        return np.where(labels == largest_label, 255, 0).astype(np.uint8)

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._largest_component(cleaned)
