import cv2
import numpy as np

from services.enhancement import ImageEnhancementService
from services.preprocessing import get_preprocessing_service


def test_preprocessing_pipeline_builds_classifier_and_segmentation_inputs():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = (200, 200, 200)
    success, encoded = cv2.imencode(".png", image)
    assert success is True

    prepared = get_preprocessing_service().prepare(image_bytes=encoded.tobytes())

    assert prepared.original_image.shape == (64, 64, 3)
    assert prepared.enhanced_image.shape == (64, 64, 3)
    assert prepared.classifier_input.shape == (1, 224, 224, 3)
    assert prepared.segmentation_input.shape == (1, 128, 128, 1)
    assert prepared.classifier_input.dtype == np.float32
    assert prepared.classifier_input.min() >= 0.0
    assert prepared.classifier_input.max() <= 1.0


def test_enhancement_service_returns_expected_steps():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    result = ImageEnhancementService().enhance(image)

    assert result["backend"] == "opencv"
    assert "clahe" in result["steps"]
    assert "gaussian_blur_light" in result["steps"]
    assert "sharpening_filter" in result["steps"]
    assert result["enhanced_image"].shape == image.shape
