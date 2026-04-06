import numpy as np

from services.segmentation import SegmentationService


class DummySegmentationModel:
    backend = "heuristic"

    def predict_mask(self, image, tumor_detected=True, model_input=None):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:60, 24:58] = 255
        return mask


def test_segmentation_service_draws_green_boundary_overlay():
    service = SegmentationService.__new__(SegmentationService)
    service.model = DummySegmentationModel()

    original_image = np.zeros((80, 80, 3), dtype=np.uint8)
    result = service.segment(original_image, original_image, tumor_detected=True)

    assert result["contour_count"] >= 1
    assert result["bounding_box"] is not None
    assert np.count_nonzero(result["mask"]) > 0
    assert np.any(np.all(result["overlay"] == [0, 255, 0], axis=2))
