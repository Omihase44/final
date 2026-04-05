from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np

from models.segmentation_model import TumorSegmentationModel
from utils.image_processing import ensure_three_channel


class SegmentationService:
    """Segment lesion masks and generate clinically readable overlays."""

    def __init__(self):
        self.model = TumorSegmentationModel()

    def segment(
        self,
        original_image: np.ndarray,
        working_image: np.ndarray,
        tumor_detected: bool = True,
    ) -> Dict[str, object]:
        original_image = ensure_three_channel(original_image)
        working_image = ensure_three_channel(working_image)
        mask = self.model.predict_mask(working_image, tumor_detected=tumor_detected)
        display_mask = self._build_display_mask(mask)
        overlay = self._draw_contours_and_box(original_image, display_mask)
        bounding_box = self._extract_bounding_box(display_mask)

        return {
            "mask": mask,
            "overlay": overlay,
            "bounding_box": bounding_box,
            "backend": self.model.backend,
            "contour_count": self._contour_count(display_mask),
        }

    def _build_display_mask(self, mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) == 0:
            return mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        smoothed = cv2.GaussianBlur(dilated, (7, 7), 0)
        _, thresholded = cv2.threshold(smoothed, 32, 255, cv2.THRESH_BINARY)
        return thresholded

    def _build_alpha_mask(self, mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
        return np.clip(blurred * 0.28, 0.0, 0.28)

    def _draw_contours_and_box(self, image: np.ndarray, display_mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(display_mask) == 0:
            return image.copy()

        overlay = image.astype(np.float32).copy()
        alpha_mask = self._build_alpha_mask(display_mask)
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 1] = 255.0
        overlay = (overlay * (1.0 - alpha_mask[..., None])) + (color_mask * alpha_mask[..., None])
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        bounding_box = self._extract_bounding_box(display_mask)
        if bounding_box:
            x = bounding_box["x"]
            y = bounding_box["y"]
            w = bounding_box["width"]
            h = bounding_box["height"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        return overlay

    def _extract_bounding_box(self, mask: np.ndarray) -> Optional[Dict[str, int]]:
        if np.count_nonzero(mask) == 0:
            return None
        x, y, width, height = cv2.boundingRect(mask)
        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }

    def _contour_count(self, mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)


@lru_cache(maxsize=1)
def get_segmentation_service() -> SegmentationService:
    return SegmentationService()
