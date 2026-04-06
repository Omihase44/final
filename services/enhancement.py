import os
from datetime import datetime
from uuid import uuid4
from typing import Dict, Optional

import cv2
import numpy as np

from utils.image_processing import ensure_three_channel


class ImageEnhancementService:
    """Enhance MRI slices for cleaner downstream inference and reporting."""

    def __init__(self, clahe_clip_limit: float = 2.5, clahe_grid_size: tuple[int, int] = (8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def enhance(self, image: np.ndarray) -> Dict[str, object]:
        image = ensure_three_channel(image)
        lightly_smoothed = cv2.GaussianBlur(image, (3, 3), 0.6)

        lab_image = cv2.cvtColor(lightly_smoothed, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab_image)
        enhanced_lightness = self._clahe.apply(lightness)
        contrast_enhanced = cv2.merge((enhanced_lightness, channel_a, channel_b))
        contrast_enhanced = cv2.cvtColor(contrast_enhanced, cv2.COLOR_LAB2BGR)

        sharpen_kernel = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 5.25, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
        final_image = np.clip(sharpened, 0, 255).astype(np.uint8)

        return {
            "original_image": image,
            "enhanced_image": final_image,
            "backend": "opencv",
            "steps": ["gaussian_blur_light", "clahe", "sharpening_filter"],
        }


def save_analysis_assets(
    images: Dict[str, np.ndarray],
    output_root: str = "uploads/analysis_assets",
    study_id: Optional[str] = None,
) -> Dict[str, object]:
    """Persist analysis images and return absolute paths."""
    study_id = study_id or f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:8]}"
    absolute_root = os.path.abspath(output_root)
    output_dir = os.path.join(absolute_root, study_id)
    os.makedirs(output_dir, exist_ok=True)

    file_paths: Dict[str, str] = {}
    for name, image in images.items():
        safe_name = f"{name}.png"
        output_path = os.path.abspath(os.path.join(output_dir, safe_name))
        success, encoded = cv2.imencode(".png", ensure_three_channel(image) if image.ndim != 2 else image)
        if not success:
            raise ValueError(f"Failed to encode {name} for saving.")
        with open(output_path, "wb") as file_handle:
            file_handle.write(encoded.tobytes())
        file_paths[name] = output_path

    return {
        "study_id": study_id,
        "output_dir": output_dir,
        "files": file_paths,
    }
