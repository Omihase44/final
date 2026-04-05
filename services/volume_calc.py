from functools import lru_cache
from typing import Dict

import numpy as np

from utils.volume_calculation import calculate_tumor_volume


class VolumeCalculationService:
    """Estimate tumor volume in cubic millimeters from the segmentation mask."""

    def calculate(
        self,
        mask: np.ndarray,
        pixel_spacing_x: float,
        pixel_spacing_y: float,
        slice_thickness: float,
    ) -> Dict[str, float]:
        return calculate_tumor_volume(
            mask=mask,
            pixel_spacing_x=pixel_spacing_x,
            pixel_spacing_y=pixel_spacing_y,
            slice_thickness=slice_thickness,
        )


@lru_cache(maxsize=1)
def get_volume_calculation_service() -> VolumeCalculationService:
    return VolumeCalculationService()

