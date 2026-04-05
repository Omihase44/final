from typing import Dict

import numpy as np


def calculate_tumor_volume(
    mask: np.ndarray,
    pixel_spacing_x: float,
    pixel_spacing_y: float,
    slice_thickness: float,
) -> Dict[str, float]:
    """Compute tumor volume in mm^3 from a binary mask and voxel metadata."""
    if mask is None:
        raise ValueError("Segmentation mask is required for volume calculation.")

    white_pixel_count = int(np.count_nonzero(mask > 0))
    voxel_volume = float(pixel_spacing_x) * float(pixel_spacing_y) * float(slice_thickness)
    tumor_volume_mm3 = white_pixel_count * voxel_volume

    return {
        "white_pixel_count": white_pixel_count,
        "voxel_volume_mm3": round(voxel_volume, 4),
        "tumor_volume_mm3": round(float(tumor_volume_mm3), 2),
    }
