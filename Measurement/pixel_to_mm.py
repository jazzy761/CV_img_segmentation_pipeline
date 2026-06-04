import cv2
import numpy as np


REAL_DIAMETER_MM = 13.5  # 1 rupee coin approx


def mask_to_bbox(mask):
    """
    Convert binary mask → bounding box in pixels
    """
    coords = np.column_stack(np.where(mask > 0))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width_px = x_max - x_min
    height_px = y_max - y_min

    return width_px, height_px


def compute_mm(width_px, height_px, ref_diameter_px):
    """
    Convert pixels → mm using calibration ratio
    """

    mm_per_px = REAL_DIAMETER_MM / ref_diameter_px

    width_mm = width_px * mm_per_px
    height_mm = height_px * mm_per_px

    return width_mm, height_mm, mm_per_px


def estimate_object_size(mask):
    """
    Full pipeline for one coin mask
    """

    width_px, height_px = mask_to_bbox(mask)

    # use max dimension as diameter reference
    ref_px = max(width_px, height_px)

    width_mm, height_mm, scale = compute_mm(
        width_px,
        height_px,
        ref_px
    )

    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "mm_per_pixel": scale
    }