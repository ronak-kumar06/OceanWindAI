"""
SAR patch tiling into overlapping patches.
"""
import numpy as np
from typing import List
from config import PATCH_SIZE, PATCH_STRIDE


def extract_patches(sigma0: np.ndarray, inc_angle: np.ndarray, bbox: list) -> List[dict]:
    """
    Extract overlapping patches from a SAR scene.

    Parameters
    ----------
    sigma0    : (H, W) float32 — linear-scale sigma0 (NaN = masked)
    inc_angle : (H, W) float32 — local incidence angle in degrees
    bbox      : [min_lon, min_lat, max_lon, max_lat]

    Returns
    -------
    List of dicts with keys: patch, inc_patch, lat, lon, valid
    """
    H, W = sigma0.shape
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_per_px = (max_lon - min_lon) / W
    lat_per_px = (max_lat - min_lat) / H
    ps, st = PATCH_SIZE, PATCH_STRIDE
    patches = []

    for row in range(0, H - ps + 1, st):
        for col in range(0, W - ps + 1, st):
            patch = sigma0[row:row + ps, col:col + ps]
            inc_p = inc_angle[row:row + ps, col:col + ps]
            cx, cy = col + ps // 2, row + ps // 2
            lon = min_lon + cx * lon_per_px
            lat = max_lat - cy * lat_per_px
            valid_frac = np.isfinite(patch).mean()
            patches.append({
                "patch":     patch,
                "inc_patch": inc_p,
                "lat":       float(lat),
                "lon":       float(lon),
                "valid":     valid_frac > 0.5,
            })
    return patches
