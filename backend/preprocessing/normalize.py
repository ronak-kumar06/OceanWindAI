"""
SAR image tiling into overlapping patches.

Splits a (H, W) sigma0 array into N×N pixel patches with configurable
stride. Returns both the patches and the geographic coordinates (lat/lon)
of each patch centre, which are needed to re-assemble the wind field grid.
"""
import numpy as np
from typing import List, Tuple
from config import PATCH_SIZE, PATCH_STRIDE


def extract_patches(
    sigma0: np.ndarray,
    inc_angle: np.ndarray,
    bbox: list,
) -> List[dict]:
    """
    Extract overlapping patches from a SAR scene.

    Parameters
    ----------
    sigma0    : (H, W) float32 — linear-scale sigma0 (NaN = masked)
    inc_angle : (H, W) float32 — local incidence angle in degrees
    bbox      : [min_lon, min_lat, max_lon, max_lat]

    Returns
    -------
    List of dicts, each with:
        patch      : (PATCH_SIZE, PATCH_SIZE) sigma0 sub-array
        inc_patch  : (PATCH_SIZE, PATCH_SIZE) incidence angle sub-array
        lat        : float — patch centre latitude
        lon        : float — patch centre longitude
        valid      : bool  — False if >50% of pixels are NaN (land / no data)
    """
    H, W = sigma0.shape
    min_lon, min_lat, max_lon, max_lat = bbox

    lon_per_px = (max_lon - min_lon) / W
    lat_per_px = (max_lat - min_lat) / H   # positive = south-to-north

    patches = []
    ps = PATCH_SIZE
    st = PATCH_STRIDE

    for row in range(0, H - ps + 1, st):
        for col in range(0, W - ps + 1, st):
            patch = sigma0[row:row + ps, col:col + ps]
            inc_p = inc_angle[row:row + ps, col:col + ps]

            # Centre pixel in geographic coordinates
            cx = col + ps // 2
            cy = row + ps // 2
            lon = min_lon + cx * lon_per_px
            lat = max_lat - cy * lat_per_px   # lat decreases downward

            valid_frac = np.isfinite(patch).mean()
            patches.append({
                "patch":     patch,
                "inc_patch": inc_p,
                "lat":       float(lat),
                "lon":       float(lon),
                "valid":     valid_frac > 0.5,
            })

    return patches
