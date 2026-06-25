"""
Land-sea masking for SAR sigma0 arrays.

Generates a binary ocean mask (True = ocean pixel) for an arbitrary
geographic bounding box using the Natural Earth 110m land polygons.
Falls back to an all-ocean mask if shapefiles are unavailable.

Performance: mask is cached per (bbox, shape) — repeated queries are instant.
"""
import os
import logging
import numpy as np
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)

# ── In-memory cache for masks ─────────────────────────────────────────────
_mask_cache: dict = {}


def _build_mask_shapely(bbox: list, shape: Tuple[int, int]) -> np.ndarray:
    """Use pyshp + shapely to rasterise the land polygons into a mask."""
    import shapefile
    from shapely.geometry import shape as geom_shape, MultiPolygon, Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    import shapely.vectorized

    min_lon, min_lat, max_lon, max_lat = bbox
    H, W = shape

    # Load Natural Earth 110m land geometries manually
    shp_path = os.path.join(os.path.dirname(__file__), "..", "ne_110m_land", "ne_110m_land.shp")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Natural Earth shapefile not found at {shp_path}")

    sf = shapefile.Reader(shp_path)
    geoms = [geom_shape(s.__geo_interface__) for s in sf.shapes()]
    land_geom = unary_union(geoms)
    prepared_land = prep(land_geom)

    logger.debug("Natural Earth land polygons successfully loaded via pyshp/shapely.")

    lons = np.linspace(min_lon, max_lon, W)
    lats = np.linspace(max_lat, min_lat, H)   # top-to-bottom
    LON, LAT = np.meshgrid(lons, lats)

    # Use vectorized contains for high performance (H*W points)
    # returns True for pixels on land
    try:
        land_mask = shapely.vectorized.contains(land_geom, LON, LAT)
    except AttributeError:
        # Fallback if vectorized is missing in some shapely builds
        logger.warning("shapely.vectorized not found, falling back to slow loops.")
        land_mask = np.zeros((H, W), dtype=bool)
        for i in range(H):
            for j in range(W):
                if prepared_land.contains(Point(lons[j], lats[i])):
                    land_mask[i, j] = True

    ocean_mask = ~land_mask

    ocean_count = int(ocean_mask.sum())
    land_count = int((~ocean_mask).sum())
    total_count = H * W
    logger.info("Mask statistics: total_pixels=%d, ocean_pixels=%d, land_pixels=%d",
                total_count, ocean_count, land_count)

    return ocean_mask


def get_ocean_mask(bbox: list, shape: Tuple[int, int]) -> np.ndarray:
    """
    Returns a boolean array (H, W) where True = ocean.
    Results are cached in memory — repeated calls with the same bbox+shape are instant.

    Parameters
    ----------
    bbox  : [min_lon, min_lat, max_lon, max_lat]
    shape : (H, W) — must match the SAR array shape
    """
    cache_key = (tuple(bbox), shape)

    if cache_key in _mask_cache:
        logger.info("Land-sea mask cache HIT for bbox=%s shape=%s", bbox, shape)
        return _mask_cache[cache_key]

    logger.info("Land-sea mask cache MISS — computing for bbox=%s shape=%s", bbox, shape)
    try:
        mask = _build_mask_shapely(bbox, shape)
    except Exception as exc:
        logger.warning("Land-sea mask unavailable (%s). Using all-ocean mask.", exc)
        mask = np.ones(shape, dtype=bool)

    _mask_cache[cache_key] = mask
    return mask


def apply_mask(sigma0: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero-out land pixels in the sigma0 array (sets them to NaN).
    sigma0 : (H, W) — linear-scale backscatter
    mask   : (H, W) — True where ocean
    """
    result = sigma0.copy()
    result[~mask] = np.nan
    return result
