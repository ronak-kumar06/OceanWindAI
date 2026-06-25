"""
180° directional ambiguity resolution for SAR wind direction.

A ResNet trained on SAR returns aliased directions in [0°, 180°) because
SAR backscatter is symmetric with respect to the wind direction:
the SAR image looks identical whether wind blows from 30° or 210°.

Resolution strategy
-------------------
Given the aliased direction θ_aliased (0–180°), the true direction is either
  θ_aliased  OR  θ_aliased + 180°

We pick whichever is closer (in angular distance) to a background reference
wind direction. The reference can come from:
  1. ERA5 10m wind (preferred — most accurate)
  2. Climatological background (fallback — see inference.py)
  3. User-supplied manual reference

Reference
---------
Koch, W. (2004). Directional analysis of SAR images aiming at wind direction.
IEEE TGRS, 42(4), 702-710.
"""
import math
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


def _angular_distance(a: float, b: float) -> float:
    """Signed angular distance between two bearings (degrees)."""
    diff = (a - b + 360.0) % 360.0
    return diff if diff <= 180.0 else diff - 360.0


def resolve_ambiguity(
    aliased_directions: List[float],
    reference_direction: Optional[float] = None,
    lat: float = 15.0,
    lon: float = 75.0,
) -> List[float]:
    """
    Resolve 180° ambiguity for each aliased direction.

    Parameters
    ----------
    aliased_directions : list of floats in [0, 180) — from ResNet inference
    reference_direction : optional scalar background direction (0–360°)
                          from ERA5 or climatology
    lat, lon : approximate scene centre (used for climatological fallback)

    Returns
    -------
    resolved : list of floats in [0, 360) — meteorological wind direction
               (direction FROM which wind blows)
    """
    # If no reference provided, use climatological fallback
    if reference_direction is None:
        reference_direction = _climatological_reference(lat, lon)
        logger.info("Using climatological reference direction: %.1f°", reference_direction)
    else:
        logger.info("Using ERA5 reference direction: %.1f°", reference_direction)

    resolved = []
    for aliased in aliased_directions:
        if not math.isfinite(aliased):
            resolved.append(float("nan"))
            continue

        candidate_1 = aliased % 360.0
        candidate_2 = (aliased + 180.0) % 360.0

        dist_1 = abs(_angular_distance(candidate_1, reference_direction))
        dist_2 = abs(_angular_distance(candidate_2, reference_direction))

        resolved.append(candidate_1 if dist_1 <= dist_2 else candidate_2)

    resolved_arr = np.array(resolved)
    finite = resolved_arr[np.isfinite(resolved_arr)]
    if finite.size > 0:
        logger.info("Ambiguity resolved: mean direction = %.1f° ± %.1f°",
                    float(np.nanmean(finite)), float(np.nanstd(finite)))

    return resolved


def _climatological_reference(lat: float, lon: float) -> float:
    """
    Return approximate seasonal-mean wind direction for Indian coastal waters.
    (June–September = SW monsoon, October–December = NE monsoon).
    This is only used as a last resort when ERA5 is unavailable.
    """
    # Arabian Sea
    if lon < 77.0:
        return 225.0   # SW monsoon
    # Bay of Bengal
    return 210.0


# ── In-memory cache for ERA5 references ───────────────────────────────────
_era5_cache: dict = {}

def fetch_era5_reference(date: str, bbox: list) -> Optional[float]:
    """
    Attempt to fetch a spatially-averaged ERA5 10m wind direction for the AOI.
    Results are cached in memory per date and bbox.

    Requires CDS_KEY environment variable to be set.
    Returns None if CDS is unavailable (triggers climatological fallback).
    """
    cache_key = (date, tuple(bbox))
    if cache_key in _era5_cache:
        logger.info("ERA5 reference cache HIT for date=%s bbox=%s", date, bbox)
        return _era5_cache[cache_key]

    from config import CDS_KEY, CDS_LIVE, CDS_URL
    if not CDS_LIVE:
        logger.info("CDS API not configured — skipping ERA5 reference fetch")
        return None

    logger.info("ERA5 reference cache MISS — fetching for date=%s bbox=%s", date, bbox)
    try:
        import cdsapi
        import netCDF4 as nc
        import tempfile, os

        min_lon, min_lat, max_lon, max_lat = bbox
        year, month, day = date.split("-")

        client = cdsapi.Client(
            url=CDS_URL,
            key=CDS_KEY,
            quiet=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": year, "month": month, "day": day,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [max_lat, min_lon, min_lat, max_lon],
                "format": "netcdf",
            },
            tmp_path,
        )

        ds = nc.Dataset(tmp_path)
        u10 = np.array(ds.variables["u10"][:]).mean()
        v10 = np.array(ds.variables["v10"][:]).mean()
        ds.close()
        os.unlink(tmp_path)

        # Meteorological direction: FROM where wind blows
        direction = (270.0 - math.degrees(math.atan2(v10, u10))) % 360.0
        logger.info("ERA5 mean wind direction for AOI: %.1f°", direction)
        
        _era5_cache[cache_key] = direction
        return direction

    except Exception as exc:
        logger.warning("ERA5 fetch failed (%s) — using climatological reference", exc)
        return None
