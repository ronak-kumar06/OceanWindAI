"""
Spatiotemporal collocation of SAR patches with ASCAT scatterometer winds.

ASCAT L2 wind products (e.g. OSI SAF ASCAT 25 km) provide ground-truth
wind speed and direction. This module matches each SAR patch center to the
nearest ASCAT observation within configurable time and distance windows.

For production training, replace load_ascat_observations() with real NetCDF
ingestion from EUMETSAT/OSI SAF archives.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class AscatObservation:
    """Single ASCAT wind observation."""

    lat: float
    lon: float
    time: datetime
    wind_speed: float      # m/s
    wind_dir: float        # degrees, meteorological (from)
    quality_flag: int = 0


@dataclass(frozen=True)
class CollocatedSample:
    """SAR patch matched to an ASCAT wind observation."""

    patch_index: int
    patch_path: str
    lat: float
    lon: float
    sar_time: datetime
    wind_dir: float
    wind_speed: float
    distance_km: float
    time_delta_min: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def alias_direction(direction_deg: float) -> float:
    """Collapse meteorological direction to 0–180° SAR ambiguity range."""
    return float(direction_deg % 180.0)


def direction_to_bin(direction_deg: float, n_bins: int = 36) -> int:
    """Map aliased direction (0–180°) to class index."""
    aliased = alias_direction(direction_deg)
    bin_idx = int(aliased // (180.0 / n_bins))
    return min(bin_idx, n_bins - 1)


def load_ascat_observations(
    bbox: Sequence[float],
    scene_time: datetime,
    *,
    simulate: bool = True,
) -> List[AscatObservation]:
    """
    Load ASCAT observations for a bounding box and time window.

    Parameters
    ----------
    bbox : [min_lon, min_lat, max_lon, max_lat]
    scene_time : SAR acquisition time
    simulate : when True, generate synthetic ASCAT grid for development

    Returns
    -------
    List of AscatObservation
    """
    if not simulate:
        raise NotImplementedError(
            "Real ASCAT ingestion not yet implemented. "
            "Integrate OSI SAF ASCAT L2 NetCDF reader here."
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    observations: List[AscatObservation] = []
    grid_step = 0.25

    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            # Synthetic monsoon-like field for scaffold testing
            base_dir = 210.0 if lon < 77.0 else 205.0
            noise = np.random.default_rng(int(lat * 1000 + lon * 100)).normal(0, 8)
            wind_dir = (base_dir + noise) % 360.0
            wind_speed = 6.0 + np.random.default_rng(int(lat * 500 + lon * 50)).normal(0, 1.5)
            observations.append(
                AscatObservation(
                    lat=lat,
                    lon=lon,
                    time=scene_time,
                    wind_speed=max(2.0, float(wind_speed)),
                    wind_dir=float(wind_dir),
                    quality_flag=0,
                )
            )
            lon += grid_step
        lat += grid_step

    logger.info("Loaded %d simulated ASCAT observations", len(observations))
    return observations


def collocate_patches(
    patches: Iterable[dict],
    ascat_obs: Sequence[AscatObservation],
    sar_time: datetime,
    *,
    max_distance_km: float = 25.0,
    max_time_min: float = 60.0,
    min_wind_speed: float = 4.0,
) -> List[CollocatedSample]:
    """
    Match SAR patches to nearest ASCAT observations.

    Parameters
    ----------
    patches : output of preprocessing.tiling.extract_patches()
    ascat_obs : ASCAT wind observations
    sar_time : SAR scene acquisition time
    max_distance_km : spatial collocation radius
    max_time_min : temporal collocation window (±minutes)
    min_wind_speed : minimum ASCAT speed for valid CMOD5.N range

    Returns
    -------
    List of CollocatedSample for patches with valid matches
    """
    if not ascat_obs:
        return []

    samples: List[CollocatedSample] = []
    for idx, patch in enumerate(patches):
        if not patch.get("valid", False):
            continue

        lat, lon = patch["lat"], patch["lon"]
        best: Optional[tuple[float, float, AscatObservation]] = None

        for obs in ascat_obs:
            if obs.quality_flag != 0:
                continue
            if obs.wind_speed < min_wind_speed:
                continue

            dt_min = abs((obs.time - sar_time).total_seconds()) / 60.0
            if dt_min > max_time_min:
                continue

            dist = haversine_km(lat, lon, obs.lat, obs.lon)
            if dist > max_distance_km:
                continue

            score = dist + dt_min * 0.1
            if best is None or score < best[0]:
                best = (score, dist, obs)

        if best is None:
            continue

        _, dist_km, obs = best
        dt_min = abs((obs.time - sar_time).total_seconds()) / 60.0
        samples.append(
            CollocatedSample(
                patch_index=idx,
                patch_path=f"patch_{idx:06d}.npy",
                lat=lat,
                lon=lon,
                sar_time=sar_time,
                wind_dir=alias_direction(obs.wind_dir),
                wind_speed=obs.wind_speed,
                distance_km=dist_km,
                time_delta_min=dt_min,
            )
        )

    logger.info("Collocated %d / %d valid patches", len(samples), sum(1 for p in patches if p.get("valid")))
    return samples


def build_manifest_from_scene(
    sigma0: np.ndarray,
    inc_angle: np.ndarray,
    bbox: Sequence[float],
    scene_time: datetime,
    output_dir: str,
    *,
    simulate_ascat: bool = True,
) -> List[CollocatedSample]:
    """
    Extract patches from a SAR scene, collocate with ASCAT, and return samples.

    Saves patch arrays to output_dir when samples are produced.
    """
    from pathlib import Path

    from preprocessing.tiling import extract_patches

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patches = extract_patches(sigma0, inc_angle, list(bbox))
    ascat = load_ascat_observations(bbox, scene_time, simulate=simulate_ascat)
    samples = collocate_patches(patches, ascat, scene_time)

    for sample in samples:
        patch = patches[sample.patch_index]
        patch_file = out / sample.patch_path
        if not patch_file.exists():
            np.save(patch_file, patch["patch"].astype(np.float32))

    return samples
