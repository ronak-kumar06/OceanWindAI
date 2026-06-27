"""
Load ASCAT L2 wind observations from OSI SAF / EUMETSAT NetCDF files.

Supports standard ASCAT L2 products (25 km, coastal) with variables:
  wind_speed, wind_dir, lat, lon, time, wvc_quality_flag

Files may be plain .nc or .nc.gz. Place archives under ASCAT_DATA_DIR.
"""
from __future__ import annotations

import gzip
import logging
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ASCAT_EPOCH = datetime(1990, 1, 1)
FILENAME_TIME_RE = re.compile(r"(\d{8})_(\d{6})")

SPEED_NAMES = ("wind_speed", "wndspeed", "ff")
DIR_NAMES = ("wind_dir", "wind_direction", "wnddir", "dd")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
TIME_NAMES = ("time",)
QUALITY_NAMES = ("wvc_quality_flag", "quality_flag", "wvc_quality")


def _open_dataset(path: Path):
    from netCDF4 import Dataset

    if path.suffix == ".gz":
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp.close()
        try:
            with gzip.open(path, "rb") as src, open(tmp.name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return Dataset(tmp.name, "r"), Path(tmp.name)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise

    return Dataset(str(path), "r"), None


def _get_var(ds, names: Sequence[str]):
    for name in names:
        if name in ds.variables:
            return ds.variables[name]
    raise KeyError(f"None of {names} found in {getattr(ds, 'filepath', 'dataset')}")


def _decode_values(var) -> np.ndarray:
    data = np.array(var[:], dtype=np.float64)
    fill = getattr(var, "_FillValue", None)
    if fill is not None:
        data[data == fill] = np.nan

    scale = getattr(var, "scale_factor", 1.0)
    offset = getattr(var, "add_offset", 0.0)
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset

    # Some PO.DAAC products store direction/speed as tenths
    units = str(getattr(var, "units", "")).lower()
    long_name = str(getattr(var, "long_name", "")).lower()
    if "degree" in units and np.nanmax(data) > 360:
        data = data / 10.0
    if "m s-1" in units or "wind speed" in long_name:
        if np.nanmax(data) > 100:
            data = data / 100.0

    return data


def _decode_time(var) -> np.ndarray:
    values = np.array(var[:], dtype=np.float64)
    units = getattr(var, "units", "seconds since 1990-01-01 00:00:00")
    if "since" in units:
        epoch_str = units.split("since", 1)[1].strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                epoch = datetime.strptime(epoch_str, fmt)
                break
            except ValueError:
                epoch = ASCAT_EPOCH
    else:
        epoch = ASCAT_EPOCH
    return values


def _time_from_filename(path: Path) -> Optional[datetime]:
    match = FILENAME_TIME_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def _quality_mask(quality: Optional[np.ndarray]) -> np.ndarray:
    if quality is None:
        return np.ones(1, dtype=bool)
    # OSI SAF: 0 = good retrieval for wvc_quality_flag in many products
    return quality.astype(np.int64) == 0


def load_ascat_from_netcdf(
    path: Path,
    bbox: Sequence[float],
    scene_time: datetime,
    *,
    max_time_min: float = 60.0,
) -> list:
    """Extract ASCAT observations from a single L2 NetCDF file."""
    from training.collocation import AscatObservation

    min_lon, min_lat, max_lon, max_lat = bbox
    buffer = 0.5
    t_min = scene_time - timedelta(minutes=max_time_min)
    t_max = scene_time + timedelta(minutes=max_time_min)

    ds, tmp_path = _open_dataset(path)
    observations: list = []
    try:
        lat = _decode_values(_get_var(ds, LAT_NAMES)).ravel()
        lon = _decode_values(_get_var(ds, LON_NAMES)).ravel()
        speed = _decode_values(_get_var(ds, SPEED_NAMES)).ravel()
        direction = _decode_values(_get_var(ds, DIR_NAMES)).ravel()

        try:
            time_var = _get_var(ds, TIME_NAMES)
            time_sec = _decode_time(time_var).ravel()
            epoch = ASCAT_EPOCH
            units = getattr(time_var, "units", "")
            if "since" in units:
                epoch_str = units.split("since", 1)[1].strip()
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                    try:
                        epoch = datetime.strptime(epoch_str, fmt)
                        break
                    except ValueError:
                        pass
            times = [epoch + timedelta(seconds=float(t)) for t in time_sec]
        except KeyError:
            file_time = _time_from_filename(path) or scene_time
            times = [file_time] * lat.size

        try:
            quality = _decode_values(_get_var(ds, QUALITY_NAMES)).ravel()
        except KeyError:
            quality = None

        good = _quality_mask(quality)
        if quality is not None and quality.size == lat.size:
            good = quality.astype(np.int64) == 0

        for i in range(lat.size):
            if not np.isfinite(lat[i]) or not np.isfinite(lon[i]):
                continue
            if quality is not None and i < quality.size and not good.flat[i % good.size]:
                continue
            if not np.isfinite(speed[i]) or not np.isfinite(direction[i]):
                continue
            if speed[i] < 0 or speed[i] > 50:
                continue

            la, lo = float(lat[i]), float(lon[i])
            if lo > 180:
                lo = ((lo + 180) % 360) - 180

            if not (min_lon - buffer <= lo <= max_lon + buffer):
                continue
            if not (min_lat - buffer <= la <= max_lat + buffer):
                continue

            obs_time = times[i] if i < len(times) else times[-1]
            if not (t_min <= obs_time <= t_max):
                continue

            observations.append(
                AscatObservation(
                    lat=la,
                    lon=lo,
                    time=obs_time,
                    wind_speed=float(speed[i]),
                    wind_dir=float(direction[i]) % 360.0,
                    quality_flag=0,
                )
            )
    finally:
        ds.close()
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    return observations


def load_ascat_from_directory(
    ascat_dir: Path,
    bbox: Sequence[float],
    scene_time: datetime,
    *,
    max_time_min: float = 60.0,
) -> list:
    """Load all ASCAT observations from NetCDF files in a directory."""
    from training.collocation import AscatObservation

    ascat_dir = Path(ascat_dir)
    if not ascat_dir.is_dir():
        raise FileNotFoundError(f"ASCAT data directory not found: {ascat_dir}")

    patterns = ("*.nc", "*.nc.gz", "**/*.nc", "**/*.nc.gz")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(ascat_dir.glob(pattern))
    files = sorted(set(files))

    if not files:
        logger.warning("No ASCAT NetCDF files found in %s", ascat_dir)
        return []

    observations: list = []
    for path in files:
        try:
            obs = load_ascat_from_netcdf(path, bbox, scene_time, max_time_min=max_time_min)
            observations.extend(obs)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    logger.info("Loaded %d ASCAT observations from %d files in %s", len(observations), len(files), ascat_dir)
    return observations
