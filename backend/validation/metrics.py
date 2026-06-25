"""
Validation metrics for OceanWind AI.
Compares estimated wind field against reference data (ECMWF, ASCAT, buoy).
"""
import numpy as np
from typing import Dict


def compute_metrics(estimated: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """
    Compute standard wind field validation metrics.

    Parameters
    ----------
    estimated  : array of estimated values (speed or direction)
    reference  : array of reference (truth) values — same shape

    Returns
    -------
    dict with: rmse, mae, bias, pearson_r, scatter_index
    """
    mask = np.isfinite(estimated) & np.isfinite(reference)
    if mask.sum() < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"),
                "pearson_r": float("nan"), "scatter_index": float("nan"), "n": 0}

    est = estimated[mask]
    ref = reference[mask]

    diff   = est - ref
    bias   = float(np.mean(diff))
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    mae    = float(np.mean(np.abs(diff)))

    corr_matrix = np.corrcoef(est, ref)
    pearson_r   = float(corr_matrix[0, 1])

    ref_mean = float(np.mean(ref))
    si = rmse / (ref_mean + 1e-10)

    return {
        "rmse":          round(rmse, 3),
        "mae":           round(mae, 3),
        "bias":          round(bias, 3),
        "pearson_r":     round(pearson_r, 3),
        "scatter_index": round(si, 3),
        "n":             int(mask.sum()),
    }


def simulate_ecmwf_validation(
    speeds: np.ndarray,
    directions: np.ndarray,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Simulate ECMWF ERA5 validation when CDS API is not available.

    Generates synthetic ERA5-like reference data by adding a small,
    physically realistic offset to the estimated wind field.
    This represents what validation against ERA5 would look like
    for typical CMOD5.N retrieval accuracy.
    """
    rng = np.random.default_rng(seed)

    # ERA5 typically differs from SAR retrieval by ±1–2 m/s (RMSE ~ 1.5 m/s)
    ref_speed = speeds + rng.normal(0.2, 1.2, size=speeds.shape)
    ref_speed = np.clip(ref_speed, 0.5, 25.0)

    # Direction uncertainty: ±15–20° for ERA5 vs SAR
    ref_dir = directions + rng.normal(0, 18.0, size=directions.shape)
    ref_dir = ref_dir % 360.0

    speed_metrics = compute_metrics(speeds, ref_speed)
    dir_metrics   = compute_metrics(directions, ref_dir)

    return {
        "source":           "ECMWF ERA5 (simulated reference)",
        "speed_metrics":    speed_metrics,
        "direction_metrics": dir_metrics,
    }


def simulate_ascat_validation(
    speeds: np.ndarray,
    directions: np.ndarray,
    seed: int = 137,
) -> Dict[str, Dict[str, float]]:
    """
    Simulate ASCAT scatterometer validation.
    ASCAT has slightly higher accuracy than ERA5 (RMSE ~ 1.2 m/s).
    """
    rng = np.random.default_rng(seed)

    ref_speed = speeds + rng.normal(0.1, 1.0, size=speeds.shape)
    ref_speed = np.clip(ref_speed, 0.5, 25.0)

    ref_dir = directions + rng.normal(0, 14.0, size=directions.shape)
    ref_dir = ref_dir % 360.0

    speed_metrics = compute_metrics(speeds, ref_speed)
    dir_metrics   = compute_metrics(directions, ref_dir)

    return {
        "source":           "ASCAT Scatterometer (simulated co-location)",
        "speed_metrics":    speed_metrics,
        "direction_metrics": dir_metrics,
    }
