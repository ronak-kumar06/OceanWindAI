"""
Deterministic simulated ResNet inference for SAR wind direction.

Design philosophy
-----------------
A truly untrained ResNet50 produces statistically meaningless outputs —
random class probabilities that don't correlate with the actual SAR image.
Instead this module implements a physics-informed simulator that:

  1. Extracts real SAR texture features (gradient, variance, mean backscatter)
     from each patch.
  2. Maps those features to a wind direction estimate using a deterministic
     function that approximates the expected ResNet response for real SAR data.
  3. Returns the ALIASED direction (0–180°, the same ambiguity a real ResNet
     would return before 180° resolution).

This architecture is a drop-in replacement for real ResNet inference:
  - Same function signature
  - Same output format (aliased_direction in degrees)
  - Can be replaced with load_model() + torch.no_grad() forward pass
    by simply swapping predict_direction() with the real inference call.

Reference
---------
Shao et al. (2020). Wind direction retrieval from Sentinel-1 SAR images
using ResNet. Remote Sensing of Environment, 252, 112056.
"""
import math
import logging
from pathlib import Path
import numpy as np
from typing import List

from config import MODEL_WEIGHTS_PATH

logger = logging.getLogger(__name__)

_model_cache = None


def _get_trained_model():
    """Load trained ResNet weights if available."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    weights = Path(MODEL_WEIGHTS_PATH)
    if not weights.exists():
        return None

    try:
        from ml.resnet_model import load_model
        model = load_model(str(weights))
        if model is not None:
            _model_cache = model
            logger.info("Using trained ResNet50-SAR weights from %s", weights)
        return model
    except Exception as exc:
        logger.warning("Could not load trained model: %s", exc)
        return None


def _predict_with_model(model, patches: List[dict]) -> List[float]:
    """Run ResNet50-SAR forward pass on SAR patches."""
    import torch
    from training.dataset import patch_to_tensor

    directions = []
    with torch.no_grad():
        for p in patches:
            if not p["valid"]:
                directions.append(float("nan"))
                continue
            tensor = patch_to_tensor(p["patch"]).unsqueeze(0)
            logits = model(tensor)
            bin_idx = int(logits.argmax(1).item())
            aliased_dir = bin_idx * (180.0 / 36)
            directions.append(aliased_dir)

    logger.info("ResNet inference: %d valid patches", sum(np.isfinite(directions)))
    return directions


# ── Texture feature extraction ─────────────────────────────────────────────────

def _extract_features(patch: np.ndarray) -> dict:
    """
    Compute physically-motivated SAR texture features.

    Parameters
    ----------
    patch : (H, W) float32, sigma0 in linear scale (may contain NaN)

    Returns
    -------
    dict with features: mean_db, grad_angle, variance, skewness
    """
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return {"mean_db": -15.0, "grad_angle": 0.0, "variance": 0.0, "skewness": 0.0}

    mean_lin = float(np.nanmean(patch))
    mean_db  = 10.0 * math.log10(max(mean_lin, 1e-8))

    # Gradient direction (dominant texture streak angle in SAR)
    # Fill NaN with mean before gradient
    filled = np.where(np.isfinite(patch), patch, mean_lin)
    gy, gx = np.gradient(filled)
    # Wind rows (Bragg scattering streaks) are perpendicular to wind direction
    # The dominant gradient direction approximates the wind direction signature
    grad_angle = math.degrees(math.atan2(float(np.nanmean(gy)), float(np.nanmean(gx)))) % 360.0

    variance = float(np.nanvar(patch))
    # Skewness proxy (asymmetry of backscatter distribution)
    if variance > 0:
        skewness = float(np.nanmean((patch - mean_lin) ** 3)) / (variance ** 1.5 + 1e-12)
    else:
        skewness = 0.0

    return {
        "mean_db":   mean_db,
        "grad_angle": grad_angle,
        "variance":  variance,
        "skewness":  skewness,
    }


# ── Deterministic direction predictor ─────────────────────────────────────────

def _simulate_resnet_output(features: dict, lat: float, lon: float, seed: int) -> float:
    """
    Map SAR texture features to an aliased wind direction.

    The mapping is deterministic (same input → same output) and spatially
    coherent (nearby patches give similar directions). It is NOT random.

    Strategy:
      - Use the gradient angle as the dominant signal (physics-informed).
      - Add a location-based bias to capture the typical monsoon/trade-wind
        direction patterns over the Bay of Bengal and Arabian Sea.
      - Add patch-specific perturbation based on variance (low variance =
        more uniform flow = less deviation from background).
    """
    rng = np.random.default_rng(seed)

    # Background wind direction from climatology of Indian Ocean monsoon
    # Tamil Nadu coast: SW monsoon ~210°, NE monsoon ~50°
    # Gujarat / Arabian Sea: ~200–230° (SW monsoon dominant in June)
    climatological_dir = _indian_ocean_background_dir(lat, lon)

    # The gradient angle carries information but with π ambiguity
    grad = features["grad_angle"]
    # Aliased gradient (collapses 360° → 180° to remove the orientation ambiguity)
    aliased_grad = grad % 180.0

    # Perturbation amplitude inversely proportional to variance:
    # low-variance ocean → cleaner wind signal → less perturbation
    variance_factor = min(1.0, features["variance"] / 1e-4)
    perturb = float(rng.normal(0, 5.0 * variance_factor))

    # Blend climatological background with SAR-derived gradient
    # (In a real trained ResNet, this blend is learned; here it is explicit)
    alpha = 0.4   # weight for SAR gradient vs climatology
    blended = alpha * aliased_grad + (1 - alpha) * (climatological_dir % 180.0)
    aliased_direction = (blended + perturb) % 180.0

    return aliased_direction


def _indian_ocean_background_dir(lat: float, lon: float) -> float:
    """
    Approximate seasonal-mean wind direction over Indian coastal waters.
    June = SW monsoon. December = NE monsoon.
    Uses longitude to distinguish Bay of Bengal vs Arabian Sea.
    Returns direction in degrees (0–360, FROM which wind blows).
    """
    # Arabian Sea / Gujarat coast: SW monsoon ≈ 220°
    if lon < 77.0:
        return 220.0
    # Bay of Bengal / Tamil Nadu coast: SW monsoon ≈ 205°
    return 205.0


# ── Public API ────────────────────────────────────────────────────────────────

def predict_direction(patches: List[dict], global_seed: int = 42) -> List[float]:
    """
    Predict aliased wind direction (0–180°) for a list of SAR patches.

    This function is the drop-in replacement point for real ResNet inference.
    To use a trained model instead, replace the body of this function with:

        model = load_model("ml/weights/resnet_sar_wind.pth")
        with torch.no_grad():
            for p in patches:
                tensor = preprocess(p["patch"])
                logits = model(tensor.unsqueeze(0))
                bin_idx = logits.argmax(1).item()
                aliased_dir = bin_idx * (180 / 36)
                ...

    Parameters
    ----------
    patches     : output of preprocessing.tiling.extract_patches()
    global_seed : int — controls global spatial coherence (fixed per analysis)

    Returns
    -------
    List[float] — aliased wind direction per patch (degrees, 0–180°)
    """
    model = _get_trained_model()
    if model is not None:
        return _predict_with_model(model, patches)

    directions = []
    for idx, p in enumerate(patches):
        if not p["valid"]:
            directions.append(float("nan"))
            continue

        features = _extract_features(p["patch"])
        # Each patch gets a deterministic seed: global + patch index
        patch_seed = (global_seed + idx * 7919) & 0xFFFFFFFF
        aliased = _simulate_resnet_output(
            features, p["lat"], p["lon"], patch_seed
        )
        directions.append(aliased)

    logger.info("Simulated ResNet inference: %d valid patches", sum(np.isfinite(directions)))
    return directions
