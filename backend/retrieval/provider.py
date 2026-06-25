"""
SentinelHub provider — real Sentinel-1 GRD data retrieval.

Uses the sentinelhub-py library with OAuth2 client credentials.
Falls back gracefully to MockSentinelProvider when credentials are absent
or when the API call fails (network timeout, quota exceeded, etc.).
"""
import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import CDSE_CLIENT_ID, CDSE_CLIENT_SECRET, CDSE_LIVE

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Abstract interface — any future provider (e.g., Google Earth Engine, ASF)
# only needs to implement fetch_data().
# ══════════════════════════════════════════════════════════════════════════════
class DataProvider(ABC):
    @abstractmethod
    def fetch_data(self, date: str, bbox: list) -> Dict[str, Any]:
        """
        Returns a dict with at minimum:
          source    : str  — human-readable source label
          sigma0    : np.ndarray (H, W) — backscatter in linear scale, or None
          inc_angle : np.ndarray (H, W) — local incidence angle in degrees, or None
          orbit_dir : str  — "ASCENDING" | "DESCENDING"
          date      : str
          bbox      : list [min_lon, min_lat, max_lon, max_lat]
        """
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Mock provider — deterministic, reproducible, no network needed
# ══════════════════════════════════════════════════════════════════════════════
class MockSentinelProvider(DataProvider):
    """
    Generates a synthetic Sentinel-1-like sigma0 field.
    Uses a fixed random seed derived from (date, bbox) for reproducibility —
    the same query always returns the same synthetic scene.
    """

    def fetch_data(self, date: str, bbox: list) -> Dict[str, Any]:
        # Seed ensures determinism per (date, region)
        seed = hash(f"{date}:{bbox[0]:.1f}:{bbox[1]:.1f}:{bbox[2]:.1f}:{bbox[3]:.1f}") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        H, W = 256, 256  # synthetic SAR scene size
        # Realistic sigma0 range for ocean in VV polarisation: ~0.001 – 0.1 (linear)
        base = rng.uniform(0.005, 0.05, size=(H, W)).astype(np.float32)
        # Add smooth spatial structure (simulate real SAR texture)
        for scale in [64, 32, 16]:
            coarse = rng.uniform(-0.01, 0.01, size=(H // scale + 1, W // scale + 1))
            from scipy.ndimage import zoom
            fine = zoom(coarse, scale, order=1)[:H, :W]
            base += fine.astype(np.float32)
        base = np.clip(base, 1e-5, None)

        # Synthetic incidence angle: linearly increasing from near-range to far-range
        inc_angle = np.tile(
            np.linspace(30.0, 45.0, W, dtype=np.float32), (H, 1)
        )

        logger.info("MockSentinelProvider: generated synthetic scene (seed=%d)", seed)
        return {
            "source":    "Simulated Sentinel-1 GRD (Mock)",
            "sigma0":    base,
            "inc_angle": inc_angle,
            "orbit_dir": "DESCENDING",
            "date":      date,
            "bbox":      bbox,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Real SentinelHub provider — OAuth2 with automatic mock fallback
# ══════════════════════════════════════════════════════════════════════════════
class SentinelHubProvider(DataProvider):
    """
    Downloads Sentinel-1 GRD sigma0 (VV) via the Copernicus Data Space Ecosystem Process API.
    Requires CDSE_CLIENT_ID and CDSE_CLIENT_SECRET env vars.
    """

    def __init__(self):
        try:
            from sentinelhub import (
                SHConfig, BBox, CRS, DataCollection, MimeType,
                SentinelHubRequest, bbox_to_dimensions
            )
            self._sh_imports = {
                "SHConfig": SHConfig, "BBox": BBox, "CRS": CRS,
                "DataCollection": DataCollection, "MimeType": MimeType,
                "SentinelHubRequest": SentinelHubRequest,
                "bbox_to_dimensions": bbox_to_dimensions,
            }
            self.config = SHConfig()
            self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
            self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            self.config.sh_client_id     = CDSE_CLIENT_ID
            self.config.sh_client_secret = CDSE_CLIENT_SECRET
        except ImportError:
            raise RuntimeError("sentinelhub package not installed. Run: pip install sentinelhub")

    def fetch_data(self, date: str, bbox: list) -> Dict[str, Any]:
        from sentinelhub import BBox, CRS, DataCollection, MimeType, SentinelHubRequest, bbox_to_dimensions
        import datetime

        sh     = self._sh_imports
        config = self.config

        sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
        size    = bbox_to_dimensions(sh_bbox, resolution=100)   # 100 m/px
        # Cap size to avoid 400 Bad Request (API limit is 2500x2500) and keep memory reasonable
        size    = (min(size[0], 1024), min(size[1], 1024))

        # ± 3-day window around the requested date
        dt      = datetime.date.fromisoformat(date)
        t_from  = (dt - datetime.timedelta(days=3)).isoformat()
        t_to    = (dt + datetime.timedelta(days=3)).isoformat()

        # Evalscript: return VV sigma0, localIncidenceAngle, and dataMask
        evalscript = """
        //VERSION=3
        function setup() {
            return { 
                input: ["VV", "localIncidenceAngle", "dataMask"], 
                output: { bands: 3, sampleType: "FLOAT32" } 
            };
        }
        function evaluatePixel(s) {
            return [s.VV, s.localIncidenceAngle, s.dataMask];
        }
        """

        CDSE_S1_IW = DataCollection.define(
            "sentinel-1-grd",
            api_id="sentinel-1-grd",
            service_url=config.sh_base_url,
            collection_type="SENTINEL1"
        )

        try:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=CDSE_S1_IW,
                        time_interval=(t_from, t_to),
                        other_args={
                            "dataFilter": {"acquisitionMode": "IW", "polarization": "DV"},
                            "processing": {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID"}
                        },
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=sh_bbox,
                size=size,
                config=config,
            )
            data = request.get_data()[0]   # shape: (H, W, 3)
            
            sigma0    = data[:, :, 0].astype(np.float32)
            inc_angle = data[:, :, 1].astype(np.float32)
            mask      = data[:, :, 2].astype(bool)
            
            sigma0[~mask] = np.nan
            inc_angle[~mask] = np.nan

            H, W  = sigma0.shape

            valid_inc = inc_angle[np.isfinite(inc_angle)]
            if valid_inc.size > 0:
                logger.info("SentinelHubProvider: downloaded scene %s x %s px. Real Incidence Angle Stats: min=%.1f°, max=%.1f°, mean=%.1f°", 
                            H, W, float(np.min(valid_inc)), float(np.max(valid_inc)), float(np.mean(valid_inc)))
            else:
                logger.info("SentinelHubProvider: downloaded scene %s x %s px. (No valid pixels)", H, W)
                
            return {
                "source":    "Sentinel-1 IW GRD (Live via CDSE)",
                "sigma0":    sigma0,
                "inc_angle": inc_angle,
                "orbit_dir": "DESCENDING",
                "date":      date,
                "bbox":      bbox,
            }
        except Exception as exc:
            logger.warning("SentinelHub download failed (%s) — falling back to mock", exc)
            return MockSentinelProvider().fetch_data(date, bbox)


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════
def get_data_provider() -> DataProvider:
    if CDSE_LIVE:
        try:
            return SentinelHubProvider()
        except Exception as exc:
            logger.warning("Cannot initialise SentinelHubProvider (%s) — using mock", exc)
    return MockSentinelProvider()
