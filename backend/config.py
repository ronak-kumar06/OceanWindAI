"""
Central configuration for OceanWind AI backend.
All secrets and tuneable parameters live here.
Set values via a .env file or OS environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (development convenience)
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Copernicus Data Space Ecosystem Credentials ───────────────────────────────
CDSE_CLIENT_ID     = os.getenv("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "")
CDSE_LIVE          = bool(CDSE_CLIENT_ID and CDSE_CLIENT_SECRET)

# ── ECMWF / CDS API (for ERA5 validation) ─────────────────────────────────────
CDS_URL  = os.getenv("CDS_URL",  "https://cds.climate.copernicus.eu/api")
CDS_KEY  = os.getenv("CDS_KEY",  "")   # format: "<uid>:<api-key>" or new CADS UUID
CDS_LIVE = bool(CDS_KEY)

# ── ML Pipeline parameters ────────────────────────────────────────────────────
PATCH_SIZE    = 64          # SAR patch size (pixels)
PATCH_STRIDE  = 32          # overlap stride between patches
GRID_STEPS    = 30          # number of grid points in each direction for output field
DIR_BINS      = 36          # ResNet output classes (10° each)
WIND_SPEED_MIN = 2.0        # m/s colorbar minimum
WIND_SPEED_MAX = 12.0       # m/s colorbar maximum

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'oceanwind.db'}")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")
