import os
import sys
import logging
from dotenv import load_dotenv

# Load the env variables explicitly
load_dotenv(os.path.join("backend", ".env"))

# Set Python path so backend modules can be found
sys.path.insert(0, os.path.abspath("backend"))

from backend.ml.pipeline import process_windfield
from backend.config import CDSE_CLIENT_ID, CDSE_CLIENT_SECRET, CDSE_LIVE

logging.basicConfig(level=logging.INFO)

print("CDSE_LIVE flag:", CDSE_LIVE)
print("CDSE_CLIENT_ID set:", bool(CDSE_CLIENT_ID))

print("\n=== Tamil Nadu Test ===")
# Use a recent date since Sentinel-1 rolling archive on CDSE covers recent data easily.
res_tn = process_windfield(date_selected="2024-06-15", bbox=[77.0, 8.0, 80.0, 13.0])
print(f"Source: {res_tn['source']}")
print(f"Vectors: {len(res_tn['vectors'])}")
print(f"Image: {res_tn['image_url']}")

print("\n=== Gujarat Test ===")
res_gj = process_windfield(date_selected="2024-06-15", bbox=[68.0, 20.0, 73.0, 24.0])
print(f"Source: {res_gj['source']}")
print(f"Vectors: {len(res_gj['vectors'])}")
print(f"Image: {res_gj['image_url']}")
