import os
import sys

# Temporarily mock the env vars to avoid failing if not set
# (We will just let the fallback happen if they aren't set properly)
# But actually, the config has already loaded. Let's just run it.

from backend.ml.pipeline import process_windfield
import logging

logging.basicConfig(level=logging.INFO)

print("=== Tamil Nadu Test ===")
res_tn = process_windfield(date_selected="2024-06-15", bbox=[77.0, 8.0, 80.0, 13.0])
print(f"Source: {res_tn['source']}")
print(f"Vectors: {len(res_tn['vectors'])}")

print("\n=== Gujarat Test ===")
res_gj = process_windfield(date_selected="2024-06-15", bbox=[68.0, 20.0, 73.0, 24.0])
print(f"Source: {res_gj['source']}")
print(f"Vectors: {len(res_gj['vectors'])}")
