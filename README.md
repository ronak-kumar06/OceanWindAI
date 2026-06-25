# OceanWind AI 🌊💨

**OceanWind AI** is a full-stack platform that performs live, high-resolution coastal wind field estimation using Synthetic Aperture Radar (SAR) imagery. 

By integrating deep learning (ResNet) with empirical geophysical model functions (CMOD5.N) and climate reanalysis data (ERA5), OceanWind AI accurately maps both wind speed and wind direction directly from ocean surface backscatter.

## 🚀 Features

* **Live Data Pipeline:** Directly pulls Sentinel-1 GRD SAR imagery from the Copernicus Data Space Ecosystem (CDSE).
* **Machine Learning & Physics:**
  * Uses a **ResNet50** architecture to predict wind direction from SAR texture patches.
  * Resolves 180° directional ambiguity using **ECMWF ERA5** background reference winds (retrieved in real-time via the Copernicus CDS API).
  * Calculates precise wind speeds using the **CMOD5.N** empirical model.
* **Highly Optimized:** Features a highly optimized backend pipeline with parallel async network fetching, in-memory LRU caching for ERA5 data, and vectorized land-sea masking.
* **Scientific Visualization:** Dynamically generates publication-ready quiver plots (topography maps with wind direction vectors and wind speed colored heatmaps).
* **Modern Tech Stack:**
  * **Backend:** FastAPI, Python, NumPy, SciPy, Shapely, PyTorch.
  * **Frontend:** Next.js, React, TailwindCSS, Leaflet mapping.

## ⚙️ Prerequisites

To run the live data pipeline, you must have active accounts for Copernicus data access:
1. **CDSE Credentials:** For Sentinel-1 SAR retrieval ([Dataspace](https://dataspace.copernicus.eu/)).
2. **CDS API Key:** For ERA5 climate data retrieval ([Climate Data Store](https://cds.climate.copernicus.eu/)).

## 🛠️ Installation & Setup

### 1. Backend Setup (FastAPI)

```bash
cd backend

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# Create your .env file
echo "CDSE_CLIENT_ID=your_id_here" >> .env
echo "CDSE_CLIENT_SECRET=your_secret_here" >> .env
echo "CDS_KEY=your_uuid:your_api_key_here" >> .env

# Run the database seeder (creates default user)
python seed.py

# Start the backend server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend Setup (Next.js)

```bash
cd frontend

# Install Node modules
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to access the dashboard.

## 🗺️ System Architecture

1. **User Request:** Frontend sends a bounding box and date to the FastAPI backend.
2. **Parallel Fetch:** Backend simultaneously queries CDSE (for SAR `sigma0` and incidence angles) and CDS (for ERA5 10m wind references).
3. **Masking:** High-resolution Natural Earth shapefiles mask out land geometries.
4. **Tiling & Inference:** The SAR scene is tiled into 64x64 patches. A ResNet model infers the aliased wind direction.
5. **Ambiguity Resolution:** ERA5 reference data resolves the 180° ambiguity to provide the true meteorological wind direction.
6. **Speed Calculation:** The CMOD5.N algorithm processes the radar backscatter, incidence angle, and resolved wind direction to calculate wind speed.
7. **Mapping:** Wind vectors (U/V) are interpolated across a regular grid, mapped via `matplotlib`, and sent back to the Next.js UI.
