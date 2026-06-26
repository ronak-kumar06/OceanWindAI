<div align="center">
  <h1>🌊 OceanWind AI</h1>
  <p><strong>High-Resolution Coastal Wind Field Estimation using Synthetic Aperture Radar (SAR)</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js" />
    <img src="https://img.shields.io/badge/PyTorch-Machine_Learning-ee4c2c.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

<hr />

## 📖 Overview

**OceanWind AI** is a state-of-the-art, full-stack platform designed to perform live, high-resolution wind field estimation over coastal and oceanic regions. 

By fusing deep learning (ResNet50) with established empirical geophysical model functions (CMOD5.N) and climate reanalysis datasets (ECMWF ERA5), OceanWind AI accurately extracts both wind speed and wind direction directly from ocean surface backscatter.

---

## ✨ Key Features

- 🛰️ **Live Data Pipeline**: Direct integration with the **Copernicus Data Space Ecosystem (CDSE)** to fetch live Sentinel-1 GRD SAR imagery.
- 🧠 **Machine Learning**: Utilizes a customized **ResNet50** architecture to predict highly localized wind directions from SAR image texture patches.
- 🌍 **Physics-Informed Ambiguity Resolution**: Resolves the inherent 180° SAR directional ambiguity by dynamically fetching and comparing against **ECMWF ERA5** background reference winds via the Copernicus CDS API.
- 💨 **Geophysical Modeling**: Calculates precise wind speeds using the **CMOD5.N** empirical model.
- ⚡ **Highly Optimized Backend**: Built for speed with parallel asynchronous network fetching, in-memory LRU caching, and vectorized Natural Earth land-sea masking via `shapely`.
- 📊 **Scientific Visualization**: Dynamically generates publication-ready quiver plots (dark topography style, wind direction vectors, and colored heatmaps) using `matplotlib`.

---

## 🛠️ Technology Stack

| Component | Technologies Used |
| :--- | :--- |
| **Backend** | Python, FastAPI, PyTorch, SciPy, NumPy, Shapely, Matplotlib |
| **Frontend** | React, Next.js, Tailwind CSS, Leaflet, Axios |
| **Data Sources** | Sentinel-1 (CDSE API), ERA5 (CDS API), Natural Earth |

---

## 🚀 Getting Started

### Prerequisites

To execute the live data pipeline, you must have active accounts for Copernicus data access:
1. **CDSE Credentials:** For Sentinel-1 SAR retrieval ([Register Here](https://dataspace.copernicus.eu/)).
2. **CDS API Key:** For ERA5 climate data retrieval ([Register Here](https://cds.climate.copernicus.eu/)).

### Environment Variables

Create a `.env` file in both the `backend/` and `frontend/` directories (if required by your frontend setup). For the backend, configure the following:

```env
# backend/.env
CDSE_CLIENT_ID=your_cdse_client_id
CDSE_CLIENT_SECRET=your_cdse_client_secret
CDS_KEY=your_cds_uuid:your_cds_api_key
```

### Installation

<details>
<summary><b>1. Backend Setup (FastAPI)</b></summary>

```bash
cd backend

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the database seeder to create default roles/users
python seed.py

# Start the development server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
</details>

<details>
<summary><b>2. Frontend Setup (Next.js)</b></summary>

```bash
cd frontend

# Install Node modules
npm install

# Start the Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to access the dashboard.
</details>

---

## 📐 System Architecture & Data Flow

1. **User Request:** The Next.js frontend sends geographic bounding box coordinates and a date string to the FastAPI backend.
2. **Parallel Fetch:** The backend simultaneously queries CDSE (for SAR `sigma0` backscatter and incidence angles) and CDS (for ERA5 10m wind references).
3. **Masking:** High-resolution Natural Earth shapefiles dynamically mask out land geometries.
4. **Tiling & Inference:** The SAR scene is tiled into 64x64 patches. A PyTorch ResNet model evaluates textures to infer the aliased wind direction.
5. **Ambiguity Resolution:** ERA5 reference data resolves the 180° ambiguity to provide the true meteorological wind direction.
6. **Speed Calculation:** The CMOD5.N physics algorithm processes the radar backscatter, incidence angle, and resolved wind direction to calculate wind speed vectors.
7. **Mapping:** Wind vectors (U/V components) are interpolated across a regular grid, mapped visually via `matplotlib`, and JSON payload + PNG are sent back to the UI.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
<div align="center">
  <i>Developed for Oceanographic SAR Analysis</i>
</div>
