<div align="center">
  <h1>🌊 OceanWind AI</h1>
  <p><strong>High-Resolution Coastal Wind Field Estimation using Synthetic Aperture Radar (SAR)</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688.svg" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js" />
    <img src="https://img.shields.io/badge/PyTorch-ResNet50-ee4c2c.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker" />
    <img src="https://img.shields.io/badge/Deploy-Render_%2B_Vercel-blueviolet.svg" alt="Deployment" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

<hr />

## 📖 Overview

**OceanWind AI** is a state-of-the-art, production-ready full-stack platform for live, high-resolution coastal wind field estimation directly from Synthetic Aperture Radar (SAR) imagery.

By fusing **Deep Learning (ResNet50)** with the empirical **CMOD5.N Geophysical Model Function** and **ECMWF ERA5** climate reanalysis data, OceanWind AI retrieves both wind speed and wind direction at sub-kilometer resolutions — far exceeding the spatial resolution of traditional scatterometry.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🛰️ **Live SAR Pipeline** | Directly queries the Copernicus Data Space Ecosystem (CDSE) for real Sentinel-1 GRD IW imagery |
| 🧠 **ResNet50 AI Model** | Custom single-channel ResNet50 predicts wind direction from SAR texture patches |
| 🌍 **ERA5 Ambiguity Resolution** | Fetches live ERA5 10m winds to resolve the inherent 180° SAR directional ambiguity |
| 💨 **CMOD5.N Physics** | Implements the Newton-Raphson iterative CMOD5.N solver for precise wind speed inversion |
| 🏋️ **Full Training Pipeline** | ASCAT NetCDF reader, SAR-ASCAT collocation engine, and PyTorch GPU training loop included |
| ⚡ **Parallel & Cached** | SAR + ERA5 fetches run in parallel; land-sea masks and ERA5 refs are LRU cached |
| 🗺️ **Scientific Visualization** | Dynamic pcolormesh heatmap + black quiver arrows for publication-ready wind field plots |
| 🐳 **Docker + PostgreSQL** | Fully containerized with Alembic database migrations for production |
| 🚀 **Render + Vercel Deploy** | One-click deployment blueprint for Render (backend) and Vercel (frontend) |

---

## 🛠️ Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, PyTorch, NumPy, SciPy, Shapely, Matplotlib |
| **ML / Physics** | ResNet50, CMOD5.N GMF, Newton-Raphson solver, ERA5 ambiguity resolution |
| **Training** | PyTorch DataLoader, ASCAT NetCDF reader, SAR-ASCAT collocation engine |
| **Frontend** | Next.js 14, React, TailwindCSS, Leaflet, Axios |
| **Database** | SQLite (dev) / PostgreSQL + Alembic (production) |
| **Infrastructure** | Docker, docker-compose, Render, Vercel, GitHub Actions CI/CD |
| **Data Sources** | Sentinel-1 CDSE API, ERA5 CDS API, ASCAT OSI SAF, Natural Earth |

---

## 🗺️ System Architecture

```
User (Browser)
     │
     ▼
Next.js Frontend (Vercel)
  ├── Interactive Leaflet Map + Draw Toolbar
  ├── AnalysisPanel (date picker, stats, history)
  └── WindMap (image overlay + bounding box)
     │
     ▼ REST API
FastAPI Backend (Render / Docker)
  ├── Step 1+5a ── PARALLEL FETCH ──────────────────────────────┐
  │     ├── CDSE API → Sentinel-1 GRD (sigma0 + incAngle)       │
  │     └── CDS API  → ERA5 10m U/V winds (cached)              │
  ├── Step 2 ── Land/Sea Masking (Shapely + Natural Earth, cached)
  ├── Step 3 ── SAR Patch Tiling (64×64 px, stride=32)
  ├── Step 4 ── ResNet50 Wind Direction Inference
  ├── Step 5b ── 180° Ambiguity Resolution (ERA5 reference)
  ├── Step 6 ── CMOD5.N Wind Speed Inversion
  ├── Step 7 ── U/V Vector Decomposition
  ├── Step 8 ── Grid Interpolation (scipy.griddata)
  └── Step 9 ── Quiver Plot (pcolormesh + black arrows)
     │
     ▼
JSON payload (wind vectors) + PNG (quiver plot)
```

---

## 🚀 Getting Started

### Prerequisites

Active accounts required for live data access:
1. **CDSE Credentials** — Sentinel-1 SAR imagery ([Register](https://dataspace.copernicus.eu/))
2. **CDS API Key** — ERA5 climate reanalysis ([Register](https://cds.climate.copernicus.eu/))

### Environment Variables

Create `backend/.env` from the example:

```bash
cp backend/.env.example backend/.env
```

```env
# backend/.env
CDSE_CLIENT_ID=your_cdse_client_id
CDSE_CLIENT_SECRET=your_cdse_client_secret
CDS_KEY=your_cds_uuid:your_cds_api_key
DATABASE_URL=sqlite:///./oceanwind.db       # or postgresql://... for production
API_BASE_URL=http://127.0.0.1:8000          # or your Render URL in production
```

### Installation

<details>
<summary><b>🐍 Option A — Local Development (Python + Node)</b></summary>

```bash
# ── Backend ──────────────────────────────────────
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python seed.py                 # create default DB tables
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# ── Frontend (new terminal) ────────────────────
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)
</details>

<details>
<summary><b>🐳 Option B — Docker Compose (Recommended for Production)</b></summary>

```bash
cp backend/.env.example backend/.env   # fill in your credentials
docker-compose up --build
```

This spins up the FastAPI backend + PostgreSQL database together.
</details>

---

## 🏋️ ResNet Training Pipeline

The full GPU training pipeline is located in `backend/training/`. Once you have SAR + ASCAT data, train the model with:

```bash
# 1. Read and index your ASCAT NetCDF files
python -m training.ascat_reader --data-dir data/ascat/

# 2. Collocate SAR patches with ASCAT wind labels
python -m training.collocation --manifest data/training/manifest.csv

# 3. Train the ResNet50 model on GPU
python -m training.train \
    --manifest data/training/manifest.csv \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --output ml/weights/resnet_sar_wind.pth
```

Once training completes, drop `resnet_sar_wind.pth` into `backend/ml/weights/` — the pipeline will automatically load the real weights and replace the simulator.

---

## ☁️ Deployment

| Platform | Service | Steps |
| :--- | :--- | :--- |
| **Render** | Backend API + PostgreSQL | Connect GitHub → select `render.yaml` → set env vars |
| **Vercel** | Next.js Frontend | Connect GitHub → set `NEXT_PUBLIC_API_URL` env var |

A GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically deploys both on every push to `main`.

---

## 📚 References

1. **Shao, W., et al. (2020).** Wind direction retrieval from Sentinel-1 SAR images using ResNet. *Remote Sensing of Environment*, 236. [DOI](https://doi.org/10.1016/j.rse.2019.111513)
2. **Hersbach, H. (2010).** Comparison of C-Band scatterometer CMOD5.N equivalent neutral winds with ECMWF. *Journal of Atmospheric and Oceanic Technology*, 27(4).
3. **Koch, W. (2004).** Directional analysis of SAR images aiming at wind direction. *IEEE TGRS*, 42(4).
4. **ESA SNAP Toolbox.** Wind Field Estimation. [SNAP Help](https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/)

---

## 🔗 Repository

**GitHub:** [https://github.com/ronak-kumar06/OceanWindAI](https://github.com/ronak-kumar06/OceanWindAI)

---
<div align="center">
  <i>Built for Oceanographic SAR Wind Field Research</i>
</div>
