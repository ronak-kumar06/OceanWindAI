"""
OceanWind AI — FastAPI entry point (Phase 2).
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database.config import engine, Base, SessionLocal
from models.all_models import AnalysisTask, WindResult, ValidationResult  # register all tables
from api import windfield
from config import DATABASE_URL, STATIC_DIR, CORS_ORIGINS

logging.basicConfig(level=logging.INFO)

# SQLite dev convenience — production PostgreSQL uses Alembic migrations
if DATABASE_URL.startswith("sqlite"):
    Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="OceanWind AI API",
    description="SAR-based offshore wind field estimation — Indian coastal waters",
    version="2.0.0",
)

_origins = CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(windfield.router, prefix="/api", tags=["Windfield"])

os.makedirs(str(STATIC_DIR), exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return {
        "service": "OceanWind AI API",
        "version": "2.0.0",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
