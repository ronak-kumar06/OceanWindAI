"""
Updated database ORM models for OceanWind AI Phase 2.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database.config import Base


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id           = Column(Integer, primary_key=True, index=True)
    date_selected= Column(String,  nullable=False)
    bbox_str     = Column(String,  nullable=True)       # "minlon,minlat,maxlon,maxlat"
    aoi_geojson  = Column(String,  nullable=True)
    status       = Column(String,  default="pending")   # pending|processing|completed|failed
    data_source  = Column(String,  default="mock")      # "mock" | "sentinel1"
    image_url    = Column(String,  nullable=True)
    mean_speed   = Column(Float,   nullable=True)
    dominant_dir = Column(Float,   nullable=True)
    n_vectors    = Column(Integer, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    results      = relationship("WindResult",      back_populates="task", cascade="all, delete-orphan")
    validations  = relationship("ValidationResult", back_populates="task", cascade="all, delete-orphan")


class WindResult(Base):
    __tablename__ = "wind_results"

    id           = Column(Integer, primary_key=True, index=True)
    task_id      = Column(Integer, ForeignKey("analysis_tasks.id"))
    latitude     = Column(Float, nullable=False)
    longitude    = Column(Float, nullable=False)
    speed        = Column(Float, nullable=False)
    direction    = Column(Float, nullable=False)
    u_component  = Column(Float, nullable=False)
    v_component  = Column(Float, nullable=False)

    task         = relationship("AnalysisTask", back_populates="results")


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id           = Column(Integer, primary_key=True, index=True)
    task_id      = Column(Integer, ForeignKey("analysis_tasks.id"))
    source       = Column(String, nullable=False)    # "ecmwf" | "ascat" | "buoy"
    rmse_speed   = Column(Float, nullable=True)
    mae_speed    = Column(Float, nullable=True)
    bias_speed   = Column(Float, nullable=True)
    corr_speed   = Column(Float, nullable=True)
    rmse_dir     = Column(Float, nullable=True)
    bias_dir     = Column(Float, nullable=True)
    n_samples    = Column(Integer, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    task         = relationship("AnalysisTask", back_populates="validations")
