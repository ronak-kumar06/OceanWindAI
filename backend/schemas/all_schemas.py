"""
Updated Pydantic schemas for OceanWind AI Phase 2.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnalysisCreate(BaseModel):
    date_selected: str
    bbox: str   # "minlon,minlat,maxlon,maxlat"


class WindVector(BaseModel):
    latitude:  float
    longitude: float
    speed:     float
    direction: float
    u:         float
    v:         float

    class Config:
        from_attributes = True


class ValidationMetrics(BaseModel):
    rmse:          Optional[float] = None
    mae:           Optional[float] = None
    bias:          Optional[float] = None
    pearson_r:     Optional[float] = None
    scatter_index: Optional[float] = None
    n:             Optional[int]   = None


class ValidationResultSchema(BaseModel):
    source:           str
    speed_metrics:    ValidationMetrics
    direction_metrics: ValidationMetrics


class WindFieldStats(BaseModel):
    mean_speed:   Optional[float] = None
    max_speed:    Optional[float] = None
    min_speed:    Optional[float] = None
    dominant_dir: Optional[float] = None
    n_vectors:    Optional[int]   = None
    n_patches:    Optional[int]   = None
    data_source:  Optional[str]   = None


class WindFieldResponse(BaseModel):
    task_id:    Optional[int]         = None
    source:     str
    vectors:    List[Dict[str, Any]]
    image_url:  Optional[str]         = None
    stats:      Optional[Dict]        = None
    validation: Optional[Dict]        = None


class AnalysisTaskSchema(BaseModel):
    id:           int
    date_selected: str
    bbox_str:     Optional[str]   = None
    status:       str
    data_source:  Optional[str]   = None
    image_url:    Optional[str]   = None
    mean_speed:   Optional[float] = None
    dominant_dir: Optional[float] = None
    n_vectors:    Optional[int]   = None
    created_at:   datetime

    class Config:
        from_attributes = True
