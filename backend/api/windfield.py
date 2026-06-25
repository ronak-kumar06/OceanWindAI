"""
Main windfield API router — OceanWind AI Phase 2.
Endpoints:
  GET  /api/windfield          — synchronous full pipeline run
  POST /api/analyze            — submit async job
  GET  /api/status/{task_id}   — poll job status
  GET  /api/history            — list all tasks with thumbnails
  GET  /api/history/{id}       — full task detail with vectors
  GET  /api/validate/{task_id} — run/return validation metrics
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from database.config import get_db
from models.all_models import AnalysisTask, WindResult, ValidationResult
from schemas.all_schemas import WindFieldResponse, AnalysisTaskSchema, AnalysisCreate
from ml.pipeline import process_windfield

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Synchronous endpoint (used by the map page) ────────────────────────────────
@router.get("/windfield", response_model=WindFieldResponse)
def get_windfield(date: str, bbox: str, db: Session = Depends(get_db)):
    """
    Run the full 10-step pipeline synchronously and persist results.
    bbox format: "min_lon,min_lat,max_lon,max_lat"
    """
    try:
        bbox_list = [float(x) for x in bbox.split(",")]
        if len(bbox_list) != 4:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox. Use: min_lon,min_lat,max_lon,max_lat")

    # Create task record
    task = AnalysisTask(
        date_selected=date,
        bbox_str=bbox,
        status="processing",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    try:
        result = process_windfield(date, bbox_list)
        _persist_result(task, result, db)
        return WindFieldResponse(
            task_id=task.id,
            source=result["source"],
            vectors=result["vectors"],
            image_url=result.get("image_url"),
            stats=result.get("stats"),
            validation=result.get("validation"),
        )
    except Exception as exc:
        logger.exception("Pipeline failed for task %d", task.id)
        task.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(exc))


# ── Async submit ───────────────────────────────────────────────────────────────
@router.post("/analyze")
def submit_analysis(body: AnalysisCreate, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """Submit an analysis job to run in the background."""
    try:
        bbox_list = [float(x) for x in body.bbox.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox format")

    task = AnalysisTask(
        date_selected=body.date_selected,
        bbox_str=body.bbox,
        status="pending",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    bg.add_task(_run_pipeline_bg, task.id, body.date_selected, bbox_list)
    return {"task_id": task.id, "status": "pending"}


def _run_pipeline_bg(task_id: int, date: str, bbox_list: list):
    from database.config import SessionLocal
    db = SessionLocal()
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        db.close()
        return
    task.status = "processing"
    db.commit()
    try:
        result = process_windfield(date, bbox_list)
        _persist_result(task, result, db)
    except Exception as exc:
        logger.exception("BG pipeline failed for task %d: %s", task_id, exc)
        task.status = "failed"
        db.commit()
    finally:
        db.close()


def _persist_result(task: AnalysisTask, result: dict, db: Session):
    """Save pipeline result into database."""
    stats = result.get("stats", {})
    task.status      = "completed"
    task.image_url   = result.get("image_url")
    task.data_source = stats.get("data_source", "mock")
    task.mean_speed  = stats.get("mean_speed")
    task.dominant_dir= stats.get("dominant_dir")
    task.n_vectors   = stats.get("n_vectors")

    # Save up to 200 wind vectors
    for v in result.get("vectors", [])[:200]:
        db.add(WindResult(
            task_id=task.id,
            latitude=v["latitude"], longitude=v["longitude"],
            speed=v["speed"],       direction=v["direction"],
            u_component=v["u"],     v_component=v["v"],
        ))

    # Save validation results
    val = result.get("validation", {})
    for src_key, val_data in val.items():
        sm = val_data.get("speed_metrics", {})
        dm = val_data.get("direction_metrics", {})
        db.add(ValidationResult(
            task_id=task.id,
            source=val_data.get("source", src_key),
            rmse_speed=sm.get("rmse"),
            mae_speed=sm.get("mae"),
            bias_speed=sm.get("bias"),
            corr_speed=sm.get("pearson_r"),
            rmse_dir=dm.get("rmse"),
            bias_dir=dm.get("bias"),
            n_samples=sm.get("n"),
        ))

    db.commit()


# ── Status polling ─────────────────────────────────────────────────────────────
@router.get("/status/{task_id}")
def get_status(task_id: int, db: Session = Depends(get_db)):
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id":   task.id,
        "status":    task.status,
        "image_url": task.image_url,
        "stats":     {
            "mean_speed":  task.mean_speed,
            "dominant_dir": task.dominant_dir,
            "n_vectors":   task.n_vectors,
            "data_source": task.data_source,
        }
    }


# ── History ────────────────────────────────────────────────────────────────────
@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    tasks = db.query(AnalysisTask).order_by(AnalysisTask.created_at.desc()).all()
    return [
        {
            "id":           t.id,
            "date_selected": t.date_selected,
            "bbox_str":     t.bbox_str,
            "status":       t.status,
            "data_source":  t.data_source,
            "image_url":    t.image_url,
            "mean_speed":   t.mean_speed,
            "dominant_dir": t.dominant_dir,
            "n_vectors":    t.n_vectors,
            "created_at":   t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]


@router.get("/history/{task_id}")
def get_history_item(task_id: int, db: Session = Depends(get_db)):
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    vectors = [
        {"lat": r.latitude, "lon": r.longitude, "speed": r.speed,
         "direction": r.direction, "u": r.u_component, "v": r.v_component}
        for r in task.results[:200]
    ]
    validations = [
        {"source": v.source, "rmse_speed": v.rmse_speed, "bias_speed": v.bias_speed,
         "corr_speed": v.corr_speed, "rmse_dir": v.rmse_dir, "n": v.n_samples}
        for v in task.validations
    ]
    return {
        "id":           task.id,
        "date_selected": task.date_selected,
        "bbox_str":     task.bbox_str,
        "status":       task.status,
        "data_source":  task.data_source,
        "image_url":    task.image_url,
        "mean_speed":   task.mean_speed,
        "dominant_dir": task.dominant_dir,
        "n_vectors":    task.n_vectors,
        "created_at":   task.created_at.isoformat() if task.created_at else None,
        "vectors":      vectors,
        "validations":  validations,
    }


# ── Validation endpoint ────────────────────────────────────────────────────────
@router.get("/validate/{task_id}")
def validate_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    saved = db.query(ValidationResult).filter(ValidationResult.task_id == task_id).all()
    return [
        {"source": v.source, "rmse_speed": v.rmse_speed, "mae_speed": v.mae_speed,
         "bias_speed": v.bias_speed, "corr_speed": v.corr_speed,
         "rmse_dir": v.rmse_dir, "bias_dir": v.bias_dir, "n": v.n_samples}
        for v in saved
    ]


# ── Report endpoint ────────────────────────────────────────────────────────────
@router.get("/report/{task_id}")
def get_report(task_id: int, db: Session = Depends(get_db)):
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    validations = db.query(ValidationResult).filter(ValidationResult.task_id == task_id).all()

    return {
        "task_id":      task_id,
        "date":         task.date_selected,
        "aoi":          task.bbox_str,
        "data_source":  task.data_source,
        "status":       task.status,
        "image_url":    task.image_url,
        "statistics": {
            "mean_wind_speed_ms":  task.mean_speed,
            "dominant_direction_deg": task.dominant_dir,
            "total_vectors":       task.n_vectors,
        },
        "pipeline": {
            "step_1": "Sentinel-1 GRD data ingestion",
            "step_2": "Land/sea masking (Natural Earth 110m)",
            "step_3": "SAR patch tiling (64×64 px, stride=32)",
            "step_4": "ResNet50-SAR wind direction (deterministic simulator)",
            "step_5": "180° ambiguity resolution (ERA5/climatology)",
            "step_6": "CMOD5.N wind speed inversion",
            "step_7": "U/V decomposition",
            "step_8": "Grid interpolation (scipy.griddata)",
            "step_9": "Quiver plot (Matplotlib + Contextily)",
        },
        "validation": [
            {"source": v.source, "rmse_speed": v.rmse_speed, "bias_speed": v.bias_speed,
             "corr_speed": v.corr_speed, "rmse_dir": v.rmse_dir, "n": v.n_samples}
            for v in validations
        ],
    }
