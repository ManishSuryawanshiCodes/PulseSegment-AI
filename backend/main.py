import os
import io
import uuid
import logging
import threading
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from dataset_manager import DatasetManager, validate_identifier
from job_manager import job_manager, validate_job_id
from models import (
    DatasetInfo,
    DatasetProfile,
    DatasetConfig,
    ConfigResponse,
    JobStatusResponse,
    DatasetSummary,
    FeatureEngineeringResult,
    ExperimentsMatrix,
    GrowthPersonas,
    UploadResponse,
    PredictSegmentRequest,
    PredictSegmentResponse,
    ModelJSONExport,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pulsesegment")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="PulseSegment AI",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# GZip middleware for high-compression payload transmission
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security: CORS restricted to origins defined in ALLOWED_ORIGINS env var
allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Security-Warning", "Content-Disposition"],
)

# Security: Optional API_TOKEN check
API_TOKEN = os.getenv("API_TOKEN")


@app.middleware("http")
async def check_api_token_middleware(request: Request, call_next):
    if API_TOKEN and request.method != "OPTIONS":
        path = request.url.path
        # Exclude documentation / health
        if not (path.startswith("/docs") or path.startswith("/openapi.json") or path == "/api/health"):
            token = request.headers.get("X-API-Key") or request.headers.get("Authorization")
            if not token:
                return ORJSONResponse(
                    status_code=401,
                    content={"detail": "Missing API access token. Provide 'X-API-Key' or 'Authorization' header."},
                )
            cleaned = token.replace("Bearer ", "").strip()
            if cleaned != API_TOKEN:
                return ORJSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API access token."},
                )
    return await call_next(request)


# Security: Global exception handler preventing stack trace leak
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return ORJSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if isinstance(exc, RateLimitExceeded):
        return ORJSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please slow down."})

    error_id = str(uuid.uuid4())
    logger.exception(f"Unhandled server error [Error ID: {error_id}] on {request.method} {request.url.path}: {exc}")
    return ORJSONResponse(
        status_code=500,
        content={
            "detail": f"An internal server error occurred. Reference Error ID: {error_id}",
            "error_id": error_id,
        },
    )


# Initialize dataset manager with data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
dataset_manager = DatasetManager(data_dir=DATA_DIR)


@app.get("/api/health")
def health():
    datasets = dataset_manager.list_datasets()
    total_records = sum(d.row_count for d in datasets)
    active_dataset = dataset_manager.active_dataset_id
    is_synth = dataset_manager._datasets.get(active_dataset, {}).get("is_synthetic", False)
    return {
        "status": "ok",
        "datasets_available": len(datasets),
        "total_records": total_records,
        "active_dataset": active_dataset,
        "is_synthetic": is_synth,
    }


# --- Dataset Management Endpoints ---

@app.get("/api/datasets", response_model=List[DatasetInfo])
def list_datasets():
    return dataset_manager.list_datasets()


@app.post("/api/datasets/upload", response_model=DatasetInfo)
@limiter.limit("20/minute")
async def upload_dataset_endpoint(request: Request, file: UploadFile = File(...)):
    # Security: File extension and content validation done in dataset_manager
    return dataset_manager.upload_dataset(file)


@app.get("/api/datasets/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    return dataset_manager.profile_dataset(dataset_id)


@app.get("/api/datasets/{dataset_id}/config", response_model=DatasetConfig)
def get_dataset_config(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    return dataset_manager.get_config(dataset_id)


@app.post("/api/datasets/{dataset_id}/configure", response_model=ConfigResponse)
def configure_dataset(dataset_id: str, config: DatasetConfig):
    validate_identifier(dataset_id, "Dataset ID")
    return dataset_manager.configure_dataset(dataset_id, config)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset_endpoint(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    success = dataset_manager.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete default benchmark dataset or dataset not found.",
        )
    return {"status": "deleted", "dataset_id": dataset_id}


# --- Background Jobs Endpoints ---

def _run_benchmark_worker(job_id: str, dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    pipe._experiments_cache = None
    pipe._personas_cache = None
    job_manager.update_job(job_id, 0, "Initializing 16-experiment benchmark matrix...", status="running")

    def cb(completed, total, step):
        job_manager.update_job(job_id, completed, step)
        return not job_manager.is_cancelled(job_id)

    def cancel_chk():
        return job_manager.is_cancelled(job_id)

    try:
        res = pipe.run_experiments(job_callback=cb, cancel_check=cancel_chk)
        if job_manager.is_cancelled(job_id):
            job_manager.update_job(
                job_id,
                completed=len(res.get("experiments", [])),
                current_step="Benchmark execution cancelled by user.",
                status="cancelled",
            )
        else:
            job_manager.update_job(
                job_id,
                completed=16,
                current_step="16-experiment benchmark completed successfully.",
                status="completed",
                result=res,
            )
            if dataset_id in dataset_manager._datasets:
                dataset_manager._datasets[dataset_id]["has_results"] = True
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            completed=0,
            current_step="Benchmark execution encountered an error.",
            status="failed",
            error=str(e),
        )


@app.post("/api/datasets/{dataset_id}/run")
@limiter.limit("15/minute")
def run_dataset_benchmark(request: Request, dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    # Verify dataset exists
    dataset_manager.get_dataset(dataset_id)
    job_id = job_manager.create_job(dataset_id)
    t = threading.Thread(target=_run_benchmark_worker, args=(job_id, dataset_id), daemon=True)
    t.start()
    return {"job_id": job_id, "dataset_id": dataset_id, "status": "pending"}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    validate_job_id(job_id)
    return job_manager.get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    validate_job_id(job_id)
    success = job_manager.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": success}


# --- Dynamic Pipeline Analysis Endpoints ---

@app.get("/api/dataset-summary", response_model=DatasetSummary)
def dataset_summary(dataset_id: str = Query("sample_credit_card")):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_dataset_summary()


@app.get("/api/datasets/{dataset_id}/summary", response_model=DatasetSummary)
def dataset_summary_scoped(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_dataset_summary()


@app.get("/api/feature-engineering", response_model=FeatureEngineeringResult)
def feature_engineering(dataset_id: str = Query("sample_credit_card")):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.feature_engineering()


@app.get("/api/datasets/{dataset_id}/feature-engineering", response_model=FeatureEngineeringResult)
def feature_engineering_scoped(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.feature_engineering()


@app.get("/api/experiments-matrix", response_model=ExperimentsMatrix)
def experiments_matrix(dataset_id: str = Query("sample_credit_card")):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    if pipe._experiments_cache is None:
        raise HTTPException(status_code=404, detail="Benchmark has not been run for this dataset yet.")
    return pipe._experiments_cache


@app.get("/api/datasets/{dataset_id}/experiments", response_model=ExperimentsMatrix)
def experiments_matrix_scoped(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    if pipe._experiments_cache is None:
        raise HTTPException(status_code=404, detail="Benchmark has not been run for this dataset yet.")
    return pipe._experiments_cache


@app.get("/api/growth-personas", response_model=GrowthPersonas)
def growth_personas(dataset_id: str = Query("sample_credit_card")):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    if pipe._experiments_cache is None:
        raise HTTPException(status_code=404, detail="Benchmark has not been run for this dataset yet.")
    return pipe.get_growth_personas()


@app.get("/api/datasets/{dataset_id}/personas", response_model=GrowthPersonas)
def growth_personas_scoped(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    if pipe._experiments_cache is None:
        raise HTTPException(status_code=404, detail="Benchmark has not been run for this dataset yet.")
    return pipe.get_growth_personas()


# --- Export and Inference Endpoints ---

@app.get("/api/datasets/{dataset_id}/export/segments.csv")
def export_segments_csv(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    csv_str = pipe.export_segments_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="pulse_segments_{dataset_id}.csv"'},
    )


@app.post("/api/datasets/{dataset_id}/predict-segment", response_model=PredictSegmentResponse)
@limiter.limit("30/minute")
def predict_segment(request: Request, dataset_id: str, payload: PredictSegmentRequest):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.predict_segment(payload.features)


@app.get("/api/datasets/{dataset_id}/model.joblib")
def export_model_joblib(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    joblib_bytes = pipe.export_model_joblib()
    return Response(
        content=joblib_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="pulse_model_{dataset_id}.joblib"',
            "X-Security-Warning": "This file is a Python pickle binary. Load only from trusted sources; arbitrary code execution is possible.",
        },
    )


@app.get("/api/datasets/{dataset_id}/model.json", response_model=ModelJSONExport)
def export_model_json(dataset_id: str):
    validate_identifier(dataset_id, "Dataset ID")
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.export_model_json()


# --- Backward Compatibility Upload ---

@app.post("/api/upload", response_model=UploadResponse)
@limiter.limit("20/minute")
async def legacy_upload(request: Request, file: UploadFile = File(...)):
    info = dataset_manager.upload_dataset(file)
    df, _, _ = dataset_manager.get_dataset(info.id)
    return UploadResponse(
        dataset_id=info.id,
        filename=info.name,
        row_count=info.row_count,
        col_count=info.col_count,
        columns=list(df.columns),
        is_synthetic=info.is_synthetic,
        message=f"Successfully uploaded and profiled {info.name} ({info.row_count} records).",
    )
