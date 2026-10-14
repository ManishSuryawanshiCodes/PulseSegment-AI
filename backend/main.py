import os
import io
import threading
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from dataset_manager import DatasetManager
from job_manager import job_manager
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
)

app = FastAPI(title="PulseSegment AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dataset manager with data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
dataset_manager = DatasetManager(data_dir=DATA_DIR)


@app.get("/api/health")
def health():
    datasets = dataset_manager.list_datasets()
    total_records = sum(d.row_count for d in datasets)
    return {
        "status": "ok",
        "datasets_available": len(datasets),
        "total_records": total_records,
        "active_dataset": "sample_credit_card",
    }


# --- Dataset Management Endpoints ---

@app.get("/api/datasets", response_model=List[DatasetInfo])
def list_datasets():
    return dataset_manager.list_datasets()


@app.post("/api/datasets/upload", response_model=DatasetInfo)
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    return dataset_manager.upload_dataset(file)


@app.get("/api/datasets/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(dataset_id: str):
    return dataset_manager.profile_dataset(dataset_id)


@app.get("/api/datasets/{dataset_id}/config", response_model=DatasetConfig)
def get_dataset_config(dataset_id: str):
    return dataset_manager.get_config(dataset_id)


@app.post("/api/datasets/{dataset_id}/configure", response_model=ConfigResponse)
def configure_dataset(dataset_id: str, config: DatasetConfig):
    return dataset_manager.configure_dataset(dataset_id, config)


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
    except Exception as e:
        job_manager.update_job(
            job_id,
            completed=0,
            current_step="Benchmark execution encountered an error.",
            status="failed",
            error=str(e),
        )


@app.post("/api/datasets/{dataset_id}/run")
def run_dataset_benchmark(dataset_id: str):
    # Verify dataset exists
    dataset_manager.get_dataset(dataset_id)
    job_id = job_manager.create_job(dataset_id)
    t = threading.Thread(target=_run_benchmark_worker, args=(job_id, dataset_id), daemon=True)
    t.start()
    return {"job_id": job_id, "dataset_id": dataset_id, "status": "pending"}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    return job_manager.get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    success = job_manager.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": success}


# --- Dynamic Pipeline Analysis Endpoints ---

@app.get("/api/dataset-summary", response_model=DatasetSummary)
def dataset_summary(dataset_id: str = Query("sample_credit_card")):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_dataset_summary()


@app.get("/api/datasets/{dataset_id}/summary", response_model=DatasetSummary)
def dataset_summary_scoped(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_dataset_summary()


@app.get("/api/feature-engineering", response_model=FeatureEngineeringResult)
def feature_engineering(dataset_id: str = Query("sample_credit_card")):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.feature_engineering()


@app.get("/api/datasets/{dataset_id}/feature-engineering", response_model=FeatureEngineeringResult)
def feature_engineering_scoped(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.feature_engineering()


@app.get("/api/experiments-matrix", response_model=ExperimentsMatrix)
def experiments_matrix(dataset_id: str = Query("sample_credit_card")):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.run_experiments()


@app.get("/api/datasets/{dataset_id}/experiments", response_model=ExperimentsMatrix)
def experiments_matrix_scoped(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.run_experiments()


@app.get("/api/growth-personas", response_model=GrowthPersonas)
def growth_personas(dataset_id: str = Query("sample_credit_card")):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_growth_personas()


@app.get("/api/datasets/{dataset_id}/personas", response_model=GrowthPersonas)
def growth_personas_scoped(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.get_growth_personas()


# --- Export and Inference Endpoints ---

@app.get("/api/datasets/{dataset_id}/export/segments.csv")
def export_segments_csv(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    csv_str = pipe.export_segments_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="pulse_segments_{dataset_id}.csv"'},
    )


@app.post("/api/datasets/{dataset_id}/predict-segment", response_model=PredictSegmentResponse)
def predict_segment(dataset_id: str, payload: PredictSegmentRequest):
    pipe = dataset_manager.get_pipeline(dataset_id)
    return pipe.predict_segment(payload.features)


@app.get("/api/datasets/{dataset_id}/model.joblib")
def export_model_joblib(dataset_id: str):
    pipe = dataset_manager.get_pipeline(dataset_id)
    joblib_bytes = pipe.export_model_joblib()
    return Response(
        content=joblib_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="pulse_model_{dataset_id}.joblib"'},
    )


# --- Backward Compatibility Upload ---

@app.post("/api/upload", response_model=UploadResponse)
async def legacy_upload(file: UploadFile = File(...)):
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
