import time
import uuid
import re
import threading
from typing import Dict, Any, Optional
from fastapi import HTTPException
from models import JobStatusResponse

JOB_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_job_id(job_id: str) -> str:
    if not job_id or not JOB_ID_REGEX.match(job_id) or ".." in job_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Job ID format '{job_id}'.",
        )
    return job_id


class JobManager:
    def __init__(self, max_concurrent_jobs: int = 2, job_timeout_seconds: float = 60.0):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_timeout_seconds = job_timeout_seconds
        self.db = None

    def set_db(self, db):
        self.db = db

    def can_start_job(self) -> bool:
        with self._lock:
            running_count = sum(1 for j in self._jobs.values() if j.get("status") == "running")
            return running_count < self.max_concurrent_jobs

    def create_job(self, dataset_id: str) -> str:
        with self._lock:
            running_count = sum(1 for j in self._jobs.values() if j.get("status") == "running")
            if running_count >= self.max_concurrent_jobs:
                raise HTTPException(
                    status_code=429,
                    detail=f"Maximum concurrent benchmark jobs ({self.max_concurrent_jobs}) reached. Please wait for an existing job to complete.",
                )

            job_id = f"job_{uuid.uuid4().hex[:10]}"
            job_data = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "pending",
                "completed": 0,
                "total": 16,
                "current_step": "Initializing 16-experiment benchmark matrix...",
                "start_time": time.perf_counter(),
                "created_at_wall": time.time(),
                "elapsed_ms": 0.0,
                "error": None,
                "cancel_event": threading.Event(),
                "result": None,
            }
            self._jobs[job_id] = job_data

            if self.db:
                self.db.save_job(job_data)

        return job_id

    def update_job(
        self,
        job_id: str,
        completed: int,
        current_step: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Any] = None,
    ):
        validate_job_id(job_id)
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["completed"] = completed
                job["current_step"] = current_step
                job["elapsed_ms"] = round((time.perf_counter() - job["start_time"]) * 1000.0, 1)
                if status:
                    job["status"] = status
                if error:
                    job["error"] = error
                if result:
                    job["result"] = result

                if self.db:
                    self.db.save_job(job)

    def get_job(self, job_id: str) -> JobStatusResponse:
        validate_job_id(job_id)
        with self._lock:
            if job_id not in self._jobs:
                if self.db:
                    db_job = self.db.get_job(job_id)
                    if db_job:
                        return JobStatusResponse(
                            job_id=db_job["job_id"],
                            dataset_id=db_job["dataset_id"],
                            status=db_job["status"],
                            completed=db_job["completed"],
                            total=db_job["total"],
                            current_step=db_job.get("current_step", ""),
                            elapsed_ms=db_job.get("elapsed_ms", 0.0),
                            error=db_job.get("error"),
                        )

                return JobStatusResponse(
                    job_id=job_id,
                    dataset_id="unknown",
                    status="failed",
                    completed=0,
                    total=16,
                    current_step="Job not found",
                    elapsed_ms=0.0,
                    error="Job ID does not exist.",
                )

            job = self._jobs[job_id]

            # Check timeout
            if job["status"] == "running":
                elapsed_sec = time.time() - job["created_at_wall"]
                if elapsed_sec > self.job_timeout_seconds:
                    job["status"] = "failed"
                    job["error"] = f"Job execution exceeded maximum timeout of {self.job_timeout_seconds} seconds."
                    job["cancel_event"].set()

            elapsed = (
                job["elapsed_ms"]
                if job["status"] in ["completed", "failed", "cancelled"]
                else round((time.perf_counter() - job["start_time"]) * 1000.0, 1)
            )
            return JobStatusResponse(
                job_id=job["job_id"],
                dataset_id=job["dataset_id"],
                status=job["status"],
                completed=job["completed"],
                total=job["total"],
                current_step=job["current_step"],
                elapsed_ms=elapsed,
                error=job["error"],
            )

    def cancel_job(self, job_id: str) -> bool:
        validate_job_id(job_id)
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["cancel_event"].set()
                self._jobs[job_id]["status"] = "cancelled"
                self._jobs[job_id]["current_step"] = "Benchmark execution cancelled by user."
                if self.db:
                    self.db.save_job(self._jobs[job_id])
                return True
            return False

    def is_cancelled(self, job_id: str) -> bool:
        validate_job_id(job_id)
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]["cancel_event"].is_set()
            return False


job_manager = JobManager(max_concurrent_jobs=2, job_timeout_seconds=60.0)
