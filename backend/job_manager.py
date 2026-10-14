import time
import uuid
import threading
from typing import Dict, Any, Optional
from models import JobStatusResponse


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, dataset_id: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": "pending",
                "completed": 0,
                "total": 16,
                "current_step": "Initializing 16-experiment benchmark matrix...",
                "start_time": time.perf_counter(),
                "elapsed_ms": 0.0,
                "error": None,
                "cancel_event": threading.Event(),
                "result": None,
            }
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

    def get_job(self, job_id: str) -> JobStatusResponse:
        with self._lock:
            if job_id not in self._jobs:
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
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["cancel_event"].set()
                self._jobs[job_id]["status"] = "cancelled"
                self._jobs[job_id]["current_step"] = "Benchmark execution cancelled by user."
                return True
            return False

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]["cancel_event"].is_set()
            return False


job_manager = JobManager()
