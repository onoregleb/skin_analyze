from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import uuid
import time
from enum import Enum
from app.utils.logging import get_logger

logger = get_logger("jobs")


class JobStatus(str, Enum):
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


@dataclass
class JobRecord:
    id: str
    status: JobStatus = JobStatus.in_progress
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    progress: Dict[str, Any] = field(default_factory=dict)  # partial results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}

    def create(self) -> JobRecord:
        job_id = str(uuid.uuid4())
        rec = JobRecord(id=job_id)
        self._jobs[job_id] = rec
        logger.info(f"Job created id={job_id}")
        return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        rec = self._jobs.get(job_id)
        if not rec:
            return
        rec.progress.update(progress)
        rec.updated_at = time.time()
        logger.info(f"Job {job_id} progress updated keys={list(progress.keys())}")

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        rec = self._jobs.get(job_id)
        if not rec:
            return
        rec.status = JobStatus.done
        rec.result = result
        rec.updated_at = time.time()
        logger.info(f"Job {job_id} completed")

    def fail(self, job_id: str, error: str) -> None:
        rec = self._jobs.get(job_id)
        if not rec:
            return
        rec.status = JobStatus.failed
        rec.error = error
        rec.updated_at = time.time()
        logger.warning(f"Job {job_id} failed: {error}")

# Global manager instance
job_manager = JobManager()
