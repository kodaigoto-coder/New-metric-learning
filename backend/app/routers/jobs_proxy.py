"""Proxy endpoints forwarding heavy ML clustering and training jobs to `trainer-service`."""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from app.config import BackendConfig
from app.schemas.api_models import ClusteringTriggerRequest, TrainingTriggerRequest

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs-proxy"])


@router.post("/clustering")
def trigger_clustering_job(req: ClusteringTriggerRequest) -> dict[str, Any]:
    """Forward Phase 1 clustering job request to `trainer-service`."""
    payload = req.model_dump()
    if not payload.get("dataset_name"):
        payload["dataset_name"] = BackendConfig.get_active_dataset()
    url = f"{BackendConfig.TRAINER_API_URL}/internal/jobs/clustering"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 409:
                raise HTTPException(status_code=409, detail=resp.json().get("detail", "Job conflict"))
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Trainer service unreachable: {exc}") from exc


@router.post("/training")
def trigger_training_job(req: TrainingTriggerRequest) -> dict[str, Any]:
    """Forward Phase 2 training job request to `trainer-service`."""
    payload = req.model_dump()
    if not payload.get("dataset_name"):
        payload["dataset_name"] = BackendConfig.get_active_dataset()
    url = f"{BackendConfig.TRAINER_API_URL}/internal/jobs/training"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 409:
                raise HTTPException(status_code=409, detail=resp.json().get("detail", "Job conflict"))
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Trainer service unreachable: {exc}") from exc


@router.get("/{job_id}")
def get_job_status_proxy(job_id: str) -> dict[str, Any]:
    """Poll job progress from `trainer-service` with local disk fallback."""
    url = f"{BackendConfig.TRAINER_API_URL}/internal/jobs/{job_id}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp.json()
    except httpx.HTTPError:
        pass

    # Fallback to shared volume job status file if trainer HTTP is momentarily busy
    job_file = BackendConfig.artifacts_dir() / "jobs" / f"{job_id}.json"
    if job_file.exists():
        return json.loads(job_file.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail=f"Job `{job_id}` not found.")
