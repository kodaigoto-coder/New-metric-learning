"""FastAPI entry point for the isolated `trainer-service` ML worker container."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers.train_jobs import get_active_job_id, router as train_jobs_router

app = FastAPI(
    title="Metric-Learning Trainer Service",
    description="Isolated heavy ML worker for 2-stage clustering and Asymmetric MS-Loss + XBM training.",
    version="0.1.0",
)

app.include_router(train_jobs_router)


@app.get("/health")
def health_check() -> dict[str, str | None]:
    return {
        "status": "ok",
        "service": "trainer-service",
        "active_job_id": get_active_job_id(),
    }
