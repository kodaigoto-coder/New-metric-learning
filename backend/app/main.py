"""FastAPI entry point for the lightweight `api-backend` gateway container."""

from __future__ import annotations

from fastapi import FastAPI

from app.routers.datasets import router as datasets_router
from app.routers.evaluation import router as evaluation_router
from app.routers.jobs_proxy import router as jobs_proxy_router

app = FastAPI(
    title="Metric-Learning Backend API Gateway",
    description="Gateway API for dataset/annotation management, job orchestration, and dynamic threshold space evaluation.",
    version="0.1.0",
)

app.include_router(datasets_router)
app.include_router(jobs_proxy_router)
app.include_router(evaluation_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "api-backend"}
