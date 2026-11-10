"""Request and response schemas for the Trainer ML worker service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SyntheticDatasetRequest(BaseModel):
    num_normal: int = Field(default=100, ge=10, le=2000)
    num_scratch: int = Field(default=3, ge=1, le=200)
    num_stain: int = Field(default=3, ge=1, le=200)
    num_dent: int = Field(default=2, ge=0, le=200)
    image_size: int = Field(default=128, ge=64, le=512)
    seed: int = Field(default=42)
    clear_existing: bool = Field(default=True)


class ClusteringJobRequest(BaseModel):
    backbone_name: str = Field(default="efficientnet_b3")
    pooling_type: Literal["gap", "gem", "avg_max"] = Field(default="gem")
    core_quantile: float = Field(default=0.85, ge=0.50, le=0.97)
    k_nn: int = Field(default=15, ge=2, le=100)
    max_k: int = Field(default=8, ge=2, le=20)
    max_pca_dim: int = Field(default=64, ge=2, le=256)
    pca_variance: float = Field(default=0.95, ge=0.50, le=0.999)
    pretrained: bool = Field(default=False)


class TrainingJobRequest(BaseModel):
    backbone_name: str = Field(default="efficientnet_b3")
    pooling_type: Literal["gap", "gem", "avg_max"] = Field(default="gem")
    pretrained: bool = Field(default=False)
    freeze_backbone: bool = Field(default=True)
    unfreeze_last_block: bool = Field(default=False)
    epochs: int = Field(default=8, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=256)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1.0)
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)
    top_k_mode: Literal["count", "ratio"] = Field(default="count")
    top_k_initial: float = Field(default=30.0, gt=0.0)
    top_k_final: float = Field(default=20.0, gt=0.0)
    gamma_initial: float = Field(default=0.4, ge=0.0)
    gamma_final: float = Field(default=1.5, ge=0.0)
    w_defect: float = Field(default=2.5, ge=1.0)
    w_aux: float = Field(default=0.5, ge=0.0)
    full_snapshot_interval: int = Field(default=4, ge=1)


class SingleInferenceRequest(BaseModel):
    image_path: str
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: Literal["clustering", "training"]
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
