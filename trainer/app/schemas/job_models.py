"""Request and response schemas for the Trainer ML worker service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SyntheticDatasetRequest(BaseModel):
    dataset_name: str | None = Field(default=None)
    num_normal: int = Field(default=100, ge=10, le=2000)
    num_scratch: int = Field(default=3, ge=1, le=200)
    num_stain: int = Field(default=3, ge=1, le=200)
    num_dent: int = Field(default=2, ge=0, le=200)
    image_size: int = Field(default=128, ge=64, le=512)
    seed: int = Field(default=42)
    clear_existing: bool = Field(default=True)


class ClusteringJobRequest(BaseModel):
    dataset_name: str | None = Field(default=None)
    backbone_name: str = Field(default="efficientnet_b3")
    pooling_type: Literal["gap", "gem", "avg_max"] = Field(default="gem")
    normal_class_name: str = Field(
        default="normal",
        description="Subfolder/class name for Normal samples (kept fixed as Cluster 0 without clustering).",
    )
    core_quantile: float = Field(default=0.85, ge=0.50, le=0.97)
    k_nn: int = Field(default=15, ge=2, le=100)
    max_k: int = Field(default=8, ge=2, le=20)
    max_pca_dim: int = Field(default=64, ge=2, le=256)
    pca_variance: float = Field(default=0.95, ge=0.50, le=0.999)
    pretrained: bool = Field(default=True)


class TrainingJobRequest(BaseModel):
    dataset_name: str | None = Field(default=None)
    session_label: str | None = Field(
        default=None,
        description="Optional user memo/label for this training session.",
    )
    loss_type: Literal["multi_proxy_anchor", "ms_loss"] = Field(default="multi_proxy_anchor")
    backbone_name: str = Field(default="efficientnet_b3")
    pooling_type: Literal["gap", "gem", "avg_max"] = Field(default="gem")
    pretrained: bool = Field(default=True)
    freeze_backbone: bool = Field(default=True)
    unfreeze_last_block: bool = Field(default=False)
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=256)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1.0)
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)
    normal_loss_ratio: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description="Power balance ratio between Normal Centripetal Force and Anomaly Repulsion Force (0.50 = 50:50).",
    )
    repel_delta: float = Field(
        default=0.0,
        ge=-0.80,
        le=0.80,
        description="Cosine margin delta for anomaly repulsion max(0, S(x, p_norm) - delta) (default 0.0 = orthogonal 90 deg).",
    )
    num_normal_proxies: int = Field(
        default=0,
        ge=0,
        le=12,
        description="Number of Normal sub-center proxies M (0 = Auto M* via epsilon-covering & silhouette; 1..12 = manual override).",
    )
    hard_anomaly_boost: float = Field(default=3.0, ge=1.0, le=20.0)
    class_balance_beta: float = Field(default=0.99, ge=0.50, le=0.9999)
    enable_boundary_aug: bool = Field(default=True)
    mixup_alpha_min: float = Field(default=0.55, ge=0.30, le=0.90)
    mixup_alpha_max: float = Field(default=0.75, ge=0.35, le=0.95)
    nab_normal_ratio: float = Field(default=0.50, ge=0.25, le=0.75)
    top_k_mode: Literal["count", "ratio"] = Field(default="ratio")
    top_k_initial: float = Field(default=0.30, gt=0.0)
    top_k_final: float = Field(default=0.15, gt=0.0)
    gamma_initial: float = Field(default=0.6, ge=0.0)
    gamma_final: float = Field(default=1.5, ge=0.0)
    w_defect: float = Field(default=1.0, ge=0.1)
    w_aux: float = Field(default=0.2, ge=0.0)
    full_snapshot_interval: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _normalize_coupled_bounds(self) -> TrainingJobRequest:
        lo = min(float(self.mixup_alpha_min), float(self.mixup_alpha_max))
        hi = max(float(self.mixup_alpha_min), float(self.mixup_alpha_max))
        self.mixup_alpha_min = round(min(0.90, max(0.30, lo)), 4)
        self.mixup_alpha_max = round(
            min(0.95, max(0.35, self.mixup_alpha_min + 0.05, hi)), 4
        )
        if self.top_k_mode == "count":
            if self.top_k_initial < 1.0:
                self.top_k_initial = float(max(1, int(round(self.top_k_initial * 100.0))))
            if self.top_k_final < 1.0:
                self.top_k_final = float(max(1, int(round(self.top_k_final * 100.0))))
        else:
            if self.top_k_initial > 1.0:
                self.top_k_initial = round(min(1.0, max(0.01, self.top_k_initial / 100.0)), 4)
            if self.top_k_final > 1.0:
                self.top_k_final = round(min(1.0, max(0.01, self.top_k_final / 100.0)), 4)
        return self


class SingleInferenceRequest(BaseModel):
    image_path: str
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)
    session_id: str | None = Field(default=None)


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: Literal["clustering", "training"]
    status: Literal["queued", "running", "completed", "failed"]
    dataset_name: str = "default"
    session_id: str | None = None
    session_number: int | None = None
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    message: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
