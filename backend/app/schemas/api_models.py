"""Request and response Pydantic models for `api-backend`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateSyntheticRequest(BaseModel):
    num_normal: int = Field(default=100, ge=10, le=2000)
    num_scratch: int = Field(default=3, ge=1, le=200)
    num_stain: int = Field(default=3, ge=1, le=200)
    num_dent: int = Field(default=2, ge=0, le=200)
    image_size: int = Field(default=128, ge=64, le=512)
    seed: int = Field(default=42)
    clear_existing: bool = Field(default=True)


class UploadedImageItem(BaseModel):
    """Single image file uploaded as part of a class-subfolder directory structure."""

    file_name: str = Field(description="Original image file name (e.g. '001.png').")
    class_name: str = Field(
        description="Subfolder name acting as the class label (e.g. 'normal', 'scratch', 'stain')."
    )
    content_base64: str = Field(description="Base64-encoded binary content of the image file.")


class FolderUploadRequest(BaseModel):
    """Batch request for uploading images organized by class subfolders."""

    items: list[UploadedImageItem] = Field(default_factory=list)
    clear_existing: bool = Field(
        default=False,
        description="If True, clears existing raw images and manifests before saving this batch.",
    )


class DirectSubfolderAnnotationRequest(BaseModel):
    """Create `annotations.json` directly using uploaded subfolder names as training class labels."""

    normal_class_name: str = Field(
        default="normal",
        description="Subfolder name to treat as the Normal (OK, label_id=0) class.",
    )
    val_ratio: float = Field(default=0.25, ge=0.1, le=0.5)
    seed: int = Field(default=42)


class AnnotationSubmissionRequest(BaseModel):
    """Cluster-level label mapping plus optional subfolder-label mode and per-image overrides."""

    cluster_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from cluster_id (as string, e.g. '0', '1') to label_name (e.g. 'normal', 'defect_scratch').",
    )
    sample_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-image override mapping from image_path or file_name to label_name.",
    )
    normal_class_name: str = Field(
        default="normal",
        description="Class name representing the Normal (OK, label_id=0) class.",
    )
    use_subfolder_labels: bool = Field(
        default=False,
        description="If True, each image's uploaded parent subfolder name is used directly as its class label.",
    )
    val_ratio: float = Field(default=0.25, ge=0.1, le=0.5)
    seed: int = Field(default=42)


class ClusteringTriggerRequest(BaseModel):
    backbone_name: str = Field(default="efficientnet_b3")
    pooling_type: Literal["gap", "gem", "avg_max"] = Field(default="gem")
    core_quantile: float = Field(default=0.85, ge=0.50, le=0.97)
    k_nn: int = Field(default=15, ge=2, le=100)
    max_k: int = Field(default=8, ge=2, le=20)
    max_pca_dim: int = Field(default=64, ge=2, le=256)
    pca_variance: float = Field(default=0.95, ge=0.50, le=0.999)
    pretrained: bool = Field(default=False)


class TrainingTriggerRequest(BaseModel):
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


class InferenceProxyRequest(BaseModel):
    image_path: str
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)


class EvaluationMetricsResponse(BaseModel):
    epoch: int
    is_full_snapshot: bool
    tau_sim: float
    d_min_centroid: float
    d_min_boundary: float
    train_loss: float
    curriculum_progress: float
    effective_top_k: int
    val_metrics: dict[str, Any]
    all_metrics: dict[str, Any]
    threshold_curve: list[dict[str, float]]
    samples: list[dict[str, Any]]
    heatmap_matrix: list[list[float]]
    heatmap_labels: list[str]
