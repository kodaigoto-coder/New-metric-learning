"""Request and response Pydantic models for `api-backend`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSelectRequest(BaseModel):
    """Select an active dataset or create/clone a new named dataset."""

    dataset_name: str = Field(
        description="Name of the dataset to activate or create (e.g. 'default', 'bottle_lotA')."
    )
    clone_from: str | None = Field(
        default=None,
        description="Optional source dataset name to copy images & manifests from when creating a new dataset.",
    )


class SubfolderClassRenameRequest(BaseModel):
    """Rename or merge a subfolder class within a dataset."""

    dataset_name: str | None = Field(default=None)
    old_class_name: str = Field(description="Existing subfolder class name to rename or merge.")
    new_class_name: str = Field(description="Target subfolder class name.")


class DirectAnnotationItemUpdate(BaseModel):
    """Single row in the interactive annotation table editor."""

    image_path: str
    file_name: str
    label_name: str
    split: Literal["train", "val", "test"] = "train"
    cluster_id: int = 0


class DirectAnnotationsEditorRequest(BaseModel):
    """Save edited per-image labels directly to `annotations.json` (and optionally sync `raw/<class>/` subfolders)."""

    dataset_name: str | None = Field(default=None)
    normal_class_name: str = Field(
        default="normal",
        description="Class name representing the Normal (OK, label_id=0) class.",
    )
    items: list[DirectAnnotationItemUpdate] = Field(default_factory=list)
    sync_raw_subfolders: bool = Field(
        default=True,
        description="If True, also moves each image file on disk into raw/<label_name>/ so future clustering runs match the edited labels.",
    )
    rebalance_splits: bool = Field(
        default=False,
        description="If True, re-runs stratified 6:2:2 (train/val/test) split assignment instead of keeping row splits.",
    )
    train_ratio: float = Field(default=0.60, ge=0.20, le=0.80)
    val_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    seed: int = Field(default=42)


class GenerateSyntheticRequest(BaseModel):
    dataset_name: str | None = Field(default=None)
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

    dataset_name: str | None = Field(default=None)
    items: list[UploadedImageItem] = Field(default_factory=list)
    clear_existing: bool = Field(
        default=False,
        description="If True, clears existing raw images and manifests before saving this batch.",
    )


class DirectSubfolderAnnotationRequest(BaseModel):
    """Create `annotations.json` directly using uploaded subfolder names as training class labels."""

    dataset_name: str | None = Field(default=None)
    normal_class_name: str = Field(
        default="normal",
        description="Subfolder name to treat as the Normal (OK, label_id=0) class.",
    )
    train_ratio: float = Field(default=0.60, ge=0.20, le=0.80)
    val_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    seed: int = Field(default=42)


class AnnotationSubmissionRequest(BaseModel):
    """Cluster-level label mapping plus optional subfolder-label mode and per-image overrides."""

    dataset_name: str | None = Field(default=None)
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
    train_ratio: float = Field(default=0.60, ge=0.20, le=0.80)
    val_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.20, ge=0.05, le=0.45)
    seed: int = Field(default=42)


class ClusteringTriggerRequest(BaseModel):
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


class TrainingTriggerRequest(BaseModel):
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
    num_normal_proxies: int = Field(default=4, ge=1, le=12)
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


class InferenceProxyRequest(BaseModel):
    image_path: str
    tau_sim: float = Field(default=0.0, ge=-1.0, le=1.0)
    session_id: str | None = Field(default=None)


class EvaluationMetricsResponse(BaseModel):
    session_id: str = "session_0001"
    epoch: int
    is_full_snapshot: bool
    tau_sim: float
    selected_split: str = "test"
    split_counts: dict[str, int] = Field(default_factory=dict)
    d_min_centroid: float
    d_min_boundary: float
    train_loss: float
    val_loss: float = 0.0
    loss_direction_feedback: str = ""
    effective_normal_ratio: float = 0.50
    curriculum_progress: float
    effective_top_k: int
    num_normal_proxies: int = 1
    dispersion_metrics: dict[str, Any] = Field(default_factory=dict)
    dispersion_by_split: dict[str, dict[str, Any]] = Field(default_factory=dict)
    active_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    val_metrics: dict[str, Any]
    train_metrics: dict[str, Any] = Field(default_factory=dict)
    all_metrics: dict[str, Any]
    threshold_curve: list[dict[str, float]]
    samples: list[dict[str, Any]]
    proxy_points: list[dict[str, Any]] = Field(default_factory=list)
    heatmap_matrix: list[list[float]]
    heatmap_labels: list[str]
