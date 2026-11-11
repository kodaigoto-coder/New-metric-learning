"""Feature extraction backbone (timm / built-in fallback), GeM/GAP pooling, dataset standardization, and 128-D hypersphere projection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


PoolingType = Literal["gap", "gem", "avg_max"]


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) spatial pooling to preserve localized defect activations."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = False) -> None:
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            # Vision Transformer token sequence [B, Tokens, C] -> pool over spatial tokens
            clamped = x.clamp(min=self.eps).pow(self.p)
            return clamped.mean(dim=1).pow(1.0 / self.p)
        # CNN feature map [B, C, H, W]
        clamped = x.clamp(min=self.eps).pow(self.p)
        pooled = F.adaptive_avg_pool2d(clamped, (1, 1)).pow(1.0 / self.p)
        return pooled.flatten(1)


class SpatialPoolingHead(nn.Module):
    """Configurable spatial pooling (GAP, GeM, or Avg+Max concatenation)."""

    def __init__(self, pooling_type: PoolingType = "gem") -> None:
        super().__init__()
        self.pooling_type: PoolingType = pooling_type
        self.gem = GeMPooling(p=3.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            if self.pooling_type == "gap":
                return feat.mean(dim=1)
            if self.pooling_type == "avg_max":
                return torch.cat([feat.mean(dim=1), feat.amax(dim=1)], dim=-1)
            return self.gem(feat)

        if self.pooling_type == "gap":
            return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        if self.pooling_type == "avg_max":
            avg_p = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            max_p = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)
            return torch.cat([avg_p, max_p], dim=-1)
        return self.gem(feat)


class LightweightInspectionStem(nn.Module):
    """Deterministic multi-scale texture + color-contrast convolutional stem."""

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def extract_handcrafted_surface_statistics(x: torch.Tensor) -> torch.Tensor:
    """Compute multi-scale spatial-grid color, local variance, and gradient descriptors.

    Preserves 8x8 spatial patch structure alongside global quantiles so localized structural
    anomalies (e.g. MVTec broken_large, broken_small, contamination, scratches, stains, dents)
    produce strong per-dimension deviations relative to the normal product template.
    """
    b, c, _h, _w = x.shape
    mean_rgb = x.mean(dim=(2, 3))
    std_rgb = x.std(dim=(2, 3)) + 1e-6

    # 8x8 spatial grid mean and max pooling [B, 3*64 = 192]
    grid_avg_8x8 = F.adaptive_avg_pool2d(x, (8, 8)).reshape(b, -1)
    grid_max_4x4 = F.adaptive_max_pool2d(x, (4, 4)).reshape(b, -1)
    grid_min_4x4 = (-F.adaptive_max_pool2d(-x, (4, 4))).reshape(b, -1)

    # Local patch variance on 6x6 grid via E[X^2] - (E[X])^2 [B, 3*36 = 108]
    e_x = F.adaptive_avg_pool2d(x, (6, 6))
    e_x2 = F.adaptive_avg_pool2d(x * x, (6, 6))
    local_std_6x6 = (e_x2 - e_x * e_x).clamp(min=0.0).sqrt().reshape(b, -1)

    # Spatial high-low quantiles across 8x8 patches
    patch_avg = F.adaptive_avg_pool2d(x, (8, 8)).view(b, c, -1)
    q95 = torch.quantile(patch_avg, 0.95, dim=-1)
    q05 = torch.quantile(patch_avg, 0.05, dim=-1)

    # Horizontal & vertical edge gradients on 6x6 spatial grid [B, 3*36 = 108]
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    gx_grid = F.adaptive_avg_pool2d(gx, (6, 6)).reshape(b, -1)
    gy_grid = F.adaptive_avg_pool2d(gy, (6, 6)).reshape(b, -1)
    gx_max = F.adaptive_max_pool2d(gx, (4, 4)).view(b, c, -1).amax(dim=-1)
    gy_max = F.adaptive_max_pool2d(gy, (4, 4)).view(b, c, -1).amax(dim=-1)

    # Chromatic differences (R-G, R-B, G-B) on 4x4 grid + global stats
    rg = x[:, 0:1] - x[:, 1:2]
    rb = x[:, 0:1] - x[:, 2:3]
    gb = x[:, 1:2] - x[:, 2:3]
    chroma_grid = torch.cat(
        [
            F.adaptive_avg_pool2d(rg, (4, 4)).reshape(b, -1),
            F.adaptive_avg_pool2d(rb, (4, 4)).reshape(b, -1),
            F.adaptive_avg_pool2d(gb, (4, 4)).reshape(b, -1),
        ],
        dim=-1,
    )

    stats = torch.cat(
        [
            mean_rgb,
            std_rgb,
            q95,
            q05,
            gx_max,
            gy_max,
            grid_avg_8x8,
            grid_max_4x4,
            grid_min_4x4,
            local_std_6x6,
            gx_grid,
            gy_grid,
            chroma_grid,
        ],
        dim=-1,
    )
    return stats


class MetricProjectionModel(nn.Module):
    """Decoupled `timm` backbone + Spatial Pooling + Feature Standardization + 128-D Projection Head."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pooling_type: PoolingType = "gem",
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_block: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pooling_type: PoolingType = pooling_type
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_block = unfreeze_last_block

        self.backbone = self._load_backbone(backbone_name=backbone_name, pretrained=pretrained)
        self.pool = SpatialPoolingHead(pooling_type=pooling_type)

        # Determine raw feature dimension dynamically
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 128, 128)
            raw_feat = self._extract_raw_concat_features(dummy)
            pooled_dim = raw_feat.shape[-1]

        self.gap_feature_dim = pooled_dim

        # Dataset standardization buffers to eliminate common background/product bias vector
        self.register_buffer("feat_mean", torch.zeros(1, pooled_dim))
        self.register_buffer("feat_std", torch.ones(1, pooled_dim))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

        # Configure backbone trainability
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_block:
                params = list(self.backbone.parameters())
                for param in params[-min(6, len(params)) :]:
                    param.requires_grad = True
            else:
                self.backbone.eval()

        hidden_dim = 384
        self.projection_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def train(self, mode: bool = True) -> MetricProjectionModel:
        """Keep frozen backbone in `.eval()` mode during training so BatchNorm running stats never drift."""
        super().train(mode)
        if self.freeze_backbone and not self.unfreeze_last_block:
            self.backbone.eval()
        return self

    def _load_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        if backbone_name == "lightweight_cnn" or timm is None:
            return LightweightInspectionStem(out_channels=256)
        try:
            return timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
        except Exception:
            try:
                return timm.create_model(
                    backbone_name, pretrained=False, num_classes=0, global_pool=""
                )
            except Exception:
                return LightweightInspectionStem(out_channels=256)

    def _extract_raw_concat_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            feat_map = self.backbone.forward_features(x)
        else:
            feat_map = self.backbone(x)
        pooled = self.pool(feat_map)
        surface_stats = extract_handcrafted_surface_statistics(x)
        combined = torch.cat(
            [F.normalize(pooled, p=2, dim=-1), F.normalize(surface_stats, p=2, dim=-1)],
            dim=-1,
        )
        return combined

    @torch.no_grad()
    def calibrate_feature_stats(self, raw_features: np.ndarray | torch.Tensor) -> None:
        """Calibrate per-dimension mean and std across the dataset to remove constant background bias."""
        if isinstance(raw_features, np.ndarray):
            feat_t = torch.from_numpy(raw_features).to(
                device=self.feat_mean.device, dtype=torch.float32
            )
        else:
            feat_t = raw_features.detach().to(
                device=self.feat_mean.device, dtype=torch.float32
            )

        if feat_t.shape[0] < 2:
            return

        mean = feat_t.mean(dim=0, keepdim=True)
        std = feat_t.std(dim=0, keepdim=True)
        # Dimensions with near-zero variance across the dataset are dead/constant channels;
        # set their scale to 1.0 after subtracting mean so they stay 0.0 instead of amplifying float noise.
        valid_std = torch.where(std > 1e-5, std, torch.ones_like(std))

        self.feat_mean.copy_(mean)
        self.feat_std.copy_(valid_std)
        self.is_calibrated.fill_(True)

    def standardize_features(self, raw_feat: torch.Tensor) -> torch.Tensor:
        """Center and standardize raw features before L2 normalization."""
        if bool(self.is_calibrated.item()):
            centered = (raw_feat - self.feat_mean) / (self.feat_std + 1e-5)
            centered = centered.clamp(min=-10.0, max=10.0)
            return F.normalize(centered, p=2, dim=-1)
        if raw_feat.shape[0] >= 4:
            mean = raw_feat.mean(dim=0, keepdim=True).detach()
            std = raw_feat.std(dim=0, keepdim=True).detach()
            valid_std = torch.where(std > 1e-5, std, torch.ones_like(std))
            centered = ((raw_feat - mean) / (valid_std + 1e-5)).clamp(min=-10.0, max=10.0)
            return F.normalize(centered, p=2, dim=-1)
        return F.normalize(raw_feat, p=2, dim=-1)

    def extract_gap_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract standardized pre-projection features h in R^{D_backbone}."""
        raw_feat = self._extract_raw_concat_features(x)
        return self.standardize_features(raw_feat)

    def forward_from_gap_features(self, gap_feat: torch.Tensor) -> torch.Tensor:
        """Project standardized GAP/surface features onto the 128-D unit hypersphere S^{127}."""
        # Handle single-sample inference when BatchNorm1d is in train mode
        if gap_feat.shape[0] == 1 and self.training:
            self.projection_head.eval()
            proj = self.projection_head(gap_feat)
            self.projection_head.train()
        else:
            proj = self.projection_head(gap_feat)
        return F.normalize(proj, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_128d_normalized, gap_features)."""
        gap_feat = self.extract_gap_features(x)
        z = self.forward_from_gap_features(gap_feat)
        return z, gap_feat


def build_image_transforms(is_train: bool = False, image_size: int = 128) -> transforms.Compose:
    """Build image transforms with gentle symmetric augmentations that preserve localized defects."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.03, contrast=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_tensor(image_path: str | Path, transform: transforms.Compose) -> torch.Tensor:
    """Load an RGB image from disk and apply the specified transform."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return transform(rgb)


@torch.no_grad()
def extract_raw_uncalibrated_features(
    image_paths: list[str],
    model: MetricProjectionModel,
    device: str = "cpu",
    batch_size: int = 32,
    image_size: int = 128,
) -> np.ndarray:
    """Extract raw uncalibrated backbone + surface features across all `image_paths`."""
    model.eval()
    model.to(device)
    transform = build_image_transforms(is_train=False, image_size=image_size)
    raw_list: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = [load_image_tensor(p, transform) for p in batch_paths]
        batch_x = torch.stack(tensors, dim=0).to(device)
        raw_feat = model._extract_raw_concat_features(batch_x)
        raw_list.append(raw_feat.detach().cpu().numpy().astype(np.float32))

    if not raw_list:
        return np.zeros((0, model.gap_feature_dim), dtype=np.float32)
    return np.concatenate(raw_list, axis=0)


@torch.no_grad()
def extract_dataset_embeddings(
    image_paths: list[str],
    model: MetricProjectionModel,
    device: str = "cpu",
    batch_size: int = 32,
    image_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both 128-D projection vectors and standardized GAP/GeM features as pure NumPy arrays."""
    model.eval()
    model.to(device)

    # Automatically calibrate standardization stats across the dataset if not yet calibrated
    if not bool(model.is_calibrated.item()) and len(image_paths) >= 2:
        raw_all = extract_raw_uncalibrated_features(
            image_paths=image_paths,
            model=model,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
        )
        model.calibrate_feature_stats(raw_all)

    transform = build_image_transforms(is_train=False, image_size=image_size)
    z_list: list[np.ndarray] = []
    gap_list: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = [load_image_tensor(p, transform) for p in batch_paths]
        batch_x = torch.stack(tensors, dim=0).to(device)
        z, gap = model(batch_x)
        z_list.append(z.detach().cpu().numpy().astype(np.float32))
        gap_list.append(gap.detach().cpu().numpy().astype(np.float32))

    if not z_list:
        return (
            np.zeros((0, model.embedding_dim), dtype=np.float32),
            np.zeros((0, model.gap_feature_dim), dtype=np.float32),
        )

    return np.concatenate(z_list, axis=0), np.concatenate(gap_list, axis=0)
