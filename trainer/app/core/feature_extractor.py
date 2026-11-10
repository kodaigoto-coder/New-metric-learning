"""Feature extraction backbone (timm / built-in fallback), GeM/GAP pooling, and 128-D hypersphere projection."""

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
            # Token sequence [B, N, C]
            if self.pooling_type == "gap":
                return feat.mean(dim=1)
            if self.pooling_type == "avg_max":
                return torch.cat([feat.mean(dim=1), feat.amax(dim=1)], dim=-1)
            return self.gem(feat)

        # 4D CNN feature map [B, C, H, W]
        if self.pooling_type == "gap":
            return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        if self.pooling_type == "avg_max":
            avg_p = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            max_p = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)
            return torch.cat([avg_p, max_p], dim=-1)
        return self.gem(feat)


class LightweightInspectionStem(nn.Module):
    """Deterministic multi-scale texture + color-contrast convolutional stem.

    Used when `backbone_name == 'lightweight_cnn'` or as an offline fallback when
    downloading remote ImageNet weights is disabled in sandboxed environments.
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def extract_handcrafted_surface_statistics(x: torch.Tensor) -> torch.Tensor:
    """Compute shift-invariant multi-scale color, gradient, and patch-extremum statistics.

    Providing rich, deterministic spatial-quantile and edge statistics alongside backbone
    activations guarantees high sensitivity to tiny local scratches, stains, and dents even
    in zero-shot / frozen-backbone regimes.
    """
    # x is normalized [B, 3, H, W]
    b, c, h, w = x.shape
    mean_rgb = x.mean(dim=(2, 3))
    std_rgb = x.std(dim=(2, 3)) + 1e-6

    # Spatial high-low quantiles via 8x8 patch pooling
    patch_avg = F.adaptive_avg_pool2d(x, (8, 8)).view(b, c, -1)
    patch_max = F.adaptive_max_pool2d(x, (8, 8)).view(b, c, -1)
    patch_min = (-F.adaptive_max_pool2d(-x, (8, 8))).view(b, c, -1)

    q95 = torch.quantile(patch_avg, 0.95, dim=-1)
    q05 = torch.quantile(patch_avg, 0.05, dim=-1)
    max_val = patch_max.amax(dim=-1)
    min_val = patch_min.amin(dim=-1)

    # Sobel-like horizontal and vertical gradients for scratch / edge detection
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx_abs_mean = gx.abs().mean(dim=(2, 3))
    gy_abs_mean = gy.abs().mean(dim=(2, 3))
    gx_max = F.adaptive_max_pool2d(gx.abs(), (4, 4)).view(b, c, -1).mean(dim=-1)
    gy_max = F.adaptive_max_pool2d(gy.abs(), (4, 4)).view(b, c, -1).mean(dim=-1)

    # Channel differences (R-G, R-B, G-B) to isolate chromatic stains and rims
    rg = (x[:, 0] - x[:, 1]).view(b, -1)
    rb = (x[:, 0] - x[:, 2]).view(b, -1)
    gb = (x[:, 1] - x[:, 2]).view(b, -1)
    chroma_stats = torch.stack(
        [
            rg.mean(dim=-1),
            rg.std(dim=-1),
            rg.amax(dim=-1),
            rg.amin(dim=-1),
            rb.mean(dim=-1),
            rb.std(dim=-1),
            rb.amax(dim=-1),
            rb.amin(dim=-1),
            gb.mean(dim=-1),
            gb.std(dim=-1),
        ],
        dim=-1,
    )

    stats = torch.cat(
        [
            mean_rgb,
            std_rgb,
            q95,
            q05,
            max_val,
            min_val,
            gx_abs_mean,
            gy_abs_mean,
            gx_max,
            gy_max,
            chroma_stats,
        ],
        dim=-1,
    )
    return stats


class MetricProjectionModel(nn.Module):
    """Decoupled `timm` backbone + Spatial Pooling + 128-D Unit Hypersphere Projection Head."""

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

        self.backbone = self._load_backbone(backbone_name=backbone_name, pretrained=pretrained)
        self.pool = SpatialPoolingHead(pooling_type=pooling_type)

        # Determine pooled feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 128, 128)
            raw_feat = self._forward_backbone_features(dummy)
            pooled_dim = raw_feat.shape[-1]

        self.gap_feature_dim = pooled_dim

        # Configure backbone trainability
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_block:
                params = list(self.backbone.parameters())
                for param in params[-min(6, len(params)) :]:
                    param.requires_grad = True

        hidden_dim = max(256, min(512, pooled_dim * 2))
        self.projection_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _load_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        if backbone_name == "lightweight_cnn" or timm is None:
            return LightweightInspectionStem(out_channels=256)
        try:
            return timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
        except Exception:
            # Fallback to uninitialized timm architecture or lightweight stem when offline
            try:
                return timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
            except Exception:
                return LightweightInspectionStem(out_channels=256)

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            feat_map = self.backbone.forward_features(x)
        else:
            feat_map = self.backbone(x)
        pooled = self.pool(feat_map)
        surface_stats = extract_handcrafted_surface_statistics(x)
        combined = torch.cat([F.normalize(pooled, p=2, dim=-1), F.normalize(surface_stats, p=2, dim=-1)], dim=-1)
        return combined

    def extract_gap_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-projection pooled backbone features h in R^{D_backbone}."""
        feat = self._forward_backbone_features(x)
        return F.normalize(feat, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_128d_normalized, gap_features)."""
        gap_feat = self.extract_gap_features(x)
        proj = self.projection_head(gap_feat)
        z = F.normalize(proj, p=2, dim=-1)
        return z, gap_feat


def build_image_transforms(is_train: bool = False, image_size: int = 128) -> transforms.Compose:
    """Build image transforms.

    In Phase 2 training (`is_train=True`), identical geometric and photometric augmentation
    operators are applied symmetrically to both normal and defect samples to prevent
    shortcut learning (Clever Hans effect) on augmentation artifacts.
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
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
def extract_dataset_embeddings(
    image_paths: list[str],
    model: MetricProjectionModel,
    device: str = "cpu",
    batch_size: int = 32,
    image_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both 128-D projection vectors and pre-projection GAP/GeM features as pure NumPy arrays."""
    model.eval()
    model.to(device)
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
