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
from torchvision.transforms import functional as TF

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


PoolingType = Literal["gap", "gem", "avg_max"]


def _safe_quantile_last_dim(x: torch.Tensor, q: float) -> torch.Tensor:
    """MPS- and fp16-safe quantile along the last dimension via linear interpolation on sorted values."""
    n = x.shape[-1]
    if n == 0:
        return x.new_zeros(x.shape[:-1])
    if n == 1:
        return x.squeeze(-1)
    sorted_x, _ = torch.sort(x.float(), dim=-1)
    pos = float(np.clip(q, 0.0, 1.0)) * float(n - 1)
    lo = int(np.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = float(pos - lo)
    out = sorted_x[..., lo] * (1.0 - frac) + sorted_x[..., hi] * frac
    return out.to(dtype=x.dtype)


def _safe_adaptive_avg_pool2d(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Adaptive 2D average pooling with MPS non-divisible input size fallback."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if x.device.type == "mps" and (x.shape[-2] % oh != 0 or x.shape[-1] % ow != 0):
        th = max(oh, int(np.ceil(x.shape[-2] / float(oh))) * oh)
        tw = max(ow, int(np.ceil(x.shape[-1] / float(ow))) * ow)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(x, (oh, ow))


def _safe_adaptive_max_pool2d(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Adaptive 2D max pooling with MPS non-divisible input size fallback."""
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if x.device.type == "mps" and (x.shape[-2] % oh != 0 or x.shape[-1] % ow != 0):
        th = max(oh, int(np.ceil(x.shape[-2] / float(oh))) * oh)
        tw = max(ow, int(np.ceil(x.shape[-1] / float(ow))) * ow)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return F.adaptive_max_pool2d(x, (oh, ow))


def _safe_adaptive_avg_pool1d(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """Adaptive 1D average pooling with MPS non-divisible input size fallback."""
    ol = max(1, int(out_len))
    if x.device.type == "mps" and (x.shape[-1] % ol != 0):
        tl = max(ol, int(np.ceil(x.shape[-1] / float(ol))) * ol)
        x = F.interpolate(x, size=tl, mode="linear", align_corners=False)
    return F.adaptive_avg_pool1d(x, ol)


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
        p_safe = self.p.float().clamp(min=1.0, max=6.0)
        eps_safe = max(float(self.eps), 1e-5)
        x_f32 = x.float()
        if x.ndim == 3:
            # Vision Transformer token sequence [B, Tokens, C] -> pool over spatial tokens
            clamped = x_f32.clamp(min=eps_safe).pow(p_safe)
            return clamped.mean(dim=1).pow(1.0 / p_safe).to(dtype=x.dtype)
        # CNN feature map [B, C, H, W]
        clamped = x_f32.clamp(min=eps_safe).pow(p_safe)
        pooled = _safe_adaptive_avg_pool2d(clamped, (1, 1)).pow(1.0 / p_safe)
        return pooled.flatten(1).to(dtype=x.dtype)


class SpatialPoolingHead(nn.Module):
    """Configurable global pooling (GAP, GeM, or Avg+Max) + 4x4 CNN spatial-grid pooling for asymmetric workpieces."""

    def __init__(
        self,
        pooling_type: PoolingType = "gem",
        enable_spatial_grid: bool = True,
        spatial_grid_size: int = 4,
        spatial_channels: int = 32,
    ) -> None:
        super().__init__()
        self.pooling_type: PoolingType = pooling_type
        self.enable_spatial_grid: bool = bool(enable_spatial_grid)
        self.spatial_grid_size: int = max(2, int(spatial_grid_size))
        self.spatial_channels: int = max(8, int(spatial_channels))
        self.gem = GeMPooling(p=3.0)

    def forward_global(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute 1x1 global pooled semantic vector."""
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            if self.pooling_type == "gap":
                return feat.mean(dim=1)
            if self.pooling_type == "avg_max":
                return torch.cat([feat.mean(dim=1), feat.amax(dim=1)], dim=-1)
            return self.gem(feat)

        if self.pooling_type == "gap":
            return _safe_adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        if self.pooling_type == "avg_max":
            avg_p = _safe_adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            max_p = _safe_adaptive_max_pool2d(feat, (1, 1)).flatten(1)
            return torch.cat([avg_p, max_p], dim=-1)
        return self.gem(feat)

    def forward_spatial_grid(self, feat: torch.Tensor) -> torch.Tensor:
        """Retain (x, y) spatial coordinates via 4x4 grid x channel-grouped semantic pooling."""
        g = self.spatial_grid_size
        c_out = self.spatial_channels
        if feat.ndim == 2:
            return _safe_adaptive_avg_pool1d(feat.unsqueeze(1), g * g * c_out).flatten(1)
        if feat.ndim == 3:
            b, n_tok, _c = feat.shape
            spatial_tok = feat
            if n_tok > 1:
                side = int(round((n_tok - 1) ** 0.5))
                if side * side == n_tok - 1:
                    spatial_tok = feat[:, 1:, :]
            # [B, C, g*g] -> [B, g*g, c_out]
            pooled_tok = _safe_adaptive_avg_pool1d(spatial_tok.transpose(1, 2).contiguous(), g * g)
            grouped = _safe_adaptive_avg_pool1d(pooled_tok.transpose(1, 2).contiguous(), c_out)
            return grouped.reshape(b, g * g * c_out)

        b = feat.shape[0]
        # [B, C, H, W] -> [B, C, g*g] -> [B, g*g, c_out] -> [B, g*g*c_out]
        grid_hw = _safe_adaptive_avg_pool2d(feat, (g, g)).flatten(2)
        grouped = _safe_adaptive_avg_pool1d(grid_hw.transpose(1, 2).contiguous(), c_out)
        return grouped.reshape(b, g * g * c_out)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        global_vec = self.forward_global(feat)
        if not self.enable_spatial_grid:
            return global_vec
        spatial_vec = self.forward_spatial_grid(feat)
        return torch.cat([global_vec, spatial_vec], dim=-1)


class LightweightInspectionStem(nn.Module):
    """Deterministic multi-scale texture + color-contrast convolutional stem."""

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels
        with torch.random.fork_rng():
            torch.manual_seed(42)
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
    b, c, h, w = x.shape
    if b == 0:
        dummy = extract_handcrafted_surface_statistics(x.new_zeros((1, 3, 16, 16)))
        return x.new_zeros((0, dummy.shape[-1]))

    if c == 1:
        x = x.expand(-1, 3, -1, -1)
    elif c == 2:
        x = torch.cat([x, x[:, :1]], dim=1)
    elif c > 3:
        x = x[:, :3]
    c = 3

    if h < 2 or w < 2:
        x = F.pad(x, (0, max(0, 2 - w), 0, max(0, 2 - h)), mode="replicate")

    mean_rgb = x.mean(dim=(2, 3))
    std_rgb = x.var(dim=(2, 3), unbiased=False).clamp(min=1e-12).sqrt()

    # 8x8 spatial grid mean and max pooling [B, 3*64 = 192]
    patch_8x8 = _safe_adaptive_avg_pool2d(x, (8, 8))
    grid_avg_8x8 = patch_8x8.reshape(b, -1)
    grid_max_4x4 = _safe_adaptive_max_pool2d(x, (4, 4)).reshape(b, -1)
    grid_min_4x4 = (-_safe_adaptive_max_pool2d(-x, (4, 4))).reshape(b, -1)

    # Local patch variance on 6x6 grid via E[X^2] - (E[X])^2 [B, 3*36 = 108]
    e_x = _safe_adaptive_avg_pool2d(x, (6, 6))
    e_x2 = _safe_adaptive_avg_pool2d(x * x, (6, 6))
    local_std_6x6 = (e_x2 - e_x * e_x).clamp(min=1e-12).sqrt().reshape(b, -1)

    # Spatial high-low quantiles across 8x8 patches (MPS- & fp16-safe)
    patch_avg = patch_8x8.reshape(b, c, -1)
    q95 = _safe_quantile_last_dim(patch_avg, 0.95)
    q05 = _safe_quantile_last_dim(patch_avg, 0.05)

    # Horizontal & vertical edge gradients on 6x6 spatial grid [B, 3*36 = 108]
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    gx_grid = _safe_adaptive_avg_pool2d(gx, (6, 6)).reshape(b, -1)
    gy_grid = _safe_adaptive_avg_pool2d(gy, (6, 6)).reshape(b, -1)
    gx_max = _safe_adaptive_max_pool2d(gx, (4, 4)).reshape(b, c, -1).amax(dim=-1)
    gy_max = _safe_adaptive_max_pool2d(gy, (4, 4)).reshape(b, c, -1).amax(dim=-1)

    # Chromatic differences (R-G, R-B, G-B) on 4x4 grid + global stats
    rg = x[:, 0:1] - x[:, 1:2]
    rb = x[:, 0:1] - x[:, 2:3]
    gb = x[:, 1:2] - x[:, 2:3]
    chroma_grid = torch.cat(
        [
            _safe_adaptive_avg_pool2d(rg, (4, 4)).reshape(b, -1),
            _safe_adaptive_avg_pool2d(rb, (4, 4)).reshape(b, -1),
            _safe_adaptive_avg_pool2d(gb, (4, 4)).reshape(b, -1),
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
        enable_spatial_grid: bool = True,
        input_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pooling_type: PoolingType = pooling_type
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_block = unfreeze_last_block
        self.enable_spatial_grid = bool(enable_spatial_grid)
        self.input_dropout_rate = float(np.clip(input_dropout, 0.0, 0.5))

        self.backbone = self._load_backbone(backbone_name=backbone_name, pretrained=pretrained)
        self.pool = SpatialPoolingHead(
            pooling_type=pooling_type,
            enable_spatial_grid=self.enable_spatial_grid,
        )

        # Determine raw feature dimension dynamically
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 128, 128)
            raw_feat = self._extract_raw_concat_features(dummy)
            pooled_dim = raw_feat.shape[-1]

        self.gap_feature_dim = pooled_dim

        # Dataset standardization buffers (calibrated strictly on train Normal samples)
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

        # Input feature dropout suppresses shortcut memorization on small defect counts
        self.feat_dropout = nn.Dropout(p=self.input_dropout_rate)

        hidden_dim = 384
        self.projection_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def train(self, mode: bool = True) -> MetricProjectionModel:
        """Keep frozen backbone (and any BatchNorm layers under partial freeze) in `.eval()` mode during training."""
        super().train(mode)
        if self.freeze_backbone:
            if not self.unfreeze_last_block:
                self.backbone.eval()
            else:
                for mod in self.backbone.modules():
                    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                        mod.eval()
        return self

    def _load_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        if backbone_name == "lightweight_cnn" or timm is None:
            return LightweightInspectionStem(out_channels=256)
        for use_pre in ([True, False] if pretrained else [False]):
            try:
                return timm.create_model(
                    backbone_name, pretrained=use_pre, num_classes=0, global_pool=""
                )
            except Exception:
                continue
        return LightweightInspectionStem(out_channels=256)

    def _extract_raw_concat_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            dim = getattr(self, "gap_feature_dim", 0)
            return x.new_zeros((0, dim))

        x_backbone = x
        patch_embed = getattr(self.backbone, "patch_embed", None)
        target_sz = getattr(patch_embed, "img_size", None) if patch_embed is not None else None
        if target_sz is None:
            default_cfg = getattr(self.backbone, "default_cfg", None) or {}
            inp_sz = default_cfg.get("input_size")
            if (
                isinstance(inp_sz, (tuple, list))
                and len(inp_sz) == 3
                and patch_embed is not None
            ):
                target_sz = (int(inp_sz[1]), int(inp_sz[2]))
        if isinstance(target_sz, int):
            target_sz = (target_sz, target_sz)
        if (
            isinstance(target_sz, (tuple, list))
            and len(target_sz) == 2
            and (x.shape[-2], x.shape[-1]) != (int(target_sz[0]), int(target_sz[1]))
        ):
            x_backbone = F.interpolate(
                x,
                size=(int(target_sz[0]), int(target_sz[1])),
                mode="bilinear",
                align_corners=False,
            )
        was_bb_training = self.backbone.training
        if x_backbone.shape[0] == 1 and was_bb_training:
            self.backbone.eval()
        try:
            if hasattr(self.backbone, "forward_features"):
                feat_map = self.backbone.forward_features(x_backbone)
            else:
                feat_map = self.backbone(x_backbone)
        finally:
            if x_backbone.shape[0] == 1 and was_bb_training:
                self.backbone.train()
        if feat_map.ndim == 4 and str(getattr(self.backbone, "output_fmt", "")).upper() == "NHWC":
            feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        elif feat_map.ndim == 3:
            b_sz, n_tok, c_dim = feat_map.shape
            n_prefix = (
                0
                if bool(getattr(self.backbone, "no_embed_class", False))
                else int(getattr(self.backbone, "num_prefix_tokens", 1))
            )
            if 0 < n_prefix < n_tok:
                rem = n_tok - n_prefix
                side = int(round(rem**0.5))
                if side * side == rem and side >= 2:
                    feat_map = (
                        feat_map[:, n_prefix:, :]
                        .transpose(1, 2)
                        .reshape(b_sz, c_dim, side, side)
                        .contiguous()
                    )
                else:
                    feat_map = feat_map[:, n_prefix:, :]
            else:
                side = int(round(n_tok**0.5))
                if side * side == n_tok and side >= 2:
                    feat_map = (
                        feat_map.transpose(1, 2).reshape(b_sz, c_dim, side, side).contiguous()
                    )
        global_pooled = self.pool.forward_global(feat_map)
        surface_stats = extract_handcrafted_surface_statistics(x)
        if self.enable_spatial_grid:
            spatial_pooled = self.pool.forward_spatial_grid(feat_map)
            combined = torch.cat(
                [
                    F.normalize(global_pooled, p=2, dim=-1, eps=1e-6),
                    0.75 * F.normalize(spatial_pooled, p=2, dim=-1, eps=1e-6),
                    F.normalize(surface_stats, p=2, dim=-1, eps=1e-6),
                ],
                dim=-1,
            )
            return combined
        combined = torch.cat(
            [
                F.normalize(global_pooled, p=2, dim=-1, eps=1e-6),
                F.normalize(surface_stats, p=2, dim=-1, eps=1e-6),
            ],
            dim=-1,
        )
        return combined

    @torch.no_grad()
    def calibrate_feature_stats(
        self,
        raw_features: np.ndarray | torch.Tensor,
        subcluster_labels: np.ndarray | list[int] | torch.Tensor | None = None,
    ) -> None:
        """Calibrate per-dimension mean and pooled within-subcluster std strictly on Normal reference samples."""
        if isinstance(raw_features, np.ndarray):
            feat_t = torch.from_numpy(raw_features).to(
                device=self.feat_mean.device, dtype=torch.float32
            )
        else:
            feat_t = raw_features.detach().to(
                device=self.feat_mean.device, dtype=torch.float32
            )
        feat_t = torch.nan_to_num(feat_t, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = feat_t.shape[0]
        if n_samples < 2:
            return

        mean = feat_t.mean(dim=0, keepdim=True)
        std = feat_t.var(dim=0, keepdim=True, unbiased=False).clamp(min=0.0).sqrt()

        if subcluster_labels is not None:
            if isinstance(subcluster_labels, torch.Tensor):
                sub_arr = subcluster_labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
            else:
                sub_arr = np.asarray(subcluster_labels, dtype=np.int64).reshape(-1)
            if len(sub_arr) == n_samples:
                valid_ids = [int(cid) for cid in sorted(set(sub_arr.tolist())) if cid >= 0]
                if len(valid_ids) >= 2:
                    mode_means: list[torch.Tensor] = []
                    sq_dev_sum = torch.zeros_like(mean)
                    total_counted = 0
                    for cid in valid_ids:
                        mask_c = torch.from_numpy(sub_arr == cid).to(device=feat_t.device)
                        cnt = int(mask_c.sum().item())
                        if cnt >= 1:
                            c_feats = feat_t[mask_c]
                            c_mu = c_feats.mean(dim=0, keepdim=True)
                            mode_means.append(c_mu)
                            sq_dev_sum = sq_dev_sum + (c_feats - c_mu).pow(2).sum(
                                dim=0, keepdim=True
                            )
                            total_counted += cnt
                    if len(mode_means) >= 2 and total_counted > len(mode_means):
                        # Balanced mode mean + pooled within-subcluster std (avoids inter-cluster std inflation)
                        mean = torch.cat(mode_means, dim=0).mean(dim=0, keepdim=True)
                        dof = max(1, total_counted - len(mode_means))
                        std = (sq_dev_sum / float(dof)).clamp(min=0.0).sqrt()

        # Dimensions with near-zero variance across normals are constant channels;
        # set their scale to 1.0 after subtracting mean so they stay 0.0 instead of amplifying float noise.
        valid_std = torch.where(std > 1e-5, std, torch.ones_like(std))

        self.feat_mean.copy_(mean)
        self.feat_std.copy_(valid_std)
        self.is_calibrated.fill_(True)

    def standardize_features(self, raw_feat: torch.Tensor) -> torch.Tensor:
        """Center and standardize raw features before L2 normalization."""
        if raw_feat.shape[0] == 0:
            return raw_feat.new_zeros((0, self.gap_feature_dim))
        raw_feat = torch.nan_to_num(raw_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(self.is_calibrated.item()):
            centered = (raw_feat - self.feat_mean) / (self.feat_std + 1e-5)
            centered = centered.clamp(min=-10.0, max=10.0)
            return F.normalize(centered, p=2, dim=-1, eps=1e-6)
        if raw_feat.shape[0] >= 4:
            mean = raw_feat.mean(dim=0, keepdim=True).detach()
            std = raw_feat.var(dim=0, keepdim=True, unbiased=False).clamp(min=0.0).sqrt().detach()
            valid_std = torch.where(std > 1e-5, std, torch.ones_like(std))
            centered = ((raw_feat - mean) / (valid_std + 1e-5)).clamp(min=-10.0, max=10.0)
            return F.normalize(centered, p=2, dim=-1, eps=1e-6)
        return F.normalize(raw_feat, p=2, dim=-1, eps=1e-6)

    def extract_gap_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract standardized pre-projection features h in R^{D_backbone}."""
        if x.shape[0] == 0:
            return x.new_zeros((0, self.gap_feature_dim))
        raw_feat = self._extract_raw_concat_features(x)
        return self.standardize_features(raw_feat)

    def forward_from_gap_features(self, gap_feat: torch.Tensor) -> torch.Tensor:
        """Project standardized GAP/surface features onto the 128-D unit hypersphere S^{127}."""
        if gap_feat.shape[0] == 0:
            return gap_feat.new_zeros((0, self.embedding_dim))
        dropped = self.feat_dropout(gap_feat)
        # Handle single-sample inference when BatchNorm1d is in train mode
        if dropped.shape[0] == 1 and self.training:
            self.projection_head.eval()
            try:
                proj = self.projection_head(dropped)
            finally:
                self.projection_head.train()
        else:
            proj = self.projection_head(dropped)
        return F.normalize(proj, p=2, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_128d_normalized, gap_features)."""
        if x.shape[0] == 0:
            return x.new_zeros((0, self.embedding_dim)), x.new_zeros((0, self.gap_feature_dim))
        gap_feat = self.extract_gap_features(x)
        z = self.forward_from_gap_features(gap_feat)
        return z, gap_feat


def build_image_transforms(is_train: bool = False, image_size: int = 128) -> transforms.Compose:
    """Build position-preserving transforms for asymmetric workpieces (micro-shift & brightness/contrast only; NO flips/rotations)."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=0.0, translate=(0.02, 0.02)),
                transforms.ColorJitter(brightness=0.04, contrast=0.04),
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


def build_asymmetric_micro_augment_transforms(image_size: int = 128) -> list[transforms.Compose]:
    """Build deterministic position-preserving micro-augmentation pipelines (tiny shift + brightness/contrast only)."""
    shift_px = max(1, int(round(image_size * 0.02)))
    specs = [
        (shift_px, 0, 1.03, 0.98),
        (-shift_px, shift_px, 0.97, 1.03),
        (0, -shift_px, 1.02, 1.02),
    ]
    pipelines: list[transforms.Compose] = []
    for dx, dy, br, ct in specs:
        pipelines.append(
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.Lambda(
                        lambda img, _dx=dx, _dy=dy, _br=br, _ct=ct: TF.adjust_contrast(
                            TF.adjust_brightness(
                                TF.affine(
                                    img,
                                    angle=0.0,
                                    translate=[_dx, _dy],
                                    scale=1.0,
                                    shear=[0.0, 0.0],
                                ),
                                _br,
                            ),
                            _ct,
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )
    return pipelines


@torch.no_grad()
def extract_micro_augmented_gap_bank(
    image_paths: list[str],
    sample_indices: list[int],
    model: MetricProjectionModel,
    device: str = "cpu",
    batch_size: int = 32,
    image_size: int = 128,
) -> dict[int, torch.Tensor]:
    """Pre-extract position-preserving micro-augmented standardized GAP features for cached-backbone training."""
    if not sample_indices:
        return {}
    step = max(1, int(batch_size))
    model.eval()
    model.to(device)
    aug_pipelines = build_asymmetric_micro_augment_transforms(image_size=image_size)
    per_view_feats: list[torch.Tensor] = []

    for tf in aug_pipelines:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(sample_indices), step):
            sub_idx = sample_indices[start : start + step]
            tensors = [load_image_tensor(image_paths[i], tf) for i in sub_idx]
            batch_x = torch.stack(tensors, dim=0).to(device)
            chunks.append(model.extract_gap_features(batch_x))
        per_view_feats.append(torch.cat(chunks, dim=0))

    # Stack views -> [N_target, N_views, D_gap]
    stacked = torch.stack(per_view_feats, dim=1)
    return {int(sample_indices[pos]): stacked[pos] for pos in range(len(sample_indices))}


def load_image_tensor(image_path: str | Path, transform: transforms.Compose) -> torch.Tensor:
    """Load an image from disk (supporting 8-bit, 16-bit industrial TIFF/PNG, and float modes) and apply `transform`."""
    with Image.open(image_path) as img:
        if img.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
            arr = np.asarray(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            max_v = float(arr.max()) if arr.size > 0 else 0.0
            min_v = float(arr.min()) if arr.size > 0 else 0.0
            if img.mode.startswith("I;16") and min_v >= 0.0 and max_v > 255.0:
                arr_u8 = np.clip(arr / 257.0, 0.0, 255.0).astype(np.uint8)
            elif max_v > min_v:
                if min_v >= 0.0 and max_v <= 1.0:
                    arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                elif min_v >= 0.0 and max_v <= 255.0:
                    arr_u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                else:
                    arr_u8 = np.clip(
                        (arr - min_v) / (max_v - min_v) * 255.0, 0.0, 255.0
                    ).astype(np.uint8)
            else:
                arr_u8 = np.zeros_like(arr, dtype=np.uint8)
            rgb = Image.fromarray(arr_u8).convert("RGB")
        else:
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
    step = max(1, int(batch_size))
    model.eval()
    model.to(device)
    transform = build_image_transforms(is_train=False, image_size=image_size)
    raw_list: list[np.ndarray] = []

    for start in range(0, len(image_paths), step):
        batch_paths = image_paths[start : start + step]
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
    step = max(1, int(batch_size))
    model.eval()
    model.to(device)

    # Automatically calibrate standardization stats across the dataset if not yet calibrated
    if not bool(model.is_calibrated.item()) and len(image_paths) >= 2:
        raw_all = extract_raw_uncalibrated_features(
            image_paths=image_paths,
            model=model,
            device=device,
            batch_size=step,
            image_size=image_size,
        )
        model.calibrate_feature_stats(raw_all)

    transform = build_image_transforms(is_train=False, image_size=image_size)
    z_list: list[np.ndarray] = []
    gap_list: list[np.ndarray] = []

    for start in range(0, len(image_paths), step):
        batch_paths = image_paths[start : start + step]
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
