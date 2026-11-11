"""Phase 3 Boundary Hard-Sample Synthesis: Mask-Free Residual-Saliency Defect Transfer & Anomalous Mixup."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_gaussian_feather_mask(
    height: int,
    width: int,
    center_y: int,
    center_x: int,
    sigma_ratio: float = 0.16,
    peak_opacity: float = 0.82,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a smooth 2D Gaussian feathering window G(u, v) in [0, peak_opacity] of shape (1, H, W)."""
    yy = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    sigma_y = max(2.0, height * sigma_ratio)
    sigma_x = max(2.0, width * sigma_ratio)
    dist_sq = ((yy - float(center_y)) ** 2) / (2.0 * sigma_y**2) + (
        (xx - float(center_x)) ** 2
    ) / (2.0 * sigma_x**2)
    mask = torch.exp(-dist_sq) * float(np.clip(peak_opacity, 0.3, 1.0))
    return mask.unsqueeze(0)


@torch.no_grad()
def synthesize_defect_transfer_images(
    normal_images: torch.Tensor,
    defect_images: torch.Tensor,
    defect_labels: torch.Tensor,
    num_synthetic: int | None = None,
    sigma_ratio: float = 0.16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize pseudo-defects via Unsupervised Residual-Saliency Defect Transfer.

    Because folder-uploaded inspection datasets do not include pixel segmentation masks:
    1. For each source defect image `I_bad`, find its closest normal reference `I_ref` in `normal_images`.
    2. Compute the smoothed spatial L1 residual map `R = AvgPool(|I_bad - I_ref|)`.
    3. Locate the peak defect activation `(u*, v*) = argmax R(u, v)`.
    4. Blend the localized defect patch onto a fresh target normal image `I_good` using a
       smooth 2D Gaussian feathering mask `G(u, v)` (with optional slight random shift).

    Returns:
        synth_images: Tensor [K, 3, H, W]
        synth_labels: Tensor [K] (inherits source defect class label_id for Class-Balanced Weight W_c)
    """
    if normal_images.shape[0] == 0 or defect_images.shape[0] == 0:
        return (
            normal_images.new_zeros((0, *normal_images.shape[1:])),
            defect_labels.new_zeros((0,), dtype=torch.long),
        )

    b_norm = normal_images.shape[0]
    b_def = defect_images.shape[0]
    _, _, h, w = defect_images.shape
    k_out = b_def if num_synthetic is None else min(max(1, int(num_synthetic)), b_def * 2)

    # Low-resolution pooling for fast nearest-normal template matching
    norm_low = F.adaptive_avg_pool2d(normal_images, (16, 16)).flatten(1)
    def_low = F.adaptive_avg_pool2d(defect_images, (16, 16)).flatten(1)
    l1_dists = torch.cdist(def_low, norm_low, p=1)  # [B_def, B_norm]
    nearest_norm_idx = torch.argmin(l1_dists, dim=1)  # [B_def]

    synth_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    for idx in range(k_out):
        d_i = idx % b_def
        img_bad = defect_images[d_i]
        img_ref = normal_images[int(nearest_norm_idx[d_i].item())]

        # Compute spatial residual saliency map [1, 1, H, W]
        residual = (img_bad - img_ref).abs().mean(dim=0, keepdim=True).unsqueeze(0)
        smoothed = F.avg_pool2d(residual, kernel_size=11, stride=1, padding=5).squeeze()

        flat_max_idx = int(torch.argmax(smoothed).item())
        peak_y = flat_max_idx // w
        peak_x = flat_max_idx % w

        mask = build_gaussian_feather_mask(
            height=h,
            width=w,
            center_y=peak_y,
            center_x=peak_x,
            sigma_ratio=sigma_ratio,
            peak_opacity=0.82,
            device=defect_images.device,
        )

        # Pick a target normal image (different from reference when possible)
        target_norm_i = ( int(nearest_norm_idx[d_i].item()) + 1 + idx ) % b_norm
        img_good = normal_images[target_norm_i]

        blended = (1.0 - mask) * img_good + mask * img_bad
        synth_list.append(blended)
        label_list.append(defect_labels[d_i])

    return torch.stack(synth_list, dim=0), torch.stack(label_list, dim=0)


@torch.no_grad()
def synthesize_anomalous_mixup_images(
    normal_images: torch.Tensor,
    defect_images: torch.Tensor,
    defect_labels: torch.Tensor,
    alpha_min: float = 0.55,
    alpha_max: float = 0.75,
    num_synthetic: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize boundary hard anomalies via Anomalous Mixup:

        I_mix = alpha * I_good + (1 - alpha) * I_bad,   alpha ~ Uniform(alpha_min, alpha_max)

    Default alpha in [0.55, 0.75] places `I_mix` on the Normal/Anomaly decision boundary
    without making it indistinguishable from pure Normal images.
    """
    if normal_images.shape[0] == 0 or defect_images.shape[0] == 0:
        return (
            normal_images.new_zeros((0, *normal_images.shape[1:])),
            defect_labels.new_zeros((0,), dtype=torch.long),
        )

    b_norm = normal_images.shape[0]
    b_def = defect_images.shape[0]
    k_out = b_def if num_synthetic is None else min(max(1, int(num_synthetic)), b_def * 2)

    a_low = float(np.clip(min(alpha_min, alpha_max), 0.30, 0.90))
    a_high = float(np.clip(max(alpha_min, alpha_max), a_low + 0.01, 0.95))

    alphas = torch.empty((k_out, 1, 1, 1), device=defect_images.device, dtype=defect_images.dtype)
    alphas.uniform_(a_low, a_high)

    norm_picks = normal_images[torch.arange(k_out, device=normal_images.device) % b_norm]
    def_picks = defect_images[torch.arange(k_out, device=defect_images.device) % b_def]
    lbl_picks = defect_labels[torch.arange(k_out, device=defect_labels.device) % b_def]

    mixed = alphas * norm_picks + (1.0 - alphas) * def_picks
    return mixed, lbl_picks


@torch.no_grad()
def synthesize_boundary_gap_features(
    normal_gap: torch.Tensor,
    defect_gap: torch.Tensor,
    defect_labels: torch.Tensor,
    alpha_min: float = 0.55,
    alpha_max: float = 0.75,
    transfer_ratio: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize Defect-Transfer and Anomalous-Mixup standardized GAP features.

    1. Feature-space Residual Defect Transfer:
       Identifies the top `transfer_ratio` (e.g. 30%) dimensions where `|h_bad - h_nearest_good|`
       is largest (corresponding to localized spatial patch & GeM defect channels) and splices
       those abnormal residual activations onto a fresh normal feature vector `h_good`.
    2. Feature-space Anomalous Mixup:
       Interpolates `alpha * h_good + (1 - alpha) * h_bad` with `alpha in [alpha_min, alpha_max]`
       and L2-normalizes onto the standardized feature sphere.

    Returns:
        synth_gap: Tensor [K_total, D_gap]
        synth_labels: Tensor [K_total] (source defect class IDs for Class-Balanced Weight W_c)
    """
    if normal_gap.shape[0] == 0 or defect_gap.shape[0] == 0:
        return (
            normal_gap.new_zeros((0, normal_gap.shape[-1])),
            defect_labels.new_zeros((0,), dtype=torch.long),
        )

    b_norm, feat_dim = normal_gap.shape
    b_def = defect_gap.shape[0]
    device = defect_gap.device

    # Pair each defect with its closest normal feature (reference) and a distinct target normal
    sim_dn = defect_gap @ normal_gap.T  # [B_def, B_norm]
    ref_idx = torch.argmax(sim_dn, dim=1)
    target_idx = (ref_idx + 1 + torch.arange(b_def, device=device)) % b_norm

    h_ref = normal_gap[ref_idx]
    h_target = normal_gap[target_idx]

    # 1. Residual-Saliency Feature Defect Transfer
    residual = (defect_gap - h_ref).abs()
    k_dims = max(8, int(round(feat_dim * float(np.clip(transfer_ratio, 0.10, 0.60)))))
    _, top_dim_indices = torch.topk(residual, k=k_dims, dim=-1, largest=True)

    transfer_mask = torch.zeros_like(defect_gap)
    transfer_mask.scatter_(dim=-1, index=top_dim_indices, value=0.85)
    h_transfer = F.normalize(
        (1.0 - transfer_mask) * h_target + transfer_mask * defect_gap, p=2, dim=-1
    )

    # 2. Boundary Anomalous Mixup (default alpha in [0.55, 0.75])
    a_low = float(np.clip(min(alpha_min, alpha_max), 0.30, 0.90))
    a_high = float(np.clip(max(alpha_min, alpha_max), a_low + 0.01, 0.95))
    alphas = torch.empty((b_def, 1), device=device, dtype=defect_gap.dtype).uniform_(a_low, a_high)
    h_mix_target = normal_gap[(target_idx + 1) % b_norm]
    h_mixup = F.normalize(alphas * h_mix_target + (1.0 - alphas) * defect_gap, p=2, dim=-1)

    synth_gap = torch.cat([h_transfer, h_mixup], dim=0)
    synth_labels = torch.cat([defect_labels, defect_labels], dim=0)
    return synth_gap, synth_labels
