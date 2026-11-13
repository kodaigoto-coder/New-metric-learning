"""Phase 3 Boundary Hard-Sample Synthesis: Position-Preserving Z-Score Saliency Defect Transfer & Same-Subcluster Residual Injection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_gaussian_feather_mask(
    height: int,
    width: int,
    center_y: int,
    center_x: int,
    sigma_ratio: float = 0.10,
    peak_opacity: float = 0.88,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a localized 2D Gaussian feathering window G(u, v) in [0, peak_opacity] of shape (1, H, W)."""
    h_eff = max(1, int(height))
    w_eff = max(1, int(width))
    yy = torch.arange(h_eff, device=device, dtype=torch.float32).reshape(h_eff, 1)
    xx = torch.arange(w_eff, device=device, dtype=torch.float32).reshape(1, w_eff)
    sigma_y = max(2.0, h_eff * float(sigma_ratio))
    sigma_x = max(2.0, w_eff * float(sigma_ratio))
    dist_sq = ((yy - float(center_y)) ** 2) / (2.0 * sigma_y**2) + (
        (xx - float(center_x)) ** 2
    ) / (2.0 * sigma_x**2)
    mask = torch.exp(-dist_sq) * float(np.clip(peak_opacity, 0.3, 1.0))
    return mask.unsqueeze(0)


def _select_same_subcluster_normal_pool(
    ref_norm_idx: int,
    norm_low_or_feat: torch.Tensor,
    normal_subclusters: torch.Tensor | np.ndarray | list[int] | None,
    metric: str = "l1",
) -> torch.Tensor:
    """Return candidate normal indices restricted to the reference normal's sub-cluster and local neighborhood."""
    b_norm = norm_low_or_feat.shape[0]
    device = norm_low_or_feat.device
    all_idx = torch.arange(b_norm, device=device)
    if b_norm <= 1:
        return all_idx

    cand_mask = torch.ones(b_norm, dtype=torch.bool, device=device)
    if normal_subclusters is not None:
        if isinstance(normal_subclusters, torch.Tensor):
            sub_t = normal_subclusters.to(device=device, dtype=torch.long).reshape(-1)
        else:
            sub_t = torch.as_tensor(normal_subclusters, device=device, dtype=torch.long).reshape(-1)
        if sub_t.numel() == b_norm:
            ref_sub = int(sub_t[ref_norm_idx].item())
            same_sub = sub_t == ref_sub
            if int(same_sub.sum().item()) >= 1:
                cand_mask = same_sub

    cand_indices = all_idx[cand_mask]
    if cand_indices.numel() <= 1:
        return cand_indices

    ref_vec = norm_low_or_feat[ref_norm_idx : ref_norm_idx + 1]
    cand_vecs = norm_low_or_feat[cand_indices]
    k_nn = min(int(cand_indices.numel()), max(2, min(6, int(cand_indices.numel()))))

    if metric == "cosine":
        ref_u = F.normalize(ref_vec.float(), p=2, dim=-1, eps=1e-6)
        cand_u = F.normalize(cand_vecs.float(), p=2, dim=-1, eps=1e-6)
        sims = (cand_u @ ref_u.T).squeeze(-1)
        top_vals, top_pos = torch.topk(sims, k=k_nn, largest=True)
        close_mask = top_vals >= 0.45
        if torch.any(close_mask):
            return cand_indices[top_pos[close_mask]]
        return cand_indices[top_pos[:1]]

    # Explicit L1 distance (MPS- & fp16-safe; avoids torch.cdist p=1 fallback bug)
    dists = (cand_vecs.float() - ref_vec.float()).abs().sum(dim=-1)
    _, top_pos = torch.topk(dists, k=k_nn, largest=False)
    return cand_indices[top_pos]


@torch.no_grad()
def synthesize_defect_transfer_images(
    normal_images: torch.Tensor,
    defect_images: torch.Tensor,
    defect_labels: torch.Tensor,
    num_synthetic: int | None = None,
    sigma_ratio: float = 0.10,
    normal_subclusters: torch.Tensor | np.ndarray | list[int] | None = None,
    max_shift_px: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize position-preserving pseudo-defects via Z-Score Residual-Saliency Defect Transfer.

    Designed for asymmetric workpieces where spatial coordinates `(x, y)` carry defect meaning:
    1. Match each source defect `I_bad` to its closest normal template `I_ref`.
    2. Estimate per-pixel normal standard deviation `sigma_norm(u, v)` across normals in the same
       sub-cluster so alignment/contour jitter (high `sigma_norm`) is suppressed.
    3. Locate the defect peak `(u*, v*) = argmax AvgPool(|I_bad - I_ref| / (sigma_norm + 0.03))`.
    4. Transplant the localized defect patch onto a target normal from the SAME normal sub-cluster
       at the SAME spatial coordinate `(u* + du, v* + dv)` (`|du|, |dv| <= max_shift_px`).
    """
    device = defect_images.device
    dtype = defect_images.dtype
    if normal_images.shape[0] == 0 or defect_images.shape[0] == 0:
        return (
            defect_images.new_zeros((0, *defect_images.shape[1:])),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    normal_images = normal_images.to(device=device, dtype=dtype)
    defect_labels = defect_labels.to(device=device, dtype=torch.long).reshape(-1)
    b_def = defect_images.shape[0]
    _, _, h, w = defect_images.shape
    if normal_images.shape[-2:] != (h, w):
        normal_images = F.interpolate(
            normal_images, size=(h, w), mode="bilinear", align_corners=False
        )
    if defect_labels.numel() < b_def:
        pad_lbl = defect_labels[-1] if defect_labels.numel() > 0 else torch.tensor(1, device=device)
        defect_labels = F.pad(defect_labels, (0, b_def - defect_labels.numel()), value=int(pad_lbl.item()))

    k_out = b_def if num_synthetic is None else min(max(1, int(num_synthetic)), b_def * 2)

    # Low-resolution pooling for nearest-normal template matching (MPS-safe)
    pool_hw = (min(16, h), min(16, w))
    if device.type == "mps" and (h % pool_hw[0] != 0 or w % pool_hw[1] != 0):
        th = max(pool_hw[0], int(np.ceil(h / float(pool_hw[0]))) * pool_hw[0])
        tw = max(pool_hw[1], int(np.ceil(w / float(pool_hw[1]))) * pool_hw[1])
        norm_low = F.adaptive_avg_pool2d(
            F.interpolate(normal_images.float(), size=(th, tw), mode="bilinear", align_corners=False),
            pool_hw,
        ).flatten(1)
        def_low = F.adaptive_avg_pool2d(
            F.interpolate(defect_images.float(), size=(th, tw), mode="bilinear", align_corners=False),
            pool_hw,
        ).flatten(1)
    else:
        norm_low = F.adaptive_avg_pool2d(normal_images.float(), pool_hw).flatten(1)
        def_low = F.adaptive_avg_pool2d(defect_images.float(), pool_hw).flatten(1)

    l1_dists = (def_low.unsqueeze(1) - norm_low.unsqueeze(0)).abs().sum(dim=-1)  # [B_def, B_norm]
    nearest_norm_idx = torch.argmin(l1_dists, dim=1)  # [B_def]

    val_min = float(min(normal_images.min().item(), defect_images.min().item()))
    val_max = float(max(normal_images.max().item(), defect_images.max().item()))
    shift_lim = max(0, int(max_shift_px))

    synth_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    for idx in range(k_out):
        d_i = idx % b_def
        ref_i = int(nearest_norm_idx[d_i].item())
        img_bad = defect_images[d_i]
        img_ref = normal_images[ref_i]

        cand_pool = _select_same_subcluster_normal_pool(
            ref_norm_idx=ref_i,
            norm_low_or_feat=norm_low,
            normal_subclusters=normal_subclusters,
            metric="l1",
        )
        if cand_pool.numel() >= 2:
            sigma_norm = (
                normal_images[cand_pool]
                .float()
                .std(dim=0, unbiased=False)
                .mean(dim=0, keepdim=True)
                .unsqueeze(0)
                .to(dtype=dtype)
            )
        else:
            sigma_norm = normal_images.new_full((1, 1, h, w), 0.05)

        # Z-score residual saliency suppresses part-edge alignment jitter
        raw_res = (img_bad - img_ref).abs().mean(dim=0, keepdim=True).unsqueeze(0)
        z_saliency = raw_res / (sigma_norm + 0.03)
        smoothed = F.avg_pool2d(z_saliency.float(), kernel_size=7, stride=1, padding=3).reshape(h, w)
        smoothed = torch.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0)

        flat_max_idx = int(torch.argmax(smoothed).item())
        peak_y = flat_max_idx // max(1, w)
        peak_x = flat_max_idx % max(1, w)

        if shift_lim > 0:
            dy = int(torch.randint(-shift_lim, shift_lim + 1, (1,), device=device).item())
            dx = int(torch.randint(-shift_lim, shift_lim + 1, (1,), device=device).item())
        else:
            dy, dx = 0, 0

        target_y = int(np.clip(peak_y + dy, 0, max(0, h - 1)))
        target_x = int(np.clip(peak_x + dx, 0, max(0, w - 1)))
        eff_dy = target_y - peak_y
        eff_dx = target_x - peak_x

        mask = build_gaussian_feather_mask(
            height=h,
            width=w,
            center_y=target_y,
            center_x=target_x,
            sigma_ratio=sigma_ratio,
            peak_opacity=0.88,
            device=device,
        ).to(dtype=dtype)

        # Pick target normal strictly from the same normal sub-cluster / local neighborhood
        if cand_pool.numel() > 1:
            non_ref = cand_pool[cand_pool != ref_i]
            pick_pool = non_ref if non_ref.numel() > 0 else cand_pool
            target_norm_i = int(pick_pool[idx % int(pick_pool.numel())].item())
        else:
            target_norm_i = ref_i
        img_good = normal_images[target_norm_i]

        shifted_bad = torch.roll(img_bad, shifts=(eff_dy, eff_dx), dims=(1, 2))
        shifted_delta = torch.roll(img_bad - img_ref, shifts=(eff_dy, eff_dx), dims=(1, 2))
        blended_patch = (1.0 - mask) * img_good + mask * shifted_bad
        residual_patch = (img_good + mask * shifted_delta).clamp(min=val_min, max=val_max)
        blended = (0.5 * blended_patch + 0.5 * residual_patch).to(dtype=dtype)

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
    normal_subclusters: torch.Tensor | np.ndarray | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy-compatible helper: delegates to same-subcluster localized residual transfer (global cross-phase Mixup disabled)."""
    a_mid = 0.5 * float(alpha_min + alpha_max)
    sigma_r = float(np.clip(0.18 - 0.10 * a_mid, 0.08, 0.14))
    return synthesize_defect_transfer_images(
        normal_images=normal_images,
        defect_images=defect_images,
        defect_labels=defect_labels,
        num_synthetic=num_synthetic,
        sigma_ratio=sigma_r,
        normal_subclusters=normal_subclusters,
        max_shift_px=2,
    )


@torch.no_grad()
def synthesize_boundary_gap_features(
    normal_gap: torch.Tensor,
    defect_gap: torch.Tensor,
    defect_labels: torch.Tensor,
    alpha_min: float = 0.55,
    alpha_max: float = 0.75,
    transfer_ratio: float = 0.30,
    normal_subclusters: torch.Tensor | np.ndarray | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize boundary anomalies via Pure Abnormal Residual Extraction & Same-Subcluster Injection.

    Eliminates global linear Mixup (which placed NG points on chords between normal sub-clusters):
    1. For each defect `h_bad`, locate its nearest normal reference `h_ref` and extract the pure
       defect residual `Delta_def = h_bad - h_ref`.
    2. Select base normals `h_base` strictly from `h_ref`'s SAME normal sub-cluster and local
       cosine neighborhood (`cos_sim >= 0.45`), never across unrelated normal modes.
    3. Add channel-dropped sparse top-residual activations `h_base + beta * (Dropout(M_top) * Delta_def)`
       and L2-normalize onto the standardized feature sphere.
    """
    device = defect_gap.device
    dtype = defect_gap.dtype
    if normal_gap.shape[0] == 0 or defect_gap.shape[0] == 0 or defect_gap.shape[-1] == 0:
        return (
            defect_gap.new_zeros((0, defect_gap.shape[-1])),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    normal_gap = normal_gap.to(device=device, dtype=dtype)
    defect_labels = defect_labels.to(device=device, dtype=torch.long).reshape(-1)
    lo = float(np.clip(min(alpha_min, alpha_max), 0.25, 0.90))
    hi = float(np.clip(max(alpha_min, alpha_max), lo + 0.05, 0.95))
    _, feat_dim = normal_gap.shape
    b_def = defect_gap.shape[0]
    if defect_labels.numel() < b_def:
        pad_lbl = defect_labels[-1] if defect_labels.numel() > 0 else torch.tensor(1, device=device)
        defect_labels = F.pad(defect_labels, (0, b_def - defect_labels.numel()), value=int(pad_lbl.item()))
    elif defect_labels.numel() > b_def:
        defect_labels = defect_labels[:b_def]

    norm_u = F.normalize(normal_gap.float(), p=2, dim=-1, eps=1e-6)
    def_u = F.normalize(defect_gap.float(), p=2, dim=-1, eps=1e-6)
    sim_dn = def_u @ norm_u.T  # [B_def, B_norm]
    ref_idx = torch.argmax(sim_dn, dim=1)  # [B_def]

    h_ref = normal_gap[ref_idx]
    delta_def = defect_gap - h_ref  # Pure abnormal residual vector [B_def, D]
    abs_delta = delta_def.abs()

    base_idx_1: list[int] = []
    base_idx_2: list[int] = []
    for d_i in range(b_def):
        r_i = int(ref_idx[d_i].item())
        cand_pool = _select_same_subcluster_normal_pool(
            ref_norm_idx=r_i,
            norm_low_or_feat=norm_u,
            normal_subclusters=normal_subclusters,
            metric="cosine",
        )
        n_cand = int(cand_pool.numel())
        if n_cand >= 2:
            pick1 = int(cand_pool[int(torch.randint(0, n_cand, (1,), device=device).item())].item())
            pick2 = int(cand_pool[int(torch.randint(0, n_cand, (1,), device=device).item())].item())
        else:
            pick1 = r_i
            pick2 = r_i
        base_idx_1.append(pick1)
        base_idx_2.append(pick2)

    t_idx_1 = torch.tensor(base_idx_1, dtype=torch.long, device=device)
    t_idx_2 = torch.tensor(base_idx_2, dtype=torch.long, device=device)
    h_base_1 = normal_gap[t_idx_1]
    h_base_2 = normal_gap[t_idx_2]

    # Variant 1: Sparse top-channel residual addition with channel dropout (guarantee >=1 peak channel survives)
    ratio_clamped = float(np.clip(transfer_ratio, 0.10, 0.60))
    k_dims_1 = min(feat_dim, max(min(8, feat_dim), int(round(feat_dim * ratio_clamped))))
    _, top_idx_1 = torch.topk(abs_delta, k=k_dims_1, dim=-1, largest=True)
    mask_1 = torch.zeros_like(defect_gap)
    mask_1.scatter_(dim=-1, index=top_idx_1, value=1.0)
    drop_1 = (torch.rand_like(mask_1) < 0.80).to(dtype=dtype) / 0.80
    drop_1.scatter_(dim=-1, index=top_idx_1[:, :1], value=1.0 / 0.80)
    beta_1 = torch.empty((b_def, 1), device=device, dtype=dtype).uniform_(lo + 0.20, hi + 0.50)
    h_res_1 = F.normalize(h_base_1 + beta_1 * (mask_1 * drop_1 * delta_def), p=2, dim=-1, eps=1e-6)

    # Variant 2: Wider residual subspace perturbation with channel dropout (guarantee >=1 peak channel survives)
    k_dims_2 = min(
        feat_dim,
        max(min(8, feat_dim), int(round(feat_dim * min(0.55, ratio_clamped * 1.5)))),
    )
    _, top_idx_2 = torch.topk(abs_delta, k=k_dims_2, dim=-1, largest=True)
    mask_2 = torch.zeros_like(defect_gap)
    mask_2.scatter_(dim=-1, index=top_idx_2, value=1.0)
    drop_2 = (torch.rand_like(mask_2) < 0.75).to(dtype=dtype) / 0.75
    drop_2.scatter_(dim=-1, index=top_idx_2[:, :1], value=1.0 / 0.75)
    beta_2 = torch.empty((b_def, 1), device=device, dtype=dtype).uniform_(lo + 0.05, hi + 0.35)
    h_res_2 = F.normalize(h_base_2 + beta_2 * (mask_2 * drop_2 * delta_def), p=2, dim=-1, eps=1e-6)

    synth_gap = torch.cat([h_res_1, h_res_2], dim=0)
    synth_labels = torch.cat([defect_labels, defect_labels], dim=0)
    return synth_gap, synth_labels
