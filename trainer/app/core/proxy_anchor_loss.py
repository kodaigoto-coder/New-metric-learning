"""50:50 Power-Balanced Multi-Proxy Anchor Loss with Adjustable Normal Centripetal vs. Anomaly Repulsion Ratio."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.core.memory_bank import CrossBatchMemoryBank, TopKMode


def _safe_sorted_quantile_1d(vals: torch.Tensor, q: float) -> torch.Tensor:
    """MPS- and fp16-safe 1D quantile via linear interpolation on sorted values."""
    flat = vals.reshape(-1).float()
    n = flat.numel()
    if n == 0:
        return vals.new_tensor(0.0)
    if n == 1:
        return flat[0].to(dtype=vals.dtype)
    sorted_v, _ = torch.sort(flat)
    pos = float(np.clip(q, 0.0, 1.0)) * float(n - 1)
    lo = int(np.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = float(pos - lo)
    out = sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac
    return out.to(dtype=vals.dtype)


def _perturb_collinear_proxies(proxies: torch.Tensor, max_sim: float = 0.98) -> torch.Tensor:
    """Break exact collinearity (`cos_sim > max_sim`) among proxy rows to avoid zero-gradient sphere saddle points."""
    m, d = proxies.shape
    if m <= 1 or d < 2:
        return F.normalize(proxies, p=2, dim=-1, eps=1e-6)
    out = F.normalize(proxies.clone().float(), p=2, dim=-1, eps=1e-6)
    for j in range(1, m):
        sims = out[:j] @ out[j]
        if float(sims.max().item()) > max_sim:
            basis = torch.zeros(d, device=out.device, dtype=out.dtype)
            basis[j % d] = 1.0
            basis[(j * 7 + 3) % d] = -0.5
            tangent = basis - torch.dot(basis, out[j]) * out[j]
            if float(tangent.norm().item()) < 1e-6:
                basis = torch.roll(basis, shifts=1, dims=0)
                tangent = basis - torch.dot(basis, out[j]) * out[j]
            tangent = F.normalize(tangent.unsqueeze(0), p=2, dim=-1, eps=1e-6).squeeze(0)
            out[j] = F.normalize((out[j] + 0.28 * tangent).unsqueeze(0), p=2, dim=-1, eps=1e-6).squeeze(0)
    return out.to(dtype=proxies.dtype)


def estimate_optimal_normal_proxies(
    norm_features: np.ndarray | torch.Tensor,
    max_proxies: int = 6,
    radius_eps: float = 0.22,
    silhouette_min: float = 0.25,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> tuple[int, np.ndarray]:
    """Automatically determine optimal Normal proxy count M* in {1..max_proxies} via epsilon-covering & cosine silhouette.

    - Keeps M* = 1 when the training normal distribution is unimodal (low silhouette or minimal
      within-cluster dispersion reduction).
    - Splits into M* >= 2 sub-clusters only when discrete normal modes exist (e.g. mold cavities,
      lot differences, surface reflection sub-modes on asymmetric workpieces).
    """
    if isinstance(norm_features, torch.Tensor):
        x_np = norm_features.detach().cpu().float().numpy()
    else:
        x_np = np.asarray(norm_features, dtype=np.float32)
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    n_norm = x_np.shape[0]
    if x_np.ndim != 2 or x_np.shape[1] == 0:
        return 1, np.zeros(n_norm, dtype=np.int64)

    eff_min_size = max(3, min(int(min_cluster_size), max(3, n_norm // 3)))
    eff_min_size = max(eff_min_size, int(np.floor(0.08 * n_norm)))
    k_upper = min(int(max_proxies), n_norm // max(1, eff_min_size))

    if n_norm < max(6, 2 * eff_min_size) or k_upper < 2:
        return 1, np.zeros(n_norm, dtype=np.int64)

    # 1. Uncentered unit vectors for physical cosine covering radius
    u_raw = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
    c0 = u_raw.mean(axis=0)
    c0 = c0 / (np.linalg.norm(c0) + 1e-8)
    r90_raw_single = float(np.quantile(1.0 - (u_raw @ c0), 0.90))
    if r90_raw_single <= 1e-4:
        return 1, np.zeros(n_norm, dtype=np.int64)

    # 2. Mean-centered PCA representation (without per-channel std whitening so low-variance noise channels stay suppressed)
    mean = np.mean(x_np, axis=0, keepdims=True)
    x_centered = (x_np - mean).astype(np.float32)
    if float(np.var(x_centered)) < 1e-12:
        return 1, np.zeros(n_norm, dtype=np.int64)

    max_comp = min(32, n_norm - 1, x_centered.shape[1])
    if max_comp >= 2:
        pca = PCA(n_components=max_comp, random_state=random_state)
        z_pca_full = np.nan_to_num(pca.fit_transform(x_centered), nan=0.0)
        evr = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
        cum_var = np.cumsum(evr)
        keep_d = max(2, min(int(np.searchsorted(cum_var, 0.90) + 1), max_comp))
        z_pca = z_pca_full[:, :keep_d].astype(np.float32)
    else:
        z_pca = x_centered

    q90_euc_single = float(np.quantile(np.linalg.norm(z_pca, axis=1), 0.90))
    if q90_euc_single <= 1e-5:
        return 1, np.zeros(n_norm, dtype=np.int64)
    q90_euc_single = q90_euc_single + 1e-8
    z_sph = z_pca / (np.linalg.norm(z_pca, axis=1, keepdims=True) + 1e-8)

    candidates: list[tuple[int, float, float, float, np.ndarray]] = []
    for k_cand in range(2, k_upper + 1):
        km = KMeans(n_clusters=k_cand, n_init=10, random_state=random_state)
        cand_labels = km.fit_predict(z_sph).astype(np.int64)
        uniq, cnts = np.unique(cand_labels, return_counts=True)
        if len(uniq) < k_cand or int(np.min(cnts)) < eff_min_size:
            continue

        try:
            sil = float(silhouette_score(z_sph, cand_labels, metric="cosine"))
        except ValueError:
            continue
        if not np.isfinite(sil):
            continue

        within_dists: list[float] = []
        raw_r90_per_cluster: list[float] = []
        for cid in uniq:
            mask_c = cand_labels == cid
            c_pca = z_pca[mask_c]
            c_mu = np.mean(c_pca, axis=0, keepdims=True)
            within_dists.extend(np.linalg.norm(c_pca - c_mu, axis=1).tolist())

            u_c = u_raw[mask_c]
            u_mu = np.mean(u_c, axis=0)
            u_mu = u_mu / (np.linalg.norm(u_mu) + 1e-8)
            raw_r90_per_cluster.append(float(np.quantile(1.0 - (u_c @ u_mu), 0.90)))

        q90_within = float(np.quantile(np.asarray(within_dists, dtype=np.float32), 0.90))
        rel_drop = 1.0 - (q90_within / q90_euc_single)
        worst_r90 = float(max(raw_r90_per_cluster))

        if sil >= float(silhouette_min) and rel_drop >= 0.18:
            candidates.append((k_cand, sil, worst_r90, rel_drop, cand_labels))

    if not candidates:
        return 1, np.zeros(n_norm, dtype=np.int64)

    best_sil = max(c[1] for c in candidates)
    # If raw spread is already compact and silhouette is not strongly multi-modal, preserve M* = 1
    if r90_raw_single <= 0.04 and best_sil < 0.45:
        return 1, np.zeros(n_norm, dtype=np.int64)

    # Prefer smallest k that satisfies epsilon-covering radius within 85% of peak silhouette
    chosen: tuple[int, float, float, float, np.ndarray] | None = None
    for cand in candidates:
        if cand[2] <= float(radius_eps) and cand[1] >= 0.85 * best_sil:
            chosen = cand
            break
    if chosen is None:
        chosen = max(candidates, key=lambda c: c[1])

    m_star = int(chosen[0])
    raw_lbls = chosen[4]
    # Re-index subclusters deterministically by descending sample count (0 = largest normal mode)
    uniq_ids, counts = np.unique(raw_lbls, return_counts=True)
    order = uniq_ids[np.argsort(-counts, kind="mergesort")]
    remap = {int(old_id): new_id for new_id, old_id in enumerate(order.tolist())}
    clean_lbls = np.array([remap[int(x)] for x in raw_lbls], dtype=np.int64)
    return m_star, clean_lbls


def assign_normal_subclusters_by_centroids(
    features: np.ndarray | torch.Tensor,
    ref_features: np.ndarray | torch.Tensor,
    ref_subclusters: np.ndarray | list[int] | torch.Tensor,
) -> np.ndarray:
    """Assign normal samples in `features` to nearest sub-cluster centroid fitted on `ref_features`."""
    feat_arr = np.nan_to_num(
        features.detach().cpu().float().numpy()
        if isinstance(features, torch.Tensor)
        else np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ref_arr = np.nan_to_num(
        ref_features.detach().cpu().float().numpy()
        if isinstance(ref_features, torch.Tensor)
        else np.asarray(ref_features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if isinstance(ref_subclusters, torch.Tensor):
        ref_sub_arr = ref_subclusters.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        ref_sub_arr = np.asarray(ref_subclusters, dtype=np.int64).reshape(-1)

    n_total = feat_arr.shape[0]
    if n_total == 0 or ref_arr.shape[0] == 0 or len(ref_sub_arr) != ref_arr.shape[0]:
        return np.zeros(n_total, dtype=np.int64)

    uniq = sorted({int(c) for c in ref_sub_arr.tolist() if int(c) >= 0})
    if len(uniq) <= 1:
        return np.zeros(n_total, dtype=np.int64)

    mean = np.mean(ref_arr, axis=0, keepdims=True)
    ref_centered = ref_arr - mean
    ref_u = ref_centered / (np.linalg.norm(ref_centered, axis=1, keepdims=True) + 1e-8)

    all_centered = feat_arr - mean
    all_u = all_centered / (np.linalg.norm(all_centered, axis=1, keepdims=True) + 1e-8)

    centroids: list[np.ndarray] = []
    for cid in uniq:
        c_vec = np.mean(ref_u[ref_sub_arr == cid], axis=0)
        c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-8)
        centroids.append(c_vec)
    c_mat = np.stack(centroids, axis=0).astype(np.float32)  # [M, D]
    sims = all_u @ c_mat.T
    best_pos = np.argmax(sims, axis=1)
    return np.array([uniq[int(p)] for p in best_pos], dtype=np.int64)


class CustomizedMultiProxyAnchorLoss(nn.Module):
    """50:50 Power-Balanced Multi-Proxy Loss for Weakly-Supervised Industrial Anomaly Detection.

    Core Formulation (50:50 Linear Cosine Power Balance):
        L_total = lambda_norm * L_centripetal + lambda_anom * L_repel + L_aux

    1. Term 1 — Normal Centripetal Force (`L_centripetal`, default 50% / lambda_norm = 1.0):
           L_centripetal = (1 / |B_norm|) * sum_{x in B_norm} (1 - S(x, p_norm))
       where `S(x, p_norm) = max_{m in 1..M} <z(x), p_norm^(m)>`.
       Unlike Softplus/LogSumExp (which saturates above a margin and concentrates 99% of the
       gradient on a single outlier) or squared error `(1 - S)^2` (whose gradient vanishes near
       S = 1.0), the linear cosine distance `1 - S(x, p_norm)` applies a constant unit inward
       gradient (`-1.0`) to every normal sample all the way to `S = 1.0`, monotonically
       shrinking normal cluster dispersion across epochs.

    2. Term 2 — Anomaly Centrifugal Repulsion (`L_repel`, default 50% / lambda_anom = 1.0):
           L_repel = (1 / |B_anom|) * sum_{x in B_anom} w_tilde(x) * max(0, S(x, p_norm.detach()) - delta)
       * `p_norm.detach()` anchors the normal proxies so anomaly repulsion never drags `p_norm`
         away from the true normal centroid.
       * `w_tilde(x)` prioritizes Top-K hard real anomalies (`hard_anomaly_boost`) while keeping
         synthetic Mixup samples at a gentle `0.25x` weight, and is mean-normalized (`mean(w_tilde) = 1.0`)
         so that Term 1 and Term 2 strictly remain on the same `[0.0, 1.0]` yardstick.
       * Zero centrifugal force (`Direction B` removed from `z_norm`) is applied to normal samples.

    3. Adjustable Power Ratio (`normal_loss_ratio` in `[0.05, 0.95]`, default `0.50` = 50:50):
           lambda_norm = 2.0 * normal_loss_ratio
           lambda_anom = 2.0 * (1.0 - normal_loss_ratio)
       At `normal_loss_ratio = 0.50`, `lambda_norm = 1.0` and `lambda_anom = 1.0` (exact 50:50).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_normal_proxies: int = 4,
        num_defect_classes: int = 3,
        class_balanced_weights: np.ndarray | list[float] | None = None,
        normal_loss_ratio: float = 0.50,
        alpha_pos: float = 16.0,
        gamma_neg: float = 20.0,
        delta_pos: float = 0.88,
        delta_neg: float = 0.0,
        w_defect: float = 1.0,
        hard_anomaly_boost: float = 3.0,
        ortho_margin_defect: float = 0.20,
        ortho_margin_normal: float = 0.75,
        normal_cone_min_sim: float = 0.20,
        synth_repel_weight: float = 0.15,
        ortho_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_normal_proxies = max(1, int(num_normal_proxies))
        self.num_defect_classes = max(1, int(num_defect_classes))
        self.normal_loss_ratio = float(np.clip(normal_loss_ratio, 0.05, 0.95))
        self.alpha_pos = float(alpha_pos)
        self.gamma_neg = float(gamma_neg)
        self.delta_pos = float(delta_pos)
        self.delta_neg = float(delta_neg)
        self.w_defect = float(w_defect)
        self.hard_anomaly_boost = max(1.0, float(hard_anomaly_boost))
        self.ortho_margin_defect = float(ortho_margin_defect)
        self.ortho_margin_normal = float(ortho_margin_normal)
        self.normal_cone_min_sim = min(float(normal_cone_min_sim), self.ortho_margin_normal - 0.05)
        self.synth_repel_weight = float(np.clip(synth_repel_weight, 0.0, 1.0))
        self.ortho_weight = float(ortho_weight)

        # Learnable Proxy Parameters on 128-D Hypersphere (isolated RNG fork)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            norm_init = F.normalize(
                torch.randn(self.num_normal_proxies, self.embedding_dim), p=2, dim=-1, eps=1e-6
            )
            def_init = F.normalize(
                torch.randn(self.num_defect_classes, self.embedding_dim), p=2, dim=-1, eps=1e-6
            )
        self.normal_proxies = nn.Parameter(_perturb_collinear_proxies(norm_init))
        self.defect_proxies = nn.Parameter(_perturb_collinear_proxies(def_init))

        # Möbius per-proxy radius calibration buffers (identity alpha=1.0 when M=1 or uncalibrated)
        self.register_buffer(
            "proxy_radii", torch.full((self.num_normal_proxies,), 0.15, dtype=torch.float32)
        )
        self.register_buffer(
            "proxy_alphas", torch.ones(self.num_normal_proxies, dtype=torch.float32)
        )

        if class_balanced_weights is not None:
            cb_arr = np.asarray(class_balanced_weights, dtype=np.float32).reshape(-1)
            cb_arr = np.maximum(np.nan_to_num(cb_arr, nan=1.0, posinf=1.0, neginf=1.0), 1e-4)
            if len(cb_arr) < self.num_defect_classes:
                padded = np.ones(self.num_defect_classes, dtype=np.float32)
                padded[: len(cb_arr)] = cb_arr
                cb_arr = padded
            else:
                cb_arr = cb_arr[: self.num_defect_classes]
            cb_tensor = torch.from_numpy(cb_arr)
        else:
            cb_tensor = torch.ones(self.num_defect_classes, dtype=torch.float32)

        self.register_buffer("class_weights", cb_tensor)

    @torch.no_grad()
    def initialize_proxies_from_embeddings(
        self,
        embeddings: np.ndarray | torch.Tensor,
        labels: np.ndarray | list[int],
        normal_subclusters: np.ndarray | list[int] | None = None,
    ) -> None:
        """Initialize M Normal sub-center proxies (using precomputed subclusters or K-Means) and C Defect proxies."""
        if isinstance(embeddings, torch.Tensor):
            emb_clean = torch.nan_to_num(embeddings.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
            z_np = F.normalize(emb_clean, p=2, dim=-1, eps=1e-6).numpy()
        else:
            emb_arr = np.nan_to_num(np.asarray(embeddings, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-8
            z_np = (emb_arr / norms).astype(np.float32)

        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        device = self.normal_proxies.device

        # 1. Initialize M Normal Proxies on normal embeddings (label_id == 0)
        norm_mask = labels_arr == 0
        norm_vecs = z_np[norm_mask]
        assigned_from_subclusters = False

        if (
            self.num_normal_proxies > 1
            and normal_subclusters is not None
            and len(norm_vecs) > 0
        ):
            sub_arr = np.asarray(normal_subclusters, dtype=np.int64).reshape(-1)
            if len(sub_arr) == len(labels_arr):
                sub_norm = sub_arr[norm_mask]
            elif len(sub_arr) == len(norm_vecs):
                sub_norm = sub_arr
            else:
                sub_norm = np.array([], dtype=np.int64)

            if len(sub_norm) == len(norm_vecs):
                centers_list: list[np.ndarray] = []
                populated_slots = 0
                global_norm = norm_vecs.mean(axis=0)
                global_norm = global_norm / (np.linalg.norm(global_norm) + 1e-8)
                for m_idx in range(self.num_normal_proxies):
                    m_mask = sub_norm == m_idx
                    if np.any(m_mask):
                        c_m = norm_vecs[m_mask].mean(axis=0)
                        c_m = c_m / (np.linalg.norm(c_m) + 1e-8)
                        populated_slots += 1
                    else:
                        c_m = global_norm
                    centers_list.append(c_m.astype(np.float32))
                if populated_slots == self.num_normal_proxies or len(norm_vecs) < self.num_normal_proxies:
                    centers = np.stack(centers_list, axis=0)
                    self.normal_proxies.copy_(
                        _perturb_collinear_proxies(torch.from_numpy(centers).to(device))
                    )
                    assigned_from_subclusters = True

        if not assigned_from_subclusters:
            if len(norm_vecs) >= self.num_normal_proxies and self.num_normal_proxies > 1:
                km = KMeans(
                    n_clusters=self.num_normal_proxies,
                    n_init=10,
                    random_state=42,
                )
                km.fit(norm_vecs)
                centers = km.cluster_centers_.astype(np.float32)
                c_norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
                centers = centers / c_norms
                self.normal_proxies.copy_(
                    _perturb_collinear_proxies(torch.from_numpy(centers).to(device))
                )
            elif len(norm_vecs) > 0:
                mean_norm = norm_vecs.mean(axis=0, keepdims=True)
                mean_norm = mean_norm / (np.linalg.norm(mean_norm, axis=1, keepdims=True) + 1e-8)
                base_t = torch.from_numpy(mean_norm).to(device).repeat(self.num_normal_proxies, 1)
                self.normal_proxies.copy_(_perturb_collinear_proxies(base_t))

        # Ensure multi-normal proxies start within shared northern cone [normal_cone_min_sim, ortho_margin_normal]
        if self.num_normal_proxies > 1:
            p_n = F.normalize(self.normal_proxies.data, p=2, dim=-1, eps=1e-6)
            mean_vec = p_n.mean(dim=0, keepdim=True)
            if float(mean_vec.norm().item()) < 1e-5:
                cone_axis = p_n[0:1].clone()
            else:
                cone_axis = F.normalize(mean_vec, p=2, dim=-1, eps=1e-6)
            for m_idx in range(self.num_normal_proxies):
                dot_c = float(torch.dot(p_n[m_idx], cone_axis.squeeze(0)).item())
                if dot_c < 0.55:
                    p_n[m_idx] = F.normalize(
                        (0.55 * p_n[m_idx] + 0.45 * cone_axis.squeeze(0)).unsqueeze(0),
                        p=2,
                        dim=-1,
                        eps=1e-6,
                    ).squeeze(0)
            self.normal_proxies.copy_(_perturb_collinear_proxies(p_n))

        mean_p_norm = self.normal_proxies.mean(dim=0, keepdim=True)
        if float(mean_p_norm.norm().item()) < 1e-5:
            p_norm_mean = F.normalize(self.normal_proxies[0:1], p=2, dim=-1, eps=1e-6)
        else:
            p_norm_mean = F.normalize(mean_p_norm, p=2, dim=-1, eps=1e-6)

        # 2. Initialize C Defect Proxies from each defect class (1..C), ensuring negative hemisphere start
        for c_idx in range(1, self.num_defect_classes + 1):
            c_vecs = z_np[labels_arr == c_idx]
            if len(c_vecs) > 0:
                c_mean = c_vecs.mean(axis=0)
                c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
                c_t = torch.from_numpy(c_mean.astype(np.float32)).to(device)
                proj = torch.dot(c_t, p_norm_mean.squeeze(0))
                if float(proj.item()) > -0.15:
                    diff_t = c_t - (proj + 0.35) * p_norm_mean.squeeze(0)
                    if float(diff_t.norm().item()) < 1e-5:
                        c_t = -p_norm_mean.squeeze(0)
                    else:
                        c_t = F.normalize(diff_t.unsqueeze(0), p=2, dim=-1, eps=1e-6).squeeze(0)
                self.defect_proxies[c_idx - 1].copy_(c_t)
        self.defect_proxies.copy_(_perturb_collinear_proxies(self.defect_proxies.data))

        self.calibrate_proxy_radii(
            embeddings=z_np,
            labels=labels_arr,
            normal_subclusters=normal_subclusters,
        )

    @torch.no_grad()
    def calibrate_proxy_radii(
        self,
        embeddings: np.ndarray | torch.Tensor,
        labels: np.ndarray | list[int] | torch.Tensor | None = None,
        normal_subclusters: np.ndarray | list[int] | torch.Tensor | None = None,
    ) -> None:
        """Fit Möbius per-proxy radius calibration factors alpha_m on training Normal embeddings."""
        device = self.normal_proxies.device
        if self.num_normal_proxies <= 1:
            self.proxy_radii.fill_(0.15)
            self.proxy_alphas.fill_(1.0)
            return

        if isinstance(embeddings, torch.Tensor):
            emb_t = torch.nan_to_num(
                embeddings.detach().to(device=device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            z_t = F.normalize(emb_t, p=2, dim=-1, eps=1e-6)
        else:
            z_arr = np.nan_to_num(
                np.asarray(embeddings, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            z_t = F.normalize(
                torch.from_numpy(z_arr).to(device=device, dtype=torch.float32),
                p=2,
                dim=-1,
                eps=1e-6,
            )

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                lbl_t = labels.detach().to(device=device, dtype=torch.long).reshape(-1)
            else:
                lbl_t = torch.as_tensor(labels, device=device, dtype=torch.long).reshape(-1)
            norm_mask = lbl_t == 0
            z_norm = z_t[norm_mask]
        else:
            norm_mask = None
            z_norm = z_t

        if z_norm.shape[0] < 2:
            self.proxy_alphas.fill_(1.0)
            return

        p_norm, _ = self.get_normalized_proxies()
        dists = (1.0 - (z_norm @ p_norm.T)).clamp(min=0.0, max=2.0)  # [N_norm, M]

        sub_norm_t: torch.Tensor | None = None
        if normal_subclusters is not None:
            if isinstance(normal_subclusters, torch.Tensor):
                sub_raw = normal_subclusters.detach().to(device=device, dtype=torch.long).reshape(-1)
            else:
                sub_raw = torch.as_tensor(normal_subclusters, device=device, dtype=torch.long).reshape(-1)
            if norm_mask is not None and sub_raw.numel() == norm_mask.numel():
                sub_norm_t = sub_raw[norm_mask]
            elif sub_raw.numel() == z_norm.shape[0]:
                sub_norm_t = sub_raw

        if sub_norm_t is None:
            sub_norm_t = torch.argmin(dists, dim=1)

        radii = torch.full((self.num_normal_proxies,), 0.15, device=device, dtype=torch.float32)
        valid_r_list: list[torch.Tensor] = []
        for m_idx in range(self.num_normal_proxies):
            m_mask = sub_norm_t == m_idx
            if int(m_mask.sum().item()) >= 2:
                r_m = _safe_sorted_quantile_1d(dists[m_mask, m_idx], 0.90).clamp(min=0.02, max=0.65)
                radii[m_idx] = r_m
                valid_r_list.append(r_m)

        if len(valid_r_list) >= 2:
            r_bar = torch.stack(valid_r_list).mean().clamp(min=0.02, max=0.65)
            for m_idx in range(self.num_normal_proxies):
                if int((sub_norm_t == m_idx).sum().item()) < 2:
                    radii[m_idx] = r_bar
            alphas = (r_bar * (2.0 - radii)) / (radii * (2.0 - r_bar) + 1e-8)
            alphas = alphas.clamp(min=0.40, max=4.00)
        else:
            alphas = torch.ones_like(radii)

        self.proxy_radii.copy_(radii)
        self.proxy_alphas.copy_(alphas)

    def compute_calibrated_normal_distances(
        self,
        z: torch.Tensor,
        p_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Möbius radius-calibrated cosine distances d_tilde [N, M] in [0, 2]."""
        raw_dists = (1.0 - (z @ p_norm.T)).clamp(min=0.0, max=2.0)  # [N, M]
        if self.num_normal_proxies <= 1:
            return raw_dists
        alphas = (
            self.proxy_alphas.detach()
            .reshape(1, -1)
            .to(dtype=z.dtype, device=z.device)
            .clamp(min=0.40, max=4.00)
        )
        numer = 2.0 * alphas * raw_dists
        denom = (alphas - 1.0) * raw_dists + 2.0
        return (numer / (denom + 1e-8)).clamp(min=0.0, max=2.0)

    def compute_calibrated_normal_similarity(
        self,
        z: torch.Tensor,
        p_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Möbius-calibrated normality similarity S_cal(z) = 1 - min_m d_tilde_m(z) in [-1, 1]."""
        if z.shape[0] == 0:
            return z.new_zeros((0,))
        d_tilde = self.compute_calibrated_normal_distances(z, p_norm)
        min_d, _ = d_tilde.min(dim=1)
        return 1.0 - min_d

    def get_normalized_proxies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalized `(p_norm [M, D], p_def [C, D])` on the unit hypersphere."""
        p_norm = F.normalize(self.normal_proxies, p=2, dim=-1, eps=1e-6)
        p_def = F.normalize(self.defect_proxies, p=2, dim=-1, eps=1e-6)
        return p_norm, p_def

    @torch.no_grad()
    def get_proxies_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Export detached unit-length `(normal_proxies, defect_proxies)` as pure float32 NumPy arrays."""
        p_norm, p_def = self.get_normalized_proxies()
        return (
            p_norm.detach().cpu().numpy().astype(np.float32),
            p_def.detach().cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def get_proxy_calibration_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Export detached `(proxy_alphas, proxy_radii)` as pure float32 NumPy arrays."""
        return (
            self.proxy_alphas.detach().cpu().numpy().astype(np.float32),
            self.proxy_radii.detach().cpu().numpy().astype(np.float32),
        )

    def _get_class_weight(self, class_id: int) -> torch.Tensor:
        idx = max(0, min(self.num_defect_classes - 1, int(class_id) - 1))
        return self.class_weights[idx]

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        synth_embeddings: torch.Tensor | None = None,
        synth_labels: torch.Tensor | None = None,
        memory_bank: CrossBatchMemoryBank | None = None,
        gamma_repel: float = 1.0,
        repel_margin: float = 0.0,
        top_k_mode: TopKMode = "count",
        top_k_value: float = 20.0,
        normal_loss_ratio: float | None = None,
        normal_subclusters: torch.Tensor | np.ndarray | list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute 50:50 Power-Balanced Centripetal-Normal / Repulsive-Anomaly Loss with Sub-Cluster Routing."""
        _ = gamma_repel
        z = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)
        labels = labels.to(device=z.device, dtype=torch.long).reshape(-1)
        batch_size = z.shape[0]
        if batch_size < 2:
            zero = z.sum() * 0.0
            return zero, {
                "loss_ms": 0.0,
                "loss_centripetal": 0.0,
                "loss_repel": 0.0,
                "loss_ortho": 0.0,
                "loss_total": 0.0,
                "lambda_norm": 1.0,
                "lambda_anom": 1.0,
                "effective_top_k": 0.0,
            }

        ratio = (
            float(np.clip(normal_loss_ratio, 0.05, 0.95))
            if normal_loss_ratio is not None
            else self.normal_loss_ratio
        )
        # At ratio = 0.50 (50:50 default) -> lambda_norm = 1.0, lambda_anom = 1.0
        lambda_norm = 2.0 * ratio
        lambda_anom = 2.0 * (1.0 - ratio)

        p_norm, p_def = self.get_normalized_proxies()  # [M, D], [C, D]
        is_defect = labels > 0
        z_norm = z[~is_defect]
        z_def = z[is_defect]
        y_def = labels[is_defect]

        # =========================================================================
        # ① 良品の求心力 (Normal Centripetal Force — Scale [0.0, 2.0], Target 0.0)
        #    Routes each normal sample to its assigned sub-cluster proxy (when M > 1)
        #    without forcing distinct normal sub-clusters onto a single pole.
        # =========================================================================
        loss_centripetal = z.new_tensor(0.0)
        if z_norm.shape[0] > 0:
            d_tilde_jn = self.compute_calibrated_normal_distances(z_norm, p_norm)  # [B_norm, M]
            routed_by_subcluster = False
            if self.num_normal_proxies > 1 and normal_subclusters is not None:
                if isinstance(normal_subclusters, torch.Tensor):
                    sub_t = normal_subclusters.to(device=z.device, dtype=torch.long).reshape(-1)
                else:
                    sub_t = torch.as_tensor(normal_subclusters, device=z.device, dtype=torch.long).reshape(-1)
                if sub_t.numel() == batch_size:
                    sub_norm = sub_t[~is_defect]
                elif sub_t.numel() == z_norm.shape[0]:
                    sub_norm = sub_t
                else:
                    sub_norm = torch.empty((0,), dtype=torch.long, device=z.device)

                if sub_norm.numel() == z_norm.shape[0]:
                    active_mode_losses: list[torch.Tensor] = []
                    for m_idx in range(self.num_normal_proxies):
                        m_mask = sub_norm == m_idx
                        if torch.any(m_mask):
                            active_mode_losses.append(d_tilde_jn[m_mask, m_idx].mean())
                    if active_mode_losses:
                        loss_centripetal = torch.stack(active_mode_losses).mean()
                        routed_by_subcluster = True

            if not routed_by_subcluster:
                min_d_jn, _ = d_tilde_jn.min(dim=1)
                loss_centripetal = min_d_jn.mean()

        # =========================================================================
        # ② 不良の斥力 (Anomaly Centrifugal Repulsion — Decoupled Real vs. Shallow Synthetic)
        #    - Strict `p_norm.detach()` prevents anomaly repulsion from drifting p_norm!
        #    - Real defects use Class-Balanced + Top-K boost normalized strictly over real defects.
        #    - Synthetic boundary samples use a dedicated positive-hemisphere shallow margin
        #      and fixed weight cap (`synth_repel_weight`) so they NEVER hijack repulsion
        #      after real defects converge!
        # =========================================================================
        n_real_def = z_def.shape[0]
        delta_margin = float(repel_margin)
        p_norm_anchor = p_norm.detach()

        loss_repel_real = z.new_tensor(0.0)
        loss_repel_synth = z.new_tensor(0.0)
        k_eff_hard = 0

        if n_real_def > 0:
            s_real_max = self.compute_calibrated_normal_similarity(z_def, p_norm_anchor)  # [N_real]
            cb_real = torch.stack(
                [self._get_class_weight(int(c.item())) for c in y_def]
            ).to(dtype=z.dtype, device=z.device)
            raw_weights_real = cb_real.clone()

            if top_k_mode == "ratio":
                ratio_val = max(0.05, min(1.0, float(top_k_value)))
                k_eff_hard = max(1, min(n_real_def, int(np.ceil(n_real_def * ratio_val))))
            else:
                k_eff_hard = max(1, min(n_real_def, int(round(float(top_k_value)))))

            _, hard_idx = torch.topk(s_real_max.detach(), k=k_eff_hard, largest=True)
            raw_weights_real[hard_idx] = raw_weights_real[hard_idx] * self.hard_anomaly_boost
            norm_weights_real = raw_weights_real / (raw_weights_real.mean().detach() + 1e-8)

            hinge_primary = F.relu(s_real_max - delta_margin)
            deep_margin = min(delta_margin - 0.25, -0.25)
            hinge_secondary = F.relu(s_real_max - deep_margin)
            per_real_repel = 0.85 * hinge_primary + 0.15 * hinge_secondary
            loss_repel_real = (norm_weights_real * per_real_repel).mean()

            # Optional XBM boundary normal repulsion (applied strictly to real defects)
            if (
                memory_bank is not None
                and len(memory_bank) > 0
                and memory_bank.normal_count > 0
            ):
                boundary_normals, _, k_xbm = memory_bank.get_top_k_boundary_normals(
                    defect_embeddings=z_def,
                    top_k_mode=top_k_mode,
                    top_k_value=top_k_value,
                    device=z.device,
                )
                if k_xbm > 0:
                    k_eff_hard = max(k_eff_hard, int(k_xbm))
                    sim_ik = torch.bmm(
                        z_def.unsqueeze(1), boundary_normals.detach().transpose(1, 2)
                    ).squeeze(1)
                    xbm_max_sim, _ = sim_ik.max(dim=1)
                    xbm_hinge = F.relu(xbm_max_sim - delta_margin).mean()
                    loss_repel_real = 0.85 * loss_repel_real + 0.15 * xbm_hinge

        if synth_embeddings is not None and synth_embeddings.shape[0] > 0:
            _ = synth_labels
            z_synth = F.normalize(
                synth_embeddings.to(device=z.device, dtype=z.dtype), p=2, dim=-1, eps=1e-6
            )
            s_synth_max = self.compute_calibrated_normal_similarity(z_synth, p_norm_anchor)
            # Shallow positive-hemisphere guardrail margin: stops gradient strictly above tau_sim=0.0
            synth_margin = max(delta_margin + 0.05, 0.05)
            loss_repel_synth = F.relu(s_synth_max - synth_margin).mean()

        if n_real_def > 0:
            loss_repel = loss_repel_real + (self.synth_repel_weight * loss_repel_synth)
        else:
            loss_repel = self.synth_repel_weight * loss_repel_synth

        # =========================================================================
        # ③ 補助項: 実不良サブクラス凝集 & 北半球共有コーン・プロキシ間正則化 (No force on z_norm!)
        # =========================================================================
        loss_def_sub = z.new_tensor(0.0)
        if n_real_def > 0:
            sim_id = z_def @ p_def.T  # [B_def, C]
            active_def_terms: list[torch.Tensor] = []
            for c_idx in range(1, self.num_defect_classes + 1):
                mask_c = y_def == c_idx
                if torch.any(mask_c):
                    sims_c = sim_id[mask_c, c_idx - 1]
                    active_def_terms.append((1.0 - sims_c).mean())
            if active_def_terms:
                loss_def_sub = torch.stack(active_def_terms).mean()

        # Keep defect proxies in the negative hemisphere relative to detached normal proxies
        p_def_norm_sim = p_def @ p_norm.detach().T  # [C, M]
        loss_def_proxy_hemisphere = F.relu(p_def_norm_sim - (-0.20)).mean()

        loss_ortho = z.new_tensor(0.0)
        if self.num_defect_classes > 1:
            sim_def_gram = p_def @ p_def.T  # [C, C]
            off_diag_mask = ~torch.eye(
                self.num_defect_classes, dtype=torch.bool, device=z.device
            )
            off_diag_sims = sim_def_gram[off_diag_mask]
            loss_ortho = loss_ortho + F.relu(off_diag_sims - self.ortho_margin_defect).pow(2).mean()

        if self.num_normal_proxies > 1:
            sim_norm_gram = p_norm @ p_norm.T  # [M, M]
            norm_off_mask = ~torch.eye(
                self.num_normal_proxies, dtype=torch.bool, device=z.device
            )
            norm_off_sims = sim_norm_gram[norm_off_mask]
            # Two-sided cone constraint: prevent fusion (<= ortho_margin_normal) AND keep within northern cone (>= normal_cone_min_sim)
            anti_collapse = F.relu(norm_off_sims - self.ortho_margin_normal).pow(2).mean()
            shared_cone = F.relu(self.normal_cone_min_sim - norm_off_sims).pow(2).mean()
            loss_ortho = loss_ortho + anti_collapse + 0.50 * shared_cone

        loss_aux = (
            0.10 * loss_def_sub
            + 0.10 * loss_def_proxy_hemisphere
            + (self.ortho_weight * loss_ortho)
        )

        # Total 50:50 Power-Balanced Loss (with user-configurable lambda_norm : lambda_anom)
        loss_total = (lambda_norm * loss_centripetal) + (lambda_anom * loss_repel) + loss_aux

        stats = {
            "loss_ms": float(loss_centripetal.detach().cpu().item()),
            "loss_centripetal": float(loss_centripetal.detach().cpu().item()),
            "loss_repel": float(loss_repel.detach().cpu().item()),
            "loss_repel_real": float(loss_repel_real.detach().cpu().item()),
            "loss_repel_synth": float(loss_repel_synth.detach().cpu().item()),
            "loss_ortho": float(loss_ortho.detach().cpu().item()),
            "loss_total": float(loss_total.detach().cpu().item()),
            "lambda_norm": round(float(lambda_norm), 4),
            "lambda_anom": round(float(lambda_anom), 4),
            "effective_top_k": float(k_eff_hard),
        }
        return loss_total, stats
