"""50:50 Power-Balanced Multi-Proxy Anchor Loss with Adjustable Normal Centripetal vs. Anomaly Repulsion Ratio."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from app.core.memory_bank import CrossBatchMemoryBank, TopKMode


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
        ortho_margin_normal: float = 0.85,
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
        self.ortho_weight = float(ortho_weight)

        # Learnable Proxy Parameters on 128-D Hypersphere
        torch.manual_seed(42)
        norm_init = F.normalize(
            torch.randn(self.num_normal_proxies, self.embedding_dim), p=2, dim=-1
        )
        def_init = F.normalize(
            torch.randn(self.num_defect_classes, self.embedding_dim), p=2, dim=-1
        )
        self.normal_proxies = nn.Parameter(norm_init)
        self.defect_proxies = nn.Parameter(def_init)

        if class_balanced_weights is not None:
            cb_arr = np.asarray(class_balanced_weights, dtype=np.float32).reshape(-1)
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
    ) -> None:
        """Initialize M Normal sub-center proxies via K-Means and C Defect proxies via mode centroids."""
        if isinstance(embeddings, torch.Tensor):
            z_np = F.normalize(embeddings.detach().cpu().float(), p=2, dim=-1).numpy()
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            z_np = (embeddings / norms).astype(np.float32)

        labels_arr = np.asarray(labels, dtype=np.int64)
        device = self.normal_proxies.device

        # 1. Initialize M Normal Proxies using K-Means on normal embeddings (label_id == 0)
        norm_vecs = z_np[labels_arr == 0]
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
            self.normal_proxies.copy_(torch.from_numpy(centers).to(device))
        elif len(norm_vecs) > 0:
            mean_norm = norm_vecs.mean(axis=0, keepdims=True)
            mean_norm = mean_norm / (np.linalg.norm(mean_norm, axis=1, keepdims=True) + 1e-8)
            base_t = torch.from_numpy(mean_norm).to(device).repeat(self.num_normal_proxies, 1)
            if self.num_normal_proxies > 1:
                perturb = torch.randn_like(base_t) * 0.05
                base_t = F.normalize(base_t + perturb, p=2, dim=-1)
            self.normal_proxies.copy_(base_t)

        # Mean normal direction for hemisphere separation check
        p_norm_mean = F.normalize(self.normal_proxies.mean(dim=0, keepdim=True), p=2, dim=-1)

        # 2. Initialize C Defect Proxies from each defect class (1..C), ensuring negative hemisphere start
        for c_idx in range(1, self.num_defect_classes + 1):
            c_vecs = z_np[labels_arr == c_idx]
            if len(c_vecs) > 0:
                c_mean = c_vecs.mean(axis=0)
                c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
                c_t = torch.from_numpy(c_mean.astype(np.float32)).to(device)
                proj = torch.dot(c_t, p_norm_mean.squeeze(0))
                if float(proj.item()) > -0.10:
                    c_t = F.normalize(c_t - (proj + 0.35) * p_norm_mean.squeeze(0), p=2, dim=-1)
                self.defect_proxies[c_idx - 1].copy_(c_t)

    def get_normalized_proxies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalized `(p_norm [M, D], p_def [C, D])` on the unit hypersphere."""
        p_norm = F.normalize(self.normal_proxies, p=2, dim=-1)
        p_def = F.normalize(self.defect_proxies, p=2, dim=-1)
        return p_norm, p_def

    @torch.no_grad()
    def get_proxies_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Export detached unit-length `(normal_proxies, defect_proxies)` as pure float32 NumPy arrays."""
        p_norm, p_def = self.get_normalized_proxies()
        return (
            p_norm.detach().cpu().numpy().astype(np.float32),
            p_def.detach().cpu().numpy().astype(np.float32),
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute 50:50 Power-Balanced Centripetal-Normal / Repulsive-Anomaly Loss."""
        z = F.normalize(embeddings, p=2, dim=-1)
        labels = labels.view(-1)
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
        #    L_centripetal = (1 / |B_norm|) * sum_{x in B_norm} (1 - S(x, p_norm))
        #    Applies constant unit gradient (-1.0) to EVERY normal sample in B_norm.
        # =========================================================================
        loss_centripetal = z.new_tensor(0.0)
        if z_norm.shape[0] > 0:
            sim_jn = z_norm @ p_norm.T  # [B_norm, M]
            # Maximum cosine similarity to the M normal sub-center proxies
            s_norm_max, _ = sim_jn.max(dim=1)  # [B_norm]
            loss_primary_centripetal = (1.0 - s_norm_max).mean()

            # Keep all M sub-centers anchored near the normal sample distribution
            if self.num_normal_proxies > 1:
                subcenter_anchor = (1.0 - sim_jn.max(dim=0).values).mean()
                loss_centripetal = 0.90 * loss_primary_centripetal + 0.10 * subcenter_anchor
            else:
                loss_centripetal = loss_primary_centripetal

        # =========================================================================
        # ② 不良の斥力 (Anomaly Centrifugal Repulsion — Scale [0.0, 1.0+], Target 0.0)
        #    L_repel = (1 / |B_anom|) * sum_{x in B_anom} w_tilde(x) * max(0, S(x, p_norm.detach()) - delta)
        #    - Strict `p_norm.detach()` prevents anomaly repulsion from drifting p_norm!
        #    - Intra-anomaly weights w_tilde(x) are mean-normalized to 1.0 so the
        #      1.0 : 1.0 (50:50) macro scale with L_centripetal is strictly preserved.
        # =========================================================================
        n_real_def = z_def.shape[0]
        if synth_embeddings is not None and synth_embeddings.shape[0] > 0:
            z_synth = F.normalize(synth_embeddings, p=2, dim=-1)
            if synth_labels is not None and synth_labels.numel() == z_synth.shape[0]:
                y_synth = synth_labels.view(-1)
            else:
                y_synth = torch.ones(z_synth.shape[0], dtype=torch.long, device=z.device)
            z_all_neg = torch.cat([z_def, z_synth], dim=0) if n_real_def > 0 else z_synth
            y_all_neg = torch.cat([y_def, y_synth], dim=0) if n_real_def > 0 else y_synth
            n_synth = z_synth.shape[0]
        else:
            z_all_neg = z_def
            y_all_neg = y_def
            n_synth = 0

        # Effective cosine margin delta (default 0.0 = orthogonal 90 deg)
        delta_margin = float(repel_margin)
        loss_repel = z.new_tensor(0.0)
        k_eff_hard = 0

        if z_all_neg.shape[0] > 0:
            # Detach p_norm so anomaly repulsion moves ONLY the feature extractor (z_all_neg),
            # never dragging the Normal proxy anchor away from the Normal cluster center.
            p_norm_anchor = p_norm.detach()
            sim_neg_norm = z_all_neg @ p_norm_anchor.T  # [N_anom, M]
            s_anom_max, _ = sim_neg_norm.max(dim=1)  # [N_anom]

            n_neg = z_all_neg.shape[0]
            raw_weights = torch.ones(n_neg, device=z.device, dtype=z.dtype)

            # Apply Class-Balanced Weight and Top-K Hard-Anomaly Boost ONLY to real defects
            if n_real_def > 0:
                cb_real = torch.stack(
                    [self._get_class_weight(int(c.item())) for c in y_all_neg[:n_real_def]]
                )
                raw_weights[:n_real_def] = cb_real

                if top_k_mode == "ratio":
                    ratio_val = max(0.05, min(1.0, float(top_k_value)))
                    k_eff_hard = max(1, min(n_real_def, int(np.ceil(n_real_def * ratio_val))))
                else:
                    k_eff_hard = max(1, min(n_real_def, int(round(float(top_k_value)))))

                real_intrusion = s_anom_max[:n_real_def].detach()
                _, hard_idx = torch.topk(real_intrusion, k=k_eff_hard, largest=True)
                raw_weights[hard_idx] = raw_weights[hard_idx] * self.hard_anomaly_boost

            # Synthetic Mixup/Transfer boundary samples get a gentle 0.25x relative weight
            if n_synth > 0:
                raw_weights[n_real_def:] = 0.25

            # Mean-normalize weights to 1.0 so the macro scale of Term ② is strictly 1.0
            norm_weights = raw_weights / (raw_weights.mean().detach() + 1e-8)

            # Primary Hinge Repulsion: max(0, S(x, p_norm) - delta) on [0.0, 1.0] scale
            hinge_primary = F.relu(s_anom_max - delta_margin)
            # Gentle secondary margin (-0.25) so anomalies do not stall exactly on delta=0.0
            deep_margin = min(delta_margin - 0.25, -0.25)
            hinge_secondary = F.relu(s_anom_max - deep_margin)
            per_sample_repel = 0.85 * hinge_primary + 0.15 * hinge_secondary

            loss_repel = (norm_weights * per_sample_repel).mean()

            # Optional XBM boundary normal repulsion (blended at 15% without changing macro scale)
            if (
                memory_bank is not None
                and len(memory_bank) > 0
                and memory_bank.normal_count > 0
                and n_real_def > 0
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
                    loss_repel = 0.85 * loss_repel + 0.15 * xbm_hinge

        # =========================================================================
        # ③ 補助項: 実不良サブクラス凝集 & プロキシ間直交正則化 (No force on z_norm!)
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
            loss_ortho = (
                loss_ortho
                + 0.25 * F.relu(norm_off_sims - self.ortho_margin_normal).pow(2).mean()
            )

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
            "loss_ortho": float(loss_ortho.detach().cpu().item()),
            "loss_total": float(loss_total.detach().cpu().item()),
            "lambda_norm": round(float(lambda_norm), 4),
            "lambda_anom": round(float(lambda_anom), 4),
            "effective_top_k": float(k_eff_hard),
        }
        return loss_total, stats
