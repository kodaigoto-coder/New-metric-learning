"""Customized Multi-Proxy Anchor Loss with Sub-Center Normal Proxies, Top-K Hard-Anomaly Boost, One-Way Synthetic Repulsion, Class-Balanced Weighting, and Proxy Orthogonality."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from app.core.memory_bank import CrossBatchMemoryBank, TopKMode


class CustomizedMultiProxyAnchorLoss(nn.Module):
    """Multi-Proxy Anchor Loss tailored for Weakly-Supervised Industrial Anomaly Detection.

    Resolves the two fundamental failure modes of sample-pair MS-Loss:
    1. Zero Inter-Defect Sample Repulsion:
       Replaces sample-vs-sample pair repulsion with sample-vs-proxy anchors and completely
       eliminates repulsion between Defect Mode A samples and Defect Mode B samples/proxies.
       100% of negative sample gradients are concentrated on separating Normal vs. Anomaly.
    2. Multi-Proxy Normal Sub-Centers (M = 3..5):
       Assigns M learnable proxies `{p_norm^(1) .. p_norm^(M)}` to the Normal class so lot,
       illumination, and positional variations are absorbed by distinct normal sub-centers.
    3. Top-K Hard Anomaly Penalty Boost:
       Identifies the Top-K (or Top-q%) anomaly samples with the highest maximum cosine
       similarity `s_norm(z_i) = max_m <z_i, p_norm^(m)>` (dangerous pseudo-normals) and
       amplifies their repulsion weight by `hard_anomaly_boost` (lambda_hard > 1).
    4. One-Way Boundary Repulsion for Synthetic Samples (Anomalous Mixup & Defect Transfer):
       Synthetic hard samples (`synth_embeddings`) are strongly repelled from all M Normal
       proxies in `L_repel`, but are strictly excluded from `L_def_pos` so they never drag
       Defect proxies toward the Normal region.
    5. Proxy-Only Orthogonality Regularizer (`L_proxy_ortho`):
       Applies a lightweight cosine hinge penalty purely between proxy parameter vectors
       `{p_def^(1) .. p_def^(C)}` so defect proxies remain distinct in 128-D space without
       consuming any sample feature gradients.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_normal_proxies: int = 4,
        num_defect_classes: int = 3,
        class_balanced_weights: np.ndarray | list[float] | None = None,
        alpha_pos: float = 16.0,
        gamma_neg: float = 20.0,
        delta_pos: float = 0.65,
        delta_neg: float = -0.15,
        w_defect: float = 2.5,
        hard_anomaly_boost: float = 3.0,
        ortho_margin_defect: float = 0.20,
        ortho_margin_normal: float = 0.85,
        ortho_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_normal_proxies = max(1, int(num_normal_proxies))
        self.num_defect_classes = max(1, int(num_defect_classes))
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
                perturb = torch.randn_like(base_t) * 0.08
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
                # Subtract positive projection onto p_norm_mean so defect proxy starts in negative hemisphere
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
        repel_margin: float = -0.15,
        top_k_mode: TopKMode = "count",
        top_k_value: float = 20.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Customized Multi-Proxy Anchor Loss and diagnostic statistics."""
        z = F.normalize(embeddings, p=2, dim=-1)
        labels = labels.view(-1)
        batch_size = z.shape[0]
        if batch_size < 2:
            zero = z.sum() * 0.0
            return zero, {
                "loss_ms": 0.0,
                "loss_repel": 0.0,
                "loss_ortho": 0.0,
                "loss_total": 0.0,
                "effective_top_k": 0.0,
            }

        p_norm, p_def = self.get_normalized_proxies()  # [M, D], [C, D]
        is_defect = labels > 0
        z_norm = z[~is_defect]
        z_def = z[is_defect]
        y_def = labels[is_defect]

        # -------------------------------------------------------------------------
        # 1. L_norm_pos: Multi-Proxy Sub-Center Attraction for Normal Samples
        # Each normal sample z_j is pulled to its nearest normal sub-center proxy m*(j)
        # -------------------------------------------------------------------------
        loss_norm_pos = z.new_tensor(0.0)
        if z_norm.shape[0] > 0:
            sim_jn = z_norm @ p_norm.T  # [B_norm, M]
            nearest_m = torch.argmax(sim_jn.detach(), dim=1)  # [B_norm]
            active_norm_terms: list[torch.Tensor] = []

            for m_idx in range(self.num_normal_proxies):
                mask_m = nearest_m == m_idx
                if torch.any(mask_m):
                    sims_m = sim_jn[mask_m, m_idx]
                    exps = -self.alpha_pos * (sims_m - self.delta_pos)
                    term_m = (1.0 / self.alpha_pos) * torch.logsumexp(
                        torch.cat([z.new_zeros(1), exps], dim=0), dim=0
                    )
                    active_norm_terms.append(term_m)

            if active_norm_terms:
                loss_norm_pos = torch.stack(active_norm_terms).mean()

        # -------------------------------------------------------------------------
        # 2. L_def_pos: Class-Balanced Defect Mode Attraction (REAL defects only!)
        # Synthetic Mixup/Transfer samples are strictly excluded so they never drag
        # p_def^(c) toward the Normal cluster.
        # -------------------------------------------------------------------------
        loss_def_pos = z.new_tensor(0.0)
        if z_def.shape[0] > 0:
            sim_id = z_def @ p_def.T  # [B_def, C]
            active_def_terms: list[torch.Tensor] = []

            for c_idx in range(1, self.num_defect_classes + 1):
                mask_c = y_def == c_idx
                if torch.any(mask_c):
                    sims_c = sim_id[mask_c, c_idx - 1]
                    w_c = self._get_class_weight(c_idx)
                    exps = -self.alpha_pos * (sims_c - self.delta_pos)
                    term_c = (
                        w_c
                        * (1.0 / self.alpha_pos)
                        * torch.logsumexp(torch.cat([z.new_zeros(1), exps], dim=0), dim=0)
                    )
                    active_def_terms.append(term_c)

            if active_def_terms:
                loss_def_pos = torch.stack(active_def_terms).mean()

        loss_pos = loss_norm_pos + loss_def_pos

        # -------------------------------------------------------------------------
        # 3. L_repel: Zero Inter-Defect Repulsion + Top-K Hard Anomaly Boost
        # Combines real defect embeddings + one-way synthetic boundary embeddings
        # -------------------------------------------------------------------------
        if synth_embeddings is not None and synth_embeddings.shape[0] > 0:
            z_synth = F.normalize(synth_embeddings, p=2, dim=-1)
            if synth_labels is not None and synth_labels.numel() == z_synth.shape[0]:
                y_synth = synth_labels.view(-1)
            else:
                y_synth = torch.ones(z_synth.shape[0], dtype=torch.long, device=z.device)
            z_all_neg = torch.cat([z_def, z_synth], dim=0) if z_def.shape[0] > 0 else z_synth
            y_all_neg = torch.cat([y_def, y_synth], dim=0) if z_def.shape[0] > 0 else y_synth
        else:
            z_all_neg = z_def
            y_all_neg = y_def

        target_neg_margin = min(float(repel_margin), self.delta_neg)
        repel_terms: list[torch.Tensor] = []
        k_eff_hard = 0

        # Direction A: Repel all Real + Synthetic Anomalies from ALL M Normal Proxies
        # with Top-K Hard Anomaly Penalty Boost for the samples closest to Normal proxies
        if z_all_neg.shape[0] > 0:
            sim_neg_norm = z_all_neg @ p_norm.T  # [N_neg, M]
            # Worst-case intrusion score into the Normal multi-proxy region
            intrusion_scores = sim_neg_norm.amax(dim=1).detach()  # [N_neg]
            n_neg = z_all_neg.shape[0]

            if top_k_mode == "ratio":
                ratio_val = max(0.05, min(1.0, float(top_k_value)))
                k_eff_hard = max(1, min(n_neg, int(np.ceil(n_neg * ratio_val))))
            else:
                k_eff_hard = max(1, min(n_neg, int(round(float(top_k_value)))))

            _, hard_idx = torch.topk(intrusion_scores, k=k_eff_hard, largest=True)
            boost_vec = torch.ones(n_neg, device=z.device, dtype=z.dtype)
            boost_vec[hard_idx] = self.hard_anomaly_boost

            # Multiply by per-class Class-Balanced Weight W_c and global w_defect
            cb_vec = torch.stack([self._get_class_weight(int(c.item())) for c in y_all_neg])
            omega_vec = (boost_vec * cb_vec * self.w_defect).clamp(min=1e-4)
            log_omega = torch.log(omega_vec)  # [N_neg]

            norm_anchor_losses: list[torch.Tensor] = []
            for m_idx in range(self.num_normal_proxies):
                exps_m = log_omega + self.gamma_neg * (sim_neg_norm[:, m_idx] - target_neg_margin)
                term_m = (1.0 / self.gamma_neg) * torch.logsumexp(
                    torch.cat([z.new_zeros(1), exps_m], dim=0), dim=0
                )
                norm_anchor_losses.append(term_m)

            repel_terms.append(torch.stack(norm_anchor_losses).mean())

        # Direction B: Repel Normal Samples from ALL C Defect Mode Proxies
        if z_norm.shape[0] > 0:
            sim_norm_def = z_norm @ p_def.T  # [B_norm, C]
            def_anchor_losses: list[torch.Tensor] = []
            for c_idx in range(1, self.num_defect_classes + 1):
                w_c = self._get_class_weight(c_idx)
                exps_c = self.gamma_neg * (sim_norm_def[:, c_idx - 1] - target_neg_margin)
                term_c = (
                    w_c
                    * (1.0 / self.gamma_neg)
                    * torch.logsumexp(torch.cat([z.new_zeros(1), exps_c], dim=0), dim=0)
                )
                def_anchor_losses.append(term_c)

            repel_terms.append(torch.stack(def_anchor_losses).mean())

        # Direction C: XBM Top-K Boundary Normal Repulsion (pushes hard anomalies away from
        # the actual boundary normal samples cached in the Cross-Batch Memory Bank)
        if (
            memory_bank is not None
            and len(memory_bank) > 0
            and memory_bank.normal_count > 0
            and z_all_neg.shape[0] > 0
        ):
            boundary_normals, _, k_xbm = memory_bank.get_top_k_boundary_normals(
                defect_embeddings=z_all_neg,
                top_k_mode=top_k_mode,
                top_k_value=top_k_value,
                device=z.device,
            )
            if k_xbm > 0:
                k_eff_hard = max(k_eff_hard, int(k_xbm))
                sim_ik = torch.bmm(
                    z_all_neg.unsqueeze(1), boundary_normals.transpose(1, 2)
                ).squeeze(1)
                exp_ik = self.gamma_neg * (sim_ik - target_neg_margin)
                zeros_col = z.new_zeros((exp_ik.shape[0], 1))
                xbm_term = (1.0 / self.gamma_neg) * torch.logsumexp(
                    torch.cat([zeros_col, exp_ik], dim=1), dim=1
                )
                repel_terms.append(xbm_term.mean())

        loss_repel = torch.stack(repel_terms).mean() if repel_terms else z.new_tensor(0.0)

        # -------------------------------------------------------------------------
        # 4. L_proxy_ortho: Proxy-Parameter-Only Orthogonality Regularizer
        # Prevents C defect proxies from collapsing onto a single antipodal point -mu_norm
        # and prevents M normal sub-centers from collapsing into a single point.
        # -------------------------------------------------------------------------
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

        repel_scale = max(0.5, float(gamma_repel))
        loss_total = loss_pos + (repel_scale * loss_repel) + (self.ortho_weight * loss_ortho)

        stats = {
            "loss_ms": float(loss_pos.detach().cpu().item()),
            "loss_repel": float(loss_repel.detach().cpu().item()),
            "loss_ortho": float(loss_ortho.detach().cpu().item()),
            "loss_total": float(loss_total.detach().cpu().item()),
            "effective_top_k": float(k_eff_hard),
        }
        return loss_total, stats
