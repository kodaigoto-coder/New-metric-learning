"""Asymmetric Multi-Similarity Loss (MS-Loss) and Bidirectional Top-K Boundary Repulsion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.memory_bank import CrossBatchMemoryBank, TopKMode


class AsymmetricMultiSimilarityLoss(nn.Module):
    """Pseudo-multiclass Asymmetric Multi-Similarity Loss on the 128-D Unit Hypersphere.

    Implements the GPW framework (Wang et al., CVPR 2019) adapted for weakly
    supervised industrial anomaly detection:
    1. Positive Pairs (P_i): Contracts normal samples tightly around the normal hemisphere,
       and contracts local same-mode defect neighbors (`top_k_pos`) so multi-modal defects
       do not collapse across the normal core even when grouped under a single 'anomaly' folder.
    2. Primary Negative Pairs (N_i^primary): Strongly repels Normal (class 0) vs. Defect modes
       (class > 0) with dynamic class imbalance weight `w_defect` and negative margin
       `lambda_primary = -0.15` to drive normal-defect cosine similarity below `tau_sim = 0.0`.
    3. Auxiliary Negative Pairs (N_i^aux): Moderately separates distinct defect modes
       (Defect Mode A vs. Defect Mode B) with weight `w_aux` and margin `lambda_aux`.
    4. Bidirectional XBM Boundary Repulsion (`L_repel`): Repels live defect anchors away from
       the global Top-K closest boundary normal embeddings in the Cross-Batch Memory Bank, and
       repels live normal anchors away from cached defect embeddings.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 16.0,
        lambda_pos: float = 0.75,
        lambda_primary: float = -0.15,
        lambda_aux: float = 0.15,
        epsilon: float = 0.1,
        w_defect: float = 2.5,
        w_aux: float = 0.5,
        beta_repel: float = 12.0,
        local_pos_k: int = 3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_pos = lambda_pos
        self.lambda_primary = lambda_primary
        self.lambda_aux = lambda_aux
        self.epsilon = epsilon
        self.w_defect = w_defect
        self.w_aux = w_aux
        self.beta_repel = beta_repel
        self.local_pos_k = local_pos_k

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        memory_bank: CrossBatchMemoryBank | None = None,
        gamma_repel: float = 0.5,
        repel_margin: float = 0.0,
        top_k_mode: TopKMode = "count",
        top_k_value: float = 20.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute `L_total = L_MS + gamma(t) * L_repel` and diagnostic metrics."""
        z = F.normalize(embeddings, p=2, dim=-1)
        batch_size = z.shape[0]
        if batch_size < 2:
            zero = z.sum() * 0.0
            return zero, {
                "loss_ms": 0.0,
                "loss_repel": 0.0,
                "loss_total": 0.0,
                "effective_top_k": 0.0,
            }

        sim_mat = z @ z.T  # [B, B] in [-1, 1]
        labels = labels.view(-1)
        is_defect = labels > 0

        loss_ms_terms: list[torch.Tensor] = []

        for i in range(batch_size):
            yi = int(labels[i].item())
            anchor_is_defect = yi > 0
            anchor_weight = self.w_defect if anchor_is_defect else 1.0

            same_mask = labels == yi
            same_mask[i] = False
            diff_mask = labels != yi

            pos_sims = sim_mat[i][same_mask]
            neg_sims_all = sim_mat[i][diff_mask]

            # For defect anchors, use local k-nearest positive mining so multi-modal defects
            # inside a single 'anomaly' folder form sub-clusters instead of pulling across the normal core
            if anchor_is_defect and pos_sims.numel() > self.local_pos_k:
                pos_sims, _ = torch.topk(pos_sims, k=self.local_pos_k, largest=True)

            # Pair mining thresholds per Wang et al. (CVPR 2019)
            if neg_sims_all.numel() > 0:
                max_neg = neg_sims_all.max().detach()
                pos_mined = pos_sims[pos_sims < (max_neg + self.epsilon)]
                if pos_mined.numel() == 0:
                    pos_mined = pos_sims
            else:
                pos_mined = pos_sims

            if pos_sims.numel() > 0:
                min_pos = pos_sims.min().detach()
            else:
                min_pos = torch.tensor(1.0, device=z.device)

            # Numerically stable Positive MS term via logsumexp
            if pos_mined.numel() > 0:
                pos_exponents = -self.alpha * (pos_mined - self.lambda_pos)
                zero_pad = z.new_zeros(1)
                pos_term = (1.0 / self.alpha) * torch.logsumexp(
                    torch.cat([zero_pad, pos_exponents], dim=0), dim=0
                )
            else:
                pos_term = z.new_tensor(0.0)

            # Split negatives into Primary (Normal vs Defect) and Auxiliary (Defect A vs Defect B)
            if anchor_is_defect:
                primary_neg_mask = labels == 0
                aux_neg_mask = (labels > 0) & (labels != yi)
            else:
                primary_neg_mask = labels > 0
                aux_neg_mask = torch.zeros_like(labels, dtype=torch.bool)

            prim_sims = sim_mat[i][primary_neg_mask]
            aux_sims = sim_mat[i][aux_neg_mask]

            neg_exponents_list: list[torch.Tensor] = [z.new_zeros(1)]
            if prim_sims.numel() > 0:
                prim_mined = prim_sims[prim_sims > (min_pos - self.epsilon)]
                if prim_mined.numel() == 0:
                    prim_mined = prim_sims
                log_w_def = torch.log(z.new_tensor(max(1e-6, self.w_defect)))
                neg_exponents_list.append(
                    log_w_def + self.beta * (prim_mined - self.lambda_primary)
                )

            if aux_sims.numel() > 0 and self.w_aux > 0.0:
                aux_mined = aux_sims[aux_sims > (min_pos - self.epsilon)]
                if aux_mined.numel() == 0:
                    aux_mined = aux_sims
                log_w_aux = torch.log(z.new_tensor(max(1e-6, self.w_aux)))
                neg_exponents_list.append(
                    log_w_aux + self.beta * (aux_mined - self.lambda_aux)
                )

            neg_term = (1.0 / self.beta) * torch.logsumexp(
                torch.cat(neg_exponents_list, dim=0), dim=0
            )
            loss_ms_terms.append(anchor_weight * (pos_term + neg_term))

        loss_ms = torch.stack(loss_ms_terms).mean()

        # In-batch Normal Prototype Hemisphere Alignment (guarantees separation across tau_sim = 0.0)
        defect_anchors = z[is_defect]
        normal_anchors = z[~is_defect]
        if normal_anchors.shape[0] > 0:
            batch_proto = F.normalize(normal_anchors.mean(dim=0), p=2, dim=-1)
            norm_proto_sim = normal_anchors @ batch_proto
            loss_proto_norm = F.relu(0.70 - norm_proto_sim).mean()
            if defect_anchors.shape[0] > 0:
                def_proto_sim = defect_anchors @ batch_proto
                loss_proto_def = F.relu(def_proto_sim - (-0.25)).mean() * self.w_defect
                loss_ms = loss_ms + loss_proto_norm + loss_proto_def
            else:
                loss_ms = loss_ms + loss_proto_norm

        # Step 5: Cross-Batch Memory Bank Bidirectional Top-K Boundary Repulsion
        loss_repel = z.new_tensor(0.0)
        k_eff = 0

        if memory_bank is not None and len(memory_bank) > 0 and gamma_repel > 0.0:
            repel_parts: list[torch.Tensor] = []
            target_margin = min(repel_margin, -0.10)

            # Direction 1: Live defect anchors vs. XBM Top-K closest boundary normals
            if defect_anchors.shape[0] > 0 and memory_bank.normal_count > 0:
                boundary_normals, _, k_eff = memory_bank.get_top_k_boundary_normals(
                    defect_embeddings=defect_anchors,
                    top_k_mode=top_k_mode,
                    top_k_value=top_k_value,
                    device=z.device,
                )
                if k_eff > 0:
                    sim_ik = torch.bmm(
                        defect_anchors.unsqueeze(1), boundary_normals.transpose(1, 2)
                    ).squeeze(1)
                    # Stable logsumexp over Top-K boundary normals
                    exp_ik = self.beta_repel * (sim_ik - target_margin)
                    zeros_col = z.new_zeros((exp_ik.shape[0], 1))
                    term_def = (1.0 / self.beta_repel) * torch.logsumexp(
                        torch.cat([zeros_col, exp_ik], dim=1), dim=1
                    )
                    repel_parts.append(term_def.mean())

            # Direction 2: Live normal anchors vs. XBM cached defect embeddings
            if normal_anchors.shape[0] > 0 and memory_bank.defect_count > 0:
                bank_defects = memory_bank.get_defect_embeddings(device=z.device)
                sim_jd = normal_anchors @ bank_defects.T
                exp_jd = self.beta_repel * (sim_jd - target_margin)
                zeros_col = z.new_zeros((exp_jd.shape[0], 1))
                term_norm = (1.0 / self.beta_repel) * torch.logsumexp(
                    torch.cat([zeros_col, exp_jd], dim=1), dim=1
                )
                repel_parts.append(term_norm.mean())

            # Direction 3: XBM global prototype hemisphere alignment around tau_sim = 0.0
            normal_proto = memory_bank.get_normal_prototype(device=z.device)
            if normal_proto is not None:
                if defect_anchors.shape[0] > 0:
                    proto_sim_def = defect_anchors @ normal_proto
                    repel_parts.append(
                        F.softplus(self.beta_repel * (proto_sim_def - target_margin)).mean()
                        / self.beta_repel
                    )
                if normal_anchors.shape[0] > 0:
                    proto_sim_norm = normal_anchors @ normal_proto
                    repel_parts.append(
                        F.softplus(self.beta_repel * (0.70 - proto_sim_norm)).mean()
                        / self.beta_repel
                    )

            if repel_parts:
                loss_repel = torch.stack(repel_parts).mean()

        loss_total = loss_ms + (gamma_repel * loss_repel)
        stats = {
            "loss_ms": float(loss_ms.detach().cpu().item()),
            "loss_repel": float(loss_repel.detach().cpu().item()),
            "loss_total": float(loss_total.detach().cpu().item()),
            "effective_top_k": float(k_eff),
        }
        return loss_total, stats
