"""Curriculum penalty scheduler coupled to validation Recall and high-dimensional minimum cosine margin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TopKMode = Literal["count", "ratio"]


def _safe_float(val: float, default: float = 0.0) -> float:
    f = float(val)
    return f if math.isfinite(f) else float(default)


@dataclass
class CurriculumState:
    """Snapshot of the Step 5 dynamic curriculum schedule & validation loss-direction feedback at epoch `t`."""

    epoch: int
    progress: float
    gamma_repel: float
    repel_margin: float
    top_k_mode: TopKMode
    top_k_value: float
    effective_normal_ratio: float = 0.50
    loss_direction_feedback: str = "initial_50_50 (初期比率)"


class CurriculumPenaltyScheduler:
    """Validation-guided penalty & loss-direction scheduler for Step 5 dynamic metric learning.

    Evaluates the model on the validation split (`val` — 20%) at the end of each epoch and
    feeds back to the next training epoch (`epoch + 1`):
    1. Curriculum boundary focus (`gamma_repel`, `repel_margin`, `top_k_value`) coupled to
       `val_recall` and validation `d_min_boundary`.
    2. Loss Direction Feedback (`effective_normal_ratio`, `loss_direction_feedback`) coupled to
       validation normal dispersion (`val_loss_centripetal`) vs. validation anomaly intrusion
       (`val_loss_repel`) and `val_recall` vs. `val_precision`.
    """

    def __init__(
        self,
        total_epochs: int = 10,
        top_k_mode: TopKMode = "count",
        top_k_initial: float = 50.0,
        top_k_final: float = 20.0,
        gamma_initial: float = 0.3,
        gamma_final: float = 1.5,
        margin_initial: float = 0.15,
        margin_final: float = -0.10,
        target_d_min: float = 1.0,
        base_normal_ratio: float = 0.50,
    ) -> None:
        self.total_epochs = max(1, int(total_epochs))
        self.top_k_mode: TopKMode = top_k_mode
        self.top_k_initial = _safe_float(top_k_initial, 50.0)
        self.top_k_final = _safe_float(top_k_final, 20.0)
        self.gamma_initial = _safe_float(gamma_initial, 0.3)
        self.gamma_final = _safe_float(gamma_final, 1.5)
        self.margin_initial = _safe_float(margin_initial, 0.15)
        self.margin_final = _safe_float(margin_final, -0.10)
        self.target_d_min = max(0.1, _safe_float(target_d_min, 1.0))
        self.base_normal_ratio = max(0.05, min(0.95, _safe_float(base_normal_ratio, 0.50)))

    def step(
        self,
        epoch: int,
        val_recall: float,
        d_min_boundary: float,
        val_loss_centripetal: float = 0.0,
        val_loss_repel: float = 0.0,
        val_precision: float = 1.0,
        has_val_defects: bool = True,
    ) -> CurriculumState:
        """Compute updated curriculum parameters & loss-direction feedback for the next training epoch."""
        recall_clamped = max(0.0, min(1.0, _safe_float(val_recall, 0.0)))
        d_min_clean = _safe_float(d_min_boundary, 0.0)
        margin_ratio = max(0.0, min(1.0, d_min_clean / self.target_d_min))
        epoch_ratio = max(0.0, min(1.0, float(epoch) / float(self.total_epochs)))

        # Composite progress prevents deadlock when early-epoch Recall@0.0 is 0
        progress = 0.45 * recall_clamped + 0.35 * margin_ratio + 0.20 * epoch_ratio
        progress = max(0.0, min(1.0, progress))

        gamma_t = self.gamma_initial + progress * (self.gamma_final - self.gamma_initial)
        margin_t = self.margin_initial + progress * (self.margin_final - self.margin_initial)
        top_k_t = self.top_k_initial + progress * (self.top_k_final - self.top_k_initial)

        if self.top_k_mode == "count":
            top_k_t = float(max(1, int(round(top_k_t))))
        else:
            top_k_t = round(max(0.01, min(1.0, top_k_t)), 4)

        if epoch <= 0:
            eff_ratio = self.base_normal_ratio
            direction = f"initial (基準比率 r={eff_ratio:.3f})"
        else:
            cent_clean = _safe_float(val_loss_centripetal, 0.0)
            rep_clean = _safe_float(val_loss_repel, 0.0)
            prec_clamped = max(0.0, min(1.0, _safe_float(val_precision, 1.0)))
            loss_gap = cent_clean - rep_clean
            # Do not starve anomaly repulsion when val has 0 defects or val_loss_repel is unmeasured (0.0) while Recall < 1
            if not has_val_defects or (rep_clean <= 1e-8 and recall_clamped < 1.0):
                loss_gap = min(0.0, loss_gap)
            metric_gap = 0.0 if not has_val_defects else (recall_clamped - prec_clamped)
            delta_r = max(-0.15, min(0.15, 0.25 * loss_gap + 0.10 * metric_gap))
            eff_ratio = round(max(0.10, min(0.90, self.base_normal_ratio + delta_r)), 4)
            if delta_r > 0.012:
                direction = f"centripetal_focus (良品凝集強化: r={eff_ratio:.3f})"
            elif delta_r < -0.012:
                direction = f"repulsion_focus (不良斥力強化: r={eff_ratio:.3f})"
            else:
                direction = f"balanced_hold (均衡維持: r={eff_ratio:.3f})"

        return CurriculumState(
            epoch=epoch,
            progress=round(progress, 4),
            gamma_repel=round(gamma_t, 4),
            repel_margin=round(margin_t, 4),
            top_k_mode=self.top_k_mode,
            top_k_value=top_k_t,
            effective_normal_ratio=eff_ratio,
            loss_direction_feedback=direction,
        )
