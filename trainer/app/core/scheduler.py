"""Curriculum penalty scheduler coupled to validation Recall and high-dimensional minimum cosine margin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TopKMode = Literal["count", "ratio"]


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
        self.total_epochs = max(1, total_epochs)
        self.top_k_mode: TopKMode = top_k_mode
        self.top_k_initial = float(top_k_initial)
        self.top_k_final = float(top_k_final)
        self.gamma_initial = float(gamma_initial)
        self.gamma_final = float(gamma_final)
        self.margin_initial = float(margin_initial)
        self.margin_final = float(margin_final)
        self.target_d_min = max(0.1, float(target_d_min))
        self.base_normal_ratio = max(0.05, min(0.95, float(base_normal_ratio)))

    def step(
        self,
        epoch: int,
        val_recall: float,
        d_min_boundary: float,
        val_loss_centripetal: float = 0.0,
        val_loss_repel: float = 0.0,
        val_precision: float = 1.0,
    ) -> CurriculumState:
        """Compute updated curriculum parameters & loss-direction feedback for the next training epoch."""
        recall_clamped = max(0.0, min(1.0, float(val_recall)))
        margin_ratio = max(0.0, min(1.0, float(d_min_boundary) / self.target_d_min))
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
            loss_gap = float(val_loss_centripetal) - float(val_loss_repel)
            metric_gap = recall_clamped - max(0.0, min(1.0, float(val_precision)))
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
