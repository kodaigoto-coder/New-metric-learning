"""Curriculum penalty scheduler coupled to validation Recall and high-dimensional minimum cosine margin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TopKMode = Literal["count", "ratio"]


@dataclass
class CurriculumState:
    """Snapshot of the Step 5 dynamic curriculum schedule at epoch `t`."""

    epoch: int
    progress: float
    gamma_repel: float
    repel_margin: float
    top_k_mode: TopKMode
    top_k_value: float


class CurriculumPenaltyScheduler:
    """Progress-driven penalty scheduler for Step 5 dynamic boundary repulsion.

    Couples validation Recall (at `tau_sim`) and the continuous minimum boundary-to-defect
    cosine distance `d_min_boundary` (plus a small epoch progress component to prevent early
    deadlocks when `Recall@0.0 == 0`) to smoothly transition from:
    - Early stage: weak repulsion `gamma_init`, relaxed margin `margin_init`, wide `top_k_initial`
    - Late stage: strong repulsion `gamma_final`, strict margin `margin_final`, focused `top_k_final`
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

    def step(
        self,
        epoch: int,
        val_recall: float,
        d_min_boundary: float,
    ) -> CurriculumState:
        """Compute updated curriculum parameters for the next training epoch."""
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

        return CurriculumState(
            epoch=epoch,
            progress=round(progress, 4),
            gamma_repel=round(gamma_t, 4),
            repel_margin=round(margin_t, 4),
            top_k_mode=self.top_k_mode,
            top_k_value=top_k_t,
        )
