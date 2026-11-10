"""Cross-Batch Memory Bank (XBM) with detached storage and per-epoch memory release."""

from __future__ import annotations

import gc
from typing import Literal

import torch
import torch.nn.functional as F

TopKMode = Literal["count", "ratio"]


class CrossBatchMemoryBank:
    """In-process embedding memory bank for mining global Top-K boundary normal samples.

    Design invariants:
    1. Embeddings are strictly detached (`tensor.detach().cpu()`) before storage so autograd
       computation graphs never accumulate in RAM/VRAM.
    2. `release_epoch_memory()` explicitly purges internal tensors and triggers garbage
       collection + accelerator cache cleanup at the end of every epoch to prevent OOM Kills.
    3. Supports both absolute count (`top_k_mode='count'`, e.g., Top 20) and percentage ratio
       (`top_k_mode='ratio'`, e.g., Top 10%) with safe clamping `K_eff = min(K, N_normal)`.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self._embeddings: dict[int, torch.Tensor] = {}
        self._labels: dict[int, int] = {}
        self._is_defect: dict[int, bool] = {}
        self.last_hard_normal_indices: list[int] = []

    def __len__(self) -> int:
        return len(self._embeddings)

    @property
    def normal_count(self) -> int:
        return sum(1 for is_def in self._is_defect.values() if not is_def)

    @property
    def defect_count(self) -> int:
        return sum(1 for is_def in self._is_defect.values() if is_def)

    def update(
        self,
        sample_indices: list[int] | torch.Tensor,
        embeddings: torch.Tensor,
        labels: list[int] | torch.Tensor,
        is_defect_flags: list[bool] | torch.Tensor,
    ) -> None:
        """Store detached L2-normalized 128-D embeddings keyed by dataset sample index."""
        detached = F.normalize(embeddings.detach().to(dtype=torch.float32, device="cpu"), p=2, dim=-1)
        if detached.requires_grad:  # pragma: no cover
            raise RuntimeError("Memory bank embeddings must never retain autograd graphs.")

        idx_list = (
            sample_indices.detach().cpu().tolist()
            if isinstance(sample_indices, torch.Tensor)
            else list(sample_indices)
        )
        lbl_list = (
            labels.detach().cpu().tolist() if isinstance(labels, torch.Tensor) else list(labels)
        )
        def_list = (
            is_defect_flags.detach().cpu().tolist()
            if isinstance(is_defect_flags, torch.Tensor)
            else list(is_defect_flags)
        )

        for i, sample_idx in enumerate(idx_list):
            sid = int(sample_idx)
            self._embeddings[sid] = detached[i].clone()
            self._labels[sid] = int(lbl_list[i])
            self._is_defect[sid] = bool(def_list[i])

    def resolve_effective_top_k(
        self,
        top_k_mode: TopKMode = "count",
        top_k_value: float = 20.0,
    ) -> int:
        """Compute effective K clamped by currently stored normal embeddings."""
        n_norm = self.normal_count
        if n_norm == 0:
            return 0
        if top_k_mode == "ratio":
            ratio = max(0.01, min(1.0, float(top_k_value)))
            target_k = max(1, int(round(n_norm * ratio)))
        else:
            target_k = max(1, int(round(top_k_value)))
        return min(target_k, n_norm)

    def get_top_k_boundary_normals(
        self,
        defect_embeddings: torch.Tensor,
        top_k_mode: TopKMode = "count",
        top_k_value: float = 20.0,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, list[int], int]:
        """Find the Top-K normal embeddings in the memory bank closest to each defect anchor.

        Returns:
            - `boundary_normals`: Tensor of shape `[N_defect_anchors, K_eff, D]` on `device`
            - `hard_normal_sample_indices`: Unique dataset indices of the mined boundary normals
            - `k_eff`: Effective K used
        """
        normal_ids = [sid for sid, is_def in self._is_defect.items() if not is_def]
        k_eff = self.resolve_effective_top_k(top_k_mode=top_k_mode, top_k_value=top_k_value)
        if not normal_ids or k_eff == 0 or defect_embeddings.numel() == 0:
            empty = torch.zeros(
                (defect_embeddings.shape[0], 0, self.embedding_dim),
                dtype=torch.float32,
                device=device,
            )
            return empty, [], 0

        bank_normals = torch.stack([self._embeddings[sid] for sid in normal_ids], dim=0).to(device)
        def_norm = F.normalize(defect_embeddings.detach(), p=2, dim=-1).to(device)

        # Cosine similarity [N_def, N_bank_normal]
        sim_matrix = def_norm @ bank_normals.T
        _, topk_cols = torch.topk(sim_matrix, k=k_eff, dim=1, largest=True, sorted=True)

        # Gather [N_def, K_eff, D]
        expanded_indices = topk_cols.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        expanded_bank = bank_normals.unsqueeze(0).expand(def_norm.shape[0], -1, -1)
        boundary_normals = torch.gather(expanded_bank, dim=1, index=expanded_indices)

        unique_cols = torch.unique(topk_cols).detach().cpu().tolist()
        hard_sample_ids = [normal_ids[c] for c in unique_cols]
        self.last_hard_normal_indices = hard_sample_ids
        return boundary_normals.detach(), hard_sample_ids, k_eff

    def get_defect_embeddings(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return all cached defect embeddings `[N_bank_defect, D]` for bidirectional repulsion."""
        defect_ids = [sid for sid, is_def in self._is_defect.items() if is_def]
        if not defect_ids:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32, device=device)
        return torch.stack([self._embeddings[sid] for sid in defect_ids], dim=0).to(device).detach()

    def get_normal_prototype(self, device: str | torch.device = "cpu") -> torch.Tensor | None:
        """Compute L2-normalized normal prototype center from cached normal embeddings."""
        normal_ids = [sid for sid, is_def in self._is_defect.items() if not is_def]
        if not normal_ids:
            return None
        normals = torch.stack([self._embeddings[sid] for sid in normal_ids], dim=0).to(device)
        proto = normals.mean(dim=0)
        return F.normalize(proto, p=2, dim=0).detach()

    def clear(self) -> None:
        """Clear all stored embeddings and metadata dictionaries."""
        self._embeddings.clear()
        self._labels.clear()
        self._is_defect.clear()

    def release_epoch_memory(self) -> dict[str, int]:
        """Explicitly release in-process memory at the end of an epoch to prevent OOM Kills."""
        released_entries = len(self._embeddings)
        self.clear()
        gc.collect()
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):  # pragma: no cover
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return {"released_entries": released_entries, "remaining_entries": len(self._embeddings)}
