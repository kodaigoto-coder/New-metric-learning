"""Normal-Anomaly Balanced Sampler (NAB-Sampler) & Effective-Number Class-Balanced Weighting."""

from __future__ import annotations

import numpy as np


def compute_class_balanced_weights(
    labels: np.ndarray | list[int],
    num_defect_classes: int,
    beta: float = 0.99,
) -> np.ndarray:
    """Compute Effective-Number Class-Balanced Weights W_c for defect modes c in {1..C}.

    Follows Cui et al. (CVPR 2019) "Class-Balanced Loss Based on Effective Number of Samples":
        E_{n_c} = (1 - beta^{n_c}) / (1 - beta)
        W_c_raw = 1 / E_{n_c} = (1 - beta) / (1 - beta^{n_c})
    Normalized so that mean_{c=1..C}(W_c) = 1.0.

    Returns:
        weights: np.ndarray of shape (num_defect_classes,) with dtype float32,
                 where index `c - 1` corresponds to defect label_id `c`.
    """
    c_total = max(1, int(num_defect_classes))
    beta_val = float(beta) if np.isfinite(float(beta)) else 0.99
    beta_clamped = float(np.clip(beta_val, 0.5, 0.9999))
    labels_arr = np.asarray(labels, dtype=np.int64)

    raw_weights = np.ones(c_total, dtype=np.float64)
    for c_idx in range(1, c_total + 1):
        n_c = int(np.sum(labels_arr == c_idx))
        n_eff = max(1, n_c)
        raw_weights[c_idx - 1] = (1.0 - beta_clamped) / max(1e-8, 1.0 - (beta_clamped**n_eff))

    mean_w = float(np.mean(raw_weights))
    if mean_w > 1e-8:
        raw_weights = raw_weights / mean_w

    return raw_weights.astype(np.float32)


class NormalAnomalyBalancedSampler:
    """Asymmetric 50/50 Normal-Anomaly Balanced Sampler (NAB-Sampler).

    Constructs each mini-batch of size `N` with:
    1. Normal Slots (default 50% = N // 2 samples):
       Drawn across all Normal training samples (`label_id == 0`) to cover lot/lighting/position
       variations, while optionally injecting up to 35% hard boundary normals identified by XBM.
    2. Anomaly Slots (default 50% = N - N_norm samples):
       Stratified equally across existing defect classes (`label_id in {1..C}`).
       Even if a rare defect mode has only 1 or 2 training images, sampling with replacement
       within that mode guarantees multiple positive anchors per step and prevents gradient starvation.
    """

    def __init__(
        self,
        labels: np.ndarray | list[int],
        is_defect: np.ndarray | list[bool],
        splits: list[str] | None = None,
        batch_size: int = 32,
        normal_ratio: float = 0.50,
        seed: int = 42,
        normal_subclusters: np.ndarray | list[int] | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.is_defect = np.asarray(is_defect, dtype=bool)
        self.batch_size = max(4, int(batch_size))
        self.normal_ratio = float(np.clip(normal_ratio, 0.25, 0.75))
        self.rng = np.random.default_rng(seed)

        n_total = len(self.labels)
        if splits is not None:
            train_mask = np.array([s == "train" for s in splits], dtype=bool)
        else:
            train_mask = np.ones(n_total, dtype=bool)

        normal_train = np.where((~self.is_defect) & train_mask)[0].tolist()
        if not normal_train:
            normal_train = np.where(~self.is_defect)[0].tolist()
        self.normal_indices: list[int] = [int(i) for i in normal_train]

        # Optional latent normal sub-cluster grouping (M* >= 2) to prevent minority normal mode starvation
        self.normal_by_subcluster: dict[int, list[int]] = {}
        if normal_subclusters is not None:
            sub_arr = np.asarray(normal_subclusters, dtype=np.int64).reshape(-1)
            if len(sub_arr) == n_total:
                for idx in self.normal_indices:
                    sc_id = max(0, int(sub_arr[idx]))
                    self.normal_by_subcluster.setdefault(sc_id, []).append(int(idx))
        if not self.normal_by_subcluster and self.normal_indices:
            self.normal_by_subcluster = {0: list(self.normal_indices)}
        self.normal_subcluster_ids: list[int] = sorted(self.normal_by_subcluster.keys())

        defect_train = np.where(self.is_defect & train_mask)[0].tolist()
        if not defect_train:
            defect_train = np.where(self.is_defect)[0].tolist()
        self.defect_indices: list[int] = [int(i) for i in defect_train]

        # Group defect indices by their defect class label_id (1..C)
        self.defect_by_class: dict[int, list[int]] = {}
        for idx in self.defect_indices:
            cid = int(self.labels[idx])
            if cid <= 0:
                cid = 1
            self.defect_by_class.setdefault(cid, []).append(int(idx))

        # Ensure every defect class present anywhere in the dataset is represented
        all_defect_classes = sorted({int(c) for c in self.labels[self.is_defect] if int(c) > 0})
        for cid in all_defect_classes:
            if cid not in self.defect_by_class:
                fallback_ids = np.where(self.labels == cid)[0].tolist()
                if fallback_ids:
                    self.defect_by_class[cid] = [int(i) for i in fallback_ids]

        self.defect_class_ids: list[int] = sorted(self.defect_by_class.keys())

    def build_epoch_batches(
        self,
        hard_normal_indices: list[int] | None = None,
        min_steps: int = 12,
    ) -> list[list[int]]:
        """Generate one epoch of NAB-Sampler mini-batches."""
        if not self.normal_indices and not self.defect_indices:
            return []
        if not self.defect_class_ids:
            shuffled = list(self.normal_indices)
            self.rng.shuffle(shuffled)
            chunks = [
                shuffled[i : i + self.batch_size]
                for i in range(0, len(shuffled), self.batch_size)
            ]
            if len(chunks) >= 2 and len(chunks[-1]) == 1 and len(shuffled) >= 2:
                chunks[-1].append(int(shuffled[0]))
            return chunks
        if not self.normal_indices:
            shuffled_def = list(self.defect_indices)
            self.rng.shuffle(shuffled_def)
            chunks_def = [
                shuffled_def[i : i + self.batch_size]
                for i in range(0, len(shuffled_def), self.batch_size)
            ]
            if len(chunks_def) >= 2 and len(chunks_def[-1]) == 1 and len(shuffled_def) >= 2:
                chunks_def[-1].append(int(shuffled_def[0]))
            return chunks_def

        n_norm_slots = max(2, int(round(self.batch_size * self.normal_ratio)))
        n_def_slots = max(2, self.batch_size - n_norm_slots)
        n_norm_slots = max(2, self.batch_size - n_def_slots)

        norm_pool = list(self.normal_indices)
        self.rng.shuffle(norm_pool)

        # Ensure all normal samples are visited at least twice per epoch
        steps_for_normals = int(np.ceil(len(norm_pool) / max(1, n_norm_slots))) * 2
        num_batches = max(min_steps, steps_for_normals)

        valid_hard_normals = [
            int(idx) for idx in (hard_normal_indices or []) if idx in self.normal_indices
        ]

        batches: list[list[int]] = []
        norm_cursor = 0
        num_modes = len(self.defect_class_ids)
        num_norm_sub = len(self.normal_subcluster_ids)

        sub_pools: dict[int, list[int]] = {}
        sub_cursors: dict[int, int] = {}
        if num_norm_sub >= 2:
            for sc_id in self.normal_subcluster_ids:
                p_list = list(self.normal_by_subcluster[sc_id])
                self.rng.shuffle(p_list)
                sub_pools[sc_id] = p_list
                sub_cursors[sc_id] = 0

        for _ in range(num_batches):
            # 1. Fill Normal slots (50% of batch), stratifying across normal subclusters when M* >= 2
            selected_normals: list[int] = []
            if valid_hard_normals:
                n_hard = min(len(valid_hard_normals), max(1, int(n_norm_slots * 0.35)))
                hard_picks = self.rng.choice(
                    valid_hard_normals, size=n_hard, replace=False
                ).tolist()
                selected_normals.extend(int(x) for x in hard_picks)

            rem_norm = n_norm_slots - len(selected_normals)
            if rem_norm > 0 and num_norm_sub >= 2:
                sc_order = list(self.normal_subcluster_ids)
                self.rng.shuffle(sc_order)
                base_sc = rem_norm // num_norm_sub
                rem_sc = rem_norm % num_norm_sub
                for s_pos, sc_id in enumerate(sc_order):
                    k_sc = base_sc + (1 if s_pos < rem_sc else 0)
                    if k_sc <= 0:
                        continue
                    pool_sc = sub_pools[sc_id]
                    picked_sc = 0
                    sc_attempts = 0
                    while picked_sc < k_sc and pool_sc:
                        cand = int(pool_sc[sub_cursors[sc_id] % len(pool_sc)])
                        sub_cursors[sc_id] += 1
                        if sub_cursors[sc_id] % len(pool_sc) == 0:
                            self.rng.shuffle(pool_sc)
                        if (
                            cand in selected_normals
                            and len(pool_sc) >= k_sc
                            and sc_attempts < len(pool_sc)
                        ):
                            sc_attempts += 1
                            continue
                        selected_normals.append(cand)
                        picked_sc += 1
            else:
                attempts = 0
                while len(selected_normals) < n_norm_slots and norm_pool:
                    cand = int(norm_pool[norm_cursor % len(norm_pool)])
                    norm_cursor += 1
                    if norm_cursor % len(norm_pool) == 0:
                        self.rng.shuffle(norm_pool)
                    if (
                        cand in selected_normals
                        and len(norm_pool) >= n_norm_slots
                        and attempts < len(norm_pool)
                    ):
                        attempts += 1
                        continue
                    selected_normals.append(cand)

            # 2. Fill Anomaly slots (50% of batch) stratified across defect classes
            selected_defects: list[int] = []
            if num_modes <= n_def_slots // 2 or num_modes <= 4:
                chosen_modes = list(self.defect_class_ids)
                self.rng.shuffle(chosen_modes)
            else:
                # If many defect modes exist, pick up to n_def_slots // 2 modes per step so each gets >=2 samples
                max_active_modes = max(1, n_def_slots // 2)
                chosen_modes = self.rng.choice(
                    self.defect_class_ids, size=max_active_modes, replace=False
                ).tolist()

            base_per_mode = n_def_slots // len(chosen_modes)
            remainder = n_def_slots % len(chosen_modes)

            for m_idx, cid in enumerate(chosen_modes):
                k_mode = base_per_mode + (1 if m_idx < remainder else 0)
                if k_mode <= 0:
                    continue
                mode_pool = self.defect_by_class[int(cid)]
                replace_flag = len(mode_pool) < k_mode
                picks = self.rng.choice(mode_pool, size=k_mode, replace=replace_flag).tolist()
                selected_defects.extend(int(x) for x in picks)

            batch_indices = selected_normals + selected_defects
            batches.append(batch_indices)

        return batches
