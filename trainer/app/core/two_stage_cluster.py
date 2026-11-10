"""Two-stage density-decoupled clustering pipeline for 100:2 imbalanced datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def compute_density_scores(features: np.ndarray, k_nn: int = 15) -> np.ndarray:
    """Compute global + local cosine density scores in [0, 1] for Stage 1 Normal Core extraction.

    Combines cosine similarity to the robust global median prototype with mean k-NN cosine
    similarity (using a sufficiently large k_nn) so that tiny clusters of identical defects
    are not misclassified as part of the majority normal core.
    """
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.zeros((0,), dtype=np.float32)
    if n_samples == 1:
        return np.ones((1,), dtype=np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = (features / norms).astype(np.float32)

    # Robust global center via component-wise median normalized to unit sphere
    median_vec = np.median(normed, axis=0)
    median_vec = median_vec / (np.linalg.norm(median_vec) + 1e-8)
    global_sim = normed @ median_vec  # [-1, 1]

    # Pairwise cosine similarity matrix
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, -np.inf)

    k_eff = min(max(1, k_nn), n_samples - 1)
    top_k_sims = np.partition(sim_matrix, -k_eff, axis=1)[:, -k_eff:]
    local_knn_sim = top_k_sims.mean(axis=1)

    combined_score = 0.65 * global_sim + 0.35 * local_knn_sim
    return combined_score.astype(np.float32)


def reduce_tail_pca(
    tail_features: np.ndarray,
    max_components: int = 64,
    variance_threshold: float = 0.95,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce tail features via PCA with upper dimension bound and 95% cumulative variance slicing."""
    n_tail, n_feat = tail_features.shape
    max_possible = min(max_components, n_tail - 1, n_feat)
    if max_possible < 2:
        return tail_features.astype(np.float32)

    pca = PCA(n_components=max_possible, random_state=random_state)
    transformed = pca.fit_transform(tail_features)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    keep_dims = int(np.searchsorted(cum_var, variance_threshold) + 1)
    keep_dims = max(2, min(keep_dims, max_possible))
    return transformed[:, :keep_dims].astype(np.float32)


def run_two_stage_clustering(
    features: np.ndarray,
    image_paths: list[str],
    core_quantile: float = 0.85,
    k_nn: int = 15,
    max_k: int = 8,
    max_pca_dim: int = 64,
    pca_variance: float = 0.95,
    backbone_name: str = "efficientnet_b3",
    ground_truth_hints: list[str | None] | None = None,
    output_json_path: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute the 2-stage clustering pipeline and optionally persist `clustering_result.json`.

    - Stage 1: Separates the high-density majority Normal Core (`cluster_id = 0`) using
      robust global + k-NN cosine density (`core_quantile`).
    - Stage 2: Runs PCA + Silhouette-score-maximized K-means on the remaining low-density
      tail (`cluster_id = 1..k*`) so minority defect modes are isolated without distorting
      the raw 100:2 distribution.
    """
    n_samples = features.shape[0]
    if n_samples != len(image_paths):
        raise ValueError("Length of features and image_paths must match.")
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for two-stage clustering.")

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed_features = (features / norms).astype(np.float32)

    # Stage 1: High-Density Normal Core Extraction
    density_scores = compute_density_scores(normed_features, k_nn=k_nn)
    clamped_quantile = float(np.clip(core_quantile, 0.50, 0.97))
    # Ensure at least 3 samples remain in the tail for Stage 2 clustering
    n_core = min(int(round(n_samples * clamped_quantile)), max(1, n_samples - 3))
    sorted_indices = np.argsort(density_scores)[::-1]
    core_indices = sorted_indices[:n_core]
    tail_indices = sorted_indices[n_core:]

    is_normal_core = np.zeros(n_samples, dtype=bool)
    is_normal_core[core_indices] = True

    cluster_ids = np.zeros(n_samples, dtype=np.int64)
    silhouette_history: dict[str, float] = {}
    best_tail_k = 1

    # Stage 2: Low-Density Tail PCA + Silhouette-Optimized K-Means
    n_tail = len(tail_indices)
    if n_tail >= 3:
        tail_feats = normed_features[tail_indices]
        tail_pca = reduce_tail_pca(
            tail_feats,
            max_components=max_pca_dim,
            variance_threshold=pca_variance,
            random_state=random_state,
        )
        tail_pca_norm = tail_pca / (np.linalg.norm(tail_pca, axis=1, keepdims=True) + 1e-8)

        upper_k = min(max_k, n_tail - 1)
        best_score = -2.0
        best_labels = np.zeros(n_tail, dtype=np.int64)

        for k_cand in range(2, max(3, upper_k + 1)):
            if k_cand >= n_tail:
                break
            km = KMeans(n_clusters=k_cand, n_init=10, random_state=random_state)
            cand_labels = km.fit_predict(tail_pca_norm)
            if len(np.unique(cand_labels)) < 2:
                continue
            score = float(silhouette_score(tail_pca_norm, cand_labels, metric="cosine"))
            silhouette_history[str(k_cand)] = round(score, 5)
            if score > best_score:
                best_score = score
                best_tail_k = k_cand
                best_labels = cand_labels.astype(np.int64)

        # Assign tail clusters 1 .. best_tail_k
        cluster_ids[tail_indices] = best_labels + 1
    else:
        cluster_ids[tail_indices] = 1

    # Compute 2D PCA preview coordinates across all samples for UI visualization
    if n_samples >= 2 and normed_features.shape[1] >= 2:
        preview_pca = PCA(n_components=2, random_state=random_state)
        coords_2d = preview_pca.fit_transform(normed_features).astype(np.float32)
    else:
        coords_2d = np.zeros((n_samples, 2), dtype=np.float32)

    items: list[dict[str, Any]] = []
    for i in range(n_samples):
        img_p = Path(image_paths[i])
        hint = ground_truth_hints[i] if ground_truth_hints and i < len(ground_truth_hints) else None
        if hint is None:
            parts = img_p.parts
            if "raw" in parts:
                raw_idx = parts.index("raw")
                if raw_idx + 2 < len(parts):
                    hint = parts[raw_idx + 1]

        if hint is None:
            fname = img_p.name
            if "defect_scratch" in fname:
                hint = "defect_scratch"
            elif "defect_stain" in fname:
                hint = "defect_stain"
            elif "defect_dent" in fname:
                hint = "defect_dent"
            elif "normal" in fname:
                hint = "normal"

        items.append(
            {
                "image_path": image_paths[i],
                "file_name": img_p.name,
                "cluster_id": int(cluster_ids[i]),
                "is_normal_core": bool(is_normal_core[i]),
                "density_score": round(float(density_scores[i]), 5),
                "preview_2d": [round(float(coords_2d[i, 0]), 4), round(float(coords_2d[i, 1]), 4)],
                "subfolder_class": hint,
                "ground_truth_hint": hint,
            }
        )

    detected_classes = sorted(
        {str(it["ground_truth_hint"]) for it in items if it.get("ground_truth_hint")}
    )

    result: dict[str, Any] = {
        "backbone_name": backbone_name,
        "total_samples": int(n_samples),
        "normal_core_count": int(n_core),
        "tail_count": int(n_tail),
        "core_quantile": float(clamped_quantile),
        "best_tail_k": int(best_tail_k),
        "total_clusters": int(1 + best_tail_k),
        "detected_classes": detected_classes,
        "silhouette_scores": silhouette_history,
        "items": items,
    }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_json_path)

    return result
