"""Phase 1 Clustering: Fixed Normal Class (Cluster 0) + Anomaly-Only Sub-Clustering via PCA & Silhouette K-Means."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

_NORMAL_CANDIDATE_NAMES = {"normal", "nomal", "good", "ok", "良品", "正常", "pass"}


def _resolve_sample_hints(
    image_paths: list[str],
    ground_truth_hints: list[str | None] | None = None,
) -> list[str | None]:
    """Resolve subfolder or filename class hints for every sample."""
    resolved: list[str | None] = []
    for i, path_str in enumerate(image_paths):
        img_p = Path(path_str)
        hint = (
            ground_truth_hints[i]
            if (ground_truth_hints is not None and i < len(ground_truth_hints))
            else None
        )
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
            elif "normal" in fname or "nomal" in fname:
                hint = "normal"
        resolved.append(hint)
    return resolved


def _detect_normal_class_name(
    hints: list[str | None],
    requested_normal_class: str = "normal",
) -> str | None:
    """Identify which subfolder/hint name represents the Normal (OK) class."""
    unique_hints = sorted({h for h in hints if h})
    if not unique_hints:
        return None

    req_clean = requested_normal_class.strip()
    for h in unique_hints:
        if h.lower() == req_clean.lower():
            return h
    for h in unique_hints:
        if h.lower() in _NORMAL_CANDIDATE_NAMES:
            return h
    for h in unique_hints:
        if any(cand in h.lower() for cand in ("normal", "nomal", "good", "ok", "良品", "正常")):
            return h
    return None


def compute_density_scores(features: np.ndarray, k_nn: int = 15) -> np.ndarray:
    """Compute cosine similarity to the Normal prototype and local k-NN cosine density."""
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.zeros((0,), dtype=np.float32)
    if n_samples == 1:
        return np.ones((1,), dtype=np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = (features / norms).astype(np.float32)

    median_vec = np.median(normed, axis=0)
    median_vec = median_vec / (np.linalg.norm(median_vec) + 1e-8)
    global_sim = normed @ median_vec

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
    """Reduce anomaly features via PCA with upper dimension bound and 95% cumulative variance slicing."""
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
    normal_class_name: str = "normal",
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
    """Execute Phase 1 clustering: keep Normal fixed as Cluster 0 and sub-cluster Anomaly samples into Cluster 1..k*.

    1. **Normal Class Preservation (`cluster_id = 0`)**:
       When Normal vs. Anomaly folders/hints are present (e.g. `nomal/` and `anomaly/`), all
       Normal samples are kept 100% intact in `Cluster 0` without clustering.
    2. **Anomaly-Only Sub-Clustering (`cluster_id = 1..k*`)**:
       Anomaly features are standardized relative to the Normal reference distribution
       (`(h - mu_normal) / std_normal`) to isolate pure defect residuals, then clustered via
       PCA + Silhouette-score-maximized K-means into `k*` distinct defect modes.
    3. **Unlabeled Fallback**:
       Only when no Normal/Anomaly folder hints exist at all, Stage 1 falls back to extracting
       the high-density core (`core_quantile`) as `Cluster 0` and clusters the low-density tail.
    """
    n_samples = features.shape[0]
    if n_samples != len(image_paths):
        raise ValueError("Length of features and image_paths must match.")
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for clustering.")

    hints = _resolve_sample_hints(image_paths=image_paths, ground_truth_hints=ground_truth_hints)
    detected_normal_cls = _detect_normal_class_name(
        hints=hints, requested_normal_class=normal_class_name
    )

    has_known_normal_and_anomaly = False
    if detected_normal_cls is not None:
        normal_mask_list = [h == detected_normal_cls for h in hints]
        n_known_norm = sum(normal_mask_list)
        n_known_anom = n_samples - n_known_norm
        if n_known_norm >= 1 and n_known_anom >= 1:
            has_known_normal_and_anomaly = True

    if has_known_normal_and_anomaly and detected_normal_cls is not None:
        clustering_mode = "anomaly_only"
        resolved_normal_name = detected_normal_cls
        core_indices = np.array(
            [i for i, h in enumerate(hints) if h == detected_normal_cls], dtype=np.int64
        )
        tail_indices = np.array(
            [i for i, h in enumerate(hints) if h != detected_normal_cls], dtype=np.int64
        )

        # Standardize features relative to the Normal class reference distribution
        norm_ref = features[core_indices]
        ref_mean = np.mean(norm_ref, axis=0, keepdims=True)
        ref_std = np.std(norm_ref, axis=0, keepdims=True)
        valid_std = np.where(ref_std > 1e-5, ref_std, 1.0)
        standardized = np.clip((features - ref_mean) / (valid_std + 1e-5), -10.0, 10.0)
        normed_features = (
            standardized / (np.linalg.norm(standardized, axis=1, keepdims=True) + 1e-8)
        ).astype(np.float32)

        # Compute similarity of every sample to the Normal core centroid for spot-checking display
        raw_norms = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        normal_centroid = raw_norms[core_indices].mean(axis=0)
        normal_centroid = normal_centroid / (np.linalg.norm(normal_centroid) + 1e-8)
        density_scores = (raw_norms @ normal_centroid).astype(np.float32)
        clamped_quantile = round(float(len(core_indices) / n_samples), 4)
    else:
        # Fallback for 100% unlabeled single-folder datasets
        clustering_mode = "density_fallback"
        resolved_normal_name = normal_class_name.strip() or "normal"
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        normed_features = (features / norms).astype(np.float32)
        density_scores = compute_density_scores(normed_features, k_nn=k_nn)
        clamped_quantile = float(np.clip(core_quantile, 0.50, 0.97))
        n_core_est = min(int(round(n_samples * clamped_quantile)), max(1, n_samples - 3))
        sorted_indices = np.argsort(density_scores)[::-1]
        core_indices = sorted_indices[:n_core_est]
        tail_indices = sorted_indices[n_core_est:]

    n_core = len(core_indices)
    n_tail = len(tail_indices)

    is_normal_core = np.zeros(n_samples, dtype=bool)
    is_normal_core[core_indices] = True

    cluster_ids = np.zeros(n_samples, dtype=np.int64)
    silhouette_history: dict[str, float] = {}
    best_tail_k = 1

    # Sub-cluster ONLY the Anomaly samples (`tail_indices`) via PCA + Silhouette-Optimized K-Means
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

        cluster_ids[tail_indices] = best_labels + 1
    elif n_tail > 0:
        cluster_ids[tail_indices] = 1

    # Compute 2D PCA preview coordinates where Normal Core is centered and Anomaly sub-clusters fan out
    if n_samples >= 2 and normed_features.shape[1] >= 2:
        preview_pca = PCA(n_components=2, random_state=random_state)
        coords_2d = preview_pca.fit_transform(normed_features).astype(np.float32)
    else:
        coords_2d = np.zeros((n_samples, 2), dtype=np.float32)

    # Build default suggested label per cluster so anomaly sub-clusters get distinct names
    anomaly_hints_set = sorted({h for h in hints if h and h != resolved_normal_name})
    suggested_cluster_labels: dict[str, str] = {"0": resolved_normal_name}
    for cid in range(1, best_tail_k + 1):
        c_hints = [hints[idx] for idx in range(n_samples) if cluster_ids[idx] == cid and hints[idx]]
        dom_hint = max(set(c_hints), key=c_hints.count) if c_hints else "anomaly"
        if len(anomaly_hints_set) <= 1:
            # Single anomaly folder (e.g. 'anomaly') -> name sub-clusters 'anomaly_cluster_1', etc.
            base_anom = anomaly_hints_set[0] if anomaly_hints_set else "anomaly"
            suggested_cluster_labels[str(cid)] = f"{base_anom}_cluster_{cid}"
        else:
            # Multiple defect subfolders -> keep dominant subfolder name or disambiguate if already used
            if dom_hint not in suggested_cluster_labels.values():
                suggested_cluster_labels[str(cid)] = dom_hint
            else:
                suggested_cluster_labels[str(cid)] = f"{dom_hint}_cluster_{cid}"

    items: list[dict[str, Any]] = []
    for i in range(n_samples):
        img_p = Path(image_paths[i])
        hint = hints[i]
        cid = int(cluster_ids[i])
        items.append(
            {
                "image_path": image_paths[i],
                "file_name": img_p.name,
                "cluster_id": cid,
                "is_normal_core": bool(is_normal_core[i]),
                "density_score": round(float(density_scores[i]), 5),
                "preview_2d": [round(float(coords_2d[i, 0]), 4), round(float(coords_2d[i, 1]), 4)],
                "subfolder_class": hint,
                "ground_truth_hint": hint,
                "suggested_label": suggested_cluster_labels.get(
                    str(cid), resolved_normal_name if cid == 0 else f"anomaly_cluster_{cid}"
                ),
            }
        )

    detected_classes = sorted({str(h) for h in hints if h})

    result: dict[str, Any] = {
        "backbone_name": backbone_name,
        "clustering_mode": clustering_mode,
        "normal_class_name": resolved_normal_name,
        "total_samples": int(n_samples),
        "normal_core_count": int(n_core),
        "tail_count": int(n_tail),
        "core_quantile": float(clamped_quantile),
        "best_tail_k": int(best_tail_k),
        "total_clusters": int(1 + best_tail_k),
        "detected_classes": detected_classes,
        "suggested_cluster_labels": suggested_cluster_labels,
        "silhouette_scores": silhouette_history,
        "items": items,
    }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_json_path)

    return result
