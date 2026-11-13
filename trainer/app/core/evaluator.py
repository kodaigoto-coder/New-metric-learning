"""Step 6 Space Evaluation: Cosine similarity thresholding, 3D UMAP, cluster dispersion metrics, and 2-tier Pure NumPy `.pkl` persistence."""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def _finite_float(val: float, default: float = 0.0) -> float:
    f = float(val)
    return f if math.isfinite(f) else float(default)


def compute_calibrated_normal_similarities_np(
    z_norm: np.ndarray,
    norm_proxies: np.ndarray,
    proxy_alphas: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Möbius radius-calibrated normality similarity S_cal(z) = 1 - min_m d_tilde_m(z) in [-1, 1].

    Reduces to exact max_m <z, p_norm^(m)> when M=1 or proxy_alphas is None / all ones.
    """
    z_clean = np.nan_to_num(np.asarray(z_norm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if z_clean.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    p_clean = np.nan_to_num(
        np.asarray(norm_proxies, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if p_clean.ndim == 1:
        p_clean = p_clean.reshape(1, -1)
    if p_clean.shape[0] == 0:
        return np.zeros((z_clean.shape[0],), dtype=np.float32)

    raw_dists = np.clip(1.0 - (z_clean @ p_clean.T), 0.0, 2.0)  # [N, M]
    if p_clean.shape[0] <= 1 or proxy_alphas is None:
        return (1.0 - np.min(raw_dists, axis=1)).astype(np.float32)

    alphas = np.nan_to_num(
        np.asarray(proxy_alphas, dtype=np.float32).reshape(1, -1),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    if alphas.shape[1] != p_clean.shape[0]:
        return (1.0 - np.min(raw_dists, axis=1)).astype(np.float32)

    alphas = np.clip(alphas, 0.40, 4.00)
    numer = 2.0 * alphas * raw_dists
    denom = (alphas - 1.0) * raw_dists + 2.0
    d_tilde = np.clip(numer / (denom + 1e-8), 0.0, 2.0)
    return (1.0 - np.min(d_tilde, axis=1)).astype(np.float32)


def compute_binary_metrics_at_threshold(
    cosine_similarities: np.ndarray,
    is_defect: np.ndarray,
    tau_sim: float = 0.0,
) -> dict[str, Any]:
    """Compute binary anomaly detection metrics (Recall, Precision, F1, Confusion Matrix) at `tau_sim`.

    Decision rule on 128-D hypersphere:
    - `cosine_similarity < tau_sim`  => Predicted Defect (NG / Positive class for anomaly recall)
    - `cosine_similarity >= tau_sim` => Predicted Normal (OK / Negative class)
    """
    sims = np.asarray(cosine_similarities, dtype=np.float32)
    gt_defect = np.asarray(is_defect, dtype=bool)
    pred_defect = sims < float(tau_sim)

    tp = int(np.sum(pred_defect & gt_defect))
    fp = int(np.sum(pred_defect & (~gt_defect)))
    tn = int(np.sum((~pred_defect) & (~gt_defect)))
    fn = int(np.sum((~pred_defect) & gt_defect))

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = float((tp + tn) / len(sims)) if len(sims) > 0 else 0.0

    return {
        "tau_sim": round(float(tau_sim), 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_space_dispersion_metrics(
    embeddings_128d: np.ndarray,
    cosine_similarities: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    is_defect: np.ndarray,
    umap_3d: np.ndarray,
) -> dict[str, Any]:
    """Compute 128-D hypersphere & 3D UMAP intra-class dispersion and inter-class separation metrics.

    Used to track how cluster variance shrinks and class separation grows across training epochs:
    - `normal_intra_disp`: Mean cosine distance of Normal samples to their nearest Normal proxy/prototype.
    - `defect_intra_disp`: Mean cosine distance of Defect samples to their respective defect mode centroid.
    - `inter_class_dist`: Mean cosine distance of Defect samples from the Normal proxy manifold (`mean(1 - s(x))`).
    - `fisher_separation_ratio`: `inter_class_dist / (normal_intra_disp + defect_intra_disp + 1e-4)`.
    - `umap_normal_radius` / `umap_defect_radius`: RMS Euclidean spread in 3D UMAP space.
    - `per_class_dispersion`: Per-class breakdown of count, 128-D cosine dispersion, and 3D UMAP spread.
    """
    norms = np.linalg.norm(embeddings_128d, axis=1, keepdims=True) + 1e-8
    z_norm = (embeddings_128d / norms).astype(np.float32)
    sims = np.asarray(cosine_similarities, dtype=np.float32)
    defect_mask = np.asarray(is_defect, dtype=bool)
    normal_mask = ~defect_mask
    labels_arr = np.asarray(labels, dtype=np.int64)
    umap_arr = np.asarray(umap_3d, dtype=np.float32)

    if np.any(normal_mask):
        normal_intra_disp = float(np.mean(1.0 - sims[normal_mask]))
        normal_intra_std = float(np.std(1.0 - sims[normal_mask]))
        u_norm = umap_arr[normal_mask]
        u_norm_center = np.mean(u_norm, axis=0, keepdims=True)
        umap_normal_radius = float(np.sqrt(np.mean(np.sum((u_norm - u_norm_center) ** 2, axis=1))))
    else:
        normal_intra_disp = 0.0
        normal_intra_std = 0.0
        umap_normal_radius = 0.0

    defect_mode_disps: list[float] = []
    if np.any(defect_mask):
        inter_class_dist = float(np.mean(1.0 - sims[defect_mask]))
        u_def = umap_arr[defect_mask]
        u_def_center = np.mean(u_def, axis=0, keepdims=True)
        umap_defect_radius = float(np.sqrt(np.mean(np.sum((u_def - u_def_center) ** 2, axis=1))))
        for cid in sorted(set(labels_arr[defect_mask].tolist())):
            c_mask = labels_arr == cid
            c_vecs = z_norm[c_mask]
            c_mean = np.mean(c_vecs, axis=0)
            c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
            c_disp = float(np.mean(1.0 - (c_vecs @ c_mean)))
            defect_mode_disps.append(c_disp)
    else:
        inter_class_dist = 0.0
        umap_defect_radius = 0.0

    defect_intra_disp = float(np.mean(defect_mode_disps)) if defect_mode_disps else 0.0
    fisher_separation_ratio = float(
        inter_class_dist / (normal_intra_disp + defect_intra_disp + 1e-4)
    )

    per_class_dispersion: list[dict[str, Any]] = []
    unique_names = sorted(set(label_names), key=lambda nm: (nm != "normal" and nm != "nomal", nm))
    for cls_name in unique_names:
        c_idx = [i for i, nm in enumerate(label_names) if nm == cls_name]
        if not c_idx:
            continue
        c_vecs = z_norm[c_idx]
        is_def_cls = bool(defect_mask[c_idx[0]])
        if is_def_cls:
            c_mean = np.mean(c_vecs, axis=0)
            c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
            c_cos_disp = float(np.mean(1.0 - (c_vecs @ c_mean)))
        else:
            # For Normal class, measure dispersion relative to assigned/calibrated normal proxies
            c_cos_disp = float(np.mean(1.0 - sims[c_idx]))
        c_u = umap_arr[c_idx]
        c_u_center = np.mean(c_u, axis=0, keepdims=True)
        c_u_rad = float(np.sqrt(np.mean(np.sum((c_u - c_u_center) ** 2, axis=1))))
        c_norm_dist = float(np.mean(1.0 - sims[c_idx]))
        per_class_dispersion.append(
            {
                "label_name": str(cls_name),
                "count": int(len(c_idx)),
                "is_defect": is_def_cls,
                "intra_cosine_disp": round(c_cos_disp, 5),
                "mean_dist_from_normal": round(c_norm_dist, 5),
                "umap_radius_3d": round(c_u_rad, 4),
            }
        )

    return {
        "normal_intra_disp": round(normal_intra_disp, 5),
        "normal_intra_std": round(normal_intra_std, 5),
        "defect_intra_disp": round(defect_intra_disp, 5),
        "inter_class_dist": round(inter_class_dist, 5),
        "fisher_separation_ratio": round(fisher_separation_ratio, 4),
        "umap_normal_radius": round(umap_normal_radius, 4),
        "umap_defect_radius": round(umap_defect_radius, 4),
        "per_class_dispersion": per_class_dispersion,
    }


def compute_3d_umap_coordinates(
    embeddings_128d: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 3D UMAP coordinates directly from 128-D embeddings (no invalid 300-D PCA).

    Includes safe sample-count guards (`n_neighbors = min(15, max(2, N - 1))`) and falls back
    cleanly to 3D PCA when sample count is too small or spectral UMAP produces non-finite values.
    """
    clean_emb = np.nan_to_num(
        np.asarray(embeddings_128d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    n_samples = clean_emb.shape[0]
    if n_samples == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n_samples < 5 or umap is None or float(np.var(clean_emb)) < 1e-12:
        return _pca_3d_fallback(clean_emb)

    n_neighbors = min(15, max(2, n_samples - 1))
    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            init="random" if n_samples < 12 else "spectral",
            random_state=random_state,
        )
        coords = reducer.fit_transform(clean_emb)
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.shape != (n_samples, 3) or not np.all(np.isfinite(coords_arr)):
            return _pca_3d_fallback(clean_emb)
        return coords_arr
    except Exception:
        return _pca_3d_fallback(clean_emb)


def _pca_3d_fallback(embeddings: np.ndarray) -> np.ndarray:
    clean_emb = np.nan_to_num(
        np.asarray(embeddings, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    n_samples, n_features = clean_emb.shape
    k = min(3, n_samples, n_features)
    out = np.zeros((n_samples, 3), dtype=np.float32)
    if k >= 1 and float(np.var(clean_emb)) >= 1e-12:
        pca = PCA(n_components=k, random_state=42)
        out[:, :k] = np.nan_to_num(pca.fit_transform(clean_emb), nan=0.0).astype(np.float32)
    return out


def _assert_no_torch_tensors(obj: Any, path: str = "root") -> None:
    """Verify recursively that no `torch.Tensor` objects leak into `.pkl` files."""
    cls_module = type(obj).__module__
    if cls_module.startswith("torch"):
        raise TypeError(f"Forbidden torch object `{type(obj)}` found in pickle payload at `{path}`.")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_torch_tensors(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_no_torch_tensors(item, f"{path}[{idx}]")


def _compute_split_margins(
    z_norm: np.ndarray,
    cosine_similarities: np.ndarray,
    defect_mask: np.ndarray,
    split_mask: np.ndarray,
    ref_normal_mask: np.ndarray,
) -> tuple[float, float]:
    """Compute `(d_min_centroid, d_min_boundary)` for defect samples within `split_mask`."""
    _ = z_norm
    split_def = split_mask & defect_mask
    split_norm = split_mask & (~defect_mask)
    norm_pool = split_norm if np.any(split_norm) else ref_normal_mask
    if np.any(split_def) and np.any(norm_pool):
        defect_scores = cosine_similarities[split_def]
        normal_scores = cosine_similarities[norm_pool]
        d_min_c = float(1.0 - np.mean(defect_scores))
        d_min_b = float(np.min(normal_scores) - np.max(defect_scores))
        return round(d_min_c, 5), round(d_min_b, 5)
    return 0.0, 0.0


def evaluate_and_persist_epoch_space(
    epoch: int,
    image_paths: list[str],
    labels: np.ndarray,
    label_names: list[str],
    is_defect: np.ndarray,
    splits: list[str],
    embeddings_128d: np.ndarray,
    gap_features: np.ndarray | None,
    artifacts_dir: Path,
    tau_sim: float = 0.0,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    loss_direction_feedback: str = "",
    effective_normal_ratio: float = 0.50,
    curriculum_progress: float = 0.0,
    effective_top_k: int = 0,
    save_full_snapshot: bool = False,
    normal_proxies: np.ndarray | None = None,
    defect_proxies: np.ndarray | None = None,
    normal_proxy_alphas: np.ndarray | None = None,
    normal_proxy_radii: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the 128-D hypersphere space across `train`/`val`/`test` (6:2:2) and persist 2-tier Pure NumPy `.pkl` snapshots."""
    emb_clean = np.nan_to_num(
        np.asarray(embeddings_128d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if emb_clean.ndim != 2 or emb_clean.shape[0] == 0:
        raise ValueError("`embeddings_128d` must be a non-empty 2D array for space evaluation.")
    norms = np.linalg.norm(emb_clean, axis=1, keepdims=True) + 1e-8
    z_norm = (emb_clean / norms).astype(np.float32)

    labels_arr = np.asarray(labels, dtype=np.int64)
    defect_mask = np.asarray(is_defect, dtype=bool)
    normal_mask = ~defect_mask
    train_mask = np.array([s == "train" for s in splits], dtype=bool)
    val_mask = np.array([s == "val" for s in splits], dtype=bool)
    test_mask = np.array([s == "test" for s in splits], dtype=bool)

    # Fit normal prototype strictly on training normal samples (fallback to all normals if empty)
    train_normal_mask = normal_mask & train_mask
    if np.any(train_normal_mask):
        proto_source = z_norm[train_normal_mask]
    elif np.any(normal_mask):
        proto_source = z_norm[normal_mask]
    else:
        proto_source = z_norm

    normal_prototype = proto_source.mean(axis=0)
    normal_prototype = (normal_prototype / (np.linalg.norm(normal_prototype) + 1e-8)).astype(
        np.float32
    )

    # Normalize Multi-Proxy Normal vectors [M, 128] if provided
    norm_proxies_clean: np.ndarray | None = None
    if normal_proxies is not None:
        np_arr = np.asarray(normal_proxies, dtype=np.float32)
        if np_arr.ndim == 1:
            np_arr = np_arr.reshape(1, -1)
        if np_arr.shape[0] > 0:
            p_norms = np.linalg.norm(np_arr, axis=1, keepdims=True) + 1e-8
            norm_proxies_clean = (np_arr / p_norms).astype(np.float32)

    def_proxies_clean: np.ndarray | None = None
    if defect_proxies is not None:
        dp_arr = np.asarray(defect_proxies, dtype=np.float32)
        if dp_arr.ndim == 1:
            dp_arr = dp_arr.reshape(1, -1)
        if dp_arr.shape[0] > 0:
            dp_norms = np.linalg.norm(dp_arr, axis=1, keepdims=True) + 1e-8
            def_proxies_clean = (dp_arr / dp_norms).astype(np.float32)

    norm_alphas_clean: np.ndarray | None = None
    if normal_proxy_alphas is not None:
        norm_alphas_clean = np.asarray(normal_proxy_alphas, dtype=np.float32).reshape(-1)

    norm_radii_clean: np.ndarray | None = None
    if normal_proxy_radii is not None:
        norm_radii_clean = np.asarray(normal_proxy_radii, dtype=np.float32).reshape(-1)

    # Möbius-calibrated Multi-Proxy normality similarity score S_cal(x) in [-1, 1]
    if norm_proxies_clean is not None and norm_proxies_clean.shape[0] > 0:
        cosine_similarities = compute_calibrated_normal_similarities_np(
            z_norm=z_norm,
            norm_proxies=norm_proxies_clean,
            proxy_alphas=norm_alphas_clean,
        )
    else:
        cosine_similarities = (z_norm @ normal_prototype).astype(np.float32)

    # Evaluate masks for val, test, train, and all
    has_val = bool(np.any(val_mask & defect_mask) and np.any(val_mask & normal_mask))
    has_test = bool(np.any(test_mask & defect_mask) and np.any(test_mask & normal_mask))
    has_train = bool(np.any(train_mask & defect_mask) and np.any(train_mask & normal_mask))

    eff_val_mask = val_mask if has_val else np.ones(len(z_norm), dtype=bool)
    eff_test_mask = test_mask if has_test else eff_val_mask
    eff_train_mask = train_mask if has_train else np.ones(len(z_norm), dtype=bool)
    all_mask = np.ones(len(z_norm), dtype=bool)

    val_d_min_centroid, val_d_min_boundary = _compute_split_margins(
        z_norm, cosine_similarities, defect_mask, eff_val_mask, train_normal_mask if np.any(train_normal_mask) else normal_mask
    )
    test_d_min_centroid, test_d_min_boundary = _compute_split_margins(
        z_norm, cosine_similarities, defect_mask, eff_test_mask, train_normal_mask if np.any(train_normal_mask) else normal_mask
    )
    train_d_min_centroid, train_d_min_boundary = _compute_split_margins(
        z_norm, cosine_similarities, defect_mask, eff_train_mask, train_normal_mask if np.any(train_normal_mask) else normal_mask
    )
    all_d_min_centroid, all_d_min_boundary = _compute_split_margins(
        z_norm, cosine_similarities, defect_mask, all_mask, normal_mask
    )

    # Use validation split margins for curriculum feedback, and store per-split metrics
    d_min_centroid = val_d_min_centroid
    d_min_boundary = val_d_min_boundary

    val_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities[eff_val_mask],
        is_defect=defect_mask[eff_val_mask],
        tau_sim=tau_sim,
    )
    val_metrics["d_min_centroid"] = val_d_min_centroid
    val_metrics["d_min_boundary"] = val_d_min_boundary
    val_metrics["sample_count"] = int(np.sum(eff_val_mask))

    test_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities[eff_test_mask],
        is_defect=defect_mask[eff_test_mask],
        tau_sim=tau_sim,
    )
    test_metrics["d_min_centroid"] = test_d_min_centroid
    test_metrics["d_min_boundary"] = test_d_min_boundary
    test_metrics["sample_count"] = int(np.sum(eff_test_mask))

    train_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities[eff_train_mask],
        is_defect=defect_mask[eff_train_mask],
        tau_sim=tau_sim,
    )
    train_metrics["d_min_centroid"] = train_d_min_centroid
    train_metrics["d_min_boundary"] = train_d_min_boundary
    train_metrics["sample_count"] = int(np.sum(eff_train_mask))

    all_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities,
        is_defect=defect_mask,
        tau_sim=tau_sim,
    )
    all_metrics["d_min_centroid"] = all_d_min_centroid
    all_metrics["d_min_boundary"] = all_d_min_boundary
    all_metrics["sample_count"] = int(len(z_norm))

    # Jointly project samples + learned proxies into 3D UMAP space so proxies can be visualized
    extra_blocks: list[np.ndarray] = []
    n_samples = z_norm.shape[0]
    n_norm_p = 0
    n_def_p = 0
    if norm_proxies_clean is not None and norm_proxies_clean.shape[0] > 0:
        extra_blocks.append(norm_proxies_clean)
        n_norm_p = norm_proxies_clean.shape[0]
    if def_proxies_clean is not None and def_proxies_clean.shape[0] > 0:
        extra_blocks.append(def_proxies_clean)
        n_def_p = def_proxies_clean.shape[0]

    if extra_blocks:
        combined_128d = np.vstack([z_norm, *extra_blocks])
        combined_3d = compute_3d_umap_coordinates(combined_128d)
        umap_3d = combined_3d[:n_samples]
        norm_proxies_umap_3d = (
            combined_3d[n_samples : n_samples + n_norm_p]
            if n_norm_p > 0
            else np.zeros((0, 3), dtype=np.float32)
        )
        def_proxies_umap_3d = (
            combined_3d[n_samples + n_norm_p : n_samples + n_norm_p + n_def_p]
            if n_def_p > 0
            else np.zeros((0, 3), dtype=np.float32)
        )
    else:
        umap_3d = compute_3d_umap_coordinates(z_norm)
        norm_proxies_umap_3d = np.zeros((0, 3), dtype=np.float32)
        def_proxies_umap_3d = np.zeros((0, 3), dtype=np.float32)

    all_label_names = [str(name) for name in label_names]
    dispersion_metrics = compute_space_dispersion_metrics(
        embeddings_128d=z_norm,
        cosine_similarities=cosine_similarities,
        labels=labels_arr,
        label_names=all_label_names,
        is_defect=defect_mask,
        umap_3d=umap_3d,
    )

    dispersion_by_split: dict[str, dict[str, Any]] = {"all": dispersion_metrics}
    for sp_key, sp_mask in (("train", eff_train_mask), ("val", eff_val_mask), ("test", eff_test_mask)):
        sp_indices = np.where(sp_mask)[0].tolist()
        if sp_indices:
            dispersion_by_split[sp_key] = compute_space_dispersion_metrics(
                embeddings_128d=z_norm[sp_mask],
                cosine_similarities=cosine_similarities[sp_mask],
                labels=labels_arr[sp_mask],
                label_names=[all_label_names[idx] for idx in sp_indices],
                is_defect=defect_mask[sp_mask],
                umap_3d=umap_3d[sp_mask],
            )
        else:
            dispersion_by_split[sp_key] = dispersion_metrics

    base_payload: dict[str, Any] = {
        "epoch": int(epoch),
        "is_full_snapshot": False,
        "image_paths": [str(p) for p in image_paths],
        "labels": labels_arr,
        "label_names": all_label_names,
        "is_defect": defect_mask,
        "splits": [str(s) for s in splits],
        "embeddings_128d": z_norm,
        "normal_prototype": normal_prototype,
        "normal_proxies": norm_proxies_clean,
        "normal_proxy_alphas": norm_alphas_clean,
        "normal_proxy_radii": norm_radii_clean,
        "defect_proxies": def_proxies_clean,
        "normal_proxies_umap_3d": norm_proxies_umap_3d,
        "defect_proxies_umap_3d": def_proxies_umap_3d,
        "cosine_similarities": cosine_similarities,
        "umap_3d": umap_3d,
        "d_min_centroid": round(_finite_float(d_min_centroid), 5),
        "d_min_boundary": round(_finite_float(d_min_boundary), 5),
        "test_d_min_centroid": round(_finite_float(test_d_min_centroid), 5),
        "test_d_min_boundary": round(_finite_float(test_d_min_boundary), 5),
        "train_loss": round(_finite_float(train_loss), 5),
        "val_loss": round(_finite_float(val_loss), 5),
        "loss_direction_feedback": str(loss_direction_feedback),
        "effective_normal_ratio": round(_finite_float(effective_normal_ratio, 0.5), 4),
        "curriculum_progress": round(_finite_float(curriculum_progress), 4),
        "effective_top_k": int(effective_top_k),
        "dispersion_metrics": dispersion_metrics,
        "dispersion_by_split": dispersion_by_split,
        "metrics_at_zero": val_metrics,
        "test_metrics_at_zero": test_metrics,
        "train_metrics_at_zero": train_metrics,
        "all_metrics_at_zero": all_metrics,
        "gap_features": None,
    }

    _assert_no_torch_tensors(base_payload)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    light_path = artifacts_dir / f"epoch_{epoch:04d}_light.pkl"
    tmp_light = light_path.with_suffix(".pkl.tmp")
    with open(tmp_light, "wb") as f:
        pickle.dump(base_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_light, light_path)

    full_path_str: str | None = None
    if save_full_snapshot and gap_features is not None:
        full_payload = dict(base_payload)
        full_payload["is_full_snapshot"] = True
        full_payload["gap_features"] = np.asarray(gap_features, dtype=np.float32)
        _assert_no_torch_tensors(full_payload)
        full_path = artifacts_dir / f"epoch_{epoch:04d}_full.pkl"
        tmp_full = full_path.with_suffix(".pkl.tmp")
        with open(tmp_full, "wb") as f:
            pickle.dump(full_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_full, full_path)
        full_path_str = str(full_path)

    return {
        "epoch": int(epoch),
        "light_pkl_path": str(light_path),
        "full_pkl_path": full_path_str,
        "d_min_centroid": round(d_min_centroid, 5),
        "d_min_boundary": round(d_min_boundary, 5),
        "test_d_min_centroid": round(test_d_min_centroid, 5),
        "test_d_min_boundary": round(test_d_min_boundary, 5),
        "dispersion_metrics": dispersion_metrics,
        "dispersion_by_split": dispersion_by_split,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "all_metrics": all_metrics,
    }
