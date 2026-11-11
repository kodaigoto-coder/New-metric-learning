"""Step 6 Space Evaluation: Cosine similarity thresholding, 3D UMAP, cluster dispersion metrics, and 2-tier Pure NumPy `.pkl` persistence."""

from __future__ import annotations

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
        c_mean = np.mean(c_vecs, axis=0)
        c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
        c_cos_disp = float(np.mean(1.0 - (c_vecs @ c_mean)))
        c_u = umap_arr[c_idx]
        c_u_center = np.mean(c_u, axis=0, keepdims=True)
        c_u_rad = float(np.sqrt(np.mean(np.sum((c_u - c_u_center) ** 2, axis=1))))
        c_norm_dist = float(np.mean(1.0 - sims[c_idx]))
        per_class_dispersion.append(
            {
                "label_name": str(cls_name),
                "count": int(len(c_idx)),
                "is_defect": bool(defect_mask[c_idx[0]]),
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
    cleanly to 3D PCA when sample count is too small for spectral UMAP initialization.
    """
    n_samples = embeddings_128d.shape[0]
    if n_samples == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n_samples < 5 or umap is None:
        return _pca_3d_fallback(embeddings_128d)

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
        coords = reducer.fit_transform(embeddings_128d)
        return np.asarray(coords, dtype=np.float32)
    except Exception:
        return _pca_3d_fallback(embeddings_128d)


def _pca_3d_fallback(embeddings: np.ndarray) -> np.ndarray:
    n_samples, n_features = embeddings.shape
    k = min(3, n_samples, n_features)
    out = np.zeros((n_samples, 3), dtype=np.float32)
    if k >= 1:
        pca = PCA(n_components=k, random_state=42)
        out[:, :k] = pca.fit_transform(embeddings).astype(np.float32)
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
    curriculum_progress: float = 0.0,
    effective_top_k: int = 0,
    save_full_snapshot: bool = False,
    normal_proxies: np.ndarray | None = None,
    defect_proxies: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the 128-D hypersphere space and atomically persist 2-tier Pure NumPy `.pkl` snapshots."""
    norms = np.linalg.norm(embeddings_128d, axis=1, keepdims=True) + 1e-8
    z_norm = (embeddings_128d / norms).astype(np.float32)

    labels_arr = np.asarray(labels, dtype=np.int64)
    defect_mask = np.asarray(is_defect, dtype=bool)
    normal_mask = ~defect_mask
    train_mask = np.array([s == "train" for s in splits], dtype=bool)
    val_mask = np.array([s == "val" for s in splits], dtype=bool)

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

    # Multi-Proxy normality cosine similarity score s(x) = max_{m=1..M} <z(x), p_norm^(m)> in [-1, 1]
    if norm_proxies_clean is not None and norm_proxies_clean.shape[0] > 0:
        sim_to_norm_proxies = z_norm @ norm_proxies_clean.T  # [N, M]
        cosine_similarities = np.max(sim_to_norm_proxies, axis=1).astype(np.float32)
    else:
        cosine_similarities = (z_norm @ normal_prototype).astype(np.float32)

    # High-dimensional quantitative cosine distances:
    # 1. Proxy/Centroid-to-Defect minimum cosine distance: min_{i in Defect} (1 - s(x_i))
    # 2. Boundary-Normal-to-Defect minimum cosine distance
    if np.any(defect_mask) and np.any(normal_mask):
        defect_vecs = z_norm[defect_mask]
        normal_vecs = z_norm[normal_mask]
        defect_scores = cosine_similarities[defect_mask]
        d_min_centroid = float(np.min(1.0 - defect_scores))
        cross_sims = defect_vecs @ normal_vecs.T
        d_min_boundary = float(np.min(1.0 - cross_sims))
    else:
        d_min_centroid = 0.0
        d_min_boundary = 0.0

    # Evaluate metrics on validation split (or full set if validation has no defects)
    eval_mask = val_mask if (np.any(val_mask & defect_mask) and np.any(val_mask & normal_mask)) else np.ones(len(z_norm), dtype=bool)
    val_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities[eval_mask],
        is_defect=defect_mask[eval_mask],
        tau_sim=tau_sim,
    )
    all_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=cosine_similarities,
        is_defect=defect_mask,
        tau_sim=tau_sim,
    )

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

    dispersion_metrics = compute_space_dispersion_metrics(
        embeddings_128d=z_norm,
        cosine_similarities=cosine_similarities,
        labels=labels_arr,
        label_names=[str(name) for name in label_names],
        is_defect=defect_mask,
        umap_3d=umap_3d,
    )

    base_payload: dict[str, Any] = {
        "epoch": int(epoch),
        "is_full_snapshot": False,
        "image_paths": [str(p) for p in image_paths],
        "labels": labels_arr,
        "label_names": [str(name) for name in label_names],
        "is_defect": defect_mask,
        "splits": [str(s) for s in splits],
        "embeddings_128d": z_norm,
        "normal_prototype": normal_prototype,
        "normal_proxies": norm_proxies_clean,
        "defect_proxies": def_proxies_clean,
        "normal_proxies_umap_3d": norm_proxies_umap_3d,
        "defect_proxies_umap_3d": def_proxies_umap_3d,
        "cosine_similarities": cosine_similarities,
        "umap_3d": umap_3d,
        "d_min_centroid": round(d_min_centroid, 5),
        "d_min_boundary": round(d_min_boundary, 5),
        "train_loss": round(float(train_loss), 5),
        "curriculum_progress": round(float(curriculum_progress), 4),
        "effective_top_k": int(effective_top_k),
        "dispersion_metrics": dispersion_metrics,
        "metrics_at_zero": val_metrics,
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
        "dispersion_metrics": dispersion_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
    }
