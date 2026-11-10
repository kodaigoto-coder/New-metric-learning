"""Step 6 Space Evaluation: Cosine similarity thresholding, 3D UMAP, and 2-tier Pure NumPy `.pkl` persistence."""

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

    # Cosine similarity to normal prototype [-1, 1]
    cosine_similarities = (z_norm @ normal_prototype).astype(np.float32)

    # High-dimensional quantitative cosine distances:
    # 1. Centroid-to-Defect minimum cosine distance
    # 2. Boundary-Normal-to-Defect minimum cosine distance
    if np.any(defect_mask) and np.any(normal_mask):
        defect_vecs = z_norm[defect_mask]
        normal_vecs = z_norm[normal_mask]
        d_min_centroid = float(np.min(1.0 - (defect_vecs @ normal_prototype)))
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

    umap_3d = compute_3d_umap_coordinates(z_norm)

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
        "cosine_similarities": cosine_similarities,
        "umap_3d": umap_3d,
        "d_min_centroid": round(d_min_centroid, 5),
        "d_min_boundary": round(d_min_boundary, 5),
        "train_loss": round(float(train_loss), 5),
        "curriculum_progress": round(float(curriculum_progress), 4),
        "effective_top_k": int(effective_top_k),
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
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
    }
