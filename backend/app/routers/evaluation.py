"""Step 6 Space Evaluation API: Pure NumPy `.pkl` loading and real-time `tau_sim` threshold recalculation."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.config import BackendConfig
from app.schemas.api_models import EvaluationMetricsResponse, InferenceProxyRequest

router = APIRouter(prefix="/api/v1", tags=["space-evaluation"])


def _recalculate_binary_metrics(
    cosine_similarities: np.ndarray,
    is_defect: np.ndarray,
    tau_sim: float,
) -> dict[str, Any]:
    """Recalculate Recall, Precision, F1, Accuracy, and Confusion Matrix at threshold `tau_sim`."""
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


@router.get("/evaluations")
def list_available_evaluations() -> dict[str, Any]:
    """List all saved epoch `.pkl` snapshots in `/app/data/artifacts`."""
    art_dir = BackendConfig.artifacts_dir()
    light_files = sorted(art_dir.glob("epoch_*_light.pkl"))
    epochs: list[dict[str, Any]] = []

    for lf in light_files:
        m = re.search(r"epoch_(\d+)_light\.pkl", lf.name)
        if not m:
            continue
        ep = int(m.group(1))
        full_file = art_dir / f"epoch_{ep:04d}_full.pkl"
        with open(lf, "rb") as f:
            snap = pickle.load(f)
        epochs.append(
            {
                "epoch": ep,
                "has_full_snapshot": full_file.exists(),
                "d_min_boundary": snap.get("d_min_boundary", 0.0),
                "d_min_centroid": snap.get("d_min_centroid", 0.0),
                "train_loss": snap.get("train_loss", 0.0),
                "curriculum_progress": snap.get("curriculum_progress", 0.0),
                "effective_top_k": snap.get("effective_top_k", 0),
                "metrics_at_zero": snap.get("metrics_at_zero", {}),
            }
        )

    return {"total_epochs": len(epochs), "epochs": epochs}


@router.get("/evaluations/{epoch}", response_model=EvaluationMetricsResponse)
def evaluate_epoch_at_threshold(
    epoch: int,
    tau_sim: float = Query(default=0.0, ge=-1.0, le=1.0),
) -> EvaluationMetricsResponse:
    """Load epoch `.pkl` snapshot and dynamically compute Recall/F1, 128-D heatmap, and 3D UMAP at `tau_sim`."""
    art_dir = BackendConfig.artifacts_dir()
    full_path = art_dir / f"epoch_{epoch:04d}_full.pkl"
    light_path = art_dir / f"epoch_{epoch:04d}_light.pkl"

    target_path = full_path if full_path.exists() else light_path
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Snapshot for epoch {epoch} not found.")

    with open(target_path, "rb") as f:
        snap = pickle.load(f)

    sims = np.asarray(snap["cosine_similarities"], dtype=np.float32)
    is_defect = np.asarray(snap["is_defect"], dtype=bool)
    splits = list(snap["splits"])
    label_names = list(snap["label_names"])
    image_paths = list(snap["image_paths"])
    z_128d = np.asarray(snap["embeddings_128d"], dtype=np.float32)
    umap_3d = np.asarray(snap["umap_3d"], dtype=np.float32)

    val_mask = np.array([s == "val" for s in splits], dtype=bool)
    eval_mask = (
        val_mask
        if (np.any(val_mask & is_defect) and np.any(val_mask & (~is_defect)))
        else np.ones(len(sims), dtype=bool)
    )

    val_metrics = _recalculate_binary_metrics(sims[eval_mask], is_defect[eval_mask], tau_sim=tau_sim)
    all_metrics = _recalculate_binary_metrics(sims, is_defect, tau_sim=tau_sim)

    # Compute Recall / Precision / F1 curve across thresholds in [-0.8, 0.8] for interactive UI plot
    threshold_curve: list[dict[str, float]] = []
    for cand_tau in np.linspace(-0.8, 0.8, 33):
        m_c = _recalculate_binary_metrics(sims[eval_mask], is_defect[eval_mask], tau_sim=float(cand_tau))
        threshold_curve.append(
            {
                "tau_sim": m_c["tau_sim"],
                "recall": m_c["recall"],
                "precision": m_c["precision"],
                "f1": m_c["f1"],
            }
        )

    samples: list[dict[str, Any]] = []
    for i in range(len(sims)):
        pred_ng = bool(sims[i] < tau_sim)
        samples.append(
            {
                "index": i,
                "file_name": Path(image_paths[i]).name,
                "image_path": image_paths[i],
                "label_name": label_names[i],
                "is_defect": bool(is_defect[i]),
                "split": splits[i],
                "cosine_similarity": round(float(sims[i]), 5),
                "cosine_distance": round(float(1.0 - sims[i]), 5),
                "predicted_defect": pred_ng,
                "umap_x": round(float(umap_3d[i, 0]), 4),
                "umap_y": round(float(umap_3d[i, 1]), 4),
                "umap_z": round(float(umap_3d[i, 2]), 4),
            }
        )

    # Sort samples by label (normals first, then defect modes) for clear 128-D heatmap comparison
    order = np.argsort(np.where(is_defect, 1, 0) * 100 + np.asarray(snap["labels"], dtype=np.int64))
    # Cap heatmap rows to at most 60 representative samples (all defects + subset of normals)
    defect_indices = [int(i) for i in order if is_defect[i]]
    normal_indices = [int(i) for i in order if not is_defect[i]]
    selected_rows = normal_indices[: max(10, 50 - len(defect_indices))] + defect_indices

    heatmap_matrix = np.round(z_128d[selected_rows], 4).tolist()
    heatmap_labels = [
        f"{label_names[i]} ({Path(image_paths[i]).stem[-8:]})" for i in selected_rows
    ]

    return EvaluationMetricsResponse(
        epoch=int(snap["epoch"]),
        is_full_snapshot=bool(snap.get("is_full_snapshot", False)),
        tau_sim=round(float(tau_sim), 4),
        d_min_centroid=float(snap.get("d_min_centroid", 0.0)),
        d_min_boundary=float(snap.get("d_min_boundary", 0.0)),
        train_loss=float(snap.get("train_loss", 0.0)),
        curriculum_progress=float(snap.get("curriculum_progress", 0.0)),
        effective_top_k=int(snap.get("effective_top_k", 0)),
        val_metrics=val_metrics,
        all_metrics=all_metrics,
        threshold_curve=threshold_curve,
        samples=samples,
        heatmap_matrix=heatmap_matrix,
        heatmap_labels=heatmap_labels,
    )


@router.post("/inference")
def run_inference_proxy(req: InferenceProxyRequest) -> dict[str, Any]:
    """Proxy single-image inference request to `trainer-service`."""
    url = f"{BackendConfig.TRAINER_API_URL}/internal/inference"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=req.model_dump())
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=resp.json().get("detail", "Inference error"),
                )
            return resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Trainer service unreachable: {exc}") from exc
