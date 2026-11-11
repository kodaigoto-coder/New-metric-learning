"""Step 6 Space Evaluation & Session History API: Pure NumPy `.pkl` loading, dispersion evolution, and dynamic `tau_sim` recalculation."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.config import BackendConfig, sanitize_identifier
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


def _compute_dispersion_from_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Compute or return cached 128-D & 3D UMAP cluster dispersion metrics for an epoch snapshot."""
    cached = snap.get("dispersion_metrics")
    if isinstance(cached, dict) and "normal_intra_disp" in cached:
        return cached

    z_raw = np.asarray(snap.get("embeddings_128d", []), dtype=np.float32)
    sims = np.asarray(snap.get("cosine_similarities", []), dtype=np.float32)
    is_defect = np.asarray(snap.get("is_defect", []), dtype=bool)
    labels_arr = np.asarray(snap.get("labels", []), dtype=np.int64)
    label_names = [str(x) for x in snap.get("label_names", [])]
    umap_arr = np.asarray(snap.get("umap_3d", []), dtype=np.float32)

    if z_raw.ndim != 2 or len(sims) == 0:
        return {
            "normal_intra_disp": 0.0,
            "normal_intra_std": 0.0,
            "defect_intra_disp": 0.0,
            "inter_class_dist": 0.0,
            "fisher_separation_ratio": 0.0,
            "umap_normal_radius": 0.0,
            "umap_defect_radius": 0.0,
            "per_class_dispersion": [],
        }

    norms = np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-8
    z_norm = (z_raw / norms).astype(np.float32)
    defect_mask = is_defect
    normal_mask = ~defect_mask

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


def _list_session_summaries(dataset_name: str | None = None) -> list[dict[str, Any]]:
    """List all saved training sessions sorted newest-first."""
    BackendConfig.ensure_legacy_session_migrated()
    sess_root = BackendConfig.sessions_dir()
    summaries: list[dict[str, Any]] = []

    for d in sess_root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^session_(\d+)$", d.name)
        if not m:
            continue
        s_num = int(m.group(1))
        meta_path = d / "session_meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        ds = str(meta.get("dataset_name", "default"))
        if dataset_name and ds != dataset_name:
            continue

        light_pkls = sorted(d.glob("epoch_*_light.pkl"))
        epochs_done = int(meta.get("epochs_completed", len(light_pkls)))
        summaries.append(
            {
                "session_id": d.name,
                "session_number": int(meta.get("session_number", s_num)),
                "session_label": str(
                    meta.get("session_label") or f"Session #{s_num:04d} ({ds})"
                ),
                "job_id": str(meta.get("job_id", "")),
                "dataset_name": ds,
                "created_at": meta.get("created_at", ""),
                "completed_at": meta.get("completed_at"),
                "status": str(meta.get("status", "completed" if light_pkls else "unknown")),
                "loss_type": str(meta.get("loss_type", "multi_proxy_anchor")),
                "backbone_name": str(meta.get("backbone_name", "efficientnet_b3")),
                "pooling_type": str(meta.get("pooling_type", "gem")),
                "epochs_completed": epochs_done,
                "total_epochs": int(meta.get("total_epochs", epochs_done)),
                "best_epoch": int(meta.get("best_epoch", epochs_done)),
                "best_val_recall": round(float(meta.get("best_val_recall", 0.0)), 4),
                "best_val_f1": round(float(meta.get("best_val_f1", 0.0)), 4),
                "best_val_precision": round(float(meta.get("best_val_precision", 0.0)), 4),
                "best_d_min_boundary": round(float(meta.get("best_d_min_boundary", 0.0)), 4),
                "best_d_min_centroid": round(float(meta.get("best_d_min_centroid", 0.0)), 4),
                "has_best_model": (d / "best_model.pt").exists(),
            }
        )

    summaries.sort(key=lambda s: int(s["session_number"]), reverse=True)
    return summaries


def _resolve_session_artifacts_dir(session_id: str | None = None) -> tuple[str, Path]:
    """Resolve `(resolved_session_id, directory_path)` for loading `.pkl` snapshots."""
    BackendConfig.ensure_legacy_session_migrated()
    if session_id:
        safe_id = sanitize_identifier(session_id, fallback="session_0001")
        s_dir = BackendConfig.session_dir(safe_id)
        if list(s_dir.glob("epoch_*_light.pkl")):
            return safe_id, s_dir

    # If no session_id or session_dir has no pkl yet, check root artifacts_dir first (for unit tests)
    art_dir = BackendConfig.artifacts_dir()
    root_pkls = list(art_dir.glob("epoch_*_light.pkl"))
    summaries = _list_session_summaries()
    if summaries and not session_id:
        latest_id = summaries[0]["session_id"]
        latest_dir = BackendConfig.session_dir(latest_id)
        if list(latest_dir.glob("epoch_*_light.pkl")):
            return latest_id, latest_dir

    if root_pkls:
        return (session_id or "session_0001"), art_dir

    if summaries:
        latest_id = summaries[0]["session_id"]
        return latest_id, BackendConfig.session_dir(latest_id)

    return "session_0001", art_dir


@router.get("/sessions")
def list_training_sessions(dataset_name: str | None = Query(default=None)) -> dict[str, Any]:
    """List all numbered training sessions (`session_0001`, `session_0002`, ...) with summary metrics."""
    sessions = _list_session_summaries(dataset_name=dataset_name)
    latest_id = sessions[0]["session_id"] if sessions else None
    return {
        "total_sessions": len(sessions),
        "latest_session_id": latest_id,
        "sessions": sessions,
    }


@router.get("/sessions/{session_id}")
def get_training_session_detail(session_id: str) -> dict[str, Any]:
    """Retrieve full training history, hyperparameters, and dispersion evolution for a numbered session."""
    BackendConfig.ensure_legacy_session_migrated()
    safe_id = sanitize_identifier(session_id, fallback="session_0001")
    s_dir = BackendConfig.session_dir(safe_id)
    meta_path = s_dir / "session_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session `{safe_id}` not found.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@router.get("/evaluations")
def list_available_evaluations(session_id: str | None = Query(default=None)) -> dict[str, Any]:
    """List all saved epoch `.pkl` snapshots and epoch-by-epoch dispersion evolution for `session_id`."""
    resolved_session_id, target_dir = _resolve_session_artifacts_dir(session_id)
    light_files = sorted(target_dir.glob("epoch_*_light.pkl"))
    epochs: list[dict[str, Any]] = []

    for lf in light_files:
        m = re.search(r"epoch_(\d+)_light\.pkl", lf.name)
        if not m:
            continue
        ep = int(m.group(1))
        full_file = target_dir / f"epoch_{ep:04d}_full.pkl"
        with open(lf, "rb") as f:
            snap = pickle.load(f)
        disp = _compute_dispersion_from_snapshot(snap)
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
                "dispersion_metrics": disp,
            }
        )

    return {
        "session_id": resolved_session_id,
        "total_epochs": len(epochs),
        "epochs": epochs,
    }


@router.get("/evaluations/{epoch}", response_model=EvaluationMetricsResponse)
def evaluate_epoch_at_threshold(
    epoch: int,
    tau_sim: float = Query(default=0.0, ge=-1.0, le=1.0),
    session_id: str | None = Query(default=None),
) -> EvaluationMetricsResponse:
    """Load epoch `.pkl` snapshot and dynamically compute Recall/F1, 3D UMAP, dispersion, and 128-D heatmap at `tau_sim`."""
    resolved_session_id, target_dir = _resolve_session_artifacts_dir(session_id)
    full_path = target_dir / f"epoch_{epoch:04d}_full.pkl"
    light_path = target_dir / f"epoch_{epoch:04d}_light.pkl"

    target_path = full_path if full_path.exists() else light_path
    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot for epoch {epoch} not found in session `{resolved_session_id}`.",
        )

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
    disp_metrics = _compute_dispersion_from_snapshot(snap)

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
        gt_ng = bool(is_defect[i])
        pred_ng = bool(sims[i] < tau_sim)
        if gt_ng and pred_ng:
            cm_cat = "TP"
        elif gt_ng and not pred_ng:
            cm_cat = "FN"
        elif (not gt_ng) and pred_ng:
            cm_cat = "FP"
        else:
            cm_cat = "TN"

        samples.append(
            {
                "index": i,
                "file_name": Path(image_paths[i]).name,
                "image_path": image_paths[i],
                "label_name": label_names[i],
                "is_defect": gt_ng,
                "split": splits[i],
                "cosine_similarity": round(float(sims[i]), 5),
                "cosine_distance": round(float(1.0 - sims[i]), 5),
                "predicted_defect": pred_ng,
                "cm_category": cm_cat,
                "umap_x": round(float(umap_3d[i, 0]), 4),
                "umap_y": round(float(umap_3d[i, 1]), 4),
                "umap_z": round(float(umap_3d[i, 2]), 4),
            }
        )

    order = np.argsort(np.where(is_defect, 1, 0) * 100 + np.asarray(snap["labels"], dtype=np.int64))
    defect_indices = [int(i) for i in order if is_defect[i]]
    normal_indices = [int(i) for i in order if not is_defect[i]]
    selected_rows = normal_indices[: max(10, 50 - len(defect_indices))] + defect_indices

    heatmap_matrix = np.round(z_128d[selected_rows], 4).tolist()
    heatmap_labels = [
        f"{label_names[i]} ({Path(image_paths[i]).stem[-8:]})" for i in selected_rows
    ]

    proxy_points: list[dict[str, Any]] = []
    norm_p_3d = snap.get("normal_proxies_umap_3d")
    def_p_3d = snap.get("defect_proxies_umap_3d")
    norm_p_raw = snap.get("normal_proxies")
    num_norm_proxies = (
        int(np.asarray(norm_p_raw).shape[0])
        if norm_p_raw is not None and len(norm_p_raw) > 0
        else 1
    )

    if norm_p_3d is not None and len(norm_p_3d) > 0:
        np_3d_arr = np.asarray(norm_p_3d, dtype=np.float32)
        for m_idx in range(np_3d_arr.shape[0]):
            proxy_points.append(
                {
                    "proxy_name": f"Normal Proxy #{m_idx + 1}",
                    "proxy_type": "normal_proxy",
                    "umap_x": round(float(np_3d_arr[m_idx, 0]), 4),
                    "umap_y": round(float(np_3d_arr[m_idx, 1]), 4),
                    "umap_z": round(float(np_3d_arr[m_idx, 2]), 4),
                }
            )

    if def_p_3d is not None and len(def_p_3d) > 0:
        dp_3d_arr = np.asarray(def_p_3d, dtype=np.float32)
        for c_idx in range(dp_3d_arr.shape[0]):
            proxy_points.append(
                {
                    "proxy_name": f"Defect Proxy #{c_idx + 1}",
                    "proxy_type": "defect_proxy",
                    "umap_x": round(float(dp_3d_arr[c_idx, 0]), 4),
                    "umap_y": round(float(dp_3d_arr[c_idx, 1]), 4),
                    "umap_z": round(float(dp_3d_arr[c_idx, 2]), 4),
                }
            )

    return EvaluationMetricsResponse(
        session_id=resolved_session_id,
        epoch=int(snap["epoch"]),
        is_full_snapshot=bool(snap.get("is_full_snapshot", False)),
        tau_sim=round(float(tau_sim), 4),
        d_min_centroid=float(snap.get("d_min_centroid", 0.0)),
        d_min_boundary=float(snap.get("d_min_boundary", 0.0)),
        train_loss=float(snap.get("train_loss", 0.0)),
        curriculum_progress=float(snap.get("curriculum_progress", 0.0)),
        effective_top_k=int(snap.get("effective_top_k", 0)),
        num_normal_proxies=num_norm_proxies,
        dispersion_metrics=disp_metrics,
        val_metrics=val_metrics,
        all_metrics=all_metrics,
        threshold_curve=threshold_curve,
        samples=samples,
        proxy_points=proxy_points,
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
