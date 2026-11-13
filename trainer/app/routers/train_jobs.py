"""Internal ML worker endpoints for synthetic dataset generation, 2-stage clustering, Multi-Proxy / MS-Loss training, and inference."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from app.config import TrainerConfig
from app.core.boundary_augmentation import (
    synthesize_boundary_gap_features,
    synthesize_defect_transfer_images,
)
from app.core.evaluator import (
    compute_calibrated_normal_similarities_np,
    evaluate_and_persist_epoch_space,
)
from app.core.feature_extractor import (
    MetricProjectionModel,
    build_image_transforms,
    extract_dataset_embeddings,
    extract_micro_augmented_gap_bank,
    extract_raw_uncalibrated_features,
    load_image_tensor,
)
from app.core.memory_bank import CrossBatchMemoryBank
from app.core.ms_loss import AsymmetricMultiSimilarityLoss
from app.core.nab_sampler import (
    NormalAnomalyBalancedSampler,
    compute_class_balanced_weights,
)
from app.core.proxy_anchor_loss import (
    CustomizedMultiProxyAnchorLoss,
    assign_normal_subclusters_by_centroids,
    estimate_optimal_normal_proxies,
)
from app.core.scheduler import CurriculumPenaltyScheduler
from app.core.synthetic_data import generate_synthetic_inspection_dataset
from app.core.two_stage_cluster import run_two_stage_clustering
from app.schemas.job_models import (
    ClusteringJobRequest,
    JobStatusResponse,
    SingleInferenceRequest,
    SyntheticDatasetRequest,
    TrainingJobRequest,
)

router = APIRouter(prefix="/internal", tags=["trainer-internal"])

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-worker")
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def get_active_job_id() -> str | None:
    with _JOBS_LOCK:
        for jid, state in _JOBS.items():
            if state["status"] in ("queued", "running"):
                return jid
    return None


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(kwargs)
            _persist_job_status(job_id, _JOBS[job_id])


def _persist_job_status(job_id: str, state: dict[str, Any]) -> None:
    jobs_dir = TrainerConfig.artifacts_dir() / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    out_file = jobs_dir / f"{job_id}.json"
    tmp_file = out_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_file, out_file)


def _persist_session_metadata(session_dir: Path, meta: dict[str, Any]) -> None:
    """Atomically write `session_meta.json` inside `session_dir`."""
    session_dir.mkdir(parents=True, exist_ok=True)
    out_file = session_dir / "session_meta.json"
    tmp_file = out_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_file, out_file)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` via a sibling `.tmp` file so readers never see partial bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def _resolve_trainer_image_path(raw_path: str, ds_name: str | None = None) -> Path | None:
    """Resolve an image path across container (`/app/data/...`), host `DATA_DIR`, or renamed class folders."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists() and p.is_file():
        return p
    if "/app/data/" in raw_path:
        rel = raw_path.split("/app/data/", 1)[1]
        cand = TrainerConfig.DATA_DIR / rel
        if cand.exists() and cand.is_file():
            return cand
    search_ds = ds_name or TrainerConfig.get_active_dataset()
    raw_root = TrainerConfig.raw_dir(search_ds)
    if raw_root.exists():
        matches = [m for m in raw_root.rglob(p.name) if m.is_file()]
        if matches:
            return matches[0]
    return None


def _execute_clustering_task(job_id: str, req: ClusteringJobRequest) -> None:
    try:
        ds_name = req.dataset_name or TrainerConfig.get_active_dataset()
        _update_job(
            job_id,
            status="running",
            dataset_name=ds_name,
            progress=0.1,
            message=f"[{ds_name}] Scanning raw images (including class subfolders)...",
        )
        raw_dir = TrainerConfig.raw_dir(ds_name)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        scanned_paths: list[Path] = []
        subfolder_hints: list[str | None] = []

        for p in sorted(raw_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or "__MACOSX" in p.parts:
                continue
            if p.suffix.lower() not in valid_exts:
                continue
            scanned_paths.append(p)
            rel_parts = p.relative_to(raw_dir).parts
            subfolder_hints.append(rel_parts[0] if len(rel_parts) >= 2 else None)

        image_files = [str(p) for p in scanned_paths]
        if len(image_files) < 3:
            raise ValueError(
                f"Found only {len(image_files)} images in {raw_dir}. At least 3 images are required."
            )

        _update_job(
            job_id,
            progress=0.3,
            message=f"[{ds_name}] Extracting {req.backbone_name} ({req.pooling_type}) features for {len(image_files)} images...",
        )
        model = MetricProjectionModel(
            backbone_name=req.backbone_name,
            pooling_type=req.pooling_type,
            embedding_dim=TrainerConfig.EMBEDDING_DIM,
            pretrained=req.pretrained,
            freeze_backbone=True,
        )
        _, gap_features = extract_dataset_embeddings(
            image_paths=image_files,
            model=model,
            device=TrainerConfig.DEVICE,
            batch_size=32,
        )

        _update_job(
            job_id,
            progress=0.7,
            message=f"[{ds_name}] Keeping Normal class fixed (Cluster 0) & running PCA + Silhouette K-Means on Anomaly class...",
        )
        out_path = TrainerConfig.manifests_dir(ds_name) / "clustering_result.json"
        cluster_res = run_two_stage_clustering(
            features=gap_features,
            image_paths=image_files,
            normal_class_name=req.normal_class_name,
            core_quantile=req.core_quantile,
            k_nn=req.k_nn,
            max_k=req.max_k,
            max_pca_dim=req.max_pca_dim,
            pca_variance=req.pca_variance,
            backbone_name=req.backbone_name,
            ground_truth_hints=subfolder_hints,
            output_json_path=out_path,
        )

        det_norm_m = int(cluster_res.get("detected_normal_subclusters", 1))
        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=(
                f"[{ds_name}] Clustering completed: Normal fixed ({cluster_res['normal_core_count']} imgs, "
                f"M*={det_norm_m} latent sub-modes) + "
                f"Anomaly split into {cluster_res['best_tail_k']} sub-clusters ({cluster_res['tail_count']} imgs)."
            ),
            result={
                "dataset_name": ds_name,
                "clustering_mode": cluster_res.get("clustering_mode", "anomaly_only"),
                "normal_class_name": cluster_res.get("normal_class_name", req.normal_class_name),
                "total_samples": cluster_res["total_samples"],
                "normal_core_count": cluster_res["normal_core_count"],
                "detected_normal_subclusters": det_norm_m,
                "tail_count": cluster_res["tail_count"],
                "best_tail_k": cluster_res["best_tail_k"],
                "total_clusters": cluster_res["total_clusters"],
                "silhouette_scores": cluster_res["silhouette_scores"],
                "manifest_path": str(out_path),
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            message=f"Clustering failed: {exc}",
            error=traceback.format_exc(),
        )


@torch.no_grad()
def _extract_pixel_boundary_augmentation_bank(
    image_paths: list[str],
    labels: np.ndarray,
    is_defect: np.ndarray,
    normal_indices: list[int],
    defect_indices: list[int],
    model: MetricProjectionModel,
    device: str,
    normal_subclusters: np.ndarray | None = None,
    alpha_min: float = 0.55,
    alpha_max: float = 0.75,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pre-generate a bank of position-preserving Z-score Saliency Defect Transfer standardized features."""
    _ = is_defect
    if not normal_indices or not defect_indices:
        return None, None
    sigma_r = float(np.clip(0.035 + 0.10 * (0.5 * float(alpha_min + alpha_max)), 0.07, 0.14))

    model.eval()
    base_tf = build_image_transforms(is_train=False, image_size=128)
    max_def = min(len(defect_indices), 48)
    max_norm = min(len(normal_indices), 64)

    # Stratify selected normal templates across normal sub-clusters so every normal mode has clean references
    if normal_subclusters is not None and len(normal_subclusters) > max(normal_indices):
        by_sub: dict[int, list[int]] = {}
        for idx in normal_indices:
            by_sub.setdefault(max(0, int(normal_subclusters[idx])), []).append(int(idx))
        selected_norm: list[int] = []
        sub_keys = sorted(by_sub.keys())
        pos = 0
        while len(selected_norm) < max_norm:
            added = False
            for sk in sub_keys:
                if pos < len(by_sub[sk]) and len(selected_norm) < max_norm:
                    selected_norm.append(by_sub[sk][pos])
                    added = True
            if not added:
                break
            pos += 1
    else:
        selected_norm = list(normal_indices[:max_norm])

    norm_tensors = [load_image_tensor(image_paths[i], base_tf) for i in selected_norm]
    def_tensors = [load_image_tensor(image_paths[i], base_tf) for i in defect_indices[:max_def]]
    def_lbls = torch.tensor(
        [int(labels[i]) for i in defect_indices[:max_def]], dtype=torch.long, device=device
    )
    norm_sub_slice: torch.Tensor | None = None
    if normal_subclusters is not None and len(normal_subclusters) > max(selected_norm):
        norm_sub_slice = torch.tensor(
            [max(0, int(normal_subclusters[i])) for i in selected_norm],
            dtype=torch.long,
            device=device,
        )

    x_norm = torch.stack(norm_tensors, dim=0).to(device)
    x_def = torch.stack(def_tensors, dim=0).to(device)

    x_synth_all, y_synth_all = synthesize_defect_transfer_images(
        normal_images=x_norm,
        defect_images=x_def,
        defect_labels=def_lbls,
        num_synthetic=min(max_def * 2, 64),
        sigma_ratio=sigma_r,
        normal_subclusters=norm_sub_slice,
        max_shift_px=2,
    )
    if x_synth_all.shape[0] == 0:
        return None, None

    gap_chunks: list[torch.Tensor] = []
    for start in range(0, x_synth_all.shape[0], 32):
        chunk = x_synth_all[start : start + 32]
        gap_chunks.append(model.extract_gap_features(chunk))

    return torch.cat(gap_chunks, dim=0), y_synth_all


def _execute_training_task(
    job_id: str,
    session_number: int,
    session_id: str,
    session_dir: Path,
    req: TrainingJobRequest,
) -> None:
    ds_name = req.dataset_name or TrainerConfig.get_active_dataset()
    created_at = datetime.now(timezone.utc).isoformat()
    session_meta: dict[str, Any] = {
        "session_id": session_id,
        "session_number": session_number,
        "session_label": req.session_label or f"Session #{session_number:04d} ({ds_name})",
        "job_id": job_id,
        "dataset_name": ds_name,
        "created_at": created_at,
        "completed_at": None,
        "status": "running",
        "loss_type": req.loss_type,
        "backbone_name": req.backbone_name,
        "pooling_type": req.pooling_type,
        "epochs_completed": 0,
        "total_epochs": req.epochs,
        "best_epoch": 0,
        "best_val_recall": 0.0,
        "best_val_f1": 0.0,
        "best_val_precision": 0.0,
        "best_val_accuracy": 0.0,
        "best_test_recall": 0.0,
        "best_test_f1": 0.0,
        "best_test_precision": 0.0,
        "best_test_accuracy": 0.0,
        "best_d_min_boundary": 0.0,
        "best_d_min_centroid": 0.0,
        "config": req.model_dump(),
        "history": [],
    }
    _persist_session_metadata(session_dir, session_meta)

    try:
        _update_job(
            job_id,
            status="running",
            dataset_name=ds_name,
            session_id=session_id,
            session_number=session_number,
            progress=0.02,
            total_epochs=req.epochs,
            message=f"[Session #{session_number:04d} | {ds_name}] Loading annotations.json manifest...",
        )
        ann_path = TrainerConfig.manifests_dir(ds_name) / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation manifest not found at {ann_path} for dataset `{ds_name}`. Complete Phase 1 annotation first."
            )

        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
        raw_items = ann_data.get("items", [])
        items: list[dict[str, Any]] = []
        for raw_it in raw_items:
            resolved_img = _resolve_trainer_image_path(str(raw_it.get("image_path", "")), ds_name)
            if resolved_img is not None:
                it_copy = dict(raw_it)
                it_copy["image_path"] = str(resolved_img)
                items.append(it_copy)
        if len(items) < 4:
            raise ValueError(
                f"At least 4 valid image files are required for Phase 2 training (found {len(items)})."
            )

        # Ensure 6:2:2 (train:val:test = 60%:20%:20%) stratified split across defect modes AND normal sub-clusters
        if not any(str(it.get("split")) == "test" for it in items) and len(items) >= 6:
            rng_split = np.random.default_rng(42)
            by_lbl: dict[tuple[str, int], list[int]] = {}
            for idx_i, it in enumerate(items):
                l_nm = str(it.get("label_name", "normal"))
                sc_i = 0 if bool(it.get("is_defect", False)) else max(0, int(it.get("normal_subcluster_id", 0)))
                by_lbl.setdefault((l_nm, sc_i), []).append(idx_i)
            sp_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
            for _stratum_key, idxs in by_lbl.items():
                arr = list(idxs)
                rng_split.shuffle(arr)
                n_c = len(arr)
                if n_c >= 3:
                    n_v = max(1, int(round(n_c * 0.20)))
                    n_t = max(1, int(round(n_c * 0.20)))
                    if n_v + n_t >= n_c:
                        n_v, n_t = 1, 1
                    n_tr = max(1, n_c - n_v - n_t)
                    for p_i in arr[:n_tr]:
                        items[p_i]["split"] = "train"
                    for p_i in arr[n_tr : n_tr + n_v]:
                        items[p_i]["split"] = "val"
                    for p_i in arr[n_tr + n_v :]:
                        items[p_i]["split"] = "test"
                elif n_c == 2:
                    items[arr[0]]["split"] = "train"
                    items[arr[1]]["split"] = "test"
                elif n_c == 1:
                    items[arr[0]]["split"] = "train"
            for it in items:
                sp = str(it.get("split", "train"))
                sp_counts[sp] = sp_counts.get(sp, 0) + 1
            ann_data["split_counts"] = sp_counts
            ann_data["split_ratios"] = {"train": 0.60, "val": 0.20, "test": 0.20}
            tmp_ann = ann_path.with_suffix(".json.tmp")
            tmp_ann.write_text(json.dumps(ann_data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_ann, ann_path)

        image_paths = [str(item["image_path"]) for item in items]
        labels = np.array([int(item["label_id"]) for item in items], dtype=np.int64)
        label_names = [str(item["label_name"]) for item in items]
        is_defect = np.array([bool(item["is_defect"]) for item in items], dtype=bool)
        if not np.any(~is_defect):
            raise ValueError(
                "At least 1 Normal (`label_id=0`, `is_defect=False`) sample is required in annotations.json."
            )
        if not np.any(is_defect):
            raise ValueError(
                "At least 1 Defect (`is_defect=True`) sample is required in annotations.json for Phase 2 training."
            )
        splits = [str(item.get("split", "train")) for item in items]
        train_mask_np = np.array([s == "train" for s in splits], dtype=bool)
        val_mask_np = np.array([s == "val" for s in splits], dtype=bool)

        num_defect_classes = max(1, int(np.max(labels)) if len(labels) > 0 else 1)
        cb_source_labels = labels[train_mask_np] if np.any(train_mask_np) else labels
        cb_weights = compute_class_balanced_weights(
            labels=cb_source_labels,
            num_defect_classes=num_defect_classes,
            beta=req.class_balance_beta,
        )

        device = TrainerConfig.DEVICE
        model = MetricProjectionModel(
            backbone_name=req.backbone_name,
            pooling_type=req.pooling_type,
            embedding_dim=TrainerConfig.EMBEDDING_DIM,
            pretrained=req.pretrained,
            freeze_backbone=req.freeze_backbone,
            unfreeze_last_block=req.unfreeze_last_block,
        ).to(device)

        _update_job(
            job_id,
            progress=0.05,
            message=(
                f"[Session #{session_number:04d} | {ds_name}] Extracting raw features & calibrating "
                "within-subcluster statistics strictly on train Normal samples..."
            ),
        )
        raw_all = extract_raw_uncalibrated_features(
            image_paths=image_paths,
            model=model,
            device=device,
            batch_size=req.batch_size,
        )

        # Strict Normal-Only Calibration: exclude ALL defect samples so localized defect channels are never damped
        train_norm_mask = train_mask_np & (~is_defect)
        if int(np.sum(train_norm_mask)) < 2:
            train_norm_mask = (
                ~is_defect if int(np.sum(~is_defect)) >= 2 else np.ones(len(items), dtype=bool)
            )
        raw_train_norm = raw_all[train_norm_mask]

        use_proxy_anchor = req.loss_type == "multi_proxy_anchor"
        if not use_proxy_anchor:
            resolved_m = 1
            train_norm_subclusters = np.zeros(len(raw_train_norm), dtype=np.int64)
        elif int(req.num_normal_proxies) <= 0:
            resolved_m, train_norm_subclusters = estimate_optimal_normal_proxies(
                raw_train_norm,
                max_proxies=6,
                random_state=42,
            )
        else:
            resolved_m = int(req.num_normal_proxies)
            if (
                resolved_m > 1
                and len(raw_train_norm) >= resolved_m
                and float(np.var(raw_train_norm)) >= 1e-12
            ):
                n_mu = np.mean(raw_train_norm, axis=0, keepdims=True)
                z_n_centered = (raw_train_norm - n_mu).astype(np.float32)
                n_comp = min(32, len(raw_train_norm) - 1, z_n_centered.shape[1])
                z_n_proj = (
                    np.nan_to_num(
                        PCA(n_components=n_comp, random_state=42).fit_transform(z_n_centered),
                        nan=0.0,
                    )
                    if n_comp >= 2
                    else z_n_centered
                )
                z_n_sph = z_n_proj / (np.linalg.norm(z_n_proj, axis=1, keepdims=True) + 1e-8)
                raw_km_lbls = KMeans(
                    n_clusters=resolved_m, n_init=10, random_state=42
                ).fit_predict(z_n_sph)
                u_ids, u_cnts = np.unique(raw_km_lbls, return_counts=True)
                order_ids = u_ids[np.argsort(-u_cnts, kind="mergesort")]
                remap_ids = {int(old_i): new_i for new_i, old_i in enumerate(order_ids.tolist())}
                train_norm_subclusters = np.array(
                    [remap_ids[int(x)] for x in raw_km_lbls], dtype=np.int64
                )
            else:
                resolved_m = (
                    1
                    if float(np.var(raw_train_norm)) < 1e-12
                    else max(1, min(resolved_m, max(1, len(raw_train_norm))))
                )
                train_norm_subclusters = np.zeros(len(raw_train_norm), dtype=np.int64)

        all_normal_subclusters = np.full(len(items), -1, dtype=np.int64)
        if np.any(~is_defect):
            if resolved_m > 1 and len(raw_train_norm) >= 2:
                assigned_all_norm = assign_normal_subclusters_by_centroids(
                    features=raw_all[~is_defect],
                    ref_features=raw_train_norm,
                    ref_subclusters=train_norm_subclusters,
                )
                all_normal_subclusters[~is_defect] = assigned_all_norm
                all_normal_subclusters[train_norm_mask] = train_norm_subclusters
            else:
                all_normal_subclusters[~is_defect] = 0

        model.calibrate_feature_stats(
            raw_train_norm,
            subcluster_labels=train_norm_subclusters if resolved_m > 1 else None,
        )

        proxy_criterion: CustomizedMultiProxyAnchorLoss | None = None
        ms_criterion: AsymmetricMultiSimilarityLoss | None = None

        if use_proxy_anchor:
            proxy_criterion = CustomizedMultiProxyAnchorLoss(
                embedding_dim=TrainerConfig.EMBEDDING_DIM,
                num_normal_proxies=resolved_m,
                num_defect_classes=num_defect_classes,
                class_balanced_weights=cb_weights,
                normal_loss_ratio=req.normal_loss_ratio,
                delta_neg=req.repel_delta,
                w_defect=req.w_defect,
                hard_anomaly_boost=req.hard_anomaly_boost,
                ortho_weight=max(0.1, req.w_aux),
            ).to(device)
            trainable_params = [p for p in model.parameters() if p.requires_grad] + list(
                proxy_criterion.parameters()
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [p for p in model.parameters() if p.requires_grad],
                        "lr": req.learning_rate,
                    },
                    {
                        "params": list(proxy_criterion.parameters()),
                        "lr": req.learning_rate * 0.5,
                    },
                ],
                weight_decay=1e-4,
            )
        else:
            ms_criterion = AsymmetricMultiSimilarityLoss(
                w_defect=req.w_defect,
                w_aux=req.w_aux,
            )
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=req.learning_rate, weight_decay=1e-4
            )

        scheduler = CurriculumPenaltyScheduler(
            total_epochs=req.epochs,
            top_k_mode=req.top_k_mode,
            top_k_initial=req.top_k_initial,
            top_k_final=req.top_k_final,
            gamma_initial=req.gamma_initial,
            gamma_final=req.gamma_final,
            base_normal_ratio=req.normal_loss_ratio,
        )
        memory_bank = CrossBatchMemoryBank(embedding_dim=TrainerConfig.EMBEDDING_DIM)
        nab_sampler = NormalAnomalyBalancedSampler(
            labels=labels,
            is_defect=is_defect,
            splits=splits,
            batch_size=req.batch_size,
            normal_ratio=req.nab_normal_ratio,
            seed=42,
            normal_subclusters=all_normal_subclusters,
        )

        train_transform = build_image_transforms(is_train=True, image_size=128)
        rng = np.random.default_rng(42)

        train_normal_indices = nab_sampler.normal_indices
        train_defect_indices = nab_sampler.defect_indices

        session_meta["resolved_num_normal_proxies"] = int(resolved_m if use_proxy_anchor else 1)
        _persist_session_metadata(session_dir, session_meta)

        model.eval()
        with torch.no_grad():
            cached_gap_tensor = model.standardize_features(
                torch.from_numpy(raw_all).to(device=device, dtype=torch.float32)
            )
            init_z = model.forward_from_gap_features(cached_gap_tensor)
            if proxy_criterion is not None:
                if np.any(train_mask_np):
                    train_idx_t = torch.tensor(
                        np.where(train_mask_np)[0], dtype=torch.long, device=device
                    )
                    proxy_criterion.initialize_proxies_from_embeddings(
                        embeddings=init_z[train_idx_t],
                        labels=labels[train_mask_np],
                        normal_subclusters=all_normal_subclusters[train_mask_np],
                    )
                else:
                    proxy_criterion.initialize_proxies_from_embeddings(
                        embeddings=init_z,
                        labels=labels,
                        normal_subclusters=all_normal_subclusters,
                    )

        cached_gap_np = cached_gap_tensor.detach().cpu().numpy().astype(np.float32)
        use_cached_backbone = bool(req.freeze_backbone and not req.unfreeze_last_block)

        # Pre-extract position-preserving micro-shifted & brightness-jittered real defect views (NO flips/rotations)
        defect_micro_aug_bank: dict[int, torch.Tensor] = {}
        if use_cached_backbone and train_defect_indices:
            defect_micro_aug_bank = extract_micro_augmented_gap_bank(
                image_paths=image_paths,
                sample_indices=train_defect_indices,
                model=model,
                device=device,
                batch_size=req.batch_size,
            )

        # Pre-generate position-preserving Z-score Saliency Defect Transfer standardized features if enabled
        pixel_synth_gap: torch.Tensor | None = None
        pixel_synth_labels: torch.Tensor | None = None
        if req.enable_boundary_aug and use_cached_backbone and train_defect_indices:
            _update_job(
                job_id,
                progress=0.08,
                message=(
                    f"[Session #{session_number:04d}] Synthesizing Phase 3 same-coordinate Z-score "
                    f"Defect Transfer boundary samples (M*={resolved_m})..."
                ),
            )
            pixel_synth_gap, pixel_synth_labels = _extract_pixel_boundary_augmentation_bank(
                image_paths=image_paths,
                labels=labels,
                is_defect=is_defect,
                normal_indices=train_normal_indices,
                defect_indices=train_defect_indices,
                model=model,
                device=device,
                normal_subclusters=all_normal_subclusters,
                alpha_min=req.mixup_alpha_min,
                alpha_max=req.mixup_alpha_max,
            )

        # Clean root artifacts_dir epoch_*.pkl before starting so root mirrors the latest session
        root_art_dir = TrainerConfig.artifacts_dir()
        for old_pkl in root_art_dir.glob("epoch_*.pkl"):
            old_pkl.unlink(missing_ok=True)

        curr_state = scheduler.step(epoch=0, val_recall=0.0, d_min_boundary=0.0)
        history: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_epoch_info: dict[str, Any] = {}

        for epoch in range(1, req.epochs + 1):
            model.train()
            if proxy_criterion is not None:
                proxy_criterion.train()

            batches = nab_sampler.build_epoch_batches(
                hard_normal_indices=memory_bank.last_hard_normal_indices,
                min_steps=14,
            )

            epoch_losses: list[float] = []
            epoch_centripetal_losses: list[float] = []
            epoch_repel_losses: list[float] = []
            last_k_eff = 0

            for batch_indices in batches:
                batch_y = torch.tensor(
                    [labels[idx] for idx in batch_indices], dtype=torch.long, device=device
                )
                batch_sub = torch.tensor(
                    [int(all_normal_subclusters[idx]) for idx in batch_indices],
                    dtype=torch.long,
                    device=device,
                )
                batch_def = [bool(is_defect[idx]) for idx in batch_indices]
                is_def_mask = batch_y > 0

                optimizer.zero_grad()
                z_synth: torch.Tensor | None = None
                y_synth: torch.Tensor | None = None

                if use_cached_backbone:
                    idx_t = torch.tensor(batch_indices, dtype=torch.long, device=device)
                    batch_gap = cached_gap_tensor[idx_t].clone()
                    if defect_micro_aug_bank:
                        for pos_i, s_idx in enumerate(batch_indices):
                            if s_idx in defect_micro_aug_bank and float(rng.random()) < 0.75:
                                views_t = defect_micro_aug_bank[s_idx]
                                v_i = int(rng.integers(0, views_t.shape[0]))
                                batch_gap[pos_i] = views_t[v_i]
                    noise = torch.randn_like(batch_gap) * 0.012
                    batch_gap_aug = F.normalize(batch_gap + noise, p=2, dim=-1)
                    z_batch = model.forward_from_gap_features(batch_gap_aug)

                    if req.enable_boundary_aug and torch.any(is_def_mask) and torch.any(~is_def_mask):
                        synth_gap_parts: list[torch.Tensor] = []
                        synth_lbl_parts: list[torch.Tensor] = []

                        f_gap, f_lbl = synthesize_boundary_gap_features(
                            normal_gap=batch_gap[~is_def_mask],
                            defect_gap=batch_gap[is_def_mask],
                            defect_labels=batch_y[is_def_mask],
                            alpha_min=req.mixup_alpha_min,
                            alpha_max=req.mixup_alpha_max,
                            normal_subclusters=batch_sub[~is_def_mask],
                        )
                        if f_gap.shape[0] > 0:
                            synth_gap_parts.append(f_gap)
                            synth_lbl_parts.append(f_lbl)

                        if pixel_synth_gap is not None and pixel_synth_labels is not None:
                            n_pix = min(8, pixel_synth_gap.shape[0])
                            pix_idx = rng.choice(pixel_synth_gap.shape[0], size=n_pix, replace=False)
                            pix_idx_t = torch.tensor(pix_idx, dtype=torch.long, device=device)
                            synth_gap_parts.append(pixel_synth_gap[pix_idx_t])
                            synth_lbl_parts.append(pixel_synth_labels[pix_idx_t])

                        if synth_gap_parts:
                            all_synth_gap = torch.cat(synth_gap_parts, dim=0)
                            y_synth = torch.cat(synth_lbl_parts, dim=0)
                            z_synth = model.forward_from_gap_features(all_synth_gap)
                else:
                    batch_tensors = [
                        load_image_tensor(image_paths[idx], train_transform)
                        for idx in batch_indices
                    ]
                    batch_x = torch.stack(batch_tensors, dim=0).to(device)
                    z_batch, _ = model(batch_x)

                    if req.enable_boundary_aug and torch.any(is_def_mask) and torch.any(~is_def_mask):
                        x_norm_b = batch_x[~is_def_mask]
                        x_def_b = batch_x[is_def_mask]
                        y_def_b = batch_y[is_def_mask]
                        sigma_r_live = float(
                            np.clip(
                                0.035 + 0.10 * (0.5 * float(req.mixup_alpha_min + req.mixup_alpha_max)),
                                0.07,
                                0.14,
                            )
                        )
                        x_syn, y_synth = synthesize_defect_transfer_images(
                            normal_images=x_norm_b,
                            defect_images=x_def_b,
                            defect_labels=y_def_b,
                            num_synthetic=max(2, x_def_b.shape[0]),
                            sigma_ratio=sigma_r_live,
                            normal_subclusters=batch_sub[~is_def_mask],
                            max_shift_px=2,
                        )
                        if x_syn.shape[0] > 0:
                            z_synth, _ = model(x_syn)

                if proxy_criterion is not None:
                    loss, loss_stats = proxy_criterion(
                        embeddings=z_batch,
                        labels=batch_y,
                        synth_embeddings=z_synth,
                        synth_labels=y_synth,
                        memory_bank=memory_bank,
                        gamma_repel=curr_state.gamma_repel,
                        repel_margin=req.repel_delta,
                        top_k_mode=curr_state.top_k_mode,
                        top_k_value=curr_state.top_k_value,
                        normal_loss_ratio=curr_state.effective_normal_ratio,
                        normal_subclusters=batch_sub,
                    )
                else:
                    assert ms_criterion is not None
                    loss, loss_stats = ms_criterion(
                        embeddings=z_batch,
                        labels=batch_y,
                        memory_bank=memory_bank,
                        gamma_repel=curr_state.gamma_repel,
                        repel_margin=curr_state.repel_margin,
                        top_k_mode=curr_state.top_k_mode,
                        top_k_value=curr_state.top_k_value,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()

                memory_bank.update(
                    sample_indices=batch_indices,
                    embeddings=z_batch.detach(),
                    labels=batch_y.detach(),
                    is_defect_flags=batch_def,
                )
                epoch_losses.append(loss_stats["loss_total"])
                epoch_centripetal_losses.append(
                    float(loss_stats.get("loss_centripetal", loss_stats.get("loss_ms", 0.0)))
                )
                epoch_repel_losses.append(float(loss_stats.get("loss_repel", 0.0)))
                if loss_stats["effective_top_k"] > 0:
                    last_k_eff = int(loss_stats["effective_top_k"])

            release_stats = memory_bank.release_epoch_memory()

            if use_cached_backbone:
                model.eval()
                with torch.no_grad():
                    z_all = (
                        model.forward_from_gap_features(cached_gap_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                gap_all = cached_gap_np
            else:
                z_all, gap_all = extract_dataset_embeddings(
                    image_paths=image_paths,
                    model=model,
                    device=device,
                    batch_size=req.batch_size,
                )

            norm_proxies_np: np.ndarray | None = None
            def_proxies_np: np.ndarray | None = None
            norm_proxy_alphas_np: np.ndarray | None = None
            norm_proxy_radii_np: np.ndarray | None = None
            if proxy_criterion is not None:
                cal_mask = (
                    train_mask_np
                    if np.any(train_mask_np & (~is_defect))
                    else np.ones(len(items), dtype=bool)
                )
                proxy_criterion.calibrate_proxy_radii(
                    embeddings=z_all[cal_mask],
                    labels=labels[cal_mask],
                    normal_subclusters=all_normal_subclusters[cal_mask],
                )
                norm_proxies_np, def_proxies_np = proxy_criterion.get_proxies_numpy()
                norm_proxy_alphas_np, norm_proxy_radii_np = (
                    proxy_criterion.get_proxy_calibration_numpy()
                )

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_centripetal = (
                float(np.mean(epoch_centripetal_losses)) if epoch_centripetal_losses else 0.0
            )
            mean_repel = float(np.mean(epoch_repel_losses)) if epoch_repel_losses else 0.0

            # Compute validation loss on the `val` (20%) split to guide the next epoch's Loss direction
            val_loss_val = 0.0
            val_loss_cent = 0.0
            val_loss_rep = 0.0
            if np.any(val_mask_np):
                with torch.no_grad():
                    z_val_t = torch.from_numpy(z_all[val_mask_np]).to(device=device, dtype=torch.float32)
                    y_val_t = torch.from_numpy(labels[val_mask_np]).to(device=device, dtype=torch.long)
                    if proxy_criterion is not None:
                        _, val_l_stats = proxy_criterion(
                            embeddings=z_val_t,
                            labels=y_val_t,
                            synth_embeddings=None,
                            synth_labels=None,
                            memory_bank=None,
                            gamma_repel=curr_state.gamma_repel,
                            repel_margin=req.repel_delta,
                            top_k_mode=curr_state.top_k_mode,
                            top_k_value=curr_state.top_k_value,
                            normal_loss_ratio=curr_state.effective_normal_ratio,
                            normal_subclusters=all_normal_subclusters[val_mask_np],
                        )
                        val_loss_val = float(val_l_stats.get("loss_total", 0.0))
                        val_loss_cent = float(val_l_stats.get("loss_centripetal", 0.0))
                        val_loss_rep = float(val_l_stats.get("loss_repel", 0.0))
                    elif ms_criterion is not None:
                        _, val_l_stats = ms_criterion(
                            embeddings=z_val_t,
                            labels=y_val_t,
                            memory_bank=None,
                            gamma_repel=curr_state.gamma_repel,
                            repel_margin=curr_state.repel_margin,
                            top_k_mode=curr_state.top_k_mode,
                            top_k_value=curr_state.top_k_value,
                        )
                        val_loss_val = float(val_l_stats.get("loss_total", 0.0))
                        val_loss_cent = float(val_l_stats.get("loss_ms", 0.0))
                        val_loss_rep = float(val_l_stats.get("loss_repel", 0.0))

            is_milestone = (
                epoch == 1
                or epoch == req.epochs
                or (epoch % req.full_snapshot_interval == 0)
            )

            # Save into session_dir and sync to root_art_dir
            eval_res = evaluate_and_persist_epoch_space(
                epoch=epoch,
                image_paths=image_paths,
                labels=labels,
                label_names=label_names,
                is_defect=is_defect,
                splits=splits,
                embeddings_128d=z_all,
                gap_features=gap_all,
                artifacts_dir=session_dir,
                tau_sim=req.tau_sim,
                train_loss=mean_loss,
                val_loss=val_loss_val,
                loss_direction_feedback=curr_state.loss_direction_feedback,
                effective_normal_ratio=curr_state.effective_normal_ratio,
                curriculum_progress=curr_state.progress,
                effective_top_k=last_k_eff,
                save_full_snapshot=is_milestone,
                normal_proxies=norm_proxies_np,
                defect_proxies=def_proxies_np,
                normal_proxy_alphas=norm_proxy_alphas_np,
                normal_proxy_radii=norm_proxy_radii_np,
            )

            # Sync light/full pkl to root artifacts_dir atomically for backward compatibility
            _atomic_copy_file(
                Path(eval_res["light_pkl_path"]),
                root_art_dir / Path(eval_res["light_pkl_path"]).name,
            )
            if eval_res["full_pkl_path"]:
                _atomic_copy_file(
                    Path(eval_res["full_pkl_path"]),
                    root_art_dir / Path(eval_res["full_pkl_path"]).name,
                )

            val_recall = float(eval_res["val_metrics"]["recall"])
            val_f1 = float(eval_res["val_metrics"]["f1"])
            val_precision = float(eval_res["val_metrics"]["precision"])
            val_accuracy = float(eval_res["val_metrics"]["accuracy"])

            test_recall = float(eval_res["test_metrics"]["recall"])
            test_f1 = float(eval_res["test_metrics"]["f1"])
            test_precision = float(eval_res["test_metrics"]["precision"])
            test_accuracy = float(eval_res["test_metrics"]["accuracy"])

            train_recall = float(eval_res["train_metrics"]["recall"])
            train_f1 = float(eval_res["train_metrics"]["f1"])
            train_accuracy = float(eval_res["train_metrics"]["accuracy"])

            d_min_boundary = float(eval_res["d_min_boundary"])
            d_min_centroid = float(eval_res["d_min_centroid"])
            test_d_min_boundary = float(eval_res.get("test_d_min_boundary", d_min_boundary))
            test_d_min_centroid = float(eval_res.get("test_d_min_centroid", d_min_centroid))
            disp_m = eval_res.get("dispersion_metrics", {})

            # Best checkpoint selection strictly uses `val` split metrics (zero test leakage!)
            composite_quality = val_recall + 0.5 * val_f1 + 0.3 * val_accuracy + 0.2 * d_min_boundary
            if composite_quality >= best_score:
                best_score = composite_quality
                ckpt_payload = {
                    "session_id": session_id,
                    "session_number": session_number,
                    "dataset_name": ds_name,
                    "epoch": epoch,
                    "loss_type": req.loss_type,
                    "backbone_name": req.backbone_name,
                    "pooling_type": req.pooling_type,
                    "embedding_dim": TrainerConfig.EMBEDDING_DIM,
                    "enable_spatial_grid": bool(model.enable_spatial_grid),
                    "input_dropout": float(model.input_dropout_rate),
                    "resolved_num_normal_proxies": int(resolved_m if use_proxy_anchor else 1),
                    "state_dict": model.state_dict(),
                    "normal_prototype": eval_res["light_pkl_path"],
                    "normal_proxies": (
                        norm_proxies_np.tolist() if norm_proxies_np is not None else None
                    ),
                    "defect_proxies": (
                        def_proxies_np.tolist() if def_proxies_np is not None else None
                    ),
                    "normal_proxy_alphas": (
                        norm_proxy_alphas_np.tolist() if norm_proxy_alphas_np is not None else None
                    ),
                    "normal_proxy_radii": (
                        norm_proxy_radii_np.tolist() if norm_proxy_radii_np is not None else None
                    ),
                }
                session_ckpt = session_dir / "best_model.pt"
                tmp_ckpt = session_dir / f".best_model.{uuid.uuid4().hex[:8]}.pt.tmp"
                torch.save(ckpt_payload, tmp_ckpt)
                os.replace(tmp_ckpt, session_ckpt)
                _atomic_copy_file(session_ckpt, root_art_dir / "best_model.pt")
                best_epoch_info = {
                    "best_epoch": epoch,
                    "best_val_recall": val_recall,
                    "best_val_f1": val_f1,
                    "best_val_precision": val_precision,
                    "best_val_accuracy": val_accuracy,
                    "best_test_recall": test_recall,
                    "best_test_f1": test_f1,
                    "best_test_precision": test_precision,
                    "best_test_accuracy": test_accuracy,
                    "best_d_min_boundary": d_min_boundary,
                    "best_d_min_centroid": d_min_centroid,
                }

            # Validation feedback -> Next Epoch's Loss direction & curriculum schedule
            has_val_def = bool(np.any(val_mask_np & is_defect))
            next_state = scheduler.step(
                epoch=epoch,
                val_recall=val_recall,
                d_min_boundary=d_min_boundary,
                val_loss_centripetal=val_loss_cent,
                val_loss_repel=val_loss_rep,
                val_precision=val_precision,
                has_val_defects=has_val_def,
            )

            # Update saved `.pkl` snapshots atomically with post-validation next-epoch feedback
            for pkl_p_str in (eval_res["light_pkl_path"], eval_res["full_pkl_path"]):
                if pkl_p_str:
                    pkl_p = Path(pkl_p_str)
                    if pkl_p.exists():
                        with open(pkl_p, "rb") as pf:
                            snap_obj = pickle.load(pf)
                        snap_obj["loss_direction_feedback"] = str(next_state.loss_direction_feedback)
                        snap_obj["effective_normal_ratio"] = round(float(next_state.effective_normal_ratio), 4)
                        tmp_pkl_p = pkl_p.with_name(f".{pkl_p.name}.{uuid.uuid4().hex[:8]}.tmp")
                        with open(tmp_pkl_p, "wb") as pf:
                            pickle.dump(snap_obj, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(tmp_pkl_p, pkl_p)
                        _atomic_copy_file(pkl_p, root_art_dir / pkl_p.name)

            epoch_record = {
                "epoch": epoch,
                "loss_type": req.loss_type,
                "train_loss": round(mean_loss, 5),
                "val_loss": round(val_loss_val, 5),
                "loss_centripetal": round(mean_centripetal, 5),
                "loss_repel": round(mean_repel, 5),
                "val_loss_centripetal": round(val_loss_cent, 5),
                "val_loss_repel": round(val_loss_rep, 5),
                "normal_loss_ratio": round(float(curr_state.effective_normal_ratio), 4),
                "effective_normal_ratio": round(float(next_state.effective_normal_ratio), 4),
                "loss_direction_feedback": next_state.loss_direction_feedback,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_accuracy": train_accuracy,
                "d_min_boundary": d_min_boundary,
                "d_min_centroid": d_min_centroid,
                "test_d_min_boundary": test_d_min_boundary,
                "test_d_min_centroid": test_d_min_centroid,
                "normal_intra_disp": float(disp_m.get("normal_intra_disp", 0.0)),
                "defect_intra_disp": float(disp_m.get("defect_intra_disp", 0.0)),
                "inter_class_dist": float(disp_m.get("inter_class_dist", 0.0)),
                "fisher_separation_ratio": float(disp_m.get("fisher_separation_ratio", 0.0)),
                "curriculum_progress": curr_state.progress,
                "gamma_repel": curr_state.gamma_repel,
                "repel_margin": float(req.repel_delta if use_proxy_anchor else curr_state.repel_margin),
                "configured_top_k": curr_state.top_k_value,
                "effective_top_k": last_k_eff,
                "num_normal_proxies": int(resolved_m if use_proxy_anchor else 1),
                "xbm_released_entries": release_stats["released_entries"],
                "xbm_remaining_entries": release_stats["remaining_entries"],
                "light_pkl_path": eval_res["light_pkl_path"],
                "full_pkl_path": eval_res["full_pkl_path"],
            }
            history.append(epoch_record)
            curr_state = next_state

            session_meta.update(
                {
                    "epochs_completed": epoch,
                    "history": history,
                    **best_epoch_info,
                }
            )
            _persist_session_metadata(session_dir, session_meta)

            _update_job(
                job_id,
                current_epoch=epoch,
                progress=round(epoch / req.epochs, 4),
                history=history,
                message=(
                    f"[Session #{session_number:04d} | {ds_name}] Ep {epoch}/{req.epochs} | "
                    f"Loss={mean_loss:.4f} (Val={val_loss_val:.4f}) | "
                    f"Val Acc={val_accuracy:.3f} F1={val_f1:.3f} | "
                    f"Test Acc={test_accuracy:.3f} Recall={test_recall:.3f} F1={test_f1:.3f} | "
                    f"次Ep方向: {curr_state.loss_direction_feedback}"
                ),
            )

        session_meta.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _persist_session_metadata(session_dir, session_meta)

        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={
                "session_id": session_id,
                "session_number": session_number,
                "dataset_name": ds_name,
                "epochs_completed": req.epochs,
                "loss_type": req.loss_type,
                "final_metrics": history[-1] if history else {},
                "best_model_path": str(session_dir / "best_model.pt"),
            },
        )
    except Exception as exc:
        session_meta.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
        _persist_session_metadata(session_dir, session_meta)
        _update_job(
            job_id,
            status="failed",
            message=f"Training failed: {exc}",
            error=traceback.format_exc(),
        )


@router.post("/datasets/generate-synthetic")
def create_synthetic_dataset(req: SyntheticDatasetRequest) -> dict[str, Any]:
    """Generate synthetic 100:2 industrial inspection dataset in the target dataset's `raw` directory."""
    ds_name = req.dataset_name or TrainerConfig.get_active_dataset()
    res = generate_synthetic_inspection_dataset(
        output_dir=TrainerConfig.raw_dir(ds_name),
        num_normal=req.num_normal,
        num_scratch=req.num_scratch,
        num_stain=req.num_stain,
        num_dent=req.num_dent,
        image_size=req.image_size,
        seed=req.seed,
        clear_existing=req.clear_existing,
    )
    res["dataset_name"] = ds_name
    return res


@router.post("/jobs/clustering", response_model=JobStatusResponse)
def start_clustering_job(req: ClusteringJobRequest) -> JobStatusResponse:
    """Start an asynchronous Phase 1 2-stage clustering job."""
    active = get_active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Another ML job ({active}) is currently active on trainer-service.",
        )

    ds_name = req.dataset_name or TrainerConfig.get_active_dataset()
    job_id = f"cluster-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {
        "job_id": job_id,
        "job_type": "clustering",
        "status": "queued",
        "dataset_name": ds_name,
        "session_id": None,
        "session_number": None,
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": 1,
        "message": f"Queued clustering job for dataset `{ds_name}`.",
        "history": [],
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = state
        _persist_job_status(job_id, state)

    _EXECUTOR.submit(_execute_clustering_task, job_id, req)
    return JobStatusResponse(**state)


@router.post("/jobs/training", response_model=JobStatusResponse)
def start_training_job(req: TrainingJobRequest) -> JobStatusResponse:
    """Start an asynchronous Phase 2 training job and assign a sequential Session Number."""
    active = get_active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Another ML job ({active}) is currently active on trainer-service.",
        )

    ds_name = req.dataset_name or TrainerConfig.get_active_dataset()
    session_number, session_id, session_dir = TrainerConfig.allocate_next_session()
    job_id = f"train-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {
        "job_id": job_id,
        "job_type": "training",
        "status": "queued",
        "dataset_name": ds_name,
        "session_id": session_id,
        "session_number": session_number,
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": req.epochs,
        "message": f"Queued training job ({session_id} on dataset `{ds_name}`).",
        "history": [],
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = state
        _persist_job_status(job_id, state)

    _EXECUTOR.submit(_execute_training_task, job_id, session_number, session_id, session_dir, req)
    return JobStatusResponse(**state)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve live status and epoch history for a clustering or training job."""
    with _JOBS_LOCK:
        if job_id in _JOBS:
            return JobStatusResponse(**_JOBS[job_id])

    persisted = TrainerConfig.artifacts_dir() / "jobs" / f"{job_id}.json"
    if persisted.exists():
        data = json.loads(persisted.read_text(encoding="utf-8"))
        return JobStatusResponse(**data)

    raise HTTPException(status_code=404, detail=f"Job `{job_id}` not found.")


@router.post("/inference")
@torch.no_grad()
def run_single_image_inference(req: SingleInferenceRequest) -> dict[str, Any]:
    """Run single-image anomaly detection inference using `best_model.pt` from `session_id` (or latest)."""
    TrainerConfig.ensure_legacy_session_migrated()
    target_dir = (
        TrainerConfig.session_dir(req.session_id)
        if req.session_id
        else TrainerConfig.artifacts_dir()
    )
    ckpt_path = target_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = TrainerConfig.artifacts_dir() / "best_model.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No trained model checkpoint (`best_model.pt`) found. Run Phase 2 training first.",
        )

    ckpt = torch.load(ckpt_path, map_location=TrainerConfig.DEVICE, weights_only=False)
    resolved_img = _resolve_trainer_image_path(
        req.image_path, str(ckpt.get("dataset_name") or "") or None
    )
    if resolved_img is None:
        raise HTTPException(status_code=404, detail=f"Image not found at {req.image_path}.")
    img_path = resolved_img
    model = MetricProjectionModel(
        backbone_name=ckpt.get("backbone_name", "efficientnet_b3"),
        pooling_type=ckpt.get("pooling_type", "gem"),
        embedding_dim=int(ckpt.get("embedding_dim", 128)),
        pretrained=False,
        freeze_backbone=True,
        enable_spatial_grid=bool(ckpt.get("enable_spatial_grid", False)),
        input_dropout=float(ckpt.get("input_dropout", 0.0)),
    ).to(TrainerConfig.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normal_proxies = ckpt.get("normal_proxies")
    normal_proxy_alphas = ckpt.get("normal_proxy_alphas")
    normal_proto: np.ndarray | None = None

    if normal_proxies is None:
        best_ep = int(ckpt.get("epoch", 0))
        candidate_pkl = target_dir / f"epoch_{best_ep:04d}_light.pkl"
        if not candidate_pkl.exists():
            light_pkl_files = sorted(target_dir.glob("epoch_*_light.pkl"))
            if not light_pkl_files:
                light_pkl_files = sorted(TrainerConfig.artifacts_dir().glob("epoch_*_light.pkl"))
            if not light_pkl_files:
                raise HTTPException(
                    status_code=400,
                    detail="No epoch snapshot `.pkl` found for prototype reference.",
                )
            candidate_pkl = light_pkl_files[-1]
        with open(candidate_pkl, "rb") as f:
            snap = pickle.load(f)
        normal_proto = np.asarray(snap["normal_prototype"], dtype=np.float32)
        normal_proxies = snap.get("normal_proxies")
        normal_proxy_alphas = snap.get("normal_proxy_alphas")

    transform = build_image_transforms(is_train=False, image_size=128)
    tensor = load_image_tensor(img_path, transform).unsqueeze(0).to(TrainerConfig.DEVICE)
    z, _ = model(tensor)
    z_np = z.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if normal_proxies is not None and len(normal_proxies) > 0:
        norm_p_arr = np.asarray(normal_proxies, dtype=np.float32)
        if norm_p_arr.ndim == 1:
            norm_p_arr = norm_p_arr.reshape(1, -1)
        p_norms = np.linalg.norm(norm_p_arr, axis=1, keepdims=True) + 1e-8
        norm_p_arr = (norm_p_arr / p_norms).astype(np.float32)
        alphas_arr = (
            np.asarray(normal_proxy_alphas, dtype=np.float32)
            if normal_proxy_alphas is not None
            else None
        )
        cos_sim = float(
            compute_calibrated_normal_similarities_np(
                z_norm=z_np.reshape(1, -1),
                norm_proxies=norm_p_arr,
                proxy_alphas=alphas_arr,
            )[0]
        )
    else:
        assert normal_proto is not None
        cos_sim = float(np.dot(z_np, normal_proto))
    cos_dist = float(1.0 - cos_sim)
    is_anomaly = cos_sim < float(req.tau_sim)

    return {
        "session_id": req.session_id or ckpt.get("session_id", "session_0001"),
        "image_path": str(img_path),
        "cosine_similarity": round(cos_sim, 5),
        "cosine_distance": round(cos_dist, 5),
        "tau_sim": round(float(req.tau_sim), 4),
        "prediction": "NG (Defect)" if is_anomaly else "OK (Normal)",
        "is_defect": bool(is_anomaly),
        "embedding_128d": z_np.tolist(),
    }
