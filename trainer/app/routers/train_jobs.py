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

from app.config import TrainerConfig
from app.core.boundary_augmentation import (
    synthesize_anomalous_mixup_images,
    synthesize_boundary_gap_features,
    synthesize_defect_transfer_images,
)
from app.core.evaluator import evaluate_and_persist_epoch_space
from app.core.feature_extractor import (
    MetricProjectionModel,
    build_image_transforms,
    extract_dataset_embeddings,
    extract_raw_uncalibrated_features,
    load_image_tensor,
)
from app.core.memory_bank import CrossBatchMemoryBank
from app.core.ms_loss import AsymmetricMultiSimilarityLoss
from app.core.nab_sampler import (
    NormalAnomalyBalancedSampler,
    compute_class_balanced_weights,
)
from app.core.proxy_anchor_loss import CustomizedMultiProxyAnchorLoss
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

        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=(
                f"[{ds_name}] Clustering completed: Normal fixed ({cluster_res['normal_core_count']} imgs) + "
                f"Anomaly split into {cluster_res['best_tail_k']} sub-clusters ({cluster_res['tail_count']} imgs)."
            ),
            result={
                "dataset_name": ds_name,
                "clustering_mode": cluster_res.get("clustering_mode", "anomaly_only"),
                "normal_class_name": cluster_res.get("normal_class_name", req.normal_class_name),
                "total_samples": cluster_res["total_samples"],
                "normal_core_count": cluster_res["normal_core_count"],
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
    alpha_min: float = 0.80,
    alpha_max: float = 0.90,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pre-generate a bank of pixel-level Defect Transfer + Anomalous Mixup standardized features."""
    if not normal_indices or not defect_indices:
        return None, None

    model.eval()
    base_tf = build_image_transforms(is_train=False, image_size=128)
    max_def = min(len(defect_indices), 48)
    max_norm = min(len(normal_indices), 48)

    norm_tensors = [load_image_tensor(image_paths[i], base_tf) for i in normal_indices[:max_norm]]
    def_tensors = [load_image_tensor(image_paths[i], base_tf) for i in defect_indices[:max_def]]
    def_lbls = torch.tensor(
        [int(labels[i]) for i in defect_indices[:max_def]], dtype=torch.long, device=device
    )

    x_norm = torch.stack(norm_tensors, dim=0).to(device)
    x_def = torch.stack(def_tensors, dim=0).to(device)

    x_transfer, y_transfer = synthesize_defect_transfer_images(
        normal_images=x_norm,
        defect_images=x_def,
        defect_labels=def_lbls,
    )
    x_mixup, y_mixup = synthesize_anomalous_mixup_images(
        normal_images=x_norm,
        defect_images=x_def,
        defect_labels=def_lbls,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    x_synth_all = torch.cat([x_transfer, x_mixup], dim=0)
    y_synth_all = torch.cat([y_transfer, y_mixup], dim=0)
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
        items = ann_data.get("items", [])
        if len(items) < 4:
            raise ValueError("At least 4 annotated samples are required for Phase 2 training.")

        image_paths = [str(item["image_path"]) for item in items]
        labels = np.array([int(item["label_id"]) for item in items], dtype=np.int64)
        label_names = [str(item["label_name"]) for item in items]
        is_defect = np.array([bool(item["is_defect"]) for item in items], dtype=bool)
        splits = [str(item.get("split", "train")) for item in items]

        num_defect_classes = max(1, int(np.max(labels)) if len(labels) > 0 else 1)
        cb_weights = compute_class_balanced_weights(
            labels=labels,
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

        use_proxy_anchor = req.loss_type == "multi_proxy_anchor"
        proxy_criterion: CustomizedMultiProxyAnchorLoss | None = None
        ms_criterion: AsymmetricMultiSimilarityLoss | None = None

        if use_proxy_anchor:
            proxy_criterion = CustomizedMultiProxyAnchorLoss(
                embedding_dim=TrainerConfig.EMBEDDING_DIM,
                num_normal_proxies=req.num_normal_proxies,
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
        )
        memory_bank = CrossBatchMemoryBank(embedding_dim=TrainerConfig.EMBEDDING_DIM)
        nab_sampler = NormalAnomalyBalancedSampler(
            labels=labels,
            is_defect=is_defect,
            splits=splits,
            batch_size=req.batch_size,
            normal_ratio=req.nab_normal_ratio,
            seed=42,
        )

        train_transform = build_image_transforms(is_train=True, image_size=128)
        rng = np.random.default_rng(42)

        train_normal_indices = nab_sampler.normal_indices
        train_defect_indices = nab_sampler.defect_indices

        _update_job(
            job_id,
            progress=0.05,
            message=f"[Session #{session_number:04d} | {ds_name}] Extracting & calibrating baseline feature statistics for {len(image_paths)} images...",
        )
        raw_all = extract_raw_uncalibrated_features(
            image_paths=image_paths,
            model=model,
            device=device,
            batch_size=req.batch_size,
        )
        model.calibrate_feature_stats(raw_all)
        with torch.no_grad():
            cached_gap_tensor = model.standardize_features(
                torch.from_numpy(raw_all).to(device=device, dtype=torch.float32)
            )
            init_z = model.forward_from_gap_features(cached_gap_tensor)
            if proxy_criterion is not None:
                proxy_criterion.initialize_proxies_from_embeddings(
                    embeddings=init_z,
                    labels=labels,
                )

        cached_gap_np = cached_gap_tensor.detach().cpu().numpy().astype(np.float32)
        use_cached_backbone = bool(req.freeze_backbone and not req.unfreeze_last_block)

        # Pre-generate pixel-level Defect Transfer + Anomalous Mixup standardized features if enabled
        pixel_synth_gap: torch.Tensor | None = None
        pixel_synth_labels: torch.Tensor | None = None
        if req.enable_boundary_aug and use_cached_backbone and train_defect_indices:
            _update_job(
                job_id,
                progress=0.08,
                message=f"[Session #{session_number:04d}] Synthesizing Phase 3 Residual-Saliency Defect Transfer & Anomalous Mixup boundary samples...",
            )
            pixel_synth_gap, pixel_synth_labels = _extract_pixel_boundary_augmentation_bank(
                image_paths=image_paths,
                labels=labels,
                is_defect=is_defect,
                normal_indices=train_normal_indices,
                defect_indices=train_defect_indices,
                model=model,
                device=device,
                alpha_min=req.mixup_alpha_min,
                alpha_max=req.mixup_alpha_max,
            )

        # Clean root artifacts_dir epoch_*.pkl before starting so root mirrors the latest session
        root_art_dir = TrainerConfig.artifacts_dir()
        for old_pkl in root_art_dir.glob("epoch_*.pkl"):
            old_pkl.unlink(missing_ok=True)

        curr_state = scheduler.step(epoch=0, val_recall=0.0, d_min_boundary=0.0)
        history: list[dict[str, Any]] = []
        best_score = -1.0
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
                batch_def = [bool(is_defect[idx]) for idx in batch_indices]
                is_def_mask = batch_y > 0

                optimizer.zero_grad()
                z_synth: torch.Tensor | None = None
                y_synth: torch.Tensor | None = None

                if use_cached_backbone:
                    idx_t = torch.tensor(batch_indices, dtype=torch.long, device=device)
                    batch_gap = cached_gap_tensor[idx_t]
                    noise = torch.randn_like(batch_gap) * 0.015
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
                        x_tr, y_tr = synthesize_defect_transfer_images(
                            normal_images=x_norm_b,
                            defect_images=x_def_b,
                            defect_labels=y_def_b,
                            num_synthetic=max(2, x_def_b.shape[0] // 2),
                        )
                        x_mx, y_mx = synthesize_anomalous_mixup_images(
                            normal_images=x_norm_b,
                            defect_images=x_def_b,
                            defect_labels=y_def_b,
                            alpha_min=req.mixup_alpha_min,
                            alpha_max=req.mixup_alpha_max,
                            num_synthetic=max(2, x_def_b.shape[0] // 2),
                        )
                        x_syn = torch.cat([x_tr, x_mx], dim=0)
                        y_synth = torch.cat([y_tr, y_mx], dim=0)
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
                        normal_loss_ratio=req.normal_loss_ratio,
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
            if proxy_criterion is not None:
                norm_proxies_np, def_proxies_np = proxy_criterion.get_proxies_numpy()

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_centripetal = (
                float(np.mean(epoch_centripetal_losses)) if epoch_centripetal_losses else 0.0
            )
            mean_repel = float(np.mean(epoch_repel_losses)) if epoch_repel_losses else 0.0
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
                curriculum_progress=curr_state.progress,
                effective_top_k=last_k_eff,
                save_full_snapshot=is_milestone,
                normal_proxies=norm_proxies_np,
                defect_proxies=def_proxies_np,
            )

            # Sync light/full pkl to root artifacts_dir for backward compatibility
            shutil.copy2(eval_res["light_pkl_path"], root_art_dir / Path(eval_res["light_pkl_path"]).name)
            if eval_res["full_pkl_path"]:
                shutil.copy2(eval_res["full_pkl_path"], root_art_dir / Path(eval_res["full_pkl_path"]).name)

            val_recall = float(eval_res["val_metrics"]["recall"])
            val_f1 = float(eval_res["val_metrics"]["f1"])
            val_precision = float(eval_res["val_metrics"]["precision"])
            d_min_boundary = float(eval_res["d_min_boundary"])
            d_min_centroid = float(eval_res["d_min_centroid"])
            disp_m = eval_res.get("dispersion_metrics", {})

            composite_quality = val_recall + 0.5 * val_f1 + 0.2 * d_min_boundary
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
                    "state_dict": model.state_dict(),
                    "normal_prototype": eval_res["light_pkl_path"],
                    "normal_proxies": (
                        norm_proxies_np.tolist() if norm_proxies_np is not None else None
                    ),
                }
                session_ckpt = session_dir / "best_model.pt"
                torch.save(ckpt_payload, session_ckpt)
                shutil.copy2(session_ckpt, root_art_dir / "best_model.pt")
                best_epoch_info = {
                    "best_epoch": epoch,
                    "best_val_recall": val_recall,
                    "best_val_f1": val_f1,
                    "best_val_precision": val_precision,
                    "best_d_min_boundary": d_min_boundary,
                    "best_d_min_centroid": d_min_centroid,
                }

            next_state = scheduler.step(
                epoch=epoch,
                val_recall=val_recall,
                d_min_boundary=d_min_boundary,
            )

            epoch_record = {
                "epoch": epoch,
                "loss_type": req.loss_type,
                "train_loss": round(mean_loss, 5),
                "loss_centripetal": round(mean_centripetal, 5),
                "loss_repel": round(mean_repel, 5),
                "normal_loss_ratio": round(float(req.normal_loss_ratio), 3),
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "d_min_boundary": d_min_boundary,
                "d_min_centroid": d_min_centroid,
                "normal_intra_disp": float(disp_m.get("normal_intra_disp", 0.0)),
                "defect_intra_disp": float(disp_m.get("defect_intra_disp", 0.0)),
                "inter_class_dist": float(disp_m.get("inter_class_dist", 0.0)),
                "fisher_separation_ratio": float(disp_m.get("fisher_separation_ratio", 0.0)),
                "curriculum_progress": curr_state.progress,
                "gamma_repel": curr_state.gamma_repel,
                "repel_margin": float(req.repel_delta if use_proxy_anchor else curr_state.repel_margin),
                "configured_top_k": curr_state.top_k_value,
                "effective_top_k": last_k_eff,
                "num_normal_proxies": int(req.num_normal_proxies if use_proxy_anchor else 1),
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
                    f"[Session #{session_number:04d} | {ds_name}] Epoch {epoch}/{req.epochs} [{req.loss_type}] | "
                    f"Loss={mean_loss:.4f} | Recall={val_recall:.3f} | F1={val_f1:.3f} | d_min={d_min_boundary:.3f}"
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
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {img_path}.")

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
    model = MetricProjectionModel(
        backbone_name=ckpt.get("backbone_name", "efficientnet_b3"),
        pooling_type=ckpt.get("pooling_type", "gem"),
        embedding_dim=int(ckpt.get("embedding_dim", 128)),
        pretrained=False,
        freeze_backbone=True,
    ).to(TrainerConfig.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    light_pkl_files = sorted(target_dir.glob("epoch_*_light.pkl"))
    if not light_pkl_files:
        light_pkl_files = sorted(TrainerConfig.artifacts_dir().glob("epoch_*_light.pkl"))
    if not light_pkl_files:
        raise HTTPException(status_code=400, detail="No epoch snapshot `.pkl` found for prototype reference.")

    with open(light_pkl_files[-1], "rb") as f:
        snap = pickle.load(f)
    normal_proto = np.asarray(snap["normal_prototype"], dtype=np.float32)
    normal_proxies = snap.get("normal_proxies")

    transform = build_image_transforms(is_train=False, image_size=128)
    tensor = load_image_tensor(img_path, transform).unsqueeze(0).to(TrainerConfig.DEVICE)
    z, _ = model(tensor)
    z_np = z.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if normal_proxies is not None and len(normal_proxies) > 0:
        norm_p_arr = np.asarray(normal_proxies, dtype=np.float32)
        if norm_p_arr.ndim == 1:
            norm_p_arr = norm_p_arr.reshape(1, -1)
        cos_sim = float(np.max(norm_p_arr @ z_np))
    else:
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
