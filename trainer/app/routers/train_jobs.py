"""Internal ML worker endpoints for synthetic dataset generation, 2-stage clustering, MS-Loss+XBM training, and inference."""

from __future__ import annotations

import json
import os
import pickle
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from app.config import TrainerConfig
from app.core.evaluator import evaluate_and_persist_epoch_space
from app.core.feature_extractor import (
    MetricProjectionModel,
    build_image_transforms,
    extract_dataset_embeddings,
    load_image_tensor,
)
from app.core.memory_bank import CrossBatchMemoryBank
from app.core.ms_loss import AsymmetricMultiSimilarityLoss
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


def _execute_clustering_task(job_id: str, req: ClusteringJobRequest) -> None:
    try:
        _update_job(job_id, status="running", progress=0.1, message="Scanning raw images (including class subfolders)...")
        raw_dir = TrainerConfig.raw_dir()
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
            message=f"Extracting {req.backbone_name} ({req.pooling_type}) features for {len(image_files)} images...",
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
            message="Running Stage 1 (Normal Core extraction) & Stage 2 (Tail PCA + Silhouette K-Means)...",
        )
        out_path = TrainerConfig.manifests_dir() / "clustering_result.json"
        cluster_res = run_two_stage_clustering(
            features=gap_features,
            image_paths=image_files,
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
            message=f"Clustering completed: {cluster_res['total_clusters']} clusters identified (best tail k={cluster_res['best_tail_k']}).",
            result={
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


def _build_balanced_batches(
    normal_indices: list[int],
    defect_indices: list[int],
    hard_normal_indices: list[int],
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Construct class-balanced + hard-negative-prioritized mini-batches.

    1. Up-samples minority defect samples so every mini-batch contains at least 25% defects
       (enabling positive and auxiliary-negative pairs inside each batch).
    2. Injects Top-K boundary normal indices discovered by the Cross-Batch Memory Bank so
       boundary normals receive live forward/backward repulsion gradients.
    """
    if not normal_indices and not defect_indices:
        return []
    if not defect_indices:
        shuffled = list(normal_indices)
        rng.shuffle(shuffled)
        return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]

    norm_pool = list(normal_indices)
    rng.shuffle(norm_pool)

    n_defect_per_batch = max(4, batch_size // 4)
    n_normal_per_batch = max(4, batch_size - n_defect_per_batch)
    num_batches = max(2, int(np.ceil(len(norm_pool) / n_normal_per_batch)))

    batches: list[list[int]] = []
    hard_set = [idx for idx in hard_normal_indices if idx in normal_indices]

    for b_idx in range(num_batches):
        start = (b_idx * n_normal_per_batch) % len(norm_pool)
        selected_normals: list[int] = []
        # Prioritize up to half of normal slots from XBM hard boundary normals
        if hard_set:
            n_hard = min(len(hard_set), n_normal_per_batch // 2)
            hard_chosen = rng.choice(hard_set, size=n_hard, replace=False).tolist()
            selected_normals.extend(int(x) for x in hard_chosen)

        while len(selected_normals) < n_normal_per_batch:
            cand = norm_pool[start % len(norm_pool)]
            start += 1
            selected_normals.append(int(cand))

        # Up-sample defect indices with replacement so every defect mode has positive pairs
        chosen_defects = rng.choice(
            defect_indices, size=n_defect_per_batch, replace=True
        ).tolist()
        batch = selected_normals + [int(x) for x in chosen_defects]
        batches.append(batch)

    return batches


def _execute_training_task(job_id: str, req: TrainingJobRequest) -> None:
    try:
        _update_job(
            job_id,
            status="running",
            progress=0.02,
            total_epochs=req.epochs,
            message="Loading annotations.json manifest...",
        )
        ann_path = TrainerConfig.manifests_dir() / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation manifest not found at {ann_path}. Complete Phase 1 annotation first."
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

        device = TrainerConfig.DEVICE
        model = MetricProjectionModel(
            backbone_name=req.backbone_name,
            pooling_type=req.pooling_type,
            embedding_dim=TrainerConfig.EMBEDDING_DIM,
            pretrained=req.pretrained,
            freeze_backbone=req.freeze_backbone,
            unfreeze_last_block=req.unfreeze_last_block,
        ).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=req.learning_rate, weight_decay=1e-4)

        criterion = AsymmetricMultiSimilarityLoss(
            w_defect=req.w_defect,
            w_aux=req.w_aux,
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

        train_transform = build_image_transforms(is_train=True, image_size=128)
        rng = np.random.default_rng(42)

        train_normal_indices = [
            i for i in range(len(items)) if (not is_defect[i]) and splits[i] == "train"
        ]
        train_defect_indices = [
            i for i in range(len(items)) if is_defect[i] and splits[i] == "train"
        ]
        if not train_normal_indices:
            train_normal_indices = [i for i in range(len(items)) if not is_defect[i]]
        if not train_defect_indices:
            train_defect_indices = [i for i in range(len(items)) if is_defect[i]]

        curr_state = scheduler.step(epoch=0, val_recall=0.0, d_min_boundary=0.0)
        history: list[dict[str, Any]] = []
        best_score = -1.0

        for epoch in range(1, req.epochs + 1):
            model.train()
            batches = _build_balanced_batches(
                normal_indices=train_normal_indices,
                defect_indices=train_defect_indices,
                hard_normal_indices=memory_bank.last_hard_normal_indices,
                batch_size=req.batch_size,
                rng=rng,
            )

            epoch_losses: list[float] = []
            last_k_eff = 0

            for batch_indices in batches:
                # Apply symmetric augmentations to both normal and defect samples
                batch_tensors = [
                    load_image_tensor(image_paths[idx], train_transform) for idx in batch_indices
                ]
                batch_x = torch.stack(batch_tensors, dim=0).to(device)
                batch_y = torch.tensor([labels[idx] for idx in batch_indices], dtype=torch.long, device=device)
                batch_def = [bool(is_defect[idx]) for idx in batch_indices]

                optimizer.zero_grad()
                z_batch, _ = model(batch_x)

                loss, loss_stats = criterion(
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

                # Update XBM with detached embeddings from this step
                memory_bank.update(
                    sample_indices=batch_indices,
                    embeddings=z_batch.detach(),
                    labels=batch_y.detach(),
                    is_defect_flags=batch_def,
                )
                epoch_losses.append(loss_stats["loss_total"])
                if loss_stats["effective_top_k"] > 0:
                    last_k_eff = int(loss_stats["effective_top_k"])

            # Mandatory OOM Prevention: Explicitly release Cross-Batch Memory Bank at epoch end
            release_stats = memory_bank.release_epoch_memory()

            # Step 6: Evaluate full dataset space and persist `.pkl`
            z_all, gap_all = extract_dataset_embeddings(
                image_paths=image_paths,
                model=model,
                device=device,
                batch_size=req.batch_size,
            )

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            is_milestone = (
                epoch == 1
                or epoch == req.epochs
                or (epoch % req.full_snapshot_interval == 0)
            )

            eval_res = evaluate_and_persist_epoch_space(
                epoch=epoch,
                image_paths=image_paths,
                labels=labels,
                label_names=label_names,
                is_defect=is_defect,
                splits=splits,
                embeddings_128d=z_all,
                gap_features=gap_all,
                artifacts_dir=TrainerConfig.artifacts_dir(),
                tau_sim=req.tau_sim,
                train_loss=mean_loss,
                curriculum_progress=curr_state.progress,
                effective_top_k=last_k_eff,
                save_full_snapshot=is_milestone,
            )

            val_recall = float(eval_res["val_metrics"]["recall"])
            val_f1 = float(eval_res["val_metrics"]["f1"])
            d_min_boundary = float(eval_res["d_min_boundary"])
            d_min_centroid = float(eval_res["d_min_centroid"])

            # Save best model checkpoint (.pt) and full snapshot when validation Recall+F1 improves
            composite_quality = val_recall + 0.5 * val_f1 + 0.2 * d_min_boundary
            if composite_quality >= best_score:
                best_score = composite_quality
                ckpt_path = TrainerConfig.artifacts_dir() / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone_name": req.backbone_name,
                        "pooling_type": req.pooling_type,
                        "embedding_dim": TrainerConfig.EMBEDDING_DIM,
                        "state_dict": model.state_dict(),
                        "normal_prototype": eval_res["light_pkl_path"],
                    },
                    ckpt_path,
                )

            # Advance Step 5 curriculum scheduler for the next epoch
            next_state = scheduler.step(
                epoch=epoch,
                val_recall=val_recall,
                d_min_boundary=d_min_boundary,
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(mean_loss, 5),
                "val_recall": val_recall,
                "val_precision": float(eval_res["val_metrics"]["precision"]),
                "val_f1": val_f1,
                "d_min_boundary": d_min_boundary,
                "d_min_centroid": d_min_centroid,
                "curriculum_progress": curr_state.progress,
                "gamma_repel": curr_state.gamma_repel,
                "repel_margin": curr_state.repel_margin,
                "configured_top_k": curr_state.top_k_value,
                "effective_top_k": last_k_eff,
                "xbm_released_entries": release_stats["released_entries"],
                "xbm_remaining_entries": release_stats["remaining_entries"],
                "light_pkl_path": eval_res["light_pkl_path"],
                "full_pkl_path": eval_res["full_pkl_path"],
            }
            history.append(epoch_record)
            curr_state = next_state

            _update_job(
                job_id,
                current_epoch=epoch,
                progress=round(epoch / req.epochs, 4),
                history=history,
                message=(
                    f"Epoch {epoch}/{req.epochs} | Loss={mean_loss:.4f} | "
                    f"Recall={val_recall:.3f} | F1={val_f1:.3f} | d_min={d_min_boundary:.3f}"
                ),
            )

        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={
                "epochs_completed": req.epochs,
                "final_metrics": history[-1] if history else {},
                "best_model_path": str(TrainerConfig.artifacts_dir() / "best_model.pt"),
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            message=f"Training failed: {exc}",
            error=traceback.format_exc(),
        )


@router.post("/datasets/generate-synthetic")
def create_synthetic_dataset(req: SyntheticDatasetRequest) -> dict[str, Any]:
    """Generate synthetic 100:2 industrial inspection dataset in `/app/data/raw`."""
    return generate_synthetic_inspection_dataset(
        output_dir=TrainerConfig.raw_dir(),
        num_normal=req.num_normal,
        num_scratch=req.num_scratch,
        num_stain=req.num_stain,
        num_dent=req.num_dent,
        image_size=req.image_size,
        seed=req.seed,
        clear_existing=req.clear_existing,
    )


@router.post("/jobs/clustering", response_model=JobStatusResponse)
def start_clustering_job(req: ClusteringJobRequest) -> JobStatusResponse:
    """Start an asynchronous Phase 1 2-stage clustering job."""
    active = get_active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Another ML job ({active}) is currently active on trainer-service.",
        )

    job_id = f"cluster-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {
        "job_id": job_id,
        "job_type": "clustering",
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": 1,
        "message": "Queued clustering job.",
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
    """Start an asynchronous Phase 2 Asymmetric MS-Loss + XBM training job."""
    active = get_active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Another ML job ({active}) is currently active on trainer-service.",
        )

    job_id = f"train-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {
        "job_id": job_id,
        "job_type": "training",
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": req.epochs,
        "message": "Queued MS-Loss + XBM training job.",
        "history": [],
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = state
        _persist_job_status(job_id, state)

    _EXECUTOR.submit(_execute_training_task, job_id, req)
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
    """Run single-image anomaly detection inference using `best_model.pt` and the latest normal prototype."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {img_path}.")

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

    light_pkl_files = sorted(TrainerConfig.artifacts_dir().glob("epoch_*_light.pkl"))
    if not light_pkl_files:
        raise HTTPException(status_code=400, detail="No epoch snapshot `.pkl` found for prototype reference.")

    with open(light_pkl_files[-1], "rb") as f:
        snap = pickle.load(f)
    normal_proto = np.asarray(snap["normal_prototype"], dtype=np.float32)

    transform = build_image_transforms(is_train=False, image_size=128)
    tensor = load_image_tensor(img_path, transform).unsqueeze(0).to(TrainerConfig.DEVICE)
    z, _ = model(tensor)
    z_np = z.squeeze(0).detach().cpu().numpy().astype(np.float32)

    cos_sim = float(np.dot(z_np, normal_proto))
    cos_dist = float(1.0 - cos_sim)
    is_anomaly = cos_sim < float(req.tau_sim)

    return {
        "image_path": str(img_path),
        "cosine_similarity": round(cos_sim, 5),
        "cosine_distance": round(cos_dist, 5),
        "tau_sim": round(float(req.tau_sim), 4),
        "prediction": "NG (Defect)" if is_anomaly else "OK (Normal)",
        "is_defect": bool(is_anomaly),
        "embedding_128d": z_np.tolist(),
    }
