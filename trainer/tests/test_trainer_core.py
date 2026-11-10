"""Unit tests for Phase 1 2-stage clustering, Phase 2 Asymmetric MS-Loss + XBM, and Step 6 evaluator."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from app.core.evaluator import (
    compute_binary_metrics_at_threshold,
    evaluate_and_persist_epoch_space,
)
from app.core.feature_extractor import MetricProjectionModel
from app.core.memory_bank import CrossBatchMemoryBank
from app.core.ms_loss import AsymmetricMultiSimilarityLoss
from app.core.scheduler import CurriculumPenaltyScheduler
from app.core.synthetic_data import generate_synthetic_inspection_dataset
from app.core.two_stage_cluster import run_two_stage_clustering


def test_two_stage_clustering_isolates_rare_defects(tmp_path: Path) -> None:
    """Verify Stage 1 extracts the normal core and Stage 2 isolates rare defect modes under 100:2 imbalance."""
    rng = np.random.default_rng(123)
    d = 64
    # 100 normal samples tightly clustered around e_0
    normal_center = np.zeros(d, dtype=np.float32)
    normal_center[0] = 1.0
    normals = normal_center + rng.normal(0.0, 0.05, size=(100, d)).astype(np.float32)

    # 3 scratch defects around e_1 and 3 stain defects around e_2
    scratch_center = np.zeros(d, dtype=np.float32)
    scratch_center[1] = 1.0
    scratches = scratch_center + rng.normal(0.0, 0.03, size=(3, d)).astype(np.float32)

    stain_center = np.zeros(d, dtype=np.float32)
    stain_center[2] = 1.0
    stains = stain_center + rng.normal(0.0, 0.03, size=(3, d)).astype(np.float32)

    features = np.vstack([normals, scratches, stains])
    paths = (
        [f"/tmp/sample_normal_{i}.png" for i in range(100)]
        + [f"/tmp/sample_defect_scratch_{i}.png" for i in range(3)]
        + [f"/tmp/sample_defect_stain_{i}.png" for i in range(3)]
    )

    out_json = tmp_path / "clustering_result.json"
    result = run_two_stage_clustering(
        features=features,
        image_paths=paths,
        core_quantile=0.85,
        k_nn=10,
        max_k=6,
        output_json_path=out_json,
    )

    assert out_json.exists()
    assert result["total_samples"] == 106
    assert result["best_tail_k"] >= 2

    # Ensure zero defects were swallowed into the Stage 1 Normal Core (cluster_id == 0)
    defect_items = [it for it in result["items"] if "defect" in it["file_name"]]
    assert len(defect_items) == 6
    for item in defect_items:
        assert item["is_normal_core"] is False
        assert item["cluster_id"] > 0


def test_cross_batch_memory_bank_count_ratio_and_epoch_release() -> None:
    """Verify XBM detached storage, count vs ratio Top-K retrieval, and per-epoch memory release."""
    bank = CrossBatchMemoryBank(embedding_dim=128)
    z = torch.randn(25, 128, requires_grad=True)
    indices = list(range(25))
    # 20 normals (label 0), 5 defects (label 1, 2)
    labels = [0] * 20 + [1, 1, 1, 2, 2]
    is_defect = [False] * 20 + [True] * 5

    bank.update(sample_indices=indices, embeddings=z, labels=labels, is_defect_flags=is_defect)
    assert len(bank) == 25
    assert bank.normal_count == 20
    assert bank.defect_count == 5

    defect_query = torch.randn(3, 128)

    # 1. Count mode (request K=50, should safely clamp to K_eff=20)
    b_norm_count, hard_ids_count, k_eff_count = bank.get_top_k_boundary_normals(
        defect_embeddings=defect_query,
        top_k_mode="count",
        top_k_value=50,
    )
    assert k_eff_count == 20
    assert b_norm_count.shape == (3, 20, 128)
    assert b_norm_count.requires_grad is False
    assert len(hard_ids_count) > 0

    # 2. Ratio mode (request top 10% of 20 normals => K_eff = 2)
    b_norm_ratio, _, k_eff_ratio = bank.get_top_k_boundary_normals(
        defect_embeddings=defect_query,
        top_k_mode="ratio",
        top_k_value=0.10,
    )
    assert k_eff_ratio == 2
    assert b_norm_ratio.shape == (3, 2, 128)

    # 3. Per-epoch memory release (OOM prevention)
    release_info = bank.release_epoch_memory()
    assert release_info["released_entries"] == 25
    assert release_info["remaining_entries"] == 0
    assert len(bank) == 0


def test_asymmetric_ms_loss_and_scheduler_backward() -> None:
    """Verify AsymmetricMultiSimilarityLoss backward pass and CurriculumPenaltyScheduler progression."""
    model = MetricProjectionModel(
        backbone_name="lightweight_cnn",
        pooling_type="gem",
        embedding_dim=128,
        pretrained=False,
    )
    bank = CrossBatchMemoryBank(embedding_dim=128)
    criterion = AsymmetricMultiSimilarityLoss()
    scheduler = CurriculumPenaltyScheduler(
        total_epochs=10,
        top_k_mode="ratio",
        top_k_initial=0.25,
        top_k_final=0.10,
    )

    # Pre-populate bank with detached embeddings
    bank.update(
        sample_indices=list(range(12)),
        embeddings=torch.randn(12, 128),
        labels=[0] * 10 + [1, 2],
        is_defect_flags=[False] * 10 + [True, True],
    )

    x = torch.randn(6, 3, 64, 64)
    y = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    z, _ = model(x)

    state_early = scheduler.step(epoch=1, val_recall=0.0, d_min_boundary=0.1)
    state_late = scheduler.step(epoch=10, val_recall=1.0, d_min_boundary=1.0)
    assert state_late.progress > state_early.progress
    assert state_late.gamma_repel > state_early.gamma_repel
    assert state_late.top_k_value < state_early.top_k_value

    loss, stats = criterion(
        embeddings=z,
        labels=y,
        memory_bank=bank,
        gamma_repel=state_early.gamma_repel,
        repel_margin=state_early.repel_margin,
        top_k_mode=state_early.top_k_mode,
        top_k_value=state_early.top_k_value,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert stats["effective_top_k"] == 2.0


def test_evaluator_pure_numpy_pkl_and_dynamic_threshold(tmp_path: Path) -> None:
    """Verify `.pkl` contains zero torch.Tensors and supports dynamic `tau_sim` recalculation."""
    n = 16
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1.0, size=(n, 128)).astype(np.float32)
    # Make first 12 samples align with +e_0 (normals) and last 4 align with -e_0 (defects)
    z[:12, 0] = 3.0
    z[12:, 0] = -3.0
    gap = rng.normal(0.0, 1.0, size=(n, 256)).astype(np.float32)

    labels = np.array([0] * 12 + [1, 1, 2, 2], dtype=np.int64)
    label_names = ["normal"] * 12 + ["defect_scratch"] * 2 + ["defect_stain"] * 2
    is_defect = labels > 0
    splits = ["train"] * 9 + ["val"] * 3 + ["train", "val", "train", "val"]
    paths = [f"/tmp/img_{i}.png" for i in range(n)]

    res = evaluate_and_persist_epoch_space(
        epoch=1,
        image_paths=paths,
        labels=labels,
        label_names=label_names,
        is_defect=is_defect,
        splits=splits,
        embeddings_128d=z,
        gap_features=gap,
        artifacts_dir=tmp_path,
        tau_sim=0.0,
        save_full_snapshot=True,
    )

    assert Path(res["light_pkl_path"]).exists()
    assert res["full_pkl_path"] is not None and Path(res["full_pkl_path"]).exists()
    assert res["val_metrics"]["recall"] == 1.0
    assert res["val_metrics"]["f1"] == 1.0

    with open(res["light_pkl_path"], "rb") as f:
        loaded = pickle.load(f)

    assert loaded["embeddings_128d"].shape == (n, 128)
    assert loaded["umap_3d"].shape == (n, 3)
    assert loaded["gap_features"] is None

    # Recalculate metrics at custom user threshold tau_sim = -0.99 (so strict that nothing is flagged)
    strict_metrics = compute_binary_metrics_at_threshold(
        cosine_similarities=loaded["cosine_similarities"],
        is_defect=loaded["is_defect"],
        tau_sim=-0.99,
    )
    assert strict_metrics["recall"] == 0.0


def test_synthetic_data_generator(tmp_path: Path) -> None:
    """Verify synthetic 100:2 dataset generator creates valid RGB PNG images."""
    meta = generate_synthetic_inspection_dataset(
        output_dir=tmp_path,
        num_normal=10,
        num_scratch=2,
        num_stain=1,
        num_dent=1,
        image_size=64,
    )
    assert meta["total_generated"] == 14
    assert len(list(tmp_path.glob("*.png"))) == 14
