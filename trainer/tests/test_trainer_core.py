"""Unit tests for Phase 1 2-stage clustering, Phase 2 Asymmetric MS-Loss + XBM, and Step 6 evaluator."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from app.core.boundary_augmentation import (
    synthesize_anomalous_mixup_images,
    synthesize_boundary_gap_features,
    synthesize_defect_transfer_images,
)
from app.core.evaluator import (
    compute_binary_metrics_at_threshold,
    evaluate_and_persist_epoch_space,
)
from app.core.feature_extractor import MetricProjectionModel
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


def test_nab_sampler_and_class_balanced_weights() -> None:
    """Verify NAB-Sampler 50:50 normal-anomaly stratification and Effective-Number Class-Balanced Weights."""
    # 100 normals (0), 20 defect mode 1, 2 defect mode 2, 1 rare defect mode 3
    labels = np.array([0] * 100 + [1] * 20 + [2] * 2 + [3] * 1, dtype=np.int64)
    is_defect = labels > 0
    splits = ["train"] * len(labels)

    cb_weights = compute_class_balanced_weights(labels=labels, num_defect_classes=3, beta=0.99)
    assert cb_weights.shape == (3,)
    # Rare defect mode 3 (n_3=1) must receive a strictly larger weight than common mode 1 (n_1=20)
    assert cb_weights[2] > cb_weights[1] > cb_weights[0]
    assert np.isclose(np.mean(cb_weights), 1.0, atol=1e-4)

    sampler = NormalAnomalyBalancedSampler(
        labels=labels,
        is_defect=is_defect,
        splits=splits,
        batch_size=24,
        normal_ratio=0.50,
        seed=42,
    )
    batches = sampler.build_epoch_batches(hard_normal_indices=[0, 1, 2], min_steps=6)
    assert len(batches) >= 6

    for batch in batches:
        assert len(batch) == 24
        b_labels = labels[batch]
        n_norm = int(np.sum(b_labels == 0))
        n_def = int(np.sum(b_labels > 0))
        assert n_norm == 12
        assert n_def == 12
        # All 3 defect modes (including 1-sample mode 3) must appear with 4 samples each
        for cid in (1, 2, 3):
            assert int(np.sum(b_labels == cid)) == 4


def test_multi_proxy_anchor_loss_zero_inter_defect_sample_repulsion_and_one_way_synth() -> None:
    """Verify CustomizedMultiProxyAnchorLoss eliminates sample-level inter-defect repulsion and supports one-way synthetic repulsion."""
    torch.manual_seed(11)
    criterion = CustomizedMultiProxyAnchorLoss(
        embedding_dim=128,
        num_normal_proxies=4,
        num_defect_classes=2,
        class_balanced_weights=[0.8, 1.2],
        hard_anomaly_boost=3.0,
    )

    init_z = torch.randn(20, 128)
    init_y = [0] * 14 + [1, 1, 1, 2, 2, 2]
    criterion.initialize_proxies_from_embeddings(init_z, init_y)

    # Create batch with Normal (0), Defect Mode 1 (1), and Defect Mode 2 (2)
    z_norm = torch.randn(4, 128, requires_grad=True)
    z_def1 = torch.randn(2, 128, requires_grad=True)
    z_def2_a = torch.randn(2, 128, requires_grad=True)

    y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.long)

    # Pass 1 with z_def2_a
    z_batch_a = torch.cat([z_norm, z_def1, z_def2_a], dim=0)
    loss_a, _ = criterion(embeddings=z_batch_a, labels=y, top_k_mode="count", top_k_value=4)
    grad_def1_a = torch.autograd.grad(loss_a, z_def1, retain_graph=True)[0]

    # Pass 2 where Defect Mode 2 samples are completely flipped (-z_def2_a), while keeping same Top-K
    z_def2_b = (-z_def2_a).detach().requires_grad_(True)
    z_batch_b = torch.cat([z_norm, z_def1, z_def2_b], dim=0)
    loss_b, _ = criterion(embeddings=z_batch_b, labels=y, top_k_mode="count", top_k_value=4)
    grad_def1_b = torch.autograd.grad(loss_b, z_def1, retain_graph=True)[0]

    # Because there is zero sample-to-sample repulsion between Defect 1 and Defect 2,
    # the gradient wrt Defect 1 samples is invariant to the direction of Defect 2 samples!
    assert torch.allclose(grad_def1_a, grad_def1_b, atol=1e-6)

    # Verify One-Way Synthetic Boundary Repulsion (Defect Transfer & Anomalous Mixup)
    z_synth = torch.randn(4, 128, requires_grad=True)
    y_synth = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    loss_with_synth, stats = criterion(
        embeddings=z_batch_a,
        labels=y,
        synth_embeddings=z_synth,
        synth_labels=y_synth,
        top_k_mode="ratio",
        top_k_value=0.50,
    )
    loss_with_synth.backward()
    assert z_synth.grad is not None
    assert float(z_synth.grad.norm().item()) > 0.0
    assert criterion.normal_proxies.grad is not None
    assert criterion.defect_proxies.grad is not None
    assert stats["loss_ortho"] >= 0.0


def test_phase3_boundary_augmentation_and_multi_proxy_evaluator(tmp_path: Path) -> None:
    """Verify Residual-Saliency Defect Transfer, Anomalous Mixup, and Multi-Proxy `.pkl` evaluation."""
    torch.manual_seed(7)
    x_norm = torch.rand(6, 3, 64, 64)
    x_def = torch.rand(3, 3, 64, 64)
    # Inject a bright localized defect patch in x_def[:, :, 20:32, 20:32]
    x_def[:, :, 20:32, 20:32] = 1.0
    y_def = torch.tensor([1, 2, 1], dtype=torch.long)

    x_transfer, y_tr = synthesize_defect_transfer_images(x_norm, x_def, y_def)
    assert x_transfer.shape == (3, 3, 64, 64)
    assert torch.equal(y_tr, y_def)

    x_mix, y_mx = synthesize_anomalous_mixup_images(
        x_norm, x_def, y_def, alpha_min=0.80, alpha_max=0.90
    )
    assert x_mix.shape == (3, 3, 64, 64)
    assert torch.equal(y_mx, y_def)

    h_norm = torch.randn(6, 128)
    h_def = torch.randn(3, 128)
    synth_gap, synth_lbl = synthesize_boundary_gap_features(h_norm, h_def, y_def)
    assert synth_gap.shape == (6, 128)
    assert synth_lbl.shape == (6,)

    # Verify Multi-Proxy Evaluator persists normal_proxies & defect_proxies as pure NumPy
    n = 12
    rng = np.random.default_rng(99)
    z = rng.normal(0.0, 0.1, size=(n, 128)).astype(np.float32)
    z[:8, 0] = 2.0  # Normals near +e_0
    z[8:, 0] = -2.0  # Defects near -e_0
    norm_proxies = np.zeros((3, 128), dtype=np.float32)
    norm_proxies[:, 0] = 1.0
    def_proxies = np.zeros((2, 128), dtype=np.float32)
    def_proxies[:, 0] = -1.0

    eval_res = evaluate_and_persist_epoch_space(
        epoch=1,
        image_paths=[f"/tmp/p_{i}.png" for i in range(n)],
        labels=np.array([0] * 8 + [1, 1, 2, 2], dtype=np.int64),
        label_names=["normal"] * 8 + ["scratch"] * 2 + ["stain"] * 2,
        is_defect=np.array([False] * 8 + [True] * 4, dtype=bool),
        splits=["train"] * 6 + ["val"] * 2 + ["train", "val", "train", "val"],
        embeddings_128d=z,
        gap_features=None,
        artifacts_dir=tmp_path,
        tau_sim=0.0,
        normal_proxies=norm_proxies,
        defect_proxies=def_proxies,
    )
    assert eval_res["val_metrics"]["recall"] == 1.0
    with open(eval_res["light_pkl_path"], "rb") as f:
        snap = pickle.load(f)
    assert snap["normal_proxies"].shape == (3, 128)
    assert snap["defect_proxies"].shape == (2, 128)
    assert snap["normal_proxies_umap_3d"].shape == (3, 3)
    assert snap["defect_proxies_umap_3d"].shape == (2, 3)
