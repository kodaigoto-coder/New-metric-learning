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
    compute_calibrated_normal_similarities_np,
    evaluate_and_persist_epoch_space,
)
from app.core.feature_extractor import (
    MetricProjectionModel,
    SpatialPoolingHead,
    build_image_transforms,
)
from app.core.memory_bank import CrossBatchMemoryBank
from app.core.ms_loss import AsymmetricMultiSimilarityLoss
from app.core.nab_sampler import (
    NormalAnomalyBalancedSampler,
    compute_class_balanced_weights,
)
from app.core.proxy_anchor_loss import (
    CustomizedMultiProxyAnchorLoss,
    estimate_optimal_normal_proxies,
)
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


def test_asymmetric_workpiece_transforms_and_spatial_grid_pooling() -> None:
    """Verify training transforms contain zero flips/rotations and SpatialPoolingHead preserves (x, y) location."""
    from torchvision import transforms

    tf_train = build_image_transforms(is_train=True, image_size=128)
    for t in tf_train.transforms:
        assert not isinstance(
            t, (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip, transforms.RandomRotation)
        )
        if isinstance(t, transforms.RandomAffine):
            assert tuple(float(d) for d in t.degrees) == (0.0, 0.0)

    head = SpatialPoolingHead(pooling_type="gem", enable_spatial_grid=True)
    # Construct two feature maps with identical global statistics but activation at top-left vs bottom-right
    fmap_tl = torch.zeros(1, 64, 8, 8)
    fmap_br = torch.zeros(1, 64, 8, 8)
    fmap_tl[:, :16, 0:2, 0:2] = 5.0
    fmap_br[:, :16, 6:8, 6:8] = 5.0

    g_tl = head.forward_global(fmap_tl)
    g_br = head.forward_global(fmap_br)
    assert torch.allclose(g_tl, g_br, atol=1e-6)

    sp_tl = head.forward_spatial_grid(fmap_tl)
    sp_br = head.forward_spatial_grid(fmap_br)
    assert sp_tl.shape == (1, 512)
    assert not torch.allclose(sp_tl, sp_br, atol=1e-3)


def test_normal_only_within_subcluster_feature_standardization() -> None:
    """Verify calibrate_feature_stats uses pooled within-subcluster std instead of inflated inter-mode std."""
    model = MetricProjectionModel(
        backbone_name="lightweight_cnn",
        pooling_type="gap",
        embedding_dim=128,
        pretrained=False,
        freeze_backbone=True,
    )
    dim = model.gap_feature_dim
    rng = np.random.default_rng(7)

    # Two normal subclusters separated by +/-10.0 on dim 0, with within-cluster std ~ 0.10
    mode_a = rng.normal(0.0, 0.10, size=(20, dim)).astype(np.float32)
    mode_b = rng.normal(0.0, 0.10, size=(20, dim)).astype(np.float32)
    mode_a[:, 0] += 10.0
    mode_b[:, 0] -= 10.0
    raw_norm = np.vstack([mode_a, mode_b])
    sub_lbls = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    model.calibrate_feature_stats(raw_norm, subcluster_labels=sub_lbls)
    std_0 = float(model.feat_std[0, 0].item())
    # Pooled within-subcluster std must reflect ~0.10 (NOT ~10.0 inter-cluster gap!)
    assert 0.04 < std_0 < 0.25


def test_auto_normal_proxies_mobius_calibration_and_same_subcluster_synthesis() -> None:
    """Verify estimate_optimal_normal_proxies, Möbius radius calibration, shallow synth guardrail, and zero cross-cluster Mixup."""
    rng = np.random.default_rng(2026)
    d = 64

    # 1. Unimodal normal distribution -> M* == 1
    unimodal = rng.normal(0.0, 0.08, size=(30, d)).astype(np.float32)
    unimodal[:, 0] += 3.0
    m_uni, lbl_uni = estimate_optimal_normal_proxies(unimodal, max_proxies=5, random_state=42)
    assert m_uni == 1
    assert np.all(lbl_uni == 0)

    # 2. Bimodal normal distribution (two distinct cavities/sub-modes) -> M* == 2
    c_a = rng.normal(0.0, 0.08, size=(20, d)).astype(np.float32)
    c_b = rng.normal(0.0, 0.08, size=(20, d)).astype(np.float32)
    c_a[:, 0] += 3.0
    c_b[:, 1] += 3.0
    bimodal = np.vstack([c_a, c_b])
    m_bi, lbl_bi = estimate_optimal_normal_proxies(bimodal, max_proxies=5, random_state=42)
    assert m_bi == 2
    assert len(set(lbl_bi.tolist())) == 2

    # 3. Möbius radius calibration equalizes tight vs loose normal sub-clusters
    criterion = CustomizedMultiProxyAnchorLoss(
        embedding_dim=128,
        num_normal_proxies=2,
        num_defect_classes=1,
        delta_neg=0.0,
    )
    z_tight = np.zeros((15, 128), dtype=np.float32)
    z_tight[:, 0] = 1.0
    z_tight += rng.normal(0.0, 0.02, size=(15, 128)).astype(np.float32)
    z_loose = np.zeros((15, 128), dtype=np.float32)
    z_loose[:, 0] = 0.7
    z_loose[:, 1] = 0.7
    z_loose += rng.normal(0.0, 0.14, size=(15, 128)).astype(np.float32)
    z_def = np.zeros((4, 128), dtype=np.float32)
    z_def[:, 0] = -1.0

    z_init = np.vstack([z_tight, z_loose, z_def])
    y_init = np.array([0] * 30 + [1] * 4, dtype=np.int64)
    sub_init = np.array([0] * 15 + [1] * 15 + [-1] * 4, dtype=np.int64)
    criterion.initialize_proxies_from_embeddings(z_init, y_init, normal_subclusters=sub_init)
    alphas_np, radii_np = criterion.get_proxy_calibration_numpy()
    assert radii_np[1] > radii_np[0]
    assert alphas_np[0] > alphas_np[1]

    p_norm_np, _ = criterion.get_proxies_numpy()
    sims_cal = compute_calibrated_normal_similarities_np(z_init[:30], p_norm_np, alphas_np)
    assert sims_cal.shape == (30,)
    assert float(np.min(sims_cal)) > 0.3

    # Shallow positive-hemisphere synthetic guardrail: synthetic vector already at S_cal <= 0.0 receives ZERO gradient
    z_batch = torch.from_numpy(z_init[:8]).float().requires_grad_(True)
    y_batch = torch.tensor([0] * 6 + [1, 1], dtype=torch.long)
    z_batch.data[6:] = torch.from_numpy(z_def[:2])
    z_synth_deep = torch.from_numpy(z_def[:2]).float().requires_grad_(True)
    y_synth_deep = torch.tensor([1, 1], dtype=torch.long)
    loss_deep, _ = criterion(
        embeddings=z_batch,
        labels=y_batch,
        synth_embeddings=z_synth_deep,
        synth_labels=y_synth_deep,
        repel_margin=0.0,
        normal_subclusters=[0] * 6 + [-1, -1],
    )
    loss_deep.backward()
    assert z_synth_deep.grad is not None
    assert float(z_synth_deep.grad.norm().item()) == 0.0

    # 4. Same-subcluster boundary synthesis NEVER places points on chord between opposite normal modes
    norm_gap = torch.zeros(8, 32)
    norm_gap[:4, 0] = 1.0   # Subcluster 0 at +e_0
    norm_gap[4:, 1] = 1.0   # Subcluster 1 at +e_1
    def_gap = torch.zeros(2, 32)
    def_gap[0, 0] = 0.9
    def_gap[0, 5] = 0.6     # Defect occurring on Subcluster 0 workpiece
    def_gap[1, 1] = 0.9
    def_gap[1, 6] = 0.6     # Defect occurring on Subcluster 1 workpiece
    synth_f, _ = synthesize_boundary_gap_features(
        normal_gap=norm_gap,
        defect_gap=def_gap,
        defect_labels=torch.tensor([1, 1], dtype=torch.long),
        normal_subclusters=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
    )
    # Defect 0 derivatives (indices 0, 2) must stay in Subcluster 0 (+e_0, zero e_1 mixing)
    assert float(synth_f[0, 1].abs().item()) < 1e-6
    assert float(synth_f[2, 1].abs().item()) < 1e-6
    # Defect 1 derivatives (indices 1, 3) must stay in Subcluster 1 (+e_1, zero e_0 mixing)
    assert float(synth_f[1, 0].abs().item()) < 1e-6
    assert float(synth_f[3, 0].abs().item()) < 1e-6


def test_nab_sampler_stratifies_minority_normal_subclusters() -> None:
    """Verify NAB-Sampler balances normal slots across normal sub-clusters when M* >= 2."""
    # 30 normal mode 0, only 3 normal mode 1, plus 4 defects
    labels = np.array([0] * 33 + [1, 1, 2, 2], dtype=np.int64)
    is_defect = labels > 0
    splits = ["train"] * 37
    norm_sub = np.array([0] * 30 + [1] * 3 + [-1] * 4, dtype=np.int64)

    sampler = NormalAnomalyBalancedSampler(
        labels=labels,
        is_defect=is_defect,
        splits=splits,
        batch_size=16,
        normal_ratio=0.50,
        seed=42,
        normal_subclusters=norm_sub,
    )
    batches = sampler.build_epoch_batches(min_steps=4)
    for b in batches:
        b_arr = np.asarray(b, dtype=np.int64)
        b_norm = b_arr[labels[b_arr] == 0]
        assert len(b_norm) == 8
        # Both normal subcluster 0 and minority subcluster 1 must receive 4 slots each!
        assert int(np.sum(norm_sub[b_norm] == 0)) == 4
        assert int(np.sum(norm_sub[b_norm] == 1)) == 4


def test_end_to_end_bimodal_asymmetric_training_and_inference(tmp_path: Path, monkeypatch) -> None:
    """Verify full clustering -> Auto-M* training -> Möbius calibrated inference on bimodal asymmetric images."""
    import json
    from PIL import Image, ImageDraw
    from app.config import TrainerConfig
    from app.routers.train_jobs import (
        _JOBS,
        _execute_clustering_task,
        _execute_training_task,
        run_single_image_inference,
    )
    from app.schemas.job_models import (
        ClusteringJobRequest,
        SingleInferenceRequest,
        TrainingJobRequest,
    )

    monkeypatch.setattr(TrainerConfig, "DATA_DIR", tmp_path)
    raw_dir = TrainerConfig.raw_dir("default")
    (raw_dir / "normal").mkdir(parents=True, exist_ok=True)
    (raw_dir / "scratch").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(101)
    # Create 2 distinct Normal modes on an asymmetric workpiece:
    # Mode A (10 imgs): bright metallic top-left boss, dark blue base
    # Mode B (10 imgs): warm copper finish with top-left boss
    for i in range(10):
        arr_a = np.clip(rng.normal(45, 4, size=(64, 64, 3)), 0, 255).astype(np.uint8)
        arr_a[:, :, 2] = np.clip(arr_a[:, :, 2] + 60, 0, 255)
        img_a = Image.fromarray(arr_a)
        draw_a = ImageDraw.Draw(img_a)
        draw_a.rectangle([6, 6, 22, 22], fill=(210, 215, 225))
        img_a.save(raw_dir / "normal" / f"modeA_{i:02d}.png")

        arr_b = np.clip(rng.normal(135, 4, size=(64, 64, 3)), 0, 255).astype(np.uint8)
        arr_b[:, :, 0] = np.clip(arr_b[:, :, 0] + 55, 0, 255)
        img_b = Image.fromarray(arr_b)
        draw_b = ImageDraw.Draw(img_b)
        draw_b.rectangle([6, 6, 22, 22], fill=(235, 190, 140))
        img_b.save(raw_dir / "normal" / f"modeB_{i:02d}.png")

    # Create 6 position-specific defects (bright red/white gash at bottom-right [42:56, 42:56])
    defect_paths: list[Path] = []
    for i in range(6):
        is_b = i >= 3
        base_val = 135 if is_b else 45
        arr_d = np.clip(rng.normal(base_val, 4, size=(64, 64, 3)), 0, 255).astype(np.uint8)
        if is_b:
            arr_d[:, :, 0] = np.clip(arr_d[:, :, 0] + 55, 0, 255)
        else:
            arr_d[:, :, 2] = np.clip(arr_d[:, :, 2] + 60, 0, 255)
        img_d = Image.fromarray(arr_d)
        draw_d = ImageDraw.Draw(img_d)
        draw_d.rectangle([6, 6, 22, 22], fill=(235, 190, 140) if is_b else (210, 215, 225))
        draw_d.rectangle([42, 42, 56, 56], fill=(255, 20, 20))
        out_p = raw_dir / "scratch" / f"ng_{i:02d}.png"
        img_d.save(out_p)
        defect_paths.append(out_p)

    # 1. Run Phase 1 Clustering
    _JOBS["c-test"] = {"job_id": "c-test", "status": "queued"}
    _execute_clustering_task(
        "c-test",
        ClusteringJobRequest(
            dataset_name="default",
            backbone_name="lightweight_cnn",
            pooling_type="gem",
            pretrained=False,
            normal_class_name="normal",
        ),
    )
    assert _JOBS["c-test"]["status"] == "completed"
    assert _JOBS["c-test"]["result"]["detected_normal_subclusters"] >= 2

    # 2. Build annotations.json (legacy 2-split without 'test' to also exercise stratified 6:2:2 auto-upgrade)
    cl_json = json.loads(
        (TrainerConfig.manifests_dir("default") / "clustering_result.json").read_text(encoding="utf-8")
    )
    ann_items = []
    for pos_i, it in enumerate(cl_json["items"]):
        is_def = int(it["cluster_id"]) > 0
        ann_items.append(
            {
                "sample_id": str(it.get("sample_id", f"s_{pos_i}")),
                "file_name": it["file_name"],
                "image_path": it["image_path"],
                "cluster_id": it["cluster_id"],
                "normal_subcluster_id": it.get("normal_subcluster_id", 0),
                "label_id": 1 if is_def else 0,
                "label_name": "scratch" if is_def else "normal",
                "is_defect": is_def,
                "split": "train",
            }
        )
    (TrainerConfig.manifests_dir("default") / "annotations.json").write_text(
        json.dumps({"dataset_name": "default", "items": ann_items}, indent=2),
        encoding="utf-8",
    )

    # 3. Run Phase 2 Training with num_normal_proxies=0 (Auto M*)
    s_num, s_id, s_dir = TrainerConfig.allocate_next_session()
    _JOBS["t-test"] = {"job_id": "t-test", "status": "queued"}
    _execute_training_task(
        "t-test",
        s_num,
        s_id,
        s_dir,
        TrainingJobRequest(
            dataset_name="default",
            backbone_name="lightweight_cnn",
            pooling_type="gem",
            pretrained=False,
            freeze_backbone=True,
            epochs=3,
            batch_size=12,
            num_normal_proxies=0,
            enable_boundary_aug=True,
        ),
    )
    assert _JOBS["t-test"]["status"] == "completed", _JOBS["t-test"].get("error")
    meta = json.loads((s_dir / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["resolved_num_normal_proxies"] >= 2
    assert (s_dir / "best_model.pt").exists()

    # 4. Run single-image inference on both normal modes and defect
    inf_norm_a = run_single_image_inference(
        SingleInferenceRequest(
            image_path=str(raw_dir / "normal" / "modeA_00.png"),
            session_id=s_id,
            tau_sim=0.0,
        )
    )
    inf_norm_b = run_single_image_inference(
        SingleInferenceRequest(
            image_path=str(raw_dir / "normal" / "modeB_00.png"),
            session_id=s_id,
            tau_sim=0.0,
        )
    )
    inf_def = run_single_image_inference(
        SingleInferenceRequest(
            image_path=str(defect_paths[0]),
            session_id=s_id,
            tau_sim=0.0,
        )
    )
    assert inf_norm_a["cosine_similarity"] > inf_def["cosine_similarity"]
    assert inf_norm_b["cosine_similarity"] > inf_def["cosine_similarity"]


def test_vit_128x128_and_zero_error_edge_cases() -> None:
    """Verify ViT 128x128 spatial interpolation, zero-variance PCA, tiny-dim synthesis, and schema parity."""
    from app.core.two_stage_cluster import reduce_tail_pca
    from app.schemas.job_models import TrainingJobRequest

    # 1. Exact 28-field UI payload validates cleanly in TrainingJobRequest (including Auto M*=0)
    req = TrainingJobRequest(
        dataset_name="default",
        session_label="UI Parity Check",
        loss_type="multi_proxy_anchor",
        backbone_name="vit_base_patch16_224",
        pooling_type="gem",
        pretrained=False,
        freeze_backbone=True,
        unfreeze_last_block=False,
        epochs=20,
        batch_size=32,
        learning_rate=0.002,
        tau_sim=0.0,
        normal_loss_ratio=0.5,
        repel_delta=0.0,
        num_normal_proxies=0,
        hard_anomaly_boost=3.0,
        class_balance_beta=0.9999,
        enable_boundary_aug=True,
        mixup_alpha_min=0.30,
        mixup_alpha_max=0.35,
        nab_normal_ratio=0.50,
        top_k_mode="ratio",
        top_k_initial=0.30,
        top_k_final=0.15,
        gamma_initial=0.6,
        gamma_final=1.5,
        w_defect=0.1,
        w_aux=0.0,
        full_snapshot_interval=10,
    )
    assert req.num_normal_proxies == 0

    # 2. ViT backbone (`224x224` patch_embed) must accept `128x128` inputs without AssertionError
    vit_model = MetricProjectionModel(
        backbone_name="vit_base_patch16_224",
        pooling_type="gem",
        embedding_dim=128,
        pretrained=False,
        freeze_backbone=True,
    )
    vit_model.train()
    x_128 = torch.rand(2, 3, 128, 128)
    z_vit, gap_vit = vit_model(x_128)
    assert z_vit.shape == (2, 128)
    assert torch.all(torch.isfinite(z_vit))
    # Empty (0, D) and single-sample (1, D) forward_from_gap_features in train() mode must never fail
    z_empty = vit_model.forward_from_gap_features(gap_vit[:0])
    assert z_empty.shape == (0, 128)
    z_single = vit_model.forward_from_gap_features(gap_vit[:1])
    assert z_single.shape == (1, 128)

    # 3. Zero-variance identical feature matrices must never produce NaN or crash PCA/KMeans
    const_norm = np.ones((16, 32), dtype=np.float32) * 2.5
    m_const, lbl_const = estimate_optimal_normal_proxies(const_norm, max_proxies=4)
    assert m_const == 1
    assert np.all(lbl_const == 0)

    const_tail = np.ones((8, 32), dtype=np.float32) * -1.2
    tail_red = reduce_tail_pca(const_tail, max_components=16)
    assert np.all(np.isfinite(tail_red))

    # 4. Tiny feature dim (< 8) in synthesize_boundary_gap_features must not overflow torch.topk
    s_tiny, y_tiny = synthesize_boundary_gap_features(
        normal_gap=torch.randn(4, 4),
        defect_gap=torch.randn(2, 4),
        defect_labels=torch.tensor([1, 2], dtype=torch.long),
    )
    assert s_tiny.shape == (4, 4)
    assert y_tiny.shape == (4,)

    # 5. NAB-Sampler with 0 defects or 0 normals must never raise ZeroDivisionError
    s_no_def = NormalAnomalyBalancedSampler(
        labels=[0, 0, 0, 0], is_defect=[False] * 4, batch_size=4
    )
    assert len(s_no_def.build_epoch_batches()) == 1
    s_no_norm = NormalAnomalyBalancedSampler(
        labels=[1, 1, 1, 1], is_defect=[True] * 4, batch_size=4
    )
    assert len(s_no_norm.build_epoch_batches()) == 1

    # 6. TrainingJobRequest model_validator normalizes count/ratio top_k & inverted alpha bounds
    req_norm = TrainingJobRequest(
        top_k_mode="count",
        top_k_initial=0.30,
        top_k_final=0.15,
        mixup_alpha_min=0.80,
        mixup_alpha_max=0.45,
    )
    assert req_norm.top_k_initial == 30.0
    assert req_norm.top_k_final == 15.0
    assert req_norm.mixup_alpha_min == 0.45
    assert req_norm.mixup_alpha_max == 0.80


def test_ml_core_adversarial_edge_cases(tmp_path: Path, monkeypatch) -> None:
    """Verify 16-bit images, [B,1] XBM tensors, list[int] subclusters, collinear proxy escape, 0-defect val, NaN UMAP, and solid-image backward."""
    from PIL import Image
    import app.core.evaluator as eval_mod
    from app.core.evaluator import compute_3d_umap_coordinates
    from app.core.feature_extractor import (
        extract_handcrafted_surface_statistics,
        load_image_tensor,
    )
    from app.core.proxy_anchor_loss import assign_normal_subclusters_by_centroids

    # 1. 16-bit industrial PNG (`I;16`) must preserve gradient contrast instead of clipping >255 to white
    grad_16 = np.linspace(1000, 60000, num=64 * 64, dtype=np.uint16).reshape(64, 64)
    p16 = tmp_path / "industrial_16bit.png"
    Image.fromarray(grad_16).save(p16)
    tf = build_image_transforms(is_train=False, image_size=64)
    t16 = load_image_tensor(p16, tf)
    assert t16.shape == (3, 64, 64)
    assert float(t16.std().item()) > 0.25

    # 2. [B, 1] 2D column-vector tensors in CrossBatchMemoryBank.update must not evaluate bool([False]) -> True
    bank = CrossBatchMemoryBank(embedding_dim=16)
    bank.update(
        sample_indices=torch.tensor([[0], [1], [2]], dtype=torch.long),
        embeddings=torch.randn(3, 16),
        labels=torch.tensor([[0], [0], [1]], dtype=torch.long),
        is_defect_flags=torch.tensor([[False], [False], [True]], dtype=torch.bool),
    )
    assert bank.normal_count == 2
    assert bank.defect_count == 1

    # 3. assign_normal_subclusters_by_centroids accepts Python list[int] without AttributeError
    feats = np.random.default_rng(3).normal(size=(8, 16)).astype(np.float32)
    feats[:4, 0] += 5.0
    feats[4:, 1] += 5.0
    assigned = assign_normal_subclusters_by_centroids(
        features=feats,
        ref_features=feats[:6],
        ref_subclusters=[0, 0, 0, 0, 1, 1],
    )
    assert assigned.shape == (8,)
    assert set(assigned.tolist()) == {0, 1}

    # 4. Collinear proxy perturbation escapes saddle point when num_normal_proxies > populated subclusters
    crit = CustomizedMultiProxyAnchorLoss(embedding_dim=32, num_normal_proxies=4, num_defect_classes=2)
    z_uni = np.zeros((6, 32), dtype=np.float32)
    z_uni[:3, 0] = 1.0
    z_uni[3:, 0] = -1.0
    crit.initialize_proxies_from_embeddings(
        z_uni, labels=[0, 0, 0, 1, 1, 1], normal_subclusters=[0, 0, 0, -1, -1, -1]
    )
    p_n, p_d = crit.get_proxies_numpy()
    gram_n = p_n @ p_n.T
    off_n = gram_n[~np.eye(4, dtype=bool)]
    assert float(np.max(off_n)) < 0.99
    gram_d = p_d @ p_d.T
    assert float(gram_d[0, 1]) < 0.99

    # 5. 0-defect val split must never starve anomaly repulsion in CurriculumPenaltyScheduler
    sched = CurriculumPenaltyScheduler(total_epochs=10, base_normal_ratio=0.50)
    st_no_def = sched.step(
        epoch=2,
        val_recall=0.0,
        d_min_boundary=0.0,
        val_loss_centripetal=0.45,
        val_loss_repel=0.0,
        val_precision=0.0,
        has_val_defects=False,
    )
    assert st_no_def.effective_normal_ratio <= 0.50

    # 6. Spectral UMAP returning silent NaN falls back cleanly to finite 3D PCA
    class _DummyNaNUMAP:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, x):
            return np.full((x.shape[0], 3), np.nan, dtype=np.float32)

    if eval_mod.umap is not None:
        monkeypatch.setattr(eval_mod.umap, "UMAP", _DummyNaNUMAP)
        coords = compute_3d_umap_coordinates(
            np.random.default_rng(5).normal(size=(16, 32)).astype(np.float32)
        )
        assert coords.shape == (16, 3)
        assert np.all(np.isfinite(coords))

    # 7. Solid constant image backward through extract_handcrafted_surface_statistics produces zero NaN gradients
    solid_x = torch.ones((2, 3, 32, 32), requires_grad=True)
    surf = extract_handcrafted_surface_statistics(solid_x)
    surf.sum().backward()
    assert solid_x.grad is not None
    assert torch.all(torch.isfinite(solid_x.grad))





