"""Unit and integration tests for `api-backend` (annotation stratification and dynamic threshold `.pkl` evaluation)."""

from __future__ import annotations

import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import BackendConfig
from app.main import app


def _make_png_base64(color: tuple[int, int, int] = (128, 128, 128)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_backend_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "api-backend"


def test_folder_upload_and_subfolder_class_annotations(tmp_path: Path, monkeypatch) -> None:
    """Verify folder upload stores images under `/app/data/raw/<subfolder>/` and maps subfolder names to training classes."""
    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    client = TestClient(app)

    b64_img = _make_png_base64()
    upload_items = [
        {"file_name": f"ok_{i}.png", "class_name": "good", "content_base64": b64_img}
        for i in range(6)
    ] + [
        {"file_name": f"ng_scr_{i}.png", "class_name": "scratch", "content_base64": b64_img}
        for i in range(3)
    ] + [
        {"file_name": f"ng_stn_{i}.png", "class_name": "stain", "content_base64": b64_img}
        for i in range(2)
    ]

    up_resp = client.post(
        "/api/v1/datasets/upload-folder",
        json={"items": upload_items, "clear_existing": True},
    )
    assert up_resp.status_code == 200
    up_data = up_resp.json()
    assert up_data["total_images"] == 11
    assert up_data["subfolder_classes"] == {"good": 6, "scratch": 3, "stain": 2}
    assert up_data["detected_class_names"] == ["good", "scratch", "stain"]

    # Confirm annotations directly from subfolder names with "good" as normal class (label_id=0)
    ann_resp = client.post(
        "/api/v1/annotations/from-subfolders",
        json={"normal_class_name": "good", "val_ratio": 0.25, "seed": 42},
    )
    assert ann_resp.status_code == 200
    ann_data = ann_resp.json()
    assert ann_data["normal_class_name"] == "good"
    assert ann_data["class_map"] == {"good": 0, "scratch": 1, "stain": 2}
    assert ann_data["class_counts"] == {"good": 6, "scratch": 3, "stain": 2}
    for item in ann_data["items"]:
        if item["label_name"] == "good":
            assert item["label_id"] == 0
            assert item["is_defect"] is False
        else:
            assert item["label_id"] in (1, 2)
            assert item["is_defect"] is True


def test_annotations_stratified_split_and_dynamic_threshold_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify Phase 1 annotation confirmation and Step 6 dynamic `tau_sim` evaluation without PyTorch."""
    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    manifests_dir = BackendConfig.manifests_dir()
    artifacts_dir = BackendConfig.artifacts_dir()

    # 1. Seed a mock clustering_result.json
    items = []
    for i in range(10):
        items.append(
            {
                "image_path": f"/app/data/raw/norm_{i}.png",
                "file_name": f"norm_{i}.png",
                "cluster_id": 0,
                "is_normal_core": True,
                "density_score": 0.92,
                "preview_2d": [0.1, 0.2],
                "ground_truth_hint": "normal",
            }
        )
    for i in range(4):
        items.append(
            {
                "image_path": f"/app/data/raw/scratch_{i}.png",
                "file_name": f"scratch_{i}.png",
                "cluster_id": 1,
                "is_normal_core": False,
                "density_score": 0.35,
                "preview_2d": [-0.8, 0.5],
                "ground_truth_hint": "defect_scratch",
            }
        )

    clustering_payload = {
        "backbone_name": "efficientnet_b3",
        "total_samples": 14,
        "normal_core_count": 10,
        "tail_count": 4,
        "core_quantile": 0.85,
        "best_tail_k": 1,
        "total_clusters": 2,
        "silhouette_scores": {"2": 0.75},
        "items": items,
    }
    (manifests_dir / "clustering_result.json").write_text(
        json.dumps(clustering_payload), encoding="utf-8"
    )

    client = TestClient(app)

    # 2. Confirm cluster annotations via POST /api/v1/annotations
    ann_resp = client.post(
        "/api/v1/annotations",
        json={
            "cluster_labels": {"0": "normal", "1": "defect_scratch"},
            "sample_overrides": {"norm_9.png": "defect_stain"},
            "val_ratio": 0.25,
            "seed": 42,
        },
    )
    assert ann_resp.status_code == 200
    ann_data = ann_resp.json()
    assert ann_data["class_map"]["normal"] == 0
    assert "defect_scratch" in ann_data["class_map"]
    assert "defect_stain" in ann_data["class_map"]
    assert ann_data["class_counts"]["defect_stain"] == 1

    # 3. Seed a Pure NumPy epoch_0001_light.pkl snapshot
    n = 14
    sims = np.array([0.75] * 9 + [-0.15] + [-0.45] * 4, dtype=np.float32)
    is_defect = np.array([False] * 9 + [True] * 5, dtype=bool)
    snap_payload = {
        "epoch": 1,
        "is_full_snapshot": False,
        "image_paths": [it["image_path"] for it in items],
        "labels": np.array([0] * 9 + [2] + [1] * 4, dtype=np.int64),
        "label_names": ["normal"] * 9 + ["defect_stain"] + ["defect_scratch"] * 4,
        "is_defect": is_defect,
        "splits": ["train"] * 7 + ["val"] * 2 + ["train"] + ["train", "train", "val", "val"],
        "embeddings_128d": np.zeros((n, 128), dtype=np.float32),
        "normal_prototype": np.zeros((128,), dtype=np.float32),
        "cosine_similarities": sims,
        "umap_3d": np.zeros((n, 3), dtype=np.float32),
        "d_min_centroid": 1.15,
        "d_min_boundary": 0.92,
        "train_loss": 0.23,
        "curriculum_progress": 0.65,
        "effective_top_k": 9,
        "metrics_at_zero": {"tau_sim": 0.0, "recall": 1.0, "precision": 1.0, "f1": 1.0},
        "gap_features": None,
    }
    with open(artifacts_dir / "epoch_0001_light.pkl", "wb") as f:
        pickle.dump(snap_payload, f)

    # 4. Query evaluation at default tau_sim = 0.0 (all defects have sim < 0.0 -> Recall = 1.0)
    eval_zero = client.get("/api/v1/evaluations/1?tau_sim=0.0")
    assert eval_zero.status_code == 200
    data_zero = eval_zero.json()
    assert data_zero["val_metrics"]["recall"] == 1.0
    assert data_zero["val_metrics"]["f1"] == 1.0

    # 5. Query evaluation at user-adjusted tau_sim = -0.30 (defect_stain at -0.15 is > -0.30)
    eval_shifted = client.get("/api/v1/evaluations/1?tau_sim=-0.30")
    assert eval_shifted.status_code == 200
    data_shifted = eval_shifted.json()
    assert data_shifted["all_metrics"]["fn"] == 1
    assert data_shifted["all_metrics"]["recall"] == 0.8
