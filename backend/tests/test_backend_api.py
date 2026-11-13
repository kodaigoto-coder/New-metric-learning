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


def test_sequential_folder_uploads_preserve_existing_subfolders(tmp_path: Path, monkeypatch) -> None:
    """Verify uploading `anomaly` first and `nomal` second with default `clear_existing=False` merges both."""
    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    client = TestClient(app)
    b64_img = _make_png_base64()

    # Step 1: Upload `anomaly` folder
    resp1 = client.post(
        "/api/v1/datasets/upload-folder",
        json={
            "items": [
                {"file_name": f"ng_{i}.png", "class_name": "anomaly", "content_base64": b64_img}
                for i in range(4)
            ],
            "clear_existing": False,
        },
    )
    assert resp1.status_code == 200
    assert resp1.json()["subfolder_classes"] == {"anomaly": 4}

    # Step 2: Upload `nomal` folder with default clear_existing=False -> must NOT delete `anomaly`!
    resp2 = client.post(
        "/api/v1/datasets/upload-folder",
        json={
            "items": [
                {"file_name": f"ok_{i}.png", "class_name": "nomal", "content_base64": b64_img}
                for i in range(6)
            ],
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["total_images"] == 10
    assert data2["subfolder_classes"] == {"anomaly": 4, "nomal": 6}


def test_multi_dataset_catalog_switching_and_session_recall(tmp_path: Path, monkeypatch) -> None:
    """Verify multi-dataset isolation, cloning, switching, session recall, cm_category, and dispersion metrics."""
    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    client = TestClient(app)
    b64_img = _make_png_base64()

    # 1. Upload 5 images to 'default' dataset
    client.post(
        "/api/v1/datasets/upload-folder",
        json={
            "items": [
                {"file_name": f"def_{i}.png", "class_name": "nomal", "content_base64": b64_img}
                for i in range(5)
            ],
            "clear_existing": True,
        },
    )

    # 2. Create and switch to a new dataset 'bottle_lotB'
    sel_resp = client.post(
        "/api/v1/datasets/select",
        json={"dataset_name": "bottle_lotB"},
    )
    assert sel_resp.status_code == 200
    assert sel_resp.json()["active_dataset"] == "bottle_lotB"
    assert sel_resp.json()["dataset"]["total_images"] == 0

    # Upload 3 images to 'bottle_lotB'
    client.post(
        "/api/v1/datasets/upload-folder",
        json={
            "items": [
                {"file_name": f"lotb_{i}.png", "class_name": "anomaly", "content_base64": b64_img}
                for i in range(3)
            ],
        },
    )

    # Verify catalog lists both 'default' (5 images) and 'bottle_lotB' (3 images)
    cat_resp = client.get("/api/v1/datasets/catalog")
    assert cat_resp.status_code == 200
    cat_data = cat_resp.json()
    assert cat_data["active_dataset"] == "bottle_lotB"
    by_name = {d["dataset_name"]: d for d in cat_data["datasets"]}
    assert by_name["default"]["total_images"] == 5
    assert by_name["bottle_lotB"]["total_images"] == 3

    # 3. Clone 'default' into 'bottle_clone'
    clone_resp = client.post(
        "/api/v1/datasets/select",
        json={"dataset_name": "bottle_clone", "clone_from": "default"},
    )
    assert clone_resp.status_code == 200
    assert clone_resp.json()["dataset"]["total_images"] == 5

    # 4. Seed a session in sessions/session_0001 and verify session recall + dispersion + cm_category
    s1_dir = BackendConfig.session_dir("session_0001")
    meta = {
        "session_id": "session_0001",
        "session_number": 1,
        "session_label": "Baseline M=4",
        "dataset_name": "default",
        "status": "completed",
        "loss_type": "multi_proxy_anchor",
        "backbone_name": "efficientnet_b3",
        "epochs_completed": 1,
        "total_epochs": 1,
        "best_epoch": 1,
        "best_val_recall": 1.0,
        "best_val_f1": 1.0,
        "history": [{"epoch": 1, "train_loss": 0.15, "val_recall": 1.0, "val_f1": 1.0}],
    }
    (s1_dir / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    snap = {
        "epoch": 1,
        "is_full_snapshot": False,
        "image_paths": [f"/app/data/raw/img_{i}.png" for i in range(6)],
        "labels": np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        "label_names": ["nomal"] * 4 + ["anomaly"] * 2,
        "is_defect": np.array([False, False, False, False, True, True], dtype=bool),
        "splits": ["train", "train", "val", "val", "train", "val"],
        "embeddings_128d": np.eye(6, 128, dtype=np.float32),
        "normal_prototype": np.zeros((128,), dtype=np.float32),
        "cosine_similarities": np.array([0.8, 0.7, 0.6, -0.1, -0.5, 0.2], dtype=np.float32),
        "umap_3d": np.zeros((6, 3), dtype=np.float32),
        "d_min_centroid": 0.8,
        "d_min_boundary": 0.7,
        "train_loss": 0.15,
        "curriculum_progress": 0.5,
        "effective_top_k": 4,
        "metrics_at_zero": {"tau_sim": 0.0, "recall": 0.5, "precision": 0.5, "f1": 0.5},
        "gap_features": None,
    }
    with open(s1_dir / "epoch_0001_light.pkl", "wb") as f:
        pickle.dump(snap, f)

    sess_list = client.get("/api/v1/sessions")
    assert sess_list.status_code == 200
    assert sess_list.json()["total_sessions"] == 1
    assert sess_list.json()["sessions"][0]["session_id"] == "session_0001"

    sess_detail = client.get("/api/v1/sessions/session_0001")
    assert sess_detail.status_code == 200
    assert sess_detail.json()["session_label"] == "Baseline M=4"

    eval_ep = client.get("/api/v1/evaluations/1?tau_sim=0.0&session_id=session_0001")
    assert eval_ep.status_code == 200
    ep_data = eval_ep.json()
    assert ep_data["session_id"] == "session_0001"
    assert "normal_intra_disp" in ep_data["dispersion_metrics"]
    assert "fisher_separation_ratio" in ep_data["dispersion_metrics"]
    cm_cats = [s["cm_category"] for s in ep_data["samples"]]
    # sims: [0.8(TN), 0.7(TN), 0.6(TN), -0.1(FP), -0.5(TP), 0.2(FN)] at tau_sim=0.0
    assert cm_cats == ["TN", "TN", "TN", "FP", "TP", "FN"]


def test_normal_subcluster_stratified_split_and_auto_proxy_schema(tmp_path: Path, monkeypatch) -> None:
    """Verify 6:2:2 split stratifies minority normal_subcluster_id across train/val/test and TrainingTriggerRequest defaults to M=0 (Auto)."""
    from app.schemas.api_models import TrainingTriggerRequest

    req_default = TrainingTriggerRequest()
    assert req_default.num_normal_proxies == 0

    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    client = TestClient(app)

    # Seed clustering_result.json with 20 normal mode 0, 5 minority normal mode 1, and 5 scratch defects
    cluster_items = []
    for i in range(20):
        cluster_items.append(
            {
                "sample_id": f"n0_{i}",
                "file_name": f"n0_{i}.png",
                "image_path": f"/app/data/raw/n0_{i}.png",
                "cluster_id": 0,
                "normal_subcluster_id": 0,
            }
        )
    for i in range(5):
        cluster_items.append(
            {
                "sample_id": f"n1_{i}",
                "file_name": f"n1_{i}.png",
                "image_path": f"/app/data/raw/n1_{i}.png",
                "cluster_id": 0,
                "normal_subcluster_id": 1,
            }
        )
    for i in range(5):
        cluster_items.append(
            {
                "sample_id": f"def_{i}",
                "file_name": f"def_{i}.png",
                "image_path": f"/app/data/raw/def_{i}.png",
                "cluster_id": 1,
                "normal_subcluster_id": -1,
            }
        )

    man_dir = BackendConfig.manifests_dir("default")
    (man_dir / "clustering_result.json").write_text(
        json.dumps(
            {
                "dataset_name": "default",
                "total_samples": len(cluster_items),
                "detected_normal_subclusters": 2,
                "items": cluster_items,
            }
        ),
        encoding="utf-8",
    )

    ann_resp = client.post(
        "/api/v1/annotations",
        json={
            "cluster_labels": {"0": "normal", "1": "scratch"},
            "normal_class_name": "normal",
        },
    )
    assert ann_resp.status_code == 200
    ann_json = json.loads((man_dir / "annotations.json").read_text(encoding="utf-8"))
    minority_splits = [
        it["split"]
        for it in ann_json["items"]
        if not it["is_defect"] and int(it.get("normal_subcluster_id", 0)) == 1
    ]
    assert len(minority_splits) == 5
    # Minority normal subcluster (5 items) must appear in train (3), val (1), and test (1)!
    assert minority_splits.count("train") == 3
    assert minority_splits.count("val") == 1
    assert minority_splits.count("test") == 1


def test_training_trigger_exact_ui_payload_and_proxy_error_forwarding(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify exact 28-field Streamlit UI payload passes `POST /api/v1/jobs/training` and error details propagate cleanly."""
    import sys
    import httpx
    from app.routers import jobs_proxy

    monkeypatch.setattr(BackendConfig, "DATA_DIR", tmp_path)
    client = TestClient(app)

    captured_payloads: list[dict] = []

    class _DummyTrainerClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, json: dict):
            captured_payloads.append(json)
            req_obj = httpx.Request("POST", url)
            if json.get("session_label") == "force-422":
                return httpx.Response(
                    422,
                    json={"detail": [{"loc": ["body", "epochs"], "msg": "mock validation error"}]},
                    request=req_obj,
                )
            return httpx.Response(
                200,
                json={
                    "job_id": "train-12345678",
                    "job_type": "training",
                    "status": "queued",
                    "dataset_name": json["dataset_name"],
                    "session_id": "session_0001",
                    "session_number": 1,
                },
                request=req_obj,
            )

    monkeypatch.setattr(jobs_proxy.httpx, "Client", _DummyTrainerClient)

    ui_payload = {
        "dataset_name": "default",
        "session_label": "Auto-M UI Run",
        "loss_type": "multi_proxy_anchor",
        "backbone_name": "efficientnet_b3",
        "pooling_type": "gem",
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_last_block": False,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.002,
        "tau_sim": 0.0,
        "normal_loss_ratio": 0.5,
        "repel_delta": 0.0,
        "num_normal_proxies": 0,
        "hard_anomaly_boost": 3.0,
        "class_balance_beta": 0.99,
        "enable_boundary_aug": True,
        "mixup_alpha_min": 0.55,
        "mixup_alpha_max": 0.75,
        "nab_normal_ratio": 0.5,
        "top_k_mode": "ratio",
        "top_k_initial": 0.3,
        "top_k_final": 0.15,
        "gamma_initial": 0.6,
        "gamma_final": 1.5,
        "w_defect": 0.1,
        "w_aux": 0.0,
        "full_snapshot_interval": 10,
    }
    resp = client.post("/api/v1/jobs/training", json=ui_payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == "train-12345678"
    assert captured_payloads[-1]["num_normal_proxies"] == 0

    # Downstream 422 must propagate as 422 with structured detail (not swallowed as 502)
    err_resp = client.post(
        "/api/v1/jobs/training",
        json={**ui_payload, "session_label": "force-422"},
    )
    assert err_resp.status_code == 422
    assert err_resp.json()["detail"][0]["msg"] == "mock validation error"

    # Also verify frontend BackendApiClient backward-compatibility retry if an old server rejects M=0
    fe_root = str(Path(__file__).resolve().parents[2] / "frontend" / "app")
    if fe_root not in sys.path:
        sys.path.insert(0, fe_root)
    import api_client as fe_api_client

    sent_m_values: list[int] = []

    class _StaleThenSuccessClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, json: dict):
            m_val = int(json["num_normal_proxies"])
            sent_m_values.append(m_val)
            req_obj = httpx.Request("POST", url)
            if m_val == 0:
                return httpx.Response(
                    422,
                    json={
                        "detail": [
                            {
                                "type": "greater_than_equal",
                                "loc": ["body", "num_normal_proxies"],
                                "msg": "Input should be greater than or equal to 1",
                            }
                        ]
                    },
                    request=req_obj,
                )
            return httpx.Response(200, json={"job_id": "train-compat", "status": "queued"}, request=req_obj)

    monkeypatch.setattr(fe_api_client.httpx, "Client", _StaleThenSuccessClient)
    fe_client = fe_api_client.BackendApiClient(base_url="http://api-backend:18000")
    compat_res = fe_client.trigger_training(ui_payload)
    assert compat_res["job_id"] == "train-compat"
    assert sent_m_values == [0, 1]

    # Verify TrainingTriggerRequest model_validator normalizes count/ratio top_k & inverted alpha bounds
    from app.schemas.api_models import TrainingTriggerRequest

    req_count_from_ratio = TrainingTriggerRequest(
        top_k_mode="count",
        top_k_initial=0.30,
        top_k_final=0.15,
        mixup_alpha_min=0.80,
        mixup_alpha_max=0.45,
    )
    assert req_count_from_ratio.top_k_initial == 30.0
    assert req_count_from_ratio.top_k_final == 15.0
    assert req_count_from_ratio.mixup_alpha_min == 0.45
    assert req_count_from_ratio.mixup_alpha_max == 0.80

    # Verify empty running session_0002 does NOT leak root artifacts from session_0001
    s2_dir = BackendConfig.session_dir("session_0002")
    (s2_dir / "session_meta.json").write_text(
        json.dumps({"session_id": "session_0002", "session_number": 2, "status": "running"}),
        encoding="utf-8",
    )
    with open(BackendConfig.artifacts_dir() / "epoch_0001_light.pkl", "wb") as f:
        pickle.dump({"epoch": 1, "cosine_similarities": [], "is_defect": [], "splits": []}, f)
    ev_s2 = client.get("/api/v1/evaluations?session_id=session_0002")
    assert ev_s2.status_code == 200
    assert ev_s2.json()["session_id"] == "session_0002"
    assert ev_s2.json()["total_epochs"] == 0




