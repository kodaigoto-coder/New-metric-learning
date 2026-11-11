"""Configuration for the FastAPI Backend Gateway service."""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sanitize_identifier(name: str, fallback: str = "default") -> str:
    """Sanitize a dataset name or session identifier to prevent path traversal."""
    cleaned = re.sub(
        r"[^a-zA-Z0-9_\-\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",
        "_",
        name.strip(),
    )
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


class BackendConfig:
    """Environment-driven settings for `api-backend`."""

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/app/data")).resolve()
    TRAINER_API_URL: str = os.getenv("TRAINER_API_URL", "http://trainer-service:18001").rstrip("/")

    @classmethod
    def active_dataset_file(cls) -> Path:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DATA_DIR / "active_dataset.json"

    @classmethod
    def datasets_root(cls) -> Path:
        path = cls.DATA_DIR / "datasets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_active_dataset(cls) -> str:
        """Return the currently active dataset name (defaults to 'default')."""
        state_file = cls.active_dataset_file()
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                return sanitize_identifier(str(data.get("active_dataset", "default")), fallback="default")
            except Exception:
                return "default"
        return "default"

    @classmethod
    def set_active_dataset(cls, dataset_name: str) -> str:
        """Persist the active dataset name and ensure its directories exist."""
        safe_name = sanitize_identifier(dataset_name, fallback="default")
        state_file = cls.active_dataset_file()
        tmp_file = state_file.with_suffix(".json.tmp")
        tmp_file.write_text(
            json.dumps(
                {
                    "active_dataset": safe_name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_file, state_file)
        cls.raw_dir(safe_name)
        cls.manifests_dir(safe_name)
        return safe_name

    @classmethod
    def raw_dir(cls, dataset_name: str | None = None) -> Path:
        """Return `raw` directory for `dataset_name` (or active dataset)."""
        ds = sanitize_identifier(dataset_name, fallback="") if dataset_name else cls.get_active_dataset()
        if not ds or ds == "default":
            path = cls.DATA_DIR / "raw"
        else:
            path = cls.datasets_root() / ds / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def manifests_dir(cls, dataset_name: str | None = None) -> Path:
        """Return `manifests` directory for `dataset_name` (or active dataset)."""
        ds = sanitize_identifier(dataset_name, fallback="") if dataset_name else cls.get_active_dataset()
        if not ds or ds == "default":
            path = cls.DATA_DIR / "manifests"
        else:
            path = cls.datasets_root() / ds / "manifests"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def artifacts_dir(cls) -> Path:
        path = cls.DATA_DIR / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def sessions_dir(cls) -> Path:
        path = cls.artifacts_dir() / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def session_dir(cls, session_id: str) -> Path:
        safe_id = sanitize_identifier(session_id, fallback="session_0001")
        path = cls.sessions_dir() / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def ensure_legacy_session_migrated(cls) -> None:
        """Auto-migrate existing root `artifacts/epoch_*.pkl` into `sessions/session_0001` if no sessions exist."""
        sess_root = cls.sessions_dir()
        existing_sessions = [
            d for d in sess_root.iterdir() if d.is_dir() and re.match(r"^session_\d+$", d.name)
        ]
        if existing_sessions:
            return

        art_dir = cls.artifacts_dir()
        light_files = sorted(art_dir.glob("epoch_*_light.pkl"))
        if not light_files:
            return

        s1_dir = sess_root / "session_0001"
        s1_dir.mkdir(parents=True, exist_ok=True)

        for pkl_file in art_dir.glob("epoch_*.pkl"):
            dest = s1_dir / pkl_file.name
            if not dest.exists():
                shutil.copy2(pkl_file, dest)

        best_pt = art_dir / "best_model.pt"
        if best_pt.exists() and not (s1_dir / "best_model.pt").exists():
            shutil.copy2(best_pt, s1_dir / "best_model.pt")

        jobs_dir = art_dir / "jobs"
        latest_train_job: dict[str, Any] = {}
        if jobs_dir.exists():
            train_js = sorted(jobs_dir.glob("train-*.json"), key=lambda p: p.stat().st_mtime)
            if train_js:
                try:
                    latest_train_job = json.loads(train_js[-1].read_text(encoding="utf-8"))
                except Exception:
                    latest_train_job = {}

        history: list[dict[str, Any]] = latest_train_job.get("history", [])
        if not history:
            for lf in light_files:
                try:
                    with open(lf, "rb") as f:
                        snap = pickle.load(f)
                    m0 = snap.get("metrics_at_zero", {})
                    history.append(
                        {
                            "epoch": int(snap.get("epoch", 1)),
                            "loss_type": "multi_proxy_anchor",
                            "train_loss": float(snap.get("train_loss", 0.0)),
                            "val_recall": float(m0.get("recall", 0.0)),
                            "val_precision": float(m0.get("precision", 0.0)),
                            "val_f1": float(m0.get("f1", 0.0)),
                            "d_min_boundary": float(snap.get("d_min_boundary", 0.0)),
                            "d_min_centroid": float(snap.get("d_min_centroid", 0.0)),
                            "curriculum_progress": float(snap.get("curriculum_progress", 0.0)),
                            "gamma_repel": 1.0,
                            "repel_margin": 0.2,
                            "configured_top_k": int(snap.get("effective_top_k", 0)),
                            "effective_top_k": int(snap.get("effective_top_k", 0)),
                            "num_normal_proxies": 4,
                            "xbm_released_entries": 0,
                            "xbm_remaining_entries": 0,
                        }
                    )
                except Exception:
                    continue

        best_ep_rec = max(
            history,
            key=lambda r: float(r.get("val_recall", 0.0)) + 0.5 * float(r.get("val_f1", 0.0)),
            default={},
        )
        meta: dict[str, Any] = {
            "session_id": "session_0001",
            "session_number": 1,
            "job_id": latest_train_job.get("job_id", "train-legacy"),
            "dataset_name": "default",
            "created_at": datetime.fromtimestamp(
                light_files[0].stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "completed_at": datetime.fromtimestamp(
                light_files[-1].stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "status": latest_train_job.get("status", "completed"),
            "loss_type": best_ep_rec.get("loss_type", "multi_proxy_anchor"),
            "backbone_name": "efficientnet_b3",
            "pooling_type": "gem",
            "epochs_completed": len(history),
            "total_epochs": int(latest_train_job.get("total_epochs", len(history))),
            "best_epoch": int(best_ep_rec.get("epoch", len(history) or 1)),
            "best_val_recall": float(best_ep_rec.get("val_recall", 0.0)),
            "best_val_f1": float(best_ep_rec.get("val_f1", 0.0)),
            "best_val_precision": float(best_ep_rec.get("val_precision", 0.0)),
            "best_d_min_boundary": float(best_ep_rec.get("d_min_boundary", 0.0)),
            "best_d_min_centroid": float(best_ep_rec.get("d_min_centroid", 0.0)),
            "config": latest_train_job.get("config", {}),
            "history": history,
        }
        meta_path = s1_dir / "session_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
