"""Configuration for the FastAPI Backend Gateway service."""

from __future__ import annotations

import os
from pathlib import Path


class BackendConfig:
    """Environment-driven settings for `api-backend`."""

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/app/data")).resolve()
    TRAINER_API_URL: str = os.getenv("TRAINER_API_URL", "http://trainer-service:18001").rstrip("/")

    @classmethod
    def raw_dir(cls) -> Path:
        path = cls.DATA_DIR / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def manifests_dir(cls) -> Path:
        path = cls.DATA_DIR / "manifests"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def artifacts_dir(cls) -> Path:
        path = cls.DATA_DIR / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path
