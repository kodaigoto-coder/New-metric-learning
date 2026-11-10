"""Configuration settings for the Trainer ML worker service."""

from __future__ import annotations

import os
from pathlib import Path


class TrainerConfig:
    """Centralized configuration loaded from environment variables."""

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/app/data")).resolve()
    DEFAULT_BACKBONE: str = os.getenv("DEFAULT_BACKBONE", "efficientnet_b3")
    DEFAULT_POOLING: str = os.getenv("DEFAULT_POOLING", "gem")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "128"))
    DEVICE: str = os.getenv("DEVICE", "cpu")

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
