"""HTTP client for communicating exclusively with `api-backend`."""

from __future__ import annotations

import os
from typing import Any

import httpx


class BackendApiClient:
    """Thin REST client wrapping all calls from Streamlit `frontend` to `api-backend`."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("BACKEND_API_URL", "http://localhost:18000")).rstrip(
            "/"
        )

    def get_health(self) -> dict[str, Any]:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    def get_datasets(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/datasets")
            resp.raise_for_status()
            return resp.json()

    def generate_synthetic_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/datasets/generate-synthetic", json=payload
            )
            resp.raise_for_status()
            return resp.json()

    def upload_folder_images(
        self,
        items: list[dict[str, str]],
        clear_existing: bool = True,
        batch_size: int = 25,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Upload folder images with their subfolder class names in batches to `api-backend`."""
        last_result: dict[str, Any] = {}
        total = len(items)
        if total == 0:
            return {"uploaded_in_batch": 0, "total_images": 0, "subfolder_classes": {}}

        with httpx.Client(timeout=60.0) as client:
            for start_idx in range(0, total, batch_size):
                chunk = items[start_idx : start_idx + batch_size]
                payload = {
                    "items": chunk,
                    "clear_existing": bool(clear_existing and start_idx == 0),
                }
                resp = client.post(f"{self.base_url}/api/v1/datasets/upload-folder", json=payload)
                resp.raise_for_status()
                last_result = resp.json()
                if progress_callback is not None:
                    progress_callback(min(start_idx + len(chunk), total), total)

        return last_result

    def save_annotations_from_subfolders(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Directly confirm training annotations (`annotations.json`) from subfolder names."""
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/annotations/from-subfolders", json=payload
            )
            resp.raise_for_status()
            return resp.json()

    def trigger_clustering(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/api/v1/jobs/clustering", json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_clusters(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/clusters")
            resp.raise_for_status()
            return resp.json()

    def get_annotations(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/annotations")
            resp.raise_for_status()
            return resp.json()

    def save_annotations(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/api/v1/annotations", json=payload)
            resp.raise_for_status()
            return resp.json()

    def trigger_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/api/v1/jobs/training", json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/jobs/{job_id}")
            resp.raise_for_status()
            return resp.json()

    def list_evaluations(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/evaluations")
            resp.raise_for_status()
            return resp.json()

    def get_epoch_evaluation(self, epoch: int, tau_sim: float = 0.0) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self.base_url}/api/v1/evaluations/{epoch}",
                params={"tau_sim": tau_sim},
            )
            resp.raise_for_status()
            return resp.json()

    def run_single_inference(self, image_path: str, tau_sim: float = 0.0) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/inference",
                json={"image_path": image_path, "tau_sim": tau_sim},
            )
            resp.raise_for_status()
            return resp.json()
