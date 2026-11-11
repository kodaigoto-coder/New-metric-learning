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

    def get_datasets_catalog(self) -> dict[str, Any]:
        """List all saved datasets and the currently active dataset."""
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/datasets/catalog")
            resp.raise_for_status()
            return resp.json()

    def select_or_create_dataset(
        self,
        dataset_name: str,
        clone_from: str | None = None,
    ) -> dict[str, Any]:
        """Switch active dataset or create/clone a new named dataset."""
        payload: dict[str, Any] = {"dataset_name": dataset_name}
        if clone_from:
            payload["clone_from"] = clone_from
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/api/v1/datasets/select", json=payload)
            resp.raise_for_status()
            return resp.json()

    def delete_dataset(self, dataset_name: str) -> dict[str, Any]:
        """Delete a named dataset or clear `'default'` dataset."""
        with httpx.Client(timeout=20.0) as client:
            resp = client.delete(f"{self.base_url}/api/v1/datasets/{dataset_name}")
            resp.raise_for_status()
            return resp.json()

    def get_datasets(self, dataset_name: str | None = None) -> dict[str, Any]:
        params = {"dataset_name": dataset_name} if dataset_name else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/datasets", params=params)
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
        clear_existing: bool = False,
        dataset_name: str | None = None,
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
                payload: dict[str, Any] = {
                    "items": chunk,
                    "clear_existing": bool(clear_existing and start_idx == 0),
                }
                if dataset_name:
                    payload["dataset_name"] = dataset_name
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

    def get_clusters(self, dataset_name: str | None = None) -> dict[str, Any]:
        params = {"dataset_name": dataset_name} if dataset_name else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/clusters", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_annotations(self, dataset_name: str | None = None) -> dict[str, Any]:
        params = {"dataset_name": dataset_name} if dataset_name else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/annotations", params=params)
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

    def list_sessions(self, dataset_name: str | None = None) -> dict[str, Any]:
        params = {"dataset_name": dataset_name} if dataset_name else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/sessions", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_session_detail(self, session_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()

    def list_evaluations(self, session_id: str | None = None) -> dict[str, Any]:
        params = {"session_id": session_id} if session_id else {}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/v1/evaluations", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_epoch_evaluation(
        self,
        epoch: int,
        tau_sim: float = 0.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"tau_sim": tau_sim}
        if session_id:
            params["session_id"] = session_id
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self.base_url}/api/v1/evaluations/{epoch}",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    def run_single_inference(
        self,
        image_path: str,
        tau_sim: float = 0.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"image_path": image_path, "tau_sim": tau_sim}
        if session_id:
            payload["session_id"] = session_id
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/inference",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
