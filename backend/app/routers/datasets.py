"""Dataset listing, folder image upload, synthetic dataset proxy, clustering result retrieval, and annotation persistence."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from app.config import BackendConfig
from app.schemas.api_models import (
    AnnotationSubmissionRequest,
    DirectSubfolderAnnotationRequest,
    FolderUploadRequest,
    GenerateSyntheticRequest,
)

router = APIRouter(prefix="/api/v1", tags=["datasets-annotations"])

_VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _sanitize_name(name: str, fallback: str = "unclassified") -> str:
    """Sanitize a subfolder class name or file name component to prevent path traversal."""
    cleaned = re.sub(r"[\\/]+", "_", name.strip())
    cleaned = cleaned.replace("..", "_").strip("._ ")
    return cleaned or fallback


def _infer_class_from_path(img_path: Path, raw_dir: Path) -> str | None:
    """Extract the subfolder name under `raw_dir` as the class name, or fallback to filename hints."""
    try:
        rel = img_path.relative_to(raw_dir)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass

    parent_name = img_path.parent.name
    if parent_name and parent_name != raw_dir.name:
        return parent_name

    fname = img_path.name
    if "defect_scratch" in fname:
        return "defect_scratch"
    if "defect_stain" in fname:
        return "defect_stain"
    if "defect_dent" in fname:
        return "defect_dent"
    if "normal" in fname:
        return "normal"
    return None


def _scan_raw_images(raw_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """Recursively scan `raw_dir` for valid images and count samples per subfolder class."""
    files: list[Path] = []
    subfolder_counts: dict[str, int] = {}

    if not raw_dir.exists():
        return files, subfolder_counts

    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() not in _VALID_IMAGE_EXTS:
            continue
        files.append(p)
        cls_name = _infer_class_from_path(p, raw_dir)
        if cls_name:
            subfolder_counts[cls_name] = subfolder_counts.get(cls_name, 0) + 1

    return files, subfolder_counts


def _clear_raw_dir_and_manifests(raw_dir: Path, manifests_dir: Path) -> None:
    """Remove existing images/subfolders in `raw_dir` and reset stale manifests."""
    if raw_dir.exists():
        for child in raw_dir.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                child.unlink(missing_ok=True)

    for m_name in ("clustering_result.json", "annotations.json"):
        m_path = manifests_dir / m_name
        if m_path.exists():
            m_path.unlink(missing_ok=True)


def _build_and_save_annotations_manifest(
    raw_items: list[dict[str, Any]],
    resolved_labels: list[str],
    normal_class_name: str,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Build deterministic `class_map` (normal_class_name -> 0, defects -> 1..C) and stratified splits."""
    norm_cls = normal_class_name.strip() or "normal"
    # If norm_cls is not present in resolved_labels but "normal" is, fallback to "normal"
    unique_all = sorted(set(resolved_labels))
    if norm_cls not in unique_all and "normal" in unique_all:
        norm_cls = "normal"

    unique_defect_names = sorted({lbl for lbl in resolved_labels if lbl != norm_cls})
    class_map: dict[str, int] = {norm_cls: 0}
    for idx, d_name in enumerate(unique_defect_names, start=1):
        class_map[d_name] = idx

    # Stratified Train/Val split with few-shot fallback (guarantees >=1 train and >=1 val when count >= 2)
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for idx, lbl in enumerate(resolved_labels):
        by_class.setdefault(lbl, []).append(idx)

    splits = ["train"] * len(raw_items)
    for _lbl, indices in by_class.items():
        arr = list(indices)
        rng.shuffle(arr)
        n_cls = len(arr)
        if n_cls >= 2:
            n_val = max(1, min(n_cls - 1, int(round(n_cls * val_ratio))))
            for val_idx in arr[:n_val]:
                splits[val_idx] = "val"

    annotated_items: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for i, item in enumerate(raw_items):
        lbl_name = resolved_labels[i]
        lbl_id = class_map.get(lbl_name, 0)
        is_def = lbl_id > 0
        class_counts[lbl_name] = class_counts.get(lbl_name, 0) + 1
        annotated_items.append(
            {
                "image_path": str(item["image_path"]),
                "file_name": str(item["file_name"]),
                "cluster_id": int(item.get("cluster_id", 0 if not is_def else lbl_id)),
                "label_name": lbl_name,
                "label_id": int(lbl_id),
                "is_defect": bool(is_def),
                "split": splits[i],
            }
        )

    manifest_payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(annotated_items),
        "normal_class_name": norm_cls,
        "class_map": class_map,
        "class_counts": class_counts,
        "items": annotated_items,
    }

    ann_path = BackendConfig.manifests_dir() / "annotations.json"
    tmp_path = ann_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, ann_path)
    return manifest_payload


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """List raw inspection images (including subfolder classes) and check availability of manifests."""
    raw_dir = BackendConfig.raw_dir()
    files, subfolder_counts = _scan_raw_images(raw_dir)
    clustering_path = BackendConfig.manifests_dir() / "clustering_result.json"
    annotations_path = BackendConfig.manifests_dir() / "annotations.json"

    return {
        "raw_dir": str(raw_dir),
        "total_images": len(files),
        "subfolder_classes": subfolder_counts,
        "detected_class_names": sorted(subfolder_counts.keys()),
        "has_clustering_result": clustering_path.exists(),
        "has_annotations": annotations_path.exists(),
        "sample_files": [str(p) for p in files[:50]],
    }


@router.post("/datasets/upload-folder")
def upload_folder_images(req: FolderUploadRequest) -> dict[str, Any]:
    """Save uploaded folder images into `/app/data/raw/<subfolder_class_name>/<file_name>`."""
    raw_dir = BackendConfig.raw_dir()
    manifests_dir = BackendConfig.manifests_dir()

    if req.clear_existing:
        _clear_raw_dir_and_manifests(raw_dir, manifests_dir)

    saved_count = 0
    for item in req.items:
        safe_class = _sanitize_name(item.class_name, fallback="unclassified")
        orig_name = Path(item.file_name).name
        safe_fname = _sanitize_name(Path(orig_name).stem, fallback=f"img_{saved_count:04d}")
        ext = Path(orig_name).suffix.lower()
        if ext not in _VALID_IMAGE_EXTS:
            ext = ".png"

        class_dir = raw_dir / safe_class
        class_dir.mkdir(parents=True, exist_ok=True)
        dest_path = class_dir / f"{safe_fname}{ext}"

        try:
            raw_bytes = base64.b64decode(item.content_base64)
            # Verify the bytes represent a valid image with PIL
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            dest_path.write_bytes(raw_bytes)
            saved_count += 1
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file `{item.file_name}` in class `{item.class_name}`: {exc}",
            ) from exc

    files, subfolder_counts = _scan_raw_images(raw_dir)
    return {
        "uploaded_in_batch": saved_count,
        "total_images": len(files),
        "subfolder_classes": subfolder_counts,
        "detected_class_names": sorted(subfolder_counts.keys()),
        "raw_dir": str(raw_dir),
    }


@router.post("/datasets/generate-synthetic")
def generate_synthetic_dataset_proxy(req: GenerateSyntheticRequest) -> dict[str, Any]:
    """Proxy synthetic 100:2 dataset generation to `trainer-service`."""
    url = f"{BackendConfig.TRAINER_API_URL}/internal/datasets/generate-synthetic"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=req.model_dump())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to call trainer-service synthetic generator: {exc}"
        ) from exc


@router.get("/clusters")
def get_clustering_result() -> dict[str, Any]:
    """Return the latest Phase 1 2-stage clustering result (`clustering_result.json`)."""
    clustering_path = BackendConfig.manifests_dir() / "clustering_result.json"
    if not clustering_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No clustering_result.json found. Run Phase 1 clustering first.",
        )
    return json.loads(clustering_path.read_text(encoding="utf-8"))


@router.get("/annotations")
def get_annotations() -> dict[str, Any]:
    """Return the confirmed pseudo-multiclass annotations (`annotations.json`)."""
    ann_path = BackendConfig.manifests_dir() / "annotations.json"
    if not ann_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No annotations.json found. Confirm cluster labels in Phase 1 first.",
        )
    return json.loads(ann_path.read_text(encoding="utf-8"))


@router.post("/annotations")
def save_annotations(req: AnnotationSubmissionRequest) -> dict[str, Any]:
    """Confirm pseudo-multiclass labels from Phase 1 clustering (or subfolder names) and create stratified splits."""
    clustering_path = BackendConfig.manifests_dir() / "clustering_result.json"
    if not clustering_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Cannot save annotations before clustering_result.json exists.",
        )

    cluster_data = json.loads(clustering_path.read_text(encoding="utf-8"))
    raw_items = cluster_data.get("items", [])
    if not raw_items:
        raise HTTPException(status_code=400, detail="clustering_result.json contains 0 items.")

    raw_dir = BackendConfig.raw_dir()
    resolved_labels: list[str] = []
    for item in raw_items:
        cid_str = str(item["cluster_id"])
        img_path = str(item["image_path"])
        fname = str(item["file_name"])
        subfolder_hint = (
            item.get("subfolder_class")
            or item.get("ground_truth_hint")
            or _infer_class_from_path(Path(img_path), raw_dir)
        )

        if req.use_subfolder_labels and subfolder_hint:
            lbl = str(subfolder_hint)
        else:
            default_lbl = (
                str(subfolder_hint)
                if subfolder_hint and cid_str not in req.cluster_labels
                else (req.normal_class_name if cid_str == "0" else f"defect_mode_{cid_str}")
            )
            lbl = req.cluster_labels.get(cid_str, default_lbl)

        if img_path in req.sample_overrides:
            lbl = req.sample_overrides[img_path]
        elif fname in req.sample_overrides:
            lbl = req.sample_overrides[fname]
        resolved_labels.append(lbl.strip() or req.normal_class_name)

    return _build_and_save_annotations_manifest(
        raw_items=raw_items,
        resolved_labels=resolved_labels,
        normal_class_name=req.normal_class_name,
        val_ratio=req.val_ratio,
        seed=req.seed,
    )


@router.post("/annotations/from-subfolders")
def save_annotations_from_subfolders(req: DirectSubfolderAnnotationRequest) -> dict[str, Any]:
    """Create `annotations.json` directly using the subfolder names under `/app/data/raw/<class_name>/`."""
    raw_dir = BackendConfig.raw_dir()
    files, _ = _scan_raw_images(raw_dir)
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No images found in /app/data/raw. Upload a folder of images first.",
        )

    # If clustering_result.json already exists, preserve cluster_id where paths match
    cluster_id_by_path: dict[str, int] = {}
    clustering_path = BackendConfig.manifests_dir() / "clustering_result.json"
    if clustering_path.exists():
        try:
            c_data = json.loads(clustering_path.read_text(encoding="utf-8"))
            for it in c_data.get("items", []):
                cluster_id_by_path[str(it["image_path"])] = int(it["cluster_id"])
        except Exception:
            pass

    raw_items: list[dict[str, Any]] = []
    resolved_labels: list[str] = []
    for p in files:
        cls_name = _infer_class_from_path(p, raw_dir) or req.normal_class_name
        resolved_labels.append(cls_name)
        raw_items.append(
            {
                "image_path": str(p),
                "file_name": p.name,
                "cluster_id": cluster_id_by_path.get(str(p), 0 if cls_name == req.normal_class_name else 1),
            }
        )

    return _build_and_save_annotations_manifest(
        raw_items=raw_items,
        resolved_labels=resolved_labels,
        normal_class_name=req.normal_class_name,
        val_ratio=req.val_ratio,
        seed=req.seed,
    )
