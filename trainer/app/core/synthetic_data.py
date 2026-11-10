"""Synthetic industrial inspection image generator for testing 100:2 imbalanced datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_synthetic_inspection_dataset(
    output_dir: Path,
    num_normal: int = 100,
    num_scratch: int = 3,
    num_stain: int = 3,
    num_dent: int = 2,
    image_size: int = 128,
    seed: int = 42,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Generate realistic synthetic industrial surface inspection images.

    Creates a brushed-metal / silicon wafer texture baseline for normal samples and
    injects distinct visual defect modes (scratch, stain, dent) to simulate the
    unlabeled initial collection stage with severe class imbalance (e.g., 100:2).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for existing in output_dir.glob("*.png"):
            existing.unlink(missing_ok=True)

    rng = np.random.default_rng(seed)
    created_files: list[dict[str, str]] = []

    def _make_base_surface() -> np.ndarray:
        """Create a brushed metallic texture with slight lighting gradient."""
        base_level = rng.uniform(155.0, 185.0)
        grad_x = np.linspace(-8.0, 8.0, image_size, dtype=np.float32)
        grad_y = np.linspace(-6.0, 6.0, image_size, dtype=np.float32)[:, None]
        streak_noise = rng.normal(0.0, 3.5, size=(image_size, 1)).astype(np.float32)
        fine_noise = rng.normal(0.0, 2.5, size=(image_size, image_size)).astype(np.float32)
        gray = np.clip(base_level + grad_x + grad_y + streak_noise + fine_noise, 0, 255).astype(
            np.uint8
        )
        rgb = np.stack([gray, gray, np.clip(gray.astype(np.int32) + 5, 0, 255).astype(np.uint8)], axis=-1)
        return rgb

    # 1. Generate Normal Samples
    for idx in range(num_normal):
        arr = _make_base_surface()
        img = Image.fromarray(arr, mode="RGB")
        file_name = f"sample_normal_{idx:04d}.png"
        file_path = output_dir / file_name
        img.save(file_path)
        created_files.append(
            {"file_name": file_name, "image_path": str(file_path), "ground_truth_hint": "normal"}
        )

    # 2. Generate Defect Mode A: Linear Scratch (high-contrast bright/dark score lines)
    for idx in range(num_scratch):
        arr = _make_base_surface()
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        for _ in range(2):
            x0 = int(rng.integers(12, image_size // 3))
            y0 = int(rng.integers(12, image_size - 12))
            x1 = int(rng.integers((2 * image_size) // 3, image_size - 12))
            y1 = int(rng.integers(12, image_size - 12))
            draw.line([(x0, y0), (x1, y1)], fill=(250, 45, 45), width=3)
            draw.line([(x0, min(image_size - 1, y0 + 2)), (x1, min(image_size - 1, y1 + 2))], fill=(25, 25, 30), width=2)
        file_name = f"sample_defect_scratch_{idx:04d}.png"
        file_path = output_dir / file_name
        img.save(file_path)
        created_files.append(
            {
                "file_name": file_name,
                "image_path": str(file_path),
                "ground_truth_hint": "defect_scratch",
            }
        )

    # 3. Generate Defect Mode B: Chemical / Oil Stain (dark brownish/yellowish patch)
    for idx in range(num_stain):
        arr = _make_base_surface()
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        cx = int(rng.integers(image_size // 4, (3 * image_size) // 4))
        cy = int(rng.integers(image_size // 4, (3 * image_size) // 4))
        rx = int(rng.integers(16, 28))
        ry = int(rng.integers(14, 26))
        draw.ellipse(
            [(cx - rx, cy - ry), (cx + rx, cy + ry)],
            fill=(40, 55, 15),
            outline=(70, 85, 25),
            width=2,
        )
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        file_name = f"sample_defect_stain_{idx:04d}.png"
        file_path = output_dir / file_name
        img.save(file_path)
        created_files.append(
            {
                "file_name": file_name,
                "image_path": str(file_path),
                "ground_truth_hint": "defect_stain",
            }
        )

    # 4. Generate Defect Mode C: Localized Pit / Dent (concentric dark crater + bright rim)
    for idx in range(num_dent):
        arr = _make_base_surface()
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        for _ in range(2):
            cx = int(rng.integers(24, image_size - 24))
            cy = int(rng.integers(24, image_size - 24))
            r = int(rng.integers(9, 15))
            draw.ellipse(
                [(cx - r - 2, cy - r - 2), (cx + r + 2, cy + r + 2)],
                fill=(30, 180, 245),
            )
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                fill=(10, 15, 65),
            )
        file_name = f"sample_defect_dent_{idx:04d}.png"
        file_path = output_dir / file_name
        img.save(file_path)
        created_files.append(
            {
                "file_name": file_name,
                "image_path": str(file_path),
                "ground_truth_hint": "defect_dent",
            }
        )

    return {
        "output_dir": str(output_dir),
        "total_generated": len(created_files),
        "counts": {
            "normal": num_normal,
            "defect_scratch": num_scratch,
            "defect_stain": num_stain,
            "defect_dent": num_dent,
        },
        "files": created_files,
    }
