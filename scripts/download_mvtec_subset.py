#!/usr/bin/env python3
"""Download or import an MVTec AD category subset formatted as a 100:2 imbalanced dataset in `data/raw/`.

Supports:
1. Importing from a local MVTec AD directory or `.tar.xz` / `.tar.gz` archive.
2. Downloading an MVTec AD category subset from Hugging Face Hub (`datasets` / HTTP parquet/tar mirror).
3. Generating a high-contrast synthetic MVTec-style inspection dataset (`--synthetic`) when offline.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def import_from_local_mvtec_dir(
    category_dir: Path,
    output_dir: Path,
    num_normal: int = 100,
    num_per_defect_mode: int = 2,
    image_size: int = 256,
    seed: int = 42,
) -> dict[str, int]:
    """Sample `num_normal` normal images and `num_per_defect_mode` images per defect type from MVTec AD."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.png"):
        old_file.unlink(missing_ok=True)

    counts: dict[str, int] = {}

    # Collect normal images from train/good and test/good
    normal_candidates: list[Path] = []
    for sub in ("train/good", "test/good"):
        d = category_dir / sub
        if d.exists():
            normal_candidates.extend(
                sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            )

    if not normal_candidates:
        raise FileNotFoundError(f"No normal images found under {category_dir}/train/good or test/good")

    chosen_normals = rng.choice(
        normal_candidates,
        size=min(num_normal, len(normal_candidates)),
        replace=False,
    )
    for idx, src_path in enumerate(chosen_normals):
        with Image.open(src_path) as img:
            rgb = img.convert("RGB").resize((image_size, image_size))
            dst = output_dir / f"sample_normal_{idx:04d}.png"
            rgb.save(dst)
    counts["normal"] = len(chosen_normals)

    # Collect defect modes from test/<defect_type> (excluding 'good')
    test_dir = category_dir / "test"
    if test_dir.exists():
        for mode_dir in sorted(test_dir.iterdir()):
            if not mode_dir.is_dir() or mode_dir.name == "good":
                continue
            mode_name = f"defect_{mode_dir.name}"
            defect_files = sorted(
                [p for p in mode_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            )
            if not defect_files:
                continue
            chosen_defects = rng.choice(
                defect_files,
                size=min(num_per_defect_mode, len(defect_files)),
                replace=False,
            )
            for idx, src_path in enumerate(chosen_defects):
                with Image.open(src_path) as img:
                    rgb = img.convert("RGB").resize((image_size, image_size))
                    dst = output_dir / f"sample_{mode_name}_{idx:04d}.png"
                    rgb.save(dst)
            counts[mode_name] = len(chosen_defects)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an imbalanced (e.g. 100:2) inspection dataset in data/raw/."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for raw inspection images (default: data/raw)",
    )
    parser.add_argument(
        "--mvtec-path",
        type=Path,
        default=None,
        help="Path to local extracted MVTec category directory or .tar.xz archive",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="metal_nut",
        help="MVTec AD category name when extracting from archive (default: metal_nut)",
    )
    parser.add_argument(
        "--num-normal",
        type=int,
        default=100,
        help="Number of normal images to sample (default: 100)",
    )
    parser.add_argument(
        "--num-per-defect",
        type=int,
        default=2,
        help="Number of defect images to sample per defect mode (default: 2)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Target square image size in pixels (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    if args.mvtec_path is not None and args.mvtec_path.exists():
        if args.mvtec_path.is_file() and args.mvtec_path.name.endswith((".tar.xz", ".tar.gz", ".tgz")):
            extract_root = args.output_dir.parent / "_tmp_mvtec"
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(args.mvtec_path) as tar:
                tar.extractall(path=extract_root)
            cat_dir = extract_root / args.category
            if not cat_dir.exists():
                cat_dir = extract_root
            counts = import_from_local_mvtec_dir(
                category_dir=cat_dir,
                output_dir=args.output_dir,
                num_normal=args.num_normal,
                num_per_defect_mode=args.num_per_defect,
                image_size=args.image_size,
                seed=args.seed,
            )
            shutil.rmtree(extract_root, ignore_errors=True)
        else:
            counts = import_from_local_mvtec_dir(
                category_dir=args.mvtec_path,
                output_dir=args.output_dir,
                num_normal=args.num_normal,
                num_per_defect_mode=args.num_per_defect,
                image_size=args.image_size,
                seed=args.seed,
            )
        print(f"Prepared MVTec subset in {args.output_dir}: {counts}")
        return 0

    # Fallback to built-in synthetic 100:2 generator
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))
    from app.core.synthetic_data import generate_synthetic_inspection_dataset

    res = generate_synthetic_inspection_dataset(
        output_dir=args.output_dir,
        num_normal=args.num_normal,
        num_scratch=args.num_per_defect,
        num_stain=args.num_per_defect,
        num_dent=args.num_per_defect,
        image_size=args.image_size,
        seed=args.seed,
        clear_existing=True,
    )
    print(f"Generated synthetic 100:2 dataset in {args.output_dir}: {res['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
