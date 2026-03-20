"""
Standalone inference script for DepthLens.

Usage:
    python inference.py --input photo.jpg --output depth.png
    python inference.py --input ./photos/ --output ./results/ --batch
    python inference.py --input photo.jpg --output depth.png --model large --colormap magma
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from models import DepthEstimator
from utils import depth_to_colormap, create_side_by_side, create_overlay


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def process_single(
    estimator: DepthEstimator,
    input_path: Path,
    output_path: Path,
    colormap: str,
    side_by_side: bool,
    overlay: bool,
    overlay_alpha: float,
    save_raw: bool,
):
    """Process a single image and save results."""
    print(f"  Processing: {input_path.name}")
    start = time.time()

    image = Image.open(input_path).convert("RGB")
    depth = estimator.predict(image)

    elapsed = time.time() - start
    print(f"  Inference: {elapsed:.2f}s")

    # Save colormapped depth
    depth_colored = depth_to_colormap(depth, colormap)

    if side_by_side:
        result = create_side_by_side(image, depth_colored)
    elif overlay:
        result = create_overlay(image, depth_colored, alpha=overlay_alpha)
    else:
        result = depth_colored

    # Determine output path
    out = Path(output_path)
    if out.suffix.lower() == ".npy" or save_raw:
        raw_path = out.with_suffix(".npy") if out.suffix else out / (input_path.stem + "_depth.npy")
        np.save(str(raw_path), depth)
        print(f"  Saved raw depth: {raw_path}")

    if out.suffix.lower() != ".npy":
        result.save(str(out))
        print(f"  Saved: {out}")


def process_batch(
    estimator: DepthEstimator,
    input_dir: Path,
    output_dir: Path,
    colormap: str,
    side_by_side: bool,
    overlay: bool,
    overlay_alpha: float,
    save_raw: bool,
):
    """Process all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images in {input_dir}")
    total_start = time.time()

    for img_path in images:
        out_name = img_path.stem + "_depth.png"
        out_path = output_dir / out_name
        process_single(
            estimator, img_path, out_path,
            colormap, side_by_side, overlay, overlay_alpha, save_raw,
        )

    total = time.time() - total_start
    avg = total / len(images)
    print(f"\nDone! {len(images)} images in {total:.1f}s (avg {avg:.2f}s/image)")


def main():
    parser = argparse.ArgumentParser(description="DepthLens — Monocular Depth Estimation")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output", required=True, help="Output path or directory")
    parser.add_argument("--model", default="small", choices=["small", "large"], help="Model size")
    parser.add_argument("--colormap", default="inferno", choices=["inferno", "magma", "viridis", "plasma"])
    parser.add_argument("--side-by-side", action="store_true", help="Generate side-by-side comparison")
    parser.add_argument("--overlay", action="store_true", help="Generate depth overlay on original")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw depth as .npy")
    parser.add_argument("--batch", action="store_true", help="Process a folder of images")
    args = parser.parse_args()

    estimator = DepthEstimator(model_size=args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.batch:
        process_batch(
            estimator, input_path, output_path,
            args.colormap, args.side_by_side, args.overlay, args.overlay_alpha, args.save_raw,
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_single(
            estimator, input_path, output_path,
            args.colormap, args.side_by_side, args.overlay, args.overlay_alpha, args.save_raw,
        )


if __name__ == "__main__":
    main()
