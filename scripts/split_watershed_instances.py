#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2


def split_components(binary_mask: np.ndarray, min_area: int = 50) -> list[np.ndarray]:
    if binary_mask.dtype != np.uint8:
        bm = (binary_mask.astype(np.uint8) > 0).astype(np.uint8)
    else:
        bm = (binary_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bm, connectivity=8)
    instances: list[np.ndarray] = []
    for lid in range(1, num):
        comp = labels == lid
        if comp.sum() >= min_area:
            instances.append(comp)
    return instances


def load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    return img


def save_binary_mask(mask: np.ndarray, out_path: Path):
    out = mask.astype(np.uint8) * 255
    cv2.imwrite(str(out_path), out)


def colorize_instances(instance_masks: list[np.ndarray]) -> np.ndarray:
    if not instance_masks:
        return None
    h, w = instance_masks[0].shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234)
    for m in instance_masks:
        c = rng.integers(0, 255, size=3, dtype=np.uint8)
        color[m] = c
    return color


def main():
    parser = argparse.ArgumentParser(
        description="Split watershed mask into instance masks using connected components"
    )
    parser.add_argument("--input", required=True, help="Path to watershed mask PNG")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write instance masks (default: same as input)",
    )
    parser.add_argument(
        "--min_area", type=int, default=50, help="Minimum component area in pixels"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_mask(in_path)

    instance_masks: list[np.ndarray] = []
    stem = in_path.stem

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colors = np.unique(rgb.reshape(-1, 3), axis=0)
        for c in colors:
            if (c == np.array([0, 0, 0])).all():
                continue
            mask = (rgb == c).all(axis=-1)
            if mask.any():
                instance_masks.extend(split_components(mask, args.min_area))
    elif img.ndim == 2:
        labels = img
        values = np.unique(labels)
        for v in values:
            if v == 0:
                continue
            mask = labels == v
            if mask.any():
                instance_masks.extend(split_components(mask, args.min_area))
    else:
        raise ValueError(
            "Unsupported mask format: expected grayscale or RGB/RGBA image"
        )

    # Save instances
    if not instance_masks:
        print("No instances found.")
        return
    for idx, m in enumerate(instance_masks):
        out_path = out_dir / f"{stem}_inst_{idx:03d}.png"
        save_binary_mask(m, out_path)

    # Save a quick colored visualization
    color = colorize_instances(instance_masks)
    if color is not None:
        viz_path = out_dir / f"{stem}_instances_viz.png"
        cv2.imwrite(str(viz_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(instance_masks)} instance masks to {out_dir}")


if __name__ == "__main__":
    main()
