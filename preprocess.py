from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
import shutil
import cv2
import json
import re

from utils.cholec_utils import get_clip_seg8k, parse_cholecseg8k_instance_mask


def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def center_crop_divisible(
    arr: np.ndarray, k: int, skip_last_dim: bool = False
) -> np.ndarray:
    """
    Center crop an array so that each dimension's shape is divisible by the specified number (with no remainder).

    Args:
        arr (np.ndarray): Input array of any shape (all dimensions must be >= k).
        k (int): The number that each dimension should be divisible by
        skip_last_dim (bool): Whether to skip the last dimension when cropping

    Returns:
        np.ndarray: Center cropped array with each dimension being divisible by 'k'
    """
    assert k > 0, "k must be > 0"
    assert (np.array(arr.shape[: -1 if skip_last_dim else None]) >= k).all(), (
        "All dimensions must be at least k"
    )

    crop_slices = []
    for dim_size in arr.shape[: -1 if skip_last_dim else None]:
        new_dim_size = (dim_size // k) * k

        total_crop = dim_size - new_dim_size
        start_crop = total_crop // 2
        end_crop = total_crop - start_crop

        crop_slices.append(slice(start_crop, dim_size - end_crop))

    cropped_arr = arr[tuple(crop_slices)]
    return cropped_arr


def estimate_crop_box(class_ids: np.ndarray):
    """Get the top, bottom, left, right indices to
    crop the black camera borders of this clip.

    Args:
        class_ids (np.ndarray): input class ids (any frame of the clip)

    Returns:
        top (int): top index to crop
        bottom (int): bottom index to crop
        left (int): left index to crop
        right (int): right index to crop
    """
    # crop top and bottom black pixel rows by using mid vertical
    mid_col = class_ids[:, class_ids.shape[1] // 2]
    vertical_keep = mid_col != 0
    top = vertical_keep.argmax()
    bottom = vertical_keep.size - np.flip(vertical_keep).argmax()

    # we need the vertically cropped class ids to find left and right
    class_ids = class_ids[top:bottom, :]

    # crop black camera borders horizontally by cutting of first row's background annotation
    horizontal_keep = class_ids[0] != 0
    assert horizontal_keep.sum() != 0
    left = horizontal_keep.argmax()
    right = horizontal_keep.size - np.flip(horizontal_keep).argmax()

    return top, bottom, left, right


def _compute_center_crop_offsets(height: int, width: int, k: int) -> tuple[int, int]:
    """Compute top/left offsets removed by center_crop_divisible given H, W, k."""
    new_h = (height // k) * k
    new_w = (width // k) * k
    off_y = (height - new_h) // 2
    off_x = (width - new_w) // 2
    return off_y, off_x


def _load_and_translate_spatial_labels(
    clip: DictConfig,
    cfg: DictConfig,
    crop_box: tuple[int, int, int, int],
) -> tuple[dict, dict]:
    """
    Load spatial labels for this clip, map original frame numbers to contiguous
    0-based preprocessed indices and translate pixel coordinates according to
    cropping operations. Returns:
      - translated_labels_json (dict): same schema filtered and updated
      - per_frame_points (dict[int, list[tuple[int,int,str]]]): for visualization
    """
    labels_path = Path(cfg.preprocess.annotation_root) / "spatial" / f"{clip.name}.json"
    if not labels_path.exists():
        return {}, {}

    with open(labels_path, "r") as f:
        original = json.load(f)

    # Compute cropping offsets (black border and qwen patch center crop)
    top, bottom, left, right = crop_box
    cropped_h = bottom - top
    cropped_w = right - left
    off_y, off_x = _compute_center_crop_offsets(cropped_h, cropped_w, cfg.preprocess.frames_divisor)
    final_h = cropped_h - 2 * off_y
    final_w = cropped_w - 2 * off_x

    translated_annotations = []
    per_frame_points: dict[int, list[tuple[int, int, str, str]]] = {}

    for annotation in original["annotations"]:
        # Translate coordinates from pil_coords [x, y]
        x_orig, y_orig = annotation.get("pil_coords")
        x_new = int(x_orig - left - off_x)
        y_new = int(y_orig - top - off_y)
        
        # Skip if out of bounds
        if not (0 <= x_new < final_w and 0 <= y_new < final_h):
            print(f"Skipping annotation {clip.name} {annotation['id']} because it's out of bounds")
            continue

        # Create translated annotation
        new_annotation = {
            "timestep": annotation["timestep"],
            "id": annotation["id"],
            "numpy_coords": [y_new, x_new],
            "pil_coords": [x_new, y_new],
            "query": annotation["query"],
        }
        translated_annotations.append(new_annotation)

        # For visualization
        per_frame_points.setdefault(int(annotation["timestep"] * cfg.preprocess.annotation_stride), []).append((x_new, y_new, new_annotation["query"], "spatial"))

    translated = {"annotations": translated_annotations}

    return translated, per_frame_points


def _render_label_visualization(
    rgb_image: np.ndarray,
    points: list[tuple[int, int, str, str]],
) -> np.ndarray:
    """Render object/action points with labels on a copy of the given RGB image.

    Args:
        rgb_image: HxWx3 uint8 array (already cropped and center-cropped)
        points: list of tuples (x, y, text, kind) where kind in {"obj", "act"}

    Returns:
        The rendered RGB image (uint8) as a new array.
    """
    img_viz = rgb_image.copy()
    for x, y, text, kind in points:
        color = (255, 0, 0) if kind == "obj" else (0, 0, 255)
        cv2.circle(img_viz, (int(x), int(y)), 4, color, thickness=-1)
        cv2.putText(
            img_viz,
            text,
            (int(x) + 6, int(y) + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return img_viz


def preprocess(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    only_update_annotations = cfg.preprocess.only_update_annotations

    if not only_update_annotations:
        out_images = clip_dir / cfg.preprocess.image_subdir
        out_images.mkdir(parents=True, exist_ok=True)

        out_sem_masks = clip_dir / cfg.preprocess.semantic_mask_subdir
        out_sem_masks.mkdir(parents=True, exist_ok=True)

        out_inst_masks = clip_dir / cfg.preprocess.instance_mask_subdir
        out_inst_masks.mkdir(parents=True, exist_ok=True)

    frame_files, semantic_mask_files = get_clip_seg8k(
        seg8k_root=Path(cfg.cholecseg8k_root),
        seg8k_video_id=clip.video_id,
        first_frame=clip.first_frame,
        last_frame=clip.last_frame,
        frame_stride=clip.frame_stride,
    )

    # estimate crops to remove black borders
    top, bottom, left, right = estimate_crop_box(
        parse_cholecseg8k_instance_mask(Image.open(semantic_mask_files[0]))
    )

    # translate spatial labels to new coordinates
    translated_labels, viz_points = _load_and_translate_spatial_labels(
        clip,
        cfg,
        crop_box=(top, bottom, left, right),
    )
    if translated_labels:
        labels_out_path = clip_dir / cfg.preprocess.spatial_labels_output_filename
        with open(labels_out_path, "w") as f:
            json.dump(translated_labels, f, indent=2)

    # Create visualization directory once if needed
    if cfg.preprocess.dump_label_visualizations and viz_points:
        viz_dir = clip_dir / cfg.preprocess.label_viz_subdir
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        viz_dir.mkdir(parents=True)

    for new_frame_id, (frame_file, semantic_mask_file) in enumerate(
        zip(frame_files, semantic_mask_files)
    ):
        rgb = Image.open(frame_file)
        instance_mask = Image.open(semantic_mask_file)

        class_ids = parse_cholecseg8k_instance_mask(instance_mask)

        rgb = np.asarray(rgb)[top:bottom, left:right]
        class_ids = class_ids[top:bottom, left:right]
        rgb = center_crop_divisible(
            rgb, cfg.preprocess.frames_divisor, skip_last_dim=True
        )  # required for qwen encoder
        class_ids = center_crop_divisible(class_ids, cfg.preprocess.frames_divisor)

        if not only_update_annotations:
            # Generate instance masks from semantic masks using connected components
            # 2d arrays, 0 is background, > 0 are instance ids
            instance_ids = np.zeros_like(class_ids, dtype=np.int32)
            instance_counter = 1
            unique_classes = np.unique(class_ids)
            unique_classes = unique_classes[unique_classes != 0]
            for class_id in unique_classes:
                class_binary = (class_ids == class_id).astype(np.uint8)
                num_components, labeled_components = cv2.connectedComponents(
                    class_binary, connectivity=8
                )
                for component_id in range(
                    1, num_components
                ):  # cv2 starts at 1, 0 is background
                    component_mask = labeled_components == component_id
                    component_area = component_mask.sum()
                    # Filter out tiny components (noise)
                    if component_area < cfg.preprocess.min_component_area:
                        continue
                    instance_ids[component_mask] = instance_counter
                    instance_counter += 1

            rgb_img_path = out_images / f"frame_{new_frame_id:06d}.png"
            Image.fromarray(rgb).save(rgb_img_path)
            np.save(out_sem_masks / f"frame_{new_frame_id:06d}.npy", class_ids)
            np.save(out_inst_masks / f"frame_{new_frame_id:06d}.npy", instance_ids)

        # Optional visualization of labels on preprocessed frames
        if (
            cfg.preprocess.dump_label_visualizations
            and new_frame_id in viz_points
        ):
            viz_dir = clip_dir / cfg.preprocess.label_viz_subdir
            img_viz = _render_label_visualization(rgb, viz_points[new_frame_id])
            cv2.imwrite(
                str(viz_dir / f"frame_{new_frame_id:06d}_viz.png"),
                cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR),
            )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        preprocess(clip, cfg)


if __name__ == "__main__":
    main()
