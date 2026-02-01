from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from hydra.core.global_hydra import GlobalHydra
import subprocess
import shutil
import sys
import torch
from loguru import logger
import zipfile
from openexr_numpy import imread
import cv2
import json
import re
from depth_anything_3.api import DepthAnything3

from cholec_utils import get_clip_seg8k, parse_cholecseg8k_instance_mask
from da3_utils import da3_to_multi_view_colmap, log_da3_rerun, filter_prediction_edge_artifacts
from scene.colmap_loader import read_points3D_binary
from scene.dataset_readers import storePly


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
    center_divisor: int,
) -> tuple[dict, dict]:
    """
    Load spatial labels for this clip, map original frame numbers to contiguous
    0-based preprocessed indices and translate pixel coordinates according to
    cropping operations. Returns:
      - translated_labels_json (dict): same schema filtered and updated
      - per_frame_points (dict[int, list[tuple[int,int,str]]]): for visualization
    """
    # Build input filename from template
    template = cfg.preprocessing.get(
        "spatial_labels_input_filename_template", "{clip_name}_spatial.json"
    )
    input_filename = template.format(clip_name=clip.name)
    labels_path = Path(cfg.preprocessing.spatial_labels_root) / input_filename
    if not labels_path.exists():
        return {}, {}

    with open(labels_path, "r") as f:
        original = json.load(f)

    first_frame = clip.first_frame
    last_frame = clip.last_frame
    stride = clip.frame_stride

    # Compute cropping offsets
    top, bottom, left, right = crop_box
    cropped_h = bottom - top
    cropped_w = right - left
    off_y, off_x = _compute_center_crop_offsets(cropped_h, cropped_w, center_divisor)
    final_h = cropped_h - 2 * off_y
    final_w = cropped_w - 2 * off_x

    translated = {}
    per_frame_points: dict[int, list[tuple[int, int, str, str]]] = {}

    for key, entry in original.items():
        orig_fn = entry.get("frame_number")
        if orig_fn is None:
            continue
        if orig_fn < first_frame or orig_fn >= last_frame:
            continue
        if (orig_fn - first_frame) % stride != 0:
            continue
        new_idx = (orig_fn - first_frame) // stride

        # Translate objects/actions coords
        def _translate_list(items: list) -> list:
            out = []
            for it in items:
                x = it.get("pixel_x")
                y = it.get("pixel_y")
                if x is None or y is None:
                    continue
                x2 = int(x - left - off_x)
                y2 = int(y - top - off_y)
                if not (0 <= x2 < final_w and 0 <= y2 < final_h):
                    continue
                new_it = dict(it)
                new_it["pixel_x"] = x2
                new_it["pixel_y"] = y2
                new_it["pixel_coords_numpy"] = [y2, x2]
                out.append(new_it)
            return out

        new_objects = _translate_list(entry.get("objects", []))
        new_actions = _translate_list(entry.get("actions", []))

        # If both empty, we still keep the entry to preserve format
        new_entry = dict(entry)
        new_entry["frame_number"] = int(new_idx)
        new_entry["objects"] = new_objects
        new_entry["actions"] = new_actions
        translated[key] = new_entry

        # For visualization, aggregate simple points with labels
        pts = []
        for it in new_objects:
            pts.append((it["pixel_x"], it["pixel_y"], it.get("query", "object"), "obj"))
        for it in new_actions:
            pts.append((it["pixel_x"], it["pixel_y"], it.get("query", "action"), "act"))
        if pts:
            per_frame_points.setdefault(int(new_idx), []).extend(pts)

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


def get_cholecseg8k_frames(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    out_rgb = clip_dir / "rgb"
    out_rgb.mkdir(parents=True, exist_ok=True)

    out_sem_masks = clip_dir / cfg.preprocessing.semantic_mask_subdir
    out_sem_masks.mkdir(parents=True, exist_ok=True)

    out_inst_masks = clip_dir / cfg.preprocessing.instance_mask_subdir
    out_inst_masks.mkdir(parents=True, exist_ok=True)

    frame_files, semantic_mask_files = get_clip_seg8k(
        seg8k_root=Path(cfg.cholecseg8k_root),
        seg8k_video_id=clip.video_id,
        first_frame=clip.first_frame,
        last_frame=clip.last_frame,
        frame_stride=clip.frame_stride,
    )

    top, bottom, left, right = estimate_crop_box(
        parse_cholecseg8k_instance_mask(Image.open(semantic_mask_files[0]))
    )

    # Pre-compute label translations and optional visualization metadata
    translated_labels, viz_points = _load_and_translate_spatial_labels(
        clip,
        cfg,
        crop_box=(top, bottom, left, right),
        center_divisor=cfg.preprocessing.frames_divisor,
    )
    # Build output filename from config
    out_filename = cfg.preprocessing.spatial_labels_output_filename
    labels_out_path = clip_dir / out_filename

    for new_frame_id, (frame_file, semantic_mask_file) in enumerate(
        zip(frame_files, semantic_mask_files)
    ):
        rgb = Image.open(frame_file)
        instance_mask = Image.open(semantic_mask_file)

        class_ids = parse_cholecseg8k_instance_mask(instance_mask)

        rgb = np.asarray(rgb)[top:bottom, left:right]
        class_ids = class_ids[top:bottom, left:right]
        rgb = center_crop_divisible(
            rgb, cfg.preprocessing.frames_divisor, skip_last_dim=True
        )  # required for qwen encoder
        class_ids = center_crop_divisible(class_ids, cfg.preprocessing.frames_divisor)

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
                if component_area < cfg.preprocessing.min_component_area:
                    continue
                instance_ids[component_mask] = instance_counter
                instance_counter += 1

        rgb_img_path = out_rgb / f"frame_{new_frame_id:06d}.png"
        Image.fromarray(rgb).save(rgb_img_path)
        np.save(out_sem_masks / f"frame_{new_frame_id:06d}.npy", class_ids)
        np.save(out_inst_masks / f"frame_{new_frame_id:06d}.npy", instance_ids)

        # Optional visualization of labels on preprocessed frames
        if (
            cfg.preprocessing.get("dump_label_visualizations", False)
            and new_frame_id in viz_points
        ):
            viz_dir = clip_dir / cfg.preprocessing.get("label_viz_subdir", "label_viz")
            viz_dir.mkdir(parents=True, exist_ok=True)
            img_viz = _render_label_visualization(rgb, viz_points[new_frame_id])
            cv2.imwrite(
                str(viz_dir / f"frame_{new_frame_id:06d}_viz.png"),
                cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR),
            )

    # Save translated labels JSON if any present
    if translated_labels:
        with open(labels_out_path, "w") as f:
            json.dump(translated_labels, f, indent=2)


def colmap_txt_to_bin(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    output_dir = clip_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(clip_dir.resolve()),
        "--output_path",
        str(output_dir.resolve()),
        "--output_type",
        "BIN",
    ]
    subprocess.run(cmd, check=True)


def delete_unused_files(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    # vipe internals
    if (clip_dir / "intrinsics").exists():
        shutil.rmtree(clip_dir / "intrinsics")
    if (clip_dir / "mask").exists():
        shutil.rmtree(clip_dir / "mask")
    if (clip_dir / "pose").exists():
        shutil.rmtree(clip_dir / "pose")
    if (clip_dir / "vipe").exists():
        shutil.rmtree(clip_dir / "vipe")
    if (clip_dir / "vipe" / "rgb_info.pkl").exists():
        (clip_dir / "vipe" / "rgb_info.pkl").unlink()
    if (clip_dir / "vipe" / "rgb_slam_map.pt").exists():
        (clip_dir / "vipe" / "rgb_slam_map.pt").unlink()

    # vipe aux vis
    if (clip_dir / "vipe_aux_vis").exists():
        shutil.rmtree(clip_dir / "vipe_aux_vis")

    # colmap txt version
    if (clip_dir / "cameras.txt").exists():
        (clip_dir / "cameras.txt").unlink()
    if (clip_dir / "images.txt").exists():
        (clip_dir / "images.txt").unlink()
    if (clip_dir / "points3D.txt").exists():
        (clip_dir / "points3D.txt").unlink()

    # rgb dir becomes images dir
    if (clip_dir / "rgb").exists():
        shutil.rmtree(clip_dir / "rgb")


def da3(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    images_dir = clip_dir / "images"

    # load da3 model
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
    model = model.to("cuda:0")

    # construct image paths and determine processing resolution close to orig
    image_filenames = sorted(
        list(images_dir.glob("*.png")), key=extract_frame_number
    )
    image_filenames = [str(img_file) for img_file in image_filenames]
    orig_w, orig_h = Image.open(image_filenames[0]).size
    processing_res = max(orig_w, orig_h)

    # da3 inference
    prediction = model.inference(
        image=image_filenames,
        ref_view_strategy="middle",  # good for video according to docs
        process_res=processing_res,
        process_res_method="upper_bound_resize",
    )

    # Apply edge filtering to prediction at processed resolution (if configured)
    edge_gradient_threshold = cfg.preprocessing.get("da3_edge_gradient_threshold", None)
    if edge_gradient_threshold is not None:
        logger.info(f"Applying depth edge filtering with threshold: {edge_gradient_threshold}")
        prediction = filter_prediction_edge_artifacts(
            prediction,
            gradient_threshold=edge_gradient_threshold,
        )

    # dump to colmap with pc consisting of multi-frame depth projection (first, middle, last)
    colmap_dir = clip_dir / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove stale PLY file so it gets regenerated from the new .bin
    stale_ply = colmap_dir / "points3D.ply"
    if stale_ply.exists():
        stale_ply.unlink()
        logger.info(f"Removed stale PLY file: {stale_ply}")
    
    # Use first, middle, and last frames for initialization
    num_frames_total = len(prediction.depth)
    init_frame_indices = [0, num_frames_total // 2, num_frames_total - 1]
    logger.info(f"Initializing point cloud from frames: {init_frame_indices}")
    
    view_point_counts = da3_to_multi_view_colmap(
        prediction,
        colmap_dir,
        image_filenames,
        view_indices=init_frame_indices,
        conf_thresh_percentile=cfg.preprocessing.da3_conf_thresh_percentile,
        pixel_stride=cfg.preprocessing.da3_pc_pixel_stride,
        densify_ratio=cfg.preprocessing.da3_densify_ratio,
    )
    logger.info(f"Point counts per view: {dict(zip(init_frame_indices, view_point_counts))}")

    # Store depth maps at both processed and original resolution
    # Processed resolution: used by point cloud and cotracker (guarantees consistency)
    # Original resolution: available for other downstream uses like depth loss supervision for rendered depths in original resolution
    depth_dir = clip_dir / cfg.preprocessing.depth_subdir
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_processed_dir = clip_dir / cfg.preprocessing.depth_processed_subdir
    depth_processed_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir = clip_dir / cfg.preprocessing.confidence_subdir
    confidence_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(prediction.depth)
    for frame_idx in range(num_frames):
        depth_proc = prediction.depth[frame_idx]
        
        # Save at processed resolution (for point cloud / cotracker consistency)
        depth_processed_path = depth_processed_dir / f"{frame_idx:06d}.npy"
        np.save(depth_processed_path, depth_proc)
        
        # Save at original resolution (for other uses)
        depth_orig = cv2.resize(
            depth_proc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        depth_path = depth_dir / f"{frame_idx:06d}.npy"
        np.save(depth_path, depth_orig)
        
        # Save confidence at original resolution
        confidence_proc = prediction.conf[frame_idx]
        confidence_orig = cv2.resize(
            confidence_proc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        confidence_path = confidence_dir / f"{frame_idx:06d}.npy"
        np.save(confidence_path, confidence_orig)


    # Clean up GPU memory from da3 model
    del model
    del prediction
    torch.cuda.empty_cache()


def pc_ply_visualization(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    bin_path = clip_dir / "sparse" / "0" / "points3D.bin"
    out_path = clip_dir / cfg.preprocessing.pc_ply_visualization_filename
    xyz, rgb, _ = read_points3D_binary(str(bin_path))
    storePly(str(out_path), xyz, rgb)


def process_clip(clip: DictConfig, cfg: DictConfig):
    get_cholecseg8k_frames(clip, cfg)
    if cfg.preprocessing.depth_estimation == "da3":
        da3(clip, cfg)
    else:
        raise ValueError(
            f"Invalid depth estimation method: {cfg.preprocessing.depth_estimation}"
        )

    if cfg.preprocessing.pc_ply_visualization_filename:
        pc_ply_visualization(clip, cfg)

    if not cfg.preprocessing.verbose_output:
        delete_unused_files(clip, cfg)


def main():
    # do hydra init manually here to avoid conflicts with vipe hydra
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        overrides = sys.argv[1:]
        cfg = hydra.compose("config.yaml", overrides=overrides)

    # Clear after composing the main config so vipe can initialize its own
    GlobalHydra.instance().clear()

    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        process_clip(clip, cfg)


if __name__ == "__main__":
    main()
