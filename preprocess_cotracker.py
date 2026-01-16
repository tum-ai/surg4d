"""
Preprocessing script for CoTracker3 control point extraction.
This script runs CoTracker3 on video frames, lifts control points to 3D using DA3 depth,
and computes Gaussian-to-control-point associations.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from hydra.core.global_hydra import GlobalHydra
import sys
from loguru import logger


# Import CoTracker3 utilities
from utils.cotracker_utils import (
    track_control_points,
    lift_control_points_to_3d,
    compute_gaussian_control_point_associations,
)
from utils.cotracker_interpolation import precompute_control_point_positions


def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    import re
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def process_clip_cotracker(clip: DictConfig, cfg: DictConfig):
    """Process a single clip to extract CoTracker3 control points."""
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / "images"
    depth_dir = clip_dir / cfg.preprocessing.depth_subdir
    instance_mask_dir = clip_dir / cfg.preprocessing.instance_mask_subdir
    
    # Output directory for CoTracker data
    cotracker_dir = clip_dir / "cotracker"
    cotracker_dir.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not depth_dir.exists():
        logger.error(f"Depth directory not found: {depth_dir}")
        return
    
    logger.info(f"Processing CoTracker3 for clip: {clip.name}")
    
    # Load images
    image_files = sorted(
        list(images_dir.glob("*.png")), key=extract_frame_number
    )
    logger.info(f"Found {len(image_files)} frames")
    
    # Load instance masks if available (convert to torch immediately)
    instance_masks = None
    if instance_mask_dir.exists():
        mask_files = sorted(
            list(instance_mask_dir.glob("*.npy")), key=extract_frame_number
        )
        if len(mask_files) == len(image_files):
            instance_masks = [torch.from_numpy(np.load(str(mf))).int() for mf in mask_files]
            logger.info(f"Loaded {len(instance_masks)} instance masks (converted to torch)")

    # TODO: check whether instance masks are actually used properly and what happens without
    
    # Track control points across all frames
    logger.info("Running CoTracker3...")
    control_points_2d = track_control_points(
        image_files,
        grid_size=cfg.preprocessing.cotracker_grid_size,
        save_dir=cotracker_dir,
    )
    # shape: (T, N_grid, 2) where N_grid = grid_size * grid_size
    
    # Lift control points to 3D using DA3 depth and camera parameters
    logger.info("Lifting control points to 3D...")
    colmap_dir = clip_dir / "sparse" / "0"
    control_points_3d, control_point_validity = lift_control_points_to_3d(
        control_points_2d,
        depth_dir,
        colmap_dir,
        image_files,
        grid_size=cfg.preprocessing.cotracker_grid_size,
        save_dir=cotracker_dir,
    )
    # control_points_3d: (T, N_grid, 3)
    # control_point_validity: (T, N_grid) boolean
    
    # Save control point trajectories
    control_points_3d_path = cotracker_dir / "control_points_3d.pth"
    torch.save(control_points_3d.cpu(), control_points_3d_path)
    logger.info(f"Saved control points 3D: {control_points_3d_path}")
    
    validity_path = cotracker_dir / "control_point_validity.pth"
    torch.save(control_point_validity.cpu(), validity_path)
    logger.info(f"Saved control point validity: {validity_path}")
    
    # Compute Gaussian-to-control-point associations for multiple init frames
    # Use first, middle, and last frames (matching preprocess.py multi-frame initialization)
    T = control_points_2d.shape[0]
    init_frame_indices = [0, T // 2, T - 1]
    logger.info(f"Computing associations for init frames: {init_frame_indices}")
    
    # Load confidence directory for filtering (matches da3_to_multi_view_colmap filtering)
    confidence_dir = clip_dir / cfg.preprocessing.confidence_subdir
    
    # Compute associations for each init frame and concatenate
    all_indices = []
    all_weights = []
    gaussians_per_frame = []
    
    for init_frame_idx in init_frame_indices:
        logger.info(f"Computing associations for frame {init_frame_idx}...")
        associations = compute_gaussian_control_point_associations(
            control_points_2d[init_frame_idx],
            depth_dir,
            init_frame_idx,
            instance_masks[init_frame_idx] if instance_masks is not None else None,
            k_neighbors=cfg.splat.cotracker_k_neighbors,
            grid_size=cfg.preprocessing.cotracker_grid_size,
            pixel_stride=cfg.preprocessing.da3_pc_pixel_stride,
            confidence_dir=confidence_dir,
            conf_thresh_percentile=cfg.preprocessing.da3_conf_thresh_percentile,
        )
        
        all_indices.append(associations["indices"])
        all_weights.append(associations["weights"])
        n_gaussians_frame = associations["indices"].shape[0]
        gaussians_per_frame.append(n_gaussians_frame)
        logger.info(f"Frame {init_frame_idx}: {n_gaussians_frame} Gaussians")
    
    # Concatenate associations from all frames
    combined_indices = torch.cat(all_indices, dim=0)  # (N_total, K)
    combined_weights = torch.cat(all_weights, dim=0)  # (N_total, K)
    
    total_gaussians = combined_indices.shape[0]
    logger.info(f"Total Gaussians from all frames: {total_gaussians}")
    logger.info(f"Gaussians per frame: {dict(zip(init_frame_indices, gaussians_per_frame))}")
    
    # Save associations
    indices_path = cotracker_dir / "gaussian_control_point_indices.pth"
    weights_path = cotracker_dir / "gaussian_control_point_weights.pth"
    
    torch.save(combined_indices.cpu(), indices_path)
    torch.save(combined_weights.cpu(), weights_path)
    
    # Also save the frame-wise counts for reference
    frame_counts_path = cotracker_dir / "gaussians_per_init_frame.pth"
    torch.save({
        "init_frame_indices": init_frame_indices,
        "gaussians_per_frame": gaussians_per_frame,
    }, frame_counts_path)
    
    logger.info(f"Saved associations: {indices_path}, {weights_path}")
    
    # Precompute Gaussian positions for all timesteps
    # control_points_3d stores 3D world coordinates
    logger.info("Precomputing Gaussian positions...")
    
    gaussian_positions = precompute_control_point_positions(
        control_points_3d,
        combined_indices,
        combined_weights,
        control_point_validity,
    )
    
    positions_path = cotracker_dir / "gaussian_positions_precomputed.pth"
    torch.save(gaussian_positions.cpu(), positions_path)
    logger.info(f"Saved precomputed positions: {positions_path}")
    
    logger.info(f"CoTracker preprocessing complete for {clip.name}")


def main():
    # do hydra init manually here to avoid conflicts
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        overrides = sys.argv[1:]
        cfg = hydra.compose("config.yaml", overrides=overrides)
    
    # Clear after composing the main config
    GlobalHydra.instance().clear()
    
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        process_clip_cotracker(clip, cfg)


if __name__ == "__main__":
    main()

