"""
Utilities for CoTracker3 control point tracking and Gaussian association.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict, Tuple, List
from loguru import logger
import rerun as rr
import cv2
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
from scene.colmap_loader import (
    read_intrinsics_binary,
    read_extrinsics_binary,
    read_intrinsics_text,
    read_extrinsics_text,
    qvec2rotmat,
)
import re


def _load_colmap_data(colmap_dir: Path):
    """
    Load raw Colmap data (cameras and images) from sparse/0/ directory.
    
    Follows the same pattern as scene.dataset_readers.readColmapSceneInfo.
    
    Args:
        colmap_dir: Directory containing sparse/0/ subdirectory with Colmap data
    
    Returns:
        cam_intrinsics: Dictionary mapping camera_id to Camera objects
        cam_extrinsics: Dictionary mapping image_id to Image objects
    """
    
    # Try binary format first, fall back to text (same pattern as readColmapSceneInfo)
    try:
        cameras_intrinsic_file = colmap_dir / "cameras.bin"
        images_extrinsic_file = colmap_dir / "images.bin"
        cam_intrinsics = read_intrinsics_binary(str(cameras_intrinsic_file))
        cam_extrinsics = read_extrinsics_binary(str(images_extrinsic_file))
    except:
        cameras_intrinsic_file = colmap_dir / "cameras.txt"
        images_extrinsic_file = colmap_dir / "images.txt"
        cam_intrinsics = read_intrinsics_text(str(cameras_intrinsic_file))
        cam_extrinsics = read_extrinsics_text(str(images_extrinsic_file))
    
    return cam_intrinsics, cam_extrinsics


def load_colmap_camera_parameters(
    colmap_dir: Path,
    image_files: List[Path],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load Colmap camera intrinsics and extrinsics, matched to image files.
    
    Args:
        colmap_dir: Directory containing sparse/0/ subdirectory with Colmap data
        image_files: List of image file paths (sorted by frame number)
    
    Returns:
        intrinsics: List of (3, 3) intrinsic matrices (K) for each image
        c2w_matrices: List of (4, 4) camera-to-world transformation matrices for each image
    """
    # Load Colmap data (reuses the same file reading pattern as readColmapSceneInfo)
    cam_intrinsics, cam_extrinsics = _load_colmap_data(colmap_dir)
    
    # Create mapping from image filename to Colmap image data
    image_name_to_colmap = {}
    for img_id, img_data in cam_extrinsics.items():
        image_name_to_colmap[img_data.name] = img_data
    
    # Extract frame numbers from image files for matching
    def extract_frame_number(filepath: Path) -> int:
        match = re.search(r"frame_(\d+)", filepath.stem)
        if match:
            return int(match.group(1))
        return 0
    
    intrinsics_list = []
    c2w_matrices_list = []
    
    for img_path in image_files:
        img_name = img_path.name
        
        # Find matching Colmap image
        if img_name in image_name_to_colmap:
            img_data = image_name_to_colmap[img_name]
        else:
            # Try without extension (e.g., "frame_000001.png" matches "frame_000001")
            img_stem = img_path.stem
            found = False
            for colmap_name, colmap_data in image_name_to_colmap.items():
                colmap_stem = Path(colmap_name).stem
                if colmap_stem == img_stem or colmap_name == img_stem:
                    img_data = colmap_data
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find Colmap data for image {img_name}")
        
        # Get camera intrinsics
        camera = cam_intrinsics[img_data.camera_id]
        
        # Extract intrinsic parameters based on camera model
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        elif camera.model == "SIMPLE_RADIAL":
            f, cx, cy, _ = camera.params
            fx = fy = f
        else:
            logger.warning(f"Unsupported camera model {camera.model}, using default intrinsics")
            fx = fy = camera.width / 2.0
            cx = camera.width / 2.0
            cy = camera.height / 2.0
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics_list.append(K)
        
        # Convert Colmap extrinsics (world-to-camera) to camera-to-world
        # Colmap stores qvec (quaternion) and tvec (translation) for world-to-camera
        R_w2c = qvec2rotmat(img_data.qvec)  # (3, 3) rotation matrix
        t_w2c = img_data.tvec  # (3,) translation
        
        # Build world-to-camera matrix
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        
        # Invert to get camera-to-world
        c2w = np.linalg.inv(w2c)
        c2w_matrices_list.append(c2w)
    
    return intrinsics_list, c2w_matrices_list


def track_control_points(
    image_files: list,
    n_points_per_frame: int = 1024,
    save_dir: Path = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Track control points across video using CoTracker3 with random point sampling.
    
    Uses random 2D points instead of a grid to ensure diversity across query frames,
    which is important when the camera moves slowly or stays static.
    
    Args:
        image_files: List of image file paths
        n_points_per_frame: Number of random points to sample per query frame
        save_dir: Directory to save visualizations
        seed: Random seed for reproducibility
    
    Returns:
        control_points_2d: (T, N_total, 2) tensor of 2D pixel coordinates on GPU
            where N_total = n_points_per_frame * number of random tracking point initializations
        visibility: (T, N_total) boolean tensor
    """
    # Initialize CoTracker3 predictor
    predictor = CoTrackerPredictor(checkpoint="checkpoints/cotracker3_scaled_offline.pth")
    predictor.to("cuda")
    
    # Load images
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        images.append(img_array)
    
    # Convert images to torch tensors
    images_torch = torch.stack([
        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        for img in images
    ], dim=0).unsqueeze(0).to("cuda")  # (B, T, C, H, W)

    logger.info(f"Image tensor shape: {images_torch.shape}")
    n_frames = images_torch.shape[1]
    H, W = images_torch.shape[3], images_torch.shape[4]
    
    # Query frames: beginning, middle, end
    query_frames = [0, n_frames // 2, n_frames - 1]
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    all_tracks = []
    all_visibility = []
    
    with torch.no_grad():
        for query_frame in query_frames:
            # Generate random 2D points for this query frame
            # queries shape: (B, N, 3) where each query is (frame_idx, x, y)
            random_x = torch.rand(n_points_per_frame) * (W - 1)  # x in [0, W-1]
            random_y = torch.rand(n_points_per_frame) * (H - 1)  # y in [0, H-1]
            # Create tensor of filled N times with the query frame
            frame_indices = torch.full((n_points_per_frame,), query_frame, dtype=torch.float32)
            # Construct queries where each query is (frame_idx, x, y)
            queries = torch.stack([frame_indices, random_x, random_y], dim=-1)  # (N, 3)
            queries = queries.unsqueeze(0).to("cuda")  # (1, N, 3)
            
            # Track with backward_tracking for middle and end frames
            backward = query_frame > 0
            tracks, visibility = predictor(images_torch, queries=queries, backward_tracking=backward)
            # tracks: (B, T, N, 2), visibility: (B, T, N)
            
            all_tracks.append(tracks)
            all_visibility.append(visibility)
            
            logger.info(f"Query frame {query_frame}: tracked {n_points_per_frame} random points")
    
    # Concatenate all tracks along the points dimension
    tracks = torch.cat(all_tracks, dim=2)  # (B, T, N_total, 2)
    visibility = torch.cat(all_visibility, dim=2)  # (B, T, N_total)
    
    logger.info(f"Total tracks shape: {tracks.shape}")

    # Visualize
    visualizer = Visualizer(save_dir=save_dir, pad_value=25)
    visualizer.visualize(video=images_torch * 255.0, tracks=tracks, visibility=visibility, filename="cotracker3_visualization_random")
    
    return tracks.squeeze(0), visibility.squeeze(0)


# TODO: refactor this, way too much happening here
def lift_control_points_to_3d(
    control_points_2d: torch.Tensor,
    visibility: torch.Tensor,
    depth_processed_dir: Path,
    colmap_dir: Path,
    image_files: List[Path],
    depth_jump_threshold: float,
    save_dir: Path = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lift 2D control points to 3D world coordinates using DA3 depth maps and camera parameters.
    
    Features camera-aware depth propagation (while accounting for camera motion), depth jump detection
    (kills points that jump between surfaces due to 2D tracking errors), and color forwarding (propagates last valid color
    for occluded points).
    
    Only returns points that are valid (have depth > 0) at ALL timesteps after forward/backward filling.
    Points that are killed or never observed are filtered out entirely.
    
    Args:
        control_points_2d: (T, N_points, 2) 2D pixel coordinates (torch.Tensor on GPU)
        visibility: (T, N_points) boolean array indicating visible/occluded tracked points
        depth_processed_dir: Directory containing depth maps at processed resolution of the depth model (*.npy files)
        colmap_dir: Directory containing sparse/0/ Colmap data with camera parameters
        image_files: List of image file paths (sorted by frame number)
        depth_jump_threshold: Absolute depth change threshold (meters) for killing points that jump
                              between surfaces. Set to None to disable.
        save_dir: Directory to save visualizations
    
    Returns:
        control_points_3d: (T, N_valid, 3) 3D coordinates in world space (only permanently valid points)
        control_points_2d: (T, N_valid, 2) 2D coordinates filtered to match valid 3D points
    """
    T, N_grid, _ = control_points_2d.shape
    
    # Load depth maps (already at processed resolution), used by the depth model
    depth_files = sorted(depth_processed_dir.glob("*.npy"))
    assert len(depth_files) == T, f"Mismatch: {len(depth_files)} depth files, {T} frames"
    
    # Load camera parameters
    intrinsics_list, c2w_list = load_colmap_camera_parameters(colmap_dir, image_files)
    assert len(intrinsics_list) == T, f"Mismatch: {len(intrinsics_list)} camera parameters, {T} frames"
    
    # Pre-convert camera parameters to tensors for efficient access
    K_tensors = [torch.from_numpy(K).float().to("cuda") for K in intrinsics_list]
    c2w_tensors = [torch.from_numpy(c2w).float().to("cuda") for c2w in c2w_list]
    # Compute w2c (world-to-camera) matrices for reprojection
    w2c_tensors = [torch.inverse(c2w) for c2w in c2w_tensors]
    
    logger.info(f"Lifting {N_grid} control points to 3D world coordinates for {T} frames...")
    
    # Get original image resolution (CoTracker coordinates are in original resolution)
    first_image = Image.open(image_files[0])
    orig_W, orig_H = first_image.size
    
    # Get processed resolution from depth maps (already at processed resolution)
    depth_map_0 = np.load(str(depth_files[0]))
    processed_H, processed_W = depth_map_0.shape
    
    # Compute scale factors from original to processed resolution
    scale_x = processed_W / orig_W
    scale_y = processed_H / orig_H
    logger.info(f"Resolution: original ({orig_W}x{orig_H}) -> processed ({processed_W}x{processed_H}), scale=({scale_x:.3f}, {scale_y:.3f})")
    
    # Pre-load all depth maps
    depth_maps = []
    for t in range(T):
        depth_map = np.load(str(depth_files[t]))
        depth_maps.append(torch.from_numpy(depth_map).to("cuda"))
    
    # Pre-load all images for color sampling
    images = []
    for t in range(T):
        img = np.array(Image.open(image_files[t]))
        images.append(img)
    
    # Sample depths at control point locations for all frames
    all_depths = torch.zeros((T, N_grid), device="cuda", dtype=torch.float32)
    visibility_torch = visibility.to("cuda")  # (T, N_grid)
    
    logger.info("Sampling depths for all frames...")
    for t in range(T):
        depth_map_torch = depth_maps[t]
        H, W = depth_map_torch.shape
        
        # Get 2D positions for this frame (in original resolution)
        points_2d = control_points_2d[t]  # (N_grid, 2)
        
        # Scale control point coordinates from original to processed resolution
        points_2d_scaled_x = points_2d[:, 0] * scale_x
        points_2d_scaled_y = points_2d[:, 1] * scale_y
        
        # Sample depth at control point locations (nearest neighbor) in processed resolution
        x_coords = torch.clamp(points_2d_scaled_x.int(), 0, W - 1)
        y_coords = torch.clamp(points_2d_scaled_y.int(), 0, H - 1)
        depths = depth_map_torch[y_coords, x_coords]  # (N_grid,)
        
        # Only use depth if point is visible and depth is valid
        visible_and_valid = visibility_torch[t] & (depths > 0)
        all_depths[t] = torch.where(visible_and_valid, depths, 0.0)
    
    # Helper functions for 3D operations
    def unproject_to_world(points_2d: torch.Tensor, depths: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """Unproject 2D points to 3D world coordinates."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        u_coords = points_2d[:, 0].float()
        v_coords = points_2d[:, 1].float()
        
        x_cam = (u_coords - cx) / fx * depths
        y_cam = (v_coords - cy) / fy * depths
        z_cam = depths
        
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        R = c2w[:3, :3]
        trans = c2w[:3, 3]
        points_world = points_cam @ R.T + trans
        return points_world
    
    def project_to_depth(world_points: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
        """Project world points to camera and return depths (z in camera space)."""
        R = w2c[:3, :3]
        trans = w2c[:3, 3]
        points_cam = world_points @ R.T + trans
        return points_cam[:, 2]  # Return z (depth)
    
    # Initialize storage for camera-aware propagation
    # Store world positions instead of raw depths for correct reprojection
    #  Using the raw depth values would be wrong if the camera moves inbetween frames
    last_valid_world_pos = torch.zeros((N_grid, 3), device="cuda", dtype=torch.float32)
    last_valid_colors = torch.zeros((N_grid, 3), device="cuda", dtype=torch.uint8)
    has_valid_forward = torch.zeros(N_grid, device="cuda", dtype=torch.bool)

    # We want to kill points that have large depth jumps between frames; otherwise, tracked points with minimal errors
    #  that slip onto another surface (especially the tool), will be incorrectly lifted onto that (tool) surface.
    #  We observe that this causes a flickering of those points along the z axis, switching back and forth between surfaces.
    killed_points = torch.zeros(N_grid, device="cuda", dtype=torch.bool)
    
    # Output arrays
    filled_depths = torch.zeros((T, N_grid), device="cuda", dtype=torch.float32)
    filled_colors = torch.zeros((T, N_grid, 3), device="cuda", dtype=torch.uint8)
    was_filled = torch.zeros((T, N_grid), device="cuda", dtype=torch.bool)  # Track which were forward/backward filled
    
    # Create initial valid mask (visible AND depth > 0)
    valid_mask = visibility_torch & (all_depths > 0)  # (T, N_grid)
    
    logger.info("Forward-filling with camera-aware depth propagation, depth jump detection, and color forwarding...")
    killed_count = 0
    
    for t in range(T):
        K = K_tensors[t]
        c2w = c2w_tensors[t]
        w2c = w2c_tensors[t]
        points_2d = control_points_2d[t]
        sampled_depth = all_depths[t]
        
        # Sample colors for this frame
        img = images[t]
        img_H, img_W = img.shape[:2]
        x_img = torch.clamp(points_2d[:, 0].long(), 0, img_W - 1).cpu().numpy()
        y_img = torch.clamp(points_2d[:, 1].long(), 0, img_H - 1).cpu().numpy()
        current_colors = torch.from_numpy(img[y_img, x_img]).to("cuda")  # (N_grid, 3)
        
        # For points with valid depth, check for depth jumps (Phase 2)
        valid_at_t = valid_mask[t] & ~killed_points  # (N_grid,)
        
        if depth_jump_threshold is not None and t > 0:
            # For points that were valid before and are valid now, check for depth jumps
            can_check_jump = has_valid_forward & valid_at_t
            
            if can_check_jump.any():
                # Compute expected depth by reprojecting last valid world position (previous frame)
                expected_depth = project_to_depth(last_valid_world_pos, w2c)
                # Compare against actual depth of the current frame
                depth_diff = torch.abs(sampled_depth - expected_depth)
                
                # Kill points where depth jump exceeds threshold
                jumped = can_check_jump & (depth_diff > depth_jump_threshold)
                newly_killed = jumped.sum().item()
                if newly_killed > 0:
                    killed_points = killed_points | jumped
                    killed_count += newly_killed
                    valid_at_t = valid_at_t & ~jumped  # Remove jumped points from valid
        
        # For valid points: compute world position, store it, use sampled color
        if valid_at_t.any():
            world_pos = unproject_to_world(points_2d, sampled_depth, K, c2w)
            last_valid_world_pos = torch.where(valid_at_t.unsqueeze(-1), world_pos, last_valid_world_pos)
            last_valid_colors = torch.where(valid_at_t.unsqueeze(-1), current_colors, last_valid_colors)
            has_valid_forward = has_valid_forward | valid_at_t
            
            # Either assign current depth and color or continue with the last observed valid depth and color
            filled_depths[t] = torch.where(valid_at_t, sampled_depth, filled_depths[t])
            filled_colors[t] = torch.where(valid_at_t.unsqueeze(-1), current_colors, filled_colors[t])
        
        # For points needing fill (invalid but have been seen before): reproject world position
        needs_fill = ~valid_at_t & has_valid_forward & ~killed_points
        if needs_fill.any():
            reprojected_depth = project_to_depth(last_valid_world_pos, w2c)
            # Either keep depth and color that was previously computed or assign reprojected depth and prev color
            filled_depths[t] = torch.where(needs_fill, reprojected_depth, filled_depths[t])
            filled_colors[t] = torch.where(needs_fill.unsqueeze(-1), last_valid_colors, filled_colors[t])
            was_filled[t] = was_filled[t] | needs_fill
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} points due to depth jumps (threshold={depth_jump_threshold}m)")
    
    # Backward-fill: for points that were never seen valid in forward pass
    # These need depth from the first future frame where they become valid
    logger.info("Backward-filling remaining points...")
    next_valid_world_pos = torch.zeros((N_grid, 3), device="cuda", dtype=torch.float32)
    next_valid_colors = torch.zeros((N_grid, 3), device="cuda", dtype=torch.uint8)
    has_valid_backward = torch.zeros(N_grid, device="cuda", dtype=torch.bool)
    
    # Starting from the back
    for t in range(T - 1, -1, -1):
        K = K_tensors[t]
        c2w = c2w_tensors[t]
        w2c = w2c_tensors[t]
        points_2d = control_points_2d[t]
        sampled_depth = all_depths[t]
        
        # Get colors for this frame (already computed in forward pass, but need for backward valid points)
        img = images[t]
        img_H, img_W = img.shape[:2]
        x_img = torch.clamp(points_2d[:, 0].long(), 0, img_W - 1).cpu().numpy()
        y_img = torch.clamp(points_2d[:, 1].long(), 0, img_H - 1).cpu().numpy()
        current_colors = torch.from_numpy(img[y_img, x_img]).to("cuda")
        
        valid_at_t = valid_mask[t] & ~killed_points
        
        # Update next valid world position for valid points
        if valid_at_t.any():
            world_pos = unproject_to_world(points_2d, sampled_depth, K, c2w)
            next_valid_world_pos = torch.where(valid_at_t.unsqueeze(-1), world_pos, next_valid_world_pos)
            next_valid_colors = torch.where(valid_at_t.unsqueeze(-1), current_colors, next_valid_colors)
            has_valid_backward = has_valid_backward | valid_at_t
        
        # For points that still need fill (weren't filled in forward pass)
        still_needs_fill = (filled_depths[t] == 0) & has_valid_backward & ~killed_points
        if still_needs_fill.any():
            reprojected_depth = project_to_depth(next_valid_world_pos, w2c)
            reprojected_depth = torch.clamp(reprojected_depth, min=0.001)
            
            filled_depths[t] = torch.where(still_needs_fill, reprojected_depth, filled_depths[t])
            filled_colors[t] = torch.where(still_needs_fill.unsqueeze(-1), next_valid_colors, filled_colors[t])
            was_filled[t] = was_filled[t] | still_needs_fill
    
    # Determine permanently valid points: valid depth at ALL frames and not killed
    permanently_valid = (filled_depths > 0).all(dim=0) & ~killed_points  # (N_grid,)
    valid_indices = permanently_valid.nonzero(as_tuple=True)[0]  # (N_valid,)
    N_valid = valid_indices.shape[0]
    
    logger.info(f"Filtered {N_grid} control points to {N_valid} permanently valid points "
                f"({N_grid - N_valid} removed: {killed_count} killed, {N_grid - N_valid - killed_count} never valid)")
    
    # Filter to keep only permanently valid points
    filled_depths_filtered = filled_depths[:, valid_indices]  # (T, N_valid)
    filled_colors_filtered = filled_colors[:, valid_indices]  # (T, N_valid, 3)
    control_points_2d_filtered = control_points_2d[:, valid_indices]  # (T, N_valid, 2)
    
    # Final pass: compute world coordinates for valid points only
    control_points_3d = torch.zeros((T, N_valid, 3), device="cuda", dtype=torch.float32)
    
    # Initialize rerun if saving
    if save_dir is not None:
        rr.init("cotracker3_visualization")
        rr.save(save_dir / "cotracker3_visualization.rrd")
    
    for t in range(T):
        K = K_tensors[t]
        c2w = c2w_tensors[t]
        points_2d = control_points_2d_filtered[t]  # (N_valid, 2)
        depths = filled_depths_filtered[t]  # (N_valid,)
        
        # Compute world positions
        world_pos = unproject_to_world(points_2d, depths, K, c2w)
        control_points_3d[t] = world_pos
        
        # Logging for visualization
        if save_dir is not None:
            rr.set_time_sequence("frame", t)
            rr.log("world/depth_map", rr.Image(depth_maps[t].cpu()))
            
            colors = filled_colors_filtered[t].cpu().numpy()
            
            rr.log("world/control_points", rr.Points3D(
                positions=control_points_3d[t].cpu(),
                colors=colors,
            ))
    
    logger.info("Successfully converted control points to 3D world coordinates")

    return control_points_3d, control_points_2d_filtered


def _compute_processed_resolution(orig_w: int, orig_h: int, patch_size: int = 14) -> Tuple[int, int]:
    """
    Compute processed resolution from original resolution using DA3's processing logic.
    
    Matches the logic in InputProcessor._process_one with upper_bound_resize:
    1. processing_res = max(orig_w, orig_h)
    2. Resize to processing_res (preserving aspect ratio)
    3. Round to nearest multiple of patch_size
    
    Args:
        orig_w: Original width
        orig_h: Original height
        patch_size: Patch size (14 for DA3)
    
    Returns:
        Tuple of (processed_w, processed_h)
    """
    processing_res = max(orig_w, orig_h)
    
    # Resize to processing_res preserving aspect ratio (upper_bound_resize)
    if orig_w >= orig_h:
        w_resized = processing_res
        h_resized = int(round(orig_h * (processing_res / orig_w)))
    else:
        h_resized = processing_res
        w_resized = int(round(orig_w * (processing_res / orig_h)))
    
    # Round to nearest multiple of patch_size
    def nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down
    
    processed_w = max(1, nearest_multiple(w_resized, patch_size))
    processed_h = max(1, nearest_multiple(h_resized, patch_size))
    
    return processed_w, processed_h


def compute_gaussian_control_point_associations(
    control_points_2d_init: torch.Tensor,
    control_points_3d_init: torch.Tensor,
    depth_processed_dir: Path,
    colmap_dir: Path,
    image_files: List[Path],
    init_frame_idx: int,
    instance_mask: Optional[torch.Tensor],
    k_neighbors: int = 4,
    idw_power: float = 2.0,
    pixel_stride: int = 1,
    confidence_dir: Optional[Path] = None,
    conf_thresh_percentile: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute associations between Gaussians (from depth map) and control points using GPU-accelerated KNN in 3D.
    
    Args:
        control_points_2d_init: (N_points, 2) 2D control points at init frame (for instance mask lookups)
        control_points_3d_init: (N_points, 3) 3D control points at init frame (for 3D distance computation)
        depth_processed_dir: Directory containing depth maps at processed resolution
        colmap_dir: Directory containing sparse/0/ Colmap data with camera parameters
        image_files: List of image file paths (sorted by frame number)
        init_frame_idx: Index of initialization frame
        instance_mask: (H, W) instance mask tensor (0=background, >0=instance IDs) on GPU
        k_neighbors: Number of nearest neighbors for association
        idw_power: Power parameter for inverse distance weighting
        pixel_stride: Stride for pixel subsampling
        confidence_dir: Optional directory containing confidence maps
        conf_thresh_percentile: Confidence threshold percentile (0.0 = minimum, matches da3_to_single_view_colmap)
    
    Returns:
        Dictionary with:
            - indices: (N_pixels, K) indices of associated control points
            - weights: (N_pixels, K) IDW weights
            - pixel_to_gaussian_map: (H, W) mapping from pixel to Gaussian index
            - instance_ids: (N_pixels,) instance ID per Gaussian (-1 for background if no mask)
    """
    device = control_points_3d_init.device
    
    # Load depth map for init frame (already at processed resolution)
    depth_files = sorted(depth_processed_dir.glob("*.npy"))
    depth_map = np.load(str(depth_files[init_frame_idx]))  # (H, W) at processed resolution
    processed_h, processed_w = depth_map.shape
    
    # Get original image resolution for coordinate scaling (COLMAP intrinsics are in original resolution)
    first_image = Image.open(image_files[0])
    W_original, H_original = first_image.size
    
    logger.info(f"Depth map at processed resolution ({processed_w}x{processed_h}), images at original ({W_original}x{H_original})")
    
    # Compute confidence threshold if confidence maps are available
    conf_thresh = None
    if confidence_dir is not None and confidence_dir.exists():
        # Load all confidence maps to compute percentile (matching da3_to_single_view_colmap)
        conf_files = sorted(confidence_dir.glob("*.npy"))
        if len(conf_files) > 0:
            all_conf = np.stack([np.load(str(cf)) for cf in conf_files])  # (T, H, W)
            print(f"max conf: {np.max(all_conf)}")
            print(f"min conf: {np.min(all_conf)}")
            conf_thresh = np.percentile(all_conf, conf_thresh_percentile)
            logger.info(f"Computed confidence threshold: {conf_thresh} (percentile={conf_thresh_percentile})")
    
    # Resize instance mask to processed resolution to match depth map
    if instance_mask is not None:
        # Instance mask comes in as torch tensor, convert to numpy for resizing
        instance_mask_np = instance_mask.cpu().numpy() if isinstance(instance_mask, torch.Tensor) else instance_mask
        H_orig_mask, W_orig_mask = instance_mask_np.shape
        # Resize to processed resolution (same as depth map)
        if (processed_w, processed_h) != (W_orig_mask, H_orig_mask):
            instance_mask_np = cv2.resize(
                instance_mask_np.astype(np.float32), (processed_w, processed_h), 
                interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve integer instance IDs
            ).astype(instance_mask_np.dtype)
            logger.info(f"Resized instance mask from ({W_orig_mask}, {H_orig_mask}) to processed resolution ({processed_w}, {processed_h})")
        # Convert back to torch if it was originally torch
        if isinstance(instance_mask, torch.Tensor):
            instance_mask = torch.from_numpy(instance_mask_np).to(instance_mask.device)
        else:
            instance_mask = instance_mask_np
    
    # Apply pixel stride subsampling to match Colmap point cloud creation (da3_to_single_view_colmap)
    if pixel_stride > 1:
        depth_map = depth_map[::pixel_stride, ::pixel_stride]
        if instance_mask is not None:
            if isinstance(instance_mask, torch.Tensor):
                instance_mask = instance_mask[::pixel_stride, ::pixel_stride]
            else:
                instance_mask = instance_mask[::pixel_stride, ::pixel_stride]
    H, W = depth_map.shape
    
    # Get valid pixels (non-zero depth)
    # Match the exact ordering used in _depths_to_world_points_with_colors:
    # Use meshgrid to create coordinates in row-major order, then filter
    valid_pixels = (np.isfinite(depth_map) & (depth_map > 0))
    
    # Apply confidence filtering if available (matching _depths_to_world_points_with_colors)
    if conf_thresh is not None:
        conf_files = sorted(confidence_dir.glob("*.npy"))
        if init_frame_idx < len(conf_files):
            conf_map_orig = np.load(str(conf_files[init_frame_idx]))  # (H_original, W_original) in original resolution
            # Resize confidence map to processed resolution to match depth map
            if (processed_w, processed_h) != (W_original, H_original):
                conf_map = cv2.resize(
                    conf_map_orig, (processed_w, processed_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                conf_map = conf_map_orig
            # Apply same pixel stride subsampling as depth map
            if pixel_stride > 1:
                conf_map = conf_map[::pixel_stride, ::pixel_stride]
            valid_pixels = valid_pixels & (conf_map >= conf_thresh)
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    # Flatten to row-major order (same as reshape(-1))
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    valid_flat = valid_pixels.reshape(-1)
    
    # Get indices of valid pixels (row-major order, matching _depths_to_world_points_with_colors)
    vidx = np.flatnonzero(valid_flat)
    x_coords = x_flat[vidx]
    y_coords = y_flat[vidx]
    
    # Scale coordinates back to processed resolution (before subsampling)
    x_coords_processed = x_coords * pixel_stride
    y_coords_processed = y_coords * pixel_stride
    
    # Scale coordinates from processed resolution to original image space
    if (processed_w, processed_h) != (W_original, H_original):
        # Scale from processed to original resolution
        scale_w = W_original / processed_w
        scale_h = H_original / processed_h
        x_coords_original = (x_coords_processed * scale_w).astype(np.int64)
        y_coords_original = (y_coords_processed * scale_h).astype(np.int64)
    else:
        x_coords_original = x_coords_processed.astype(np.int64)
        y_coords_original = y_coords_processed.astype(np.int64)
    
    # Clamp to original image bounds
    x_coords_original = np.clip(x_coords_original, 0, W_original - 1)
    y_coords_original = np.clip(y_coords_original, 0, H_original - 1)
    
    N_pixels = len(vidx)
    
    logger.info(f"Computing associations for {N_pixels} valid pixels (after pixel_stride={pixel_stride} subsampling, GPU-accelerated, 3D distances)...")
    
    # Load camera parameters for unprojecting pixels to 3D
    intrinsics_list, c2w_list = load_colmap_camera_parameters(colmap_dir, image_files)
    K = torch.from_numpy(intrinsics_list[init_frame_idx]).float().to(device)  # (3, 3)
    c2w = torch.from_numpy(c2w_list[init_frame_idx]).float().to(device)  # (4, 4)
    
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Get depths for valid pixels (in subsampled space, need to sample from depth_map)
    depth_map_torch = torch.from_numpy(depth_map).float().to(device)  # (H, W) subsampled
    depths = depth_map_torch[torch.from_numpy(y_coords).long().to(device), 
                             torch.from_numpy(x_coords).long().to(device)]  # (N_pixels,)
    
    # Unproject valid pixels to 3D world coordinates
    # Use original resolution coordinates for unprojection (camera intrinsics are in original space)
    u_coords = torch.from_numpy(x_coords_original).float().to(device)  # (N_pixels,)
    v_coords = torch.from_numpy(y_coords_original).float().to(device)  # (N_pixels,)
    
    # Camera space coordinates
    x_cam = (u_coords - cx) / fx * depths
    y_cam = (v_coords - cy) / fy * depths
    z_cam = depths
    
    # Stack to (N_pixels, 3) camera coordinates
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N_pixels, 3)
    
    # Transform to world coordinates
    R = c2w[:3, :3]  # (3, 3)
    trans = c2w[:3, 3]  # (3,)
    gaussian_positions_3d = points_cam @ R.T + trans  # (N_pixels, 3)
    
    # Control points are already in 3D world space
    control_points_torch = control_points_3d_init.float()  # (N_grid, 3)
    
    # Compute pairwise 3D distances using torch.cdist (GPU-accelerated)
    # cdist computes pairwise distances: (N_pixels, N_grid)
    distances_matrix = torch.cdist(gaussian_positions_3d, control_points_torch)  # (N_pixels, N_grid)
    
    # Find K nearest neighbors using topk
    distances, indices = torch.topk(distances_matrix, k=k_neighbors, dim=1, largest=False)
    # distances: (N_pixels, K), indices: (N_pixels, K)
    
    # TODO: we may want to get rid of this
    # Filter by instance mask if provided
    if instance_mask is not None:
        # Instance mask is already resized and subsampled, ensure it's on the right device and dtype
        instance_mask_torch = instance_mask.int().to(device)  # (H, W) in subsampled space
        
        # Get instance IDs for control points using their 2D pixel coordinates
        # control_points_2d_init is in original image space, need to convert to subsampled space
        control_point_coords = control_points_2d_init.long().to(device)  # (N_grid, 2) in original image space
        # First convert to processed resolution (if depth was resized)
        # Then convert to subsampled space
        if (processed_w, processed_h) != (W_original, H_original):
            # Scale from original to processed resolution
            scale_w = processed_w / W_original
            scale_h = processed_h / H_original
            control_point_coords_processed = control_point_coords.float()
            control_point_coords_processed[:, 0] = control_point_coords_processed[:, 0] * scale_w
            control_point_coords_processed[:, 1] = control_point_coords_processed[:, 1] * scale_h
            control_point_coords_processed = control_point_coords_processed.long()
        else:
            control_point_coords_processed = control_point_coords
        
        # Now convert to subsampled space
        control_point_coords_subsampled = control_point_coords_processed // pixel_stride
        control_point_coords_subsampled[:, 0] = torch.clamp(control_point_coords_subsampled[:, 0], 0, W - 1)
        control_point_coords_subsampled[:, 1] = torch.clamp(control_point_coords_subsampled[:, 1], 0, H - 1)
        control_point_instance_ids = instance_mask_torch[
            control_point_coords_subsampled[:, 1], control_point_coords_subsampled[:, 0]
        ]  # (N_grid,)
        
        # Get instance IDs for pixels (x_coords, y_coords are already in subsampled space)
        # Convert to torch and ensure they're within bounds
        y_coords_torch = torch.from_numpy(y_coords).long().to(device)
        x_coords_torch = torch.from_numpy(x_coords).long().to(device)
        y_coords_torch = torch.clamp(y_coords_torch, 0, H - 1)
        x_coords_torch = torch.clamp(x_coords_torch, 0, W - 1)
        pixel_instance_ids = instance_mask_torch[y_coords_torch, x_coords_torch]  # (N_pixels,)
        # Store for returning
        gaussian_instance_ids = pixel_instance_ids
        
        # Filter: only allow associations within same instance
        # Get instance IDs of K nearest control points for each pixel
        cp_instances_selected = control_point_instance_ids[indices]  # (N_pixels, K)
        pixel_instance_ids_expanded = pixel_instance_ids.unsqueeze(1).expand(-1, k_neighbors)  # (N_pixels, K)
        
        # Mask: same instance or background (0)
        same_instance_mask = (
            (cp_instances_selected == pixel_instance_ids_expanded) |
            (cp_instances_selected == 0) |
            (pixel_instance_ids_expanded == 0)
        )  # (N_pixels, K)
        
        # For pixels with no valid neighbors, allow all (fallback)
        has_valid = same_instance_mask.any(dim=1)  # (N_pixels,)
        same_instance_mask[~has_valid] = True
        
        # Set distances to inf for invalid neighbors and re-sort
        distances = distances.clone()
        distances[~same_instance_mask] = float('inf')
        
        # Re-sort to get valid neighbors first
        sorted_distances, sorted_idx = torch.sort(distances, dim=1)
        indices = torch.gather(indices, 1, sorted_idx)
        distances = sorted_distances
    else:
        # No instance mask: assign all Gaussians to background (-1)
        gaussian_instance_ids = torch.full((N_pixels,), -1, dtype=torch.long, device=device)
    
    # Compute IDW weights on GPU
    eps = 1e-8
    weights = 1.0 / (distances ** idw_power + eps)  # (N_pixels, K)
    
    # Normalize weights
    weights_sum = weights.sum(dim=1, keepdim=True)  # (N_pixels, 1)
    weights = weights / (weights_sum + eps)
    
    # Keep as torch tensors
    indices_torch = indices.long()  # (N_pixels, K)
    weights_torch = weights.float()  # (N_pixels, K)
    
    # Create pixel-to-Gaussian mapping (for later use, keep as torch)
    # Map is in original image space
    pixel_to_gaussian_map = torch.full((H_original, W_original), -1, dtype=torch.long, device=device)
    pixel_coords_torch_final = torch.stack([
        torch.from_numpy(x_coords_original).long().to(device),
        torch.from_numpy(y_coords_original).long().to(device)
    ], dim=-1)  # (N_pixels, 2) in original image space

    gaussian_idx = 0
    for y, x in zip(y_coords_original, x_coords_original):
        pixel_to_gaussian_map[y, x] = gaussian_idx
        gaussian_idx += 1
    
    logger.info(f"Computed associations: {N_pixels} pixels -> {k_neighbors} control points each")
    
    return {
        "indices": indices_torch,  # (N_pixels, K)
        "weights": weights_torch,  # (N_pixels, K)
        "pixel_to_gaussian_map": pixel_to_gaussian_map,  # (H_original, W_original)
        "valid_pixel_coords": pixel_coords_torch_final,  # (N_pixels, 2) in original image space
        "instance_ids": gaussian_instance_ids,  # (N_pixels,) instance ID per Gaussian
    }

