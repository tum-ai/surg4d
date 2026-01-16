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
    grid_size: int = 32,
    save_dir: Path = None,
) -> torch.Tensor:
    """
    Track control points across video using CoTracker3.
    
    Args:
        image_files: List of image file paths
        grid_size: Size of control point grid (grid_size x grid_size)
        save_dir: Directory to save visualizations
    
    Returns:
        control_points_2d: (T, N_grid, 2) tensor of 2D pixel coordinates on GPU
            where N_grid = grid_size * grid_size
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

    print(f"shape of images_torch: {images_torch.shape}")
    n_frames = images_torch.shape[1]
    
    with torch.no_grad():
        # each track: (B, T, N, 2) - (batch, time, points, xy)
        # each visibility: (B, T, N) - (batch, time, points)
        tracks_beginning, visibility_beginning = predictor(images_torch, grid_size=grid_size, grid_query_frame=0, backward_tracking=False)
        tracks_end, visibility_end = predictor(images_torch, grid_size=grid_size, grid_query_frame=n_frames - 1, backward_tracking=True)
        tracks_middle, visibility_middle = predictor(images_torch, grid_size=grid_size, grid_query_frame=(n_frames // 2), backward_tracking=True)
        tracks = torch.cat([tracks_beginning, tracks_middle, tracks_end], dim=2)
        visibility = torch.cat([visibility_beginning, visibility_middle, visibility_end], dim=2)

    print(f"shape of tracks: {tracks.shape}")

    # Visualize
    visualizer = Visualizer(save_dir=save_dir, pad_value=25)
    visualizer.visualize(video= images_torch * 255.0, tracks=tracks, visibility=visibility, filename="cotracker3_visualization")
    
    return tracks.squeeze(0)


def lift_control_points_to_3d(
    control_points_2d: torch.Tensor,
    depth_dir: Path,
    colmap_dir: Path,
    image_files: List[Path],
    grid_size: int = 32,
    save_dir: Path = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lift 2D control points to 3D world coordinates using DA3 depth maps and camera parameters.
    
    Args:
        control_points_2d: (T, N_grid, 2) 2D pixel coordinates (torch.Tensor on GPU)
        depth_dir: Directory containing depth maps (*.npy files)
        colmap_dir: Directory containing sparse/0/ Colmap data with camera parameters
        image_files: List of image file paths (sorted by frame number)
        grid_size: Size of control point grid
        save_dir: Directory to save visualizations
    
    Returns:
        control_points_3d: (T, N_grid, 3) 3D coordinates in world space
        validity: (T, N_grid) boolean array indicating valid points
    """
    T, N_grid, _ = control_points_2d.shape
    
    # Load depth maps
    depth_files = sorted(depth_dir.glob("*.npy"))
    assert len(depth_files) == T, f"Mismatch: {len(depth_files)} depth files, {T} frames"
    
    # Load camera parameters
    intrinsics_list, c2w_list = load_colmap_camera_parameters(colmap_dir, image_files)
    assert len(intrinsics_list) == T, f"Mismatch: {len(intrinsics_list)} camera parameters, {T} frames"
    
    control_points_3d = torch.zeros((T, N_grid, 3), device="cuda", dtype=torch.float32)
    validity = torch.zeros((T, N_grid), device="cuda", dtype=torch.bool)
    
    logger.info(f"Lifting {N_grid} control points to 3D world coordinates for {T} frames...")

    if save_dir is not None:
        rr.init("cotracker3_visualization")
        rr.save(save_dir / "cotracker3_visualization.rrd")
    
    for t in range(T):
        depth_map = np.load(str(depth_files[t]))  # (H, W)
        depth_map_torch = torch.from_numpy(depth_map).to("cuda")
        H, W = depth_map_torch.shape
        
        # Get 2D positions for this frame
        points_2d = control_points_2d[t]  # (N_grid, 2)
        
        # Sample depth at control point locations (nearest neighbor)
        x_coords = torch.clamp(points_2d[:, 0].int(), 0, W - 1)
        y_coords = torch.clamp(points_2d[:, 1].int(), 0, H - 1)
        depths = depth_map_torch[y_coords, x_coords]  # (N_grid,)
        
        # Mark valid points (non-zero depth)
        valid_mask = depths > 0
        validity[t] = valid_mask.to(torch.bool)
        
        # 3D unprojection using camera parameters
        K = torch.from_numpy(intrinsics_list[t]).float().to("cuda")  # (3, 3)
        c2w = torch.from_numpy(c2w_list[t]).float().to("cuda")  # (4, 4)
        
        # Extract intrinsic parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Convert pixel coordinates to camera space
        u_coords = points_2d[:, 0].float().to("cuda")  # (N_grid,)
        v_coords = points_2d[:, 1].float().to("cuda")  # (N_grid,)
        
        x_cam = (u_coords - cx) / fx * depths
        y_cam = (v_coords - cy) / fy * depths
        z_cam = depths
        
        # Stack to (N_grid, 3) camera coordinates
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N_grid, 3)
        
        # Transform to world coordinates
        R = c2w[:3, :3]  # (3, 3)
        trans = c2w[:3, 3]   # (3,)
        points_world = points_cam @ R.T + trans  # (N_grid, 3)
        
        print(f"t: {t}, points_world: {points_world[:10, :]}")
        print(f"type of t: {type(t)}")
        control_points_3d[t] = points_world
        
        # Logging for visualization
        if save_dir is not None:
            rr.set_time_sequence("frame", t)
            rr.log("world/depth_map", rr.Image(depth_map_torch.cpu()))
            rr.log("world/control_points", rr.Points3D(control_points_3d[t].cpu()))
    
    logger.info("Successfully converted control points to 3D world coordinates")

    return control_points_3d, validity


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
    depth_dir: Path,
    init_frame_idx: int,
    instance_mask: Optional[torch.Tensor],
    k_neighbors: int = 4,
    grid_size: int = 32,
    idw_power: float = 2.0,
    pixel_stride: int = 1,
    confidence_dir: Optional[Path] = None,
    conf_thresh_percentile: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute associations between Gaussians (from depth map) and control points using GPU-accelerated KNN.
    
    Args:
        control_points_2d_init: (N_grid, 2) 2D control points at init frame (torch.Tensor on GPU)
        depth_dir: Directory containing depth maps
        init_frame_idx: Index of initialization frame
        instance_mask: (H, W) instance mask tensor (0=background, >0=instance IDs) on GPU
        k_neighbors: Number of nearest neighbors for association
        grid_size: Size of control point grid
        idw_power: Power parameter for inverse distance weighting
        pixel_stride: Stride for pixel subsampling
        confidence_dir: Optional directory containing confidence maps
        conf_thresh_percentile: Confidence threshold percentile (0.0 = minimum, matches da3_to_single_view_colmap)
    
    Returns:
        Dictionary with:
            - indices: (N_pixels, K) indices of associated control points
            - weights: (N_pixels, K) IDW weights
            - pixel_to_gaussian_map: (H, W) mapping from pixel to Gaussian index
    """
    device = control_points_2d_init.device
    
    # Load depth map for init frame
    depth_files = sorted(depth_dir.glob("*.npy"))
    depth_map_orig = np.load(str(depth_files[init_frame_idx]))  # (H, W) in original resolution
    H_original, W_original = depth_map_orig.shape
    
    # Resize depth map to processed resolution to match da3_to_single_view_colmap
    # (da3_to_single_view_colmap uses prediction.depth which is in processed resolution)
    processed_w, processed_h = _compute_processed_resolution(W_original, H_original)
    if (processed_w, processed_h) != (W_original, H_original):
        depth_map = cv2.resize(
            depth_map_orig, (processed_w, processed_h), interpolation=cv2.INTER_LINEAR
        )
        logger.info(f"Resized depth map from ({W_original}, {H_original}) to processed resolution ({processed_w}, {processed_h})")
    else:
        depth_map = depth_map_orig
    
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
                    conf_map_orig, (processed_w, processed_h), interpolation=cv2.INTER_LINEAR
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
    
    logger.info(f"Computing associations for {N_pixels} valid pixels (after pixel_stride={pixel_stride} subsampling, GPU-accelerated)...")
    
    # Control points are already on GPU, ensure they're float
    # Control points are in original image space, so use original coordinates for KNN
    control_points_torch = control_points_2d_init.float()  # (N_grid, 2)
    pixel_coords_torch = torch.stack([
        torch.from_numpy(x_coords_original).float().to(device),
        torch.from_numpy(y_coords_original).float().to(device)
    ], dim=-1)  # (N_pixels, 2) in original image space
    
    # Compute pairwise distances using torch.cdist (GPU-accelerated)
    # cdist computes pairwise distances: (N_pixels, N_grid)
    distances_matrix = torch.cdist(pixel_coords_torch, control_points_torch)  # (N_pixels, N_grid)
    
    # Find K nearest neighbors using topk
    distances, indices = torch.topk(distances_matrix, k=k_neighbors, dim=1, largest=False)
    # distances: (N_pixels, K), indices: (N_pixels, K)
    
    # Filter by instance mask if provided
    if instance_mask is not None:
        # Instance mask is already resized and subsampled, ensure it's on the right device and dtype
        instance_mask_torch = instance_mask.int().to(device)  # (H, W) in subsampled space
        
        # Get instance IDs for control points
        # Control points are in original image space, need to convert to subsampled space
        control_point_coords = control_points_torch.long()  # (N_grid, 2) in original image space
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
    }

