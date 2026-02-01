import numpy as np
import pycolmap
import rerun as rr
from PIL import Image
from pathlib import Path
from typing import List
import cv2

from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from depth_anything_3.utils.export.colmap import _create_xyf
from depth_anything_3.specs import Prediction


def filter_depth_edge_artifacts(
    depth: np.ndarray,
    gradient_threshold: float = 0.05,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Filter edge floaters in depth maps by masking pixels with high depth gradients.
    
    Edge floaters occur when depth changes drastically at object boundaries, causing
    intermediate depth values between foreground and background. This function detects
    such regions by computing depth gradients and masking pixels where the absolute
    gradient exceeds a threshold.
    
    Args:
        depth: Depth map of shape (H, W) in meters (DA3 metric output)
        gradient_threshold: Absolute gradient threshold in meters per pixel.
            Pixels with gradient above this are masked. Default 0.05 means
            depth changes > 5cm per pixel are filtered.
        kernel_size: Size of Sobel kernel for gradient computation (must be 1, 3, 5, or 7)
    
    Returns:
        Filtered depth map of same shape, with edge artifacts set to 0 (invalid)
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    
    # Compute depth gradients using Sobel operators (meters per pixel)
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=kernel_size)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Mask pixels where absolute gradient exceeds threshold
    # These are likely edge floaters (sharp depth discontinuities)
    valid_mask = grad_magnitude <= gradient_threshold
    
    # Preserve original valid depth pixels (depth > 0)
    valid_mask = valid_mask & (depth > 0)
    
    # Apply mask: set invalid pixels to 0
    filtered_depth = depth.copy()
    filtered_depth[~valid_mask] = 0.0
    
    return filtered_depth


def filter_prediction_edge_artifacts(
    prediction: Prediction,
    gradient_threshold: float,
) -> Prediction:
    """
    Filter edge artifacts from a DA3 prediction object by modifying depth maps in-place.
    
    This function applies edge filtering to all depth maps in the prediction at processed
    resolution. The filtered prediction can then be used for point cloud generation and
    saving to disk (where it will just be resized).
    
    Args:
        prediction: DA3 Prediction object with depth maps at processed resolution
        gradient_threshold: Relative gradient threshold for filtering edge floaters
    
    Returns:
        The same prediction object with filtered depth maps (modified in-place)
    """
    num_frames = len(prediction.depth)
    
    for frame_idx in range(num_frames):
        depth_2d = prediction.depth[frame_idx]  # (H, W)
        depth_filtered = filter_depth_edge_artifacts(
            depth_2d,
            gradient_threshold=gradient_threshold,
        )
        # Modify in-place
        prediction.depth[frame_idx] = depth_filtered
    
    return prediction


def da3_to_multi_view_colmap(
    prediction,
    export_dir: str,
    image_paths: list[str],
    view_indices: List[int],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
    pixel_stride: int = 1,
    densify_ratio: int = 1,
):
    """
    Export COLMAP format with points from multiple views (e.g., first, middle, last).
    All cameras/images are still added, but only the selected views have point observations.
    Points from different views are simply concatenated (may have duplicates in overlapping regions).
    
    Args:
        prediction: DepthAnything3 prediction object
        export_dir: Directory to export COLMAP files
        image_paths: List of image file paths
        view_indices: List of frame indices to use for point cloud (e.g., [0, T//2, T-1])
        conf_thresh_percentile: Confidence threshold percentile for filtering
        pixel_stride: Stride for subsampling pixels (e.g., 3 = every 3rd pixel = 1/9 points)
        densify_ratio: Densification ratio for point cloud (1 = no densification, 5 = 5x more points, ...)
    """
    # 1. Data preparation - process each selected view
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]

    # Collect points from all selected views
    all_points = []
    all_colors = []
    all_points_xyf = []  # For 2D observations
    view_point_counts = []  # Track how many points come from each view

    # Write colored point clouds to rerun
    rr.init("depth_anything_3_visualization")
    rr.save(export_dir / "da3_viz.rrd")

    for view_idx in view_indices:
        # Extract this view's data
        view_depth = prediction.depth[view_idx : view_idx + 1]
        view_intrinsics = prediction.intrinsics[view_idx : view_idx + 1]
        view_extrinsics = prediction.extrinsics[view_idx : view_idx + 1]
        view_images = prediction.processed_images[view_idx : view_idx + 1]
        view_conf = prediction.conf[view_idx : view_idx + 1]

        # Apply pixel stride subsampling if requested
        if pixel_stride > 1:
            view_depth = view_depth[:, ::pixel_stride, ::pixel_stride]
            view_conf = view_conf[:, ::pixel_stride, ::pixel_stride]
            view_images = view_images[:, ::pixel_stride, ::pixel_stride, :]
            
            # Adjust intrinsics for subsampled resolution
            view_intrinsics = view_intrinsics.copy()
            view_intrinsics[:, 0, 0] /= pixel_stride  # fx
            view_intrinsics[:, 1, 1] /= pixel_stride  # fy
            view_intrinsics[:, 0, 2] /= pixel_stride  # cx
            view_intrinsics[:, 1, 2] /= pixel_stride  # cy

        # Get points from this view
        points, colors = _depths_to_world_points_with_colors(
            view_depth,
            view_intrinsics,
            view_extrinsics,
            view_images,
            view_conf,
            conf_thresh,
        )

        rr.set_time_sequence("frame", view_idx)

        rr.log("world/original_points", rr.Points3D(
            positions=points,
            colors=colors,
            radii=0.001,
        ))

        num_points_init = len(points)
        num_points_to_add = num_points_init * (densify_ratio - 1)
        print(f"adding {num_points_to_add} points (densify_ratio={densify_ratio})")
        # TODO: this is fine for now but should be dependent on scene extent!
        random_noise = np.random.randn(num_points_to_add, 3) * 0.01
        # Tile copies blockwise, so [p0, p1, p2] -> [p0, p1, p2, p0, p1, p2, ...]
        random_points = np.tile(points, (densify_ratio - 1, 1)) + random_noise
        additional_colors = np.tile(colors, (densify_ratio - 1, 1))
        rr.log("world/additional_points", rr.Points3D(
            positions=random_points,
            colors=additional_colors,
            radii=0.001,
        ))

        points = np.concatenate([points, random_points], axis=0)
        colors = np.tile(colors, (densify_ratio, 1))

        rr.log("world/final_points", rr.Points3D(
            positions=points,
            colors=colors,
            radii=0.001,
        ))
        
        # Get the actual height and width after subsampling
        h_subsampled, w_subsampled = view_depth.shape[1:3]

        # Create xyf mapping for this view
        points_xyf = _create_xyf(1, h_subsampled, w_subsampled).reshape(-1, 3)
        # Filter by confidence (same as in _depths_to_world_points_with_colors)
        valid_mask = (
            (view_conf.reshape(-1) >= conf_thresh)
            & np.isfinite(view_depth.reshape(-1))
            & (view_depth.reshape(-1) > 0)
        )
        points_xyf = points_xyf[valid_mask]

        # Duplicate xyf to match densified points
        # TODO: this might become problematic if noisy points are mapping to different pixels
        points_xyf = np.tile(points_xyf, (densify_ratio, 1))

        all_points.append(points)
        all_colors.append(colors)
        all_points_xyf.append(points_xyf)
        view_point_counts.append(len(points))

    # Concatenate all points
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    num_points = len(all_points)

    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add all cameras first
    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic = intrinsic.copy()
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h
        elif process_res_method == "crop":
            raise NotImplementedError("COLMAP export for crop method is not implemented")
        else:
            raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

    # Add 3D points first so we can reference them
    point3d_ids = []
    for vidx in range(num_points):
        track = pycolmap.Track()
        point3d_id = reconstruction.add_point3D(all_points[vidx], track, all_colors[vidx])
        point3d_ids.append(point3d_id)

    # Now add images with point2d observations
    point_offset = 0
    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic = intrinsic.copy()
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3]
        )

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = fidx + 1

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = fidx + 1
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # Check if this frame is one of the selected views
        if fidx in view_indices:
            view_list_idx = view_indices.index(fidx)
            points_xyf_view = all_points_xyf[view_list_idx]
            num_points_view = view_point_counts[view_list_idx]
            
            # Calculate the offset into all_points for this view
            view_point_offset = sum(view_point_counts[:view_list_idx])
            
            # Get subsampled dimensions for this view
            if pixel_stride > 1:
                h_subsampled = prediction.depth.shape[1] // pixel_stride
                w_subsampled = prediction.depth.shape[2] // pixel_stride
            else:
                h_subsampled, w_subsampled = h, w
            
            point2d_list = []
            for vidx in range(num_points_view):
                point2d = points_xyf_view[vidx][:2].copy()
                # Scale from subsampled resolution to original resolution
                point2d[0] *= (w / w_subsampled) * (orig_w / w)
                point2d[1] *= (h / h_subsampled) * (orig_h / h)
                point3d_id = point3d_ids[view_point_offset + vidx]
                point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
                # Update the track for this point
                reconstruction.point3D(point3d_id).track.add_element(
                    image.image_id, len(point2d_list) - 1
                )
            image.points2D = pycolmap.Point2DList(point2d_list)
        else:
            # Empty point2d list for other views
            image.points2D = pycolmap.Point2DList([])

        # set and add image
        image.frame_id = image.image_id
        image.name = Path(image_paths[fidx]).name
        reconstruction.add_image(image)

    # 3. Export
    reconstruction.write(export_dir)
    
    return view_point_counts  # Return counts for downstream use


def da3_to_single_view_colmap(
    prediction,
    export_dir: str,
    image_paths: list[str],
    single_view_idx: int,
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
    pixel_stride: int = 1,
):
    """
    Export COLMAP format with points only from a single view.
    All cameras/images are still added, but only the selected view has point observations.
    
    Args:
        pixel_stride: Stride for subsampling pixels (e.g., 3 = every 3rd pixel = 1/9 points)
    """
    # 1. Data preparation - only process the single view
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    # Extract only the single view's data
    single_depth = prediction.depth[single_view_idx : single_view_idx + 1]  # Keep dims
    single_intrinsics = prediction.intrinsics[single_view_idx : single_view_idx + 1]
    single_extrinsics = prediction.extrinsics[single_view_idx : single_view_idx + 1]
    single_images = prediction.processed_images[single_view_idx : single_view_idx + 1]
    single_conf = prediction.conf[single_view_idx : single_view_idx + 1]

    # Apply pixel stride subsampling if requested
    if pixel_stride > 1:
        # Subsample depth, confidence, and images
        single_depth = single_depth[:, ::pixel_stride, ::pixel_stride]
        single_conf = single_conf[:, ::pixel_stride, ::pixel_stride]
        single_images = single_images[:, ::pixel_stride, ::pixel_stride, :]
        
        # Adjust intrinsics for subsampled resolution
        single_intrinsics = single_intrinsics.copy()
        # Scale focal lengths and principal point by the stride
        single_intrinsics[:, 0, 0] /= pixel_stride  # fx
        single_intrinsics[:, 1, 1] /= pixel_stride  # fy
        single_intrinsics[:, 0, 2] /= pixel_stride  # cx
        single_intrinsics[:, 1, 2] /= pixel_stride  # cy

    # Get points only from the single view
    points, colors = _depths_to_world_points_with_colors(
        single_depth,
        single_intrinsics,
        single_extrinsics,  # w2c
        single_images,
        single_conf,
        conf_thresh,
    )
    num_points = len(points)

    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]

    # Get the actual height and width after subsampling
    h_subsampled, w_subsampled = single_depth.shape[1:3]

    # Create xyf mapping only for the single view
    # Reshape to match the filtering used in _depths_to_world_points_with_colors
    points_xyf_single = _create_xyf(1, h_subsampled, w_subsampled)  # Shape: (1, h, w, 3)
    points_xyf_single = points_xyf_single.reshape(-1, 3)  # Shape: (h*w, 3)
    # Filter by confidence (same as in _depths_to_world_points_with_colors)
    valid_mask = (
        (single_conf.reshape(-1) >= conf_thresh)
        & np.isfinite(single_depth.reshape(-1))
        & (single_depth.reshape(-1) > 0)
    )
    points_xyf_single = points_xyf_single[valid_mask]  # Shape: (num_points, 3)

    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add all cameras first
    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic = intrinsic.copy()
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h
        elif process_res_method == "crop":
            raise NotImplementedError(
                "COLMAP export for crop method is not implemented"
            )
        else:
            raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

    # Add 3D points first (only from the single view) so we can reference them
    point3d_ids = []
    for vidx in range(num_points):
        track = pycolmap.Track()
        point3d_id = reconstruction.add_point3D(points[vidx], track, colors[vidx])
        point3d_ids.append(point3d_id)

    # Now add images with point2d observations
    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic = intrinsic.copy()
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3]
        )

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = fidx + 1

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = fidx + 1
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # Only add point2d observations for the selected view
        if fidx == single_view_idx:
            point2d_list = []
            # Add all points from this view
            for vidx in range(num_points):
                point2d = points_xyf_single[vidx][:2].copy()
                # Scale from subsampled resolution to original resolution
                # First scale from subsampled to full processing resolution, then to original
                point2d[0] *= (w / w_subsampled) * (orig_w / w)
                point2d[1] *= (h / h_subsampled) * (orig_h / h)
                point3d_id = point3d_ids[vidx]
                point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
                # Update the track for this point
                reconstruction.point3D(point3d_id).track.add_element(
                    image.image_id, len(point2d_list) - 1
                )
            image.points2D = pycolmap.Point2DList(point2d_list)
        else:
            # Empty point2d list for other views
            image.points2D = pycolmap.Point2DList([])

        # set and add image
        image.frame_id = image.image_id
        image.name = Path(image_paths[fidx]).name
        reconstruction.add_image(image)

    # 3. Export
    reconstruction.write(export_dir)


def log_da3_rerun(
    prediction,
    image_paths: list[str],
    output_rrd_path: str,
    conf_thresh_percentile: float = 0.0,
    subsample_points: int = 1,  # Subsample point cloud for performance
):
    """
    Visualize cameras, intrinsics, and point cloud to Rerun file.

    Args:
        prediction: DepthAnything3 prediction object
        image_paths: List of image file paths
        output_rrd_path: Path to save .rrd file
        conf_thresh_percentile: Confidence threshold for point filtering
        subsample_points: Subsample factor for point cloud (1 = no subsampling)
    """
    # Initialize Rerun with file sink
    rr.init("depth_anything_3_visualization")
    rr.save(output_rrd_path)

    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]

    # Compute confidence threshold once
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    # Log camera trajectory as a static line (all camera positions)
    camera_centers = []
    for fidx in range(num_frames):
        ext_w2c = prediction.extrinsics[fidx]
        ext_w2c_4x4 = np.eye(4)
        ext_w2c_4x4[:3, :] = ext_w2c
        ext_c2w = np.linalg.inv(ext_w2c_4x4)
        camera_centers.append(ext_c2w[:3, 3])

    camera_centers = np.array(camera_centers)
    rr.log(
        "world/camera_trajectory",
        rr.LineStrips3D([camera_centers]),
        static=True,
    )

    # Log cameras and point clouds with timestepping
    for fidx in range(num_frames):
        # Set time for this frame
        rr.set_time_sequence("frame", fidx)

        # Get original image size
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        # Get extrinsics (w2c) and convert to c2w for Rerun
        ext_w2c = prediction.extrinsics[fidx]  # (3, 4)
        # Convert to 4x4 homogeneous
        ext_w2c_4x4 = np.eye(4)
        ext_w2c_4x4[:3, :] = ext_w2c
        # Get c2w (camera to world)
        ext_c2w = np.linalg.inv(ext_w2c_4x4)

        # Get intrinsics and scale to original image size
        K = prediction.intrinsics[fidx].copy()
        K[0, 0] *= orig_w / w  # fx
        K[1, 1] *= orig_h / h  # fy
        K[0, 2] *= orig_w / w  # cx
        K[1, 2] *= orig_h / h  # cy

        # Log camera transform and intrinsics at actual pose
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=ext_c2w[:3, 3],
                mat3x3=ext_c2w[:3, :3],
            ),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(
                resolution=[orig_w, orig_h],
                image_from_camera=K,
            ),
        )

        # Log the image
        img = prediction.processed_images[fidx]
        rr.log(
            "world/camera/image",
            rr.Image(img),
        )

        # Log camera center as a point for visibility
        rr.log(
            "world/camera/center",
            rr.Points3D(
                positions=ext_c2w[:3, 3:4].T,
                colors=[255, 0, 0],  # Red
                radii=0.01,
            ),
        )

        # Get point cloud from THIS frame's depth projection
        frame_depth = prediction.depth[fidx : fidx + 1]  # Keep dims
        frame_intrinsics = prediction.intrinsics[fidx : fidx + 1]
        frame_extrinsics = prediction.extrinsics[fidx : fidx + 1]
        frame_images = prediction.processed_images[fidx : fidx + 1]
        frame_conf = prediction.conf[fidx : fidx + 1]

        # Project depth to 3D points for this frame
        frame_points, frame_colors = _depths_to_world_points_with_colors(
            frame_depth,
            frame_intrinsics,
            frame_extrinsics,
            frame_images,
            frame_conf,
            conf_thresh,
        )

        # Subsample if requested
        if subsample_points > 1 and len(frame_points) > 0:
            indices = np.arange(0, len(frame_points), subsample_points)
            frame_points = frame_points[indices]
            frame_colors = frame_colors[indices]

        # Log point cloud for this frame
        if len(frame_points) > 0:
            rr.log(
                "world/point_cloud",
                rr.Points3D(
                    positions=frame_points,
                    colors=frame_colors,
                    radii=0.001,
                ),
            )