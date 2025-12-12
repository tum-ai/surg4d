import numpy as np
import pycolmap
import rerun as rr
from PIL import Image
from pathlib import Path

from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from depth_anything_3.utils.export.colmap import _create_xyf


def da3_to_single_view_colmap(
    prediction,
    export_dir: str,
    image_paths: list[str],
    single_view_idx: int,
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
):
    """
    Export COLMAP format with points only from a single view.
    All cameras/images are still added, but only the selected view has point observations.
    """
    # 1. Data preparation - only process the single view
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    # Extract only the single view's data
    single_depth = prediction.depth[single_view_idx : single_view_idx + 1]  # Keep dims
    single_intrinsics = prediction.intrinsics[single_view_idx : single_view_idx + 1]
    single_extrinsics = prediction.extrinsics[single_view_idx : single_view_idx + 1]
    single_images = prediction.processed_images[single_view_idx : single_view_idx + 1]
    single_conf = prediction.conf[single_view_idx : single_view_idx + 1]

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

    # Create xyf mapping only for the single view
    # Reshape to match the filtering used in _depths_to_world_points_with_colors
    points_xyf_single = _create_xyf(1, h, w)  # Shape: (1, h, w, 3)
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
                point2d[0] *= orig_w / w
                point2d[1] *= orig_h / h
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