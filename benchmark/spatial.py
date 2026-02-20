import gc
import json
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
from typing import Dict, Any
import cv2
import re
from PIL import Image
import pycolmap

from benchmark.graph_utils import get_coord_transformations
from benchmark.serialization_utils import sanitize_tool_calls, parse_json
from llm.qwen_utils import (
    prompt_with_image,
    prompt_graph_agent_with_semantic_labels,
)
from llm.tools import GraphTools
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def project_3d_to_2d(
    positions: np.ndarray, proj_matrix: torch.Tensor, img_width: int, img_height: int
) -> np.ndarray:
    # Expecting positions to be (N, 3)
    assert positions.shape[1] == 3, "Positions must be (N, 3)"

    # Conver to homogeneous coordinates
    positions = torch.tensor(
        positions, device=proj_matrix.device, dtype=proj_matrix.dtype
    )
    ones = torch.ones(
        (positions.shape[0], 1), dtype=positions.dtype, device=positions.device
    )
    positions = torch.cat([positions, ones], dim=1)  # (N, 4)

    # Apply full projection transform: world to image space
    # Apparently full_proj_transform is transposed, seems to be correct
    coords = (proj_matrix.T @ positions.T).T  # (N, 4)

    # Perspective division to get NDC (Normalized Device Coordinates)
    w = coords[:, 3]
    ndc = coords[:, :3] / (w.unsqueeze(1) + 1e-7)  # (N, 3) [x, y, z] in [-1, 1]

    # Convert NDC to pixel coordinates
    # NDC: x, y in [-1, 1] → Pixel: u in [0, width], v in [0, height]
    pixels_x = (ndc[:, 0] + 1.0) * 0.5 * img_width
    pixels_y = (ndc[:, 1] + 1.0) * 0.5 * img_height

    pixels = np.stack([pixels_x, pixels_y], axis=-1)  # (N, 2)
    return pixels


def project_3d_to_2d_and_mask(
    positions: np.ndarray, proj_matrix: torch.Tensor, img_width: int, img_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D positions to pixels and return an in-frame visibility mask.

    Returns:
        pixels: (N,2) float pixel coordinates (may fall outside if not masked)
        in_frame_mask: (N,) boolean numpy array for points inside image and in front of camera
    """
    assert positions.shape[1] == 3, "Positions must be (N, 3)"

    positions_t = torch.tensor(
        positions, device=proj_matrix.device, dtype=proj_matrix.dtype
    )
    ones = torch.ones(
        (positions_t.shape[0], 1), dtype=positions_t.dtype, device=positions_t.device
    )
    positions_h = torch.cat([positions_t, ones], dim=1)  # (N, 4)

    coords = (proj_matrix.T @ positions_h.T).T  # (N, 4)
    w = coords[:, 3]
    ndc = coords[:, :3] / (w.unsqueeze(1) + 1e-7)

    pixels_x = (ndc[:, 0] + 1.0) * 0.5 * img_width
    pixels_y = (ndc[:, 1] + 1.0) * 0.5 * img_height
    pixels = torch.stack([pixels_x, pixels_y], dim=-1)

    in_front = w > 0
    in_bounds_x = (pixels[..., 0] >= 0) & (pixels[..., 0] < img_width)
    in_bounds_y = (pixels[..., 1] >= 0) & (pixels[..., 1] < img_height)
    in_frame = (in_front & in_bounds_x & in_bounds_y).detach().cpu().numpy()

    return pixels.detach().cpu().numpy(), in_frame


def load_colmap_cams(colmap_sparse_dir: Path) -> dict:
    reconstruction = pycolmap.Reconstruction(str(colmap_sparse_dir))
    camera_index = {}

    for _, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]

        model_name = str(camera.model)
        params = np.asarray(camera.params)
        if "PINHOLE" in model_name or "OPENCV" in model_name:
            fx = float(params[0])
            fy = float(params[1])
        elif "SIMPLE_PINHOLE" in model_name or "SIMPLE_RADIAL" in model_name:
            fx = float(params[0])
            fy = float(params[0])
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model_name}")

        fovx = focal2fov(fx, camera.width)
        fovy = focal2fov(fy, camera.height)

        cam_from_world = image.cam_from_world()
        r_w2c = np.asarray(cam_from_world.rotation.matrix())
        t_w2c = np.asarray(cam_from_world.translation)

        camera_info = {
            "R": r_w2c.T,
            "T": t_w2c,
            "FovX": fovx,
            "FovY": fovy,
            "width": int(camera.width),
            "height": int(camera.height),
        }

        image_name = Path(image.name).name
        image_stem = Path(image.name).stem
        camera_index[image_name] = camera_info
        camera_index[image_stem] = camera_info

    return camera_index


def get_proj_matrix_from_timestep(
    timestep: int, camera_index: dict, frame: str
) -> torch.Tensor:
    del timestep
    frame_stem = Path(frame).stem
    camera_info = camera_index[frame] if frame in camera_index else camera_index[frame_stem]

    world_view_transform = torch.tensor(
        getWorld2View2(camera_info["R"], camera_info["T"]), dtype=torch.float32
    ).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=camera_info["FovX"],
        fovY=camera_info["FovY"],
    ).transpose(0, 1)

    full_proj_matrix = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return full_proj_matrix, camera_info["width"], camera_info["height"]

def qwen3_coords_to_pixels(
    x: float, y: float, img_width: int, img_height: int
) -> np.ndarray:
    """Convert Qwen3's normalized [0, 1000] coordinates back to pixel coordinates.
    
    Args:
        coords: Array of shape (N, 2) with [x, y] in [0, 1000] range
        img_width: Original image width in pixels
        img_height: Original image height in pixels
    
    Returns:
        Array of shape (N, 2) with [x, y] pixel coordinates
    """
    # Scale from [0, 1000] to [0, 1] then denormalize to pixel coordinates
    px = (x / 1000.0) * img_width
    py = (y / 1000.0) * img_height
    return px, py


def graph_agent_feat_queries(
    model,
    processor,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
    use_semantic_labels: bool = False,
    semantic_method_name: str = "graph_agent_semantics",
):
    """Run graph agent with tools across all queries for a clip.

    Uses prompt_graph_agent which requires qwen3 for agentic tool use.
    Loads static graph artifacts and additional data needed for tools.
    """
    graph_dir = Path(graph_dir)

    # Required static graph artifacts
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"
    positions_path = graph_dir / "positions.npy"
    clusters_path = graph_dir / "clusters.npy"
    adjacency_path = graph_dir / "graph.npy"
    bhattacharyya_path = graph_dir / "bhattacharyya_coeffs.npy"

    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)

    if use_semantic_labels:
        semantic_labels_path = graph_dir / "cluster_semantics.json"
        with open(semantic_labels_path, "r") as f:
            node_semantic_labels = json.load(f)

    if semantic_method_name == "graph_agent_semantics_vision":
        max_iterations = cfg.eval.spatial.graph_agent_semantics_vision_max_iterations
        tool_config = cfg.eval.spatial.graph_agent_semantics_vision_tools
        system_prompt = cfg.eval.spatial.graph_agent_semantics_vision_system_prompt
        prompt_template = cfg.eval.spatial.graph_agent_semantics_vision_prompt_template
    elif semantic_method_name == "graph_agent_semantics":
        max_iterations = cfg.eval.spatial.graph_agent_semantics_max_iterations
        tool_config = cfg.eval.spatial.graph_agent_semantics_tools
        system_prompt = cfg.eval.spatial.graph_agent_semantics_system_prompt
        prompt_template = cfg.eval.spatial.graph_agent_semantics_prompt_template

    # Keep agent context/tool responses in the same normalized space; convert
    # final model prediction back to original space right before projection/logging.
    point_o2n, point_n2o, distance_o2n, _ = get_coord_transformations(positions)

    # Create GraphTools instance for tool management
    images_dir = Path(cfg.preprocessed_root) / clip.name / cfg.eval.paths.images_subdir
    video_frames = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    graph_tools = GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        video_frames=video_frames,
        annotation_stride=cfg.eval.annotation_stride,
    )

    # Setup tool visualization directory if configured
    tool_viz_enabled = cfg.eval.spatial.tool_viz_dir is not None
    tool_viz_dir = None
    if tool_viz_enabled:
        tool_viz_dir = Path(cfg.eval.spatial.tool_viz_dir) / clip.name
        tool_viz_dir.mkdir(parents=True, exist_ok=True)

    # Parse selected graph-agent tool config (objects with name and max_calls)
    tool_names = []
    tool_call_limits = {}
    for tool_entry in tool_config:
        tool_name = tool_entry.name
        tool_names.append(tool_name)
        max_calls = getattr(tool_entry, "max_calls", None)
        if max_calls is not None:
            tool_call_limits[tool_name] = max_calls
    
    # Get the specific tools needed for graph agent
    tools = graph_tools.get_tools_by_name(tool_names)
    
    # Convert to None if no limits specified
    if len(tool_call_limits) == 0:
        tool_call_limits = None

    # Cameras for projection (directly from COLMAP sparse/0)
    colmap_sparse_dir = Path(cfg.preprocessed_root) / clip.name / "sparse" / "0"
    camera_index = load_colmap_cams(colmap_sparse_dir)

    results = []
    for annotation in clip_gt:
        # prompt and other data
        query_id = annotation["id"]
        method_name = semantic_method_name if use_semantic_labels else "graph_agent"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Running [{query_id}] with method [{method_name}]")
        timestep = annotation["timestep"]
        frame_number = timestep * cfg.eval.annotation_stride
        frame_name = f"frame_{frame_number:06d}.png"
        proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
            int(frame_number), camera_index, frame_name
        )
        question = annotation["query"]
        prompt = prompt_template.format(question=question)
        
        # start recording tool calls
        if tool_viz_dir is not None and graph_tools is not None:
            sanitized_query = re.sub(r'[^\w\s-]', '', question)  # Remove special chars
            sanitized_query = re.sub(r'\s+', '_', sanitized_query)  # Replace whitespace with _
            sanitized_query = sanitized_query[:50]  # Limit length
            rrd_filename = f"{method_name}_t{timestep:03d}_{query_id}_{sanitized_query}.rrd"
            rrd_file = tool_viz_dir / rrd_filename
            graph_tools.start_recording(str(rrd_file))

        agent_result = prompt_graph_agent_with_semantic_labels(
            question=prompt,
            initial_timestep_idx=timestep,
            node_centers=point_o2n(node_centers),
            node_centroids=point_o2n(node_centroids),
            node_extents=distance_o2n(node_extents),
            node_semantic_labels=node_semantic_labels,
            model=model,
            processor=processor,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            tool_call_limits=tool_call_limits,
        )

        # parse and convert to pixels
        json_data = parse_json(agent_result["final_answer"])
        if json_data is None or "x" not in json_data or "y" not in json_data or "z" not in json_data:
            px, py = None, None
            pos_arr_original = None
            if tool_viz_dir is not None and graph_tools is not None:
                graph_tools.stop_recording()
        else:
            # grab point in original coordinates
            x, y, z = json_data["x"], json_data["y"], json_data["z"]
            pos_arr = np.array([x, y, z], dtype=np.float32).reshape(1, 3)
            pos_arr_original = point_n2o(pos_arr)

            # log final prediction to rerun
            if tool_viz_dir is not None and graph_tools is not None:
                graph_tools.log_final_prediction(
                    position=pos_arr_original,
                    timestep_idx=timestep,
                    label=question,
                )
                graph_tools.stop_recording()

            # project to pixels
            pixels = project_3d_to_2d(pos_arr_original, proj_matrix, img_width, img_height)
            px, py = float(pixels[0, 0]), float(pixels[0, 1])

        results.append({
            "id": query_id,
            "timestep": timestep,
            "query": question,
            "predicted": [px, py],
            "predicted_3d_original": pos_arr_original.tolist() if pos_arr_original is not None else [None, None, None],
            "raw_response": agent_result["final_answer"],
            "message_history": agent_result["message_history"],
            "tool_calls": sanitize_tool_calls(agent_result.get("tool_calls", [])),
        })

        # Clear VRAM after each timestep to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def frame_direct_feat_queries(
    model,
    processor,
    preprocessed_root: Path | str,
    images_subdir: str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
    use_semantic_labels: bool = False,
):
    """Run direct Qwen prompting on the frame to return a single pixel per query."""
    images_dir = Path(preprocessed_root) / clip.name / images_subdir

    system_prompt = cfg.eval.spatial.frame_direct_system_prompt
    prompt_template = cfg.eval.spatial.frame_direct_prompt_template

    results = []
    for annotation in clip_gt:
        query_id = annotation["id"]
        method_name = "frame_direct"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Running [{query_id}] with method [{method_name}]")
        # prompt
        timestep = annotation["timestep"]
        frame_number = timestep * cfg.eval.annotation_stride
        frame_path = images_dir / f"frame_{frame_number:06d}.png"
        image = Image.open(frame_path).convert("RGB")
        prompt = prompt_template.format(question=annotation["query"])

        # llm answer
        response = prompt_with_image(
            image=image,
            prompt=prompt,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )

        # parse and convert to pixels
        json_data = parse_json(response)
        if json_data is None:
            px, py = None, None
            x, y = None, None
        else:
            x = float(json_data["x"])
            y = float(json_data["y"])
            px, py = qwen3_coords_to_pixels(x, y, image.width, image.height)
            px, py = float(px), float(py)

        results.append({
            "id": query_id,
            "timestep": timestep,
            "query": annotation["query"],
            "predicted": [px, py],
            "predicted_qwen3coords": [x, y],
            "raw_response": response,
            "message_history": [],
            "tool_calls": [],
        })

        # Clear VRAM after each timestep to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def dump_spatial_prediction_visualizations(
    cfg,
    results_splat: Dict[str, Any],
    clip_name: str,
    preprocessed_root: Path | str,
    images_subdir: str,
    viz_dir: Path | str,
    method_name: str | None = None,
) -> None:
    """Render top-k predicted points onto the corresponding frames and save images.

    Args:
        results_splat: Predictions dictionary returned by splat_feat_queries for a clip.
        clip_name: Name of the clip.
        preprocessed_root: Root directory for preprocessed data.
        images_subdir: Subdirectory name containing images for the clip.
        gt_data: Ground-truth dict used during evaluation; must contain frame_number per timestep.
        viz_dir: Output directory root for visualizations.
    """

    def _sanitize_filename(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9._-]", "", text)
        return text[:120] if len(text) > 120 else text

    def _draw_points(
        img_bgr,
        coords,
        color_bgr=(255, 0, 0),
        radius: int = 5,
        draw_indices: bool = False,
    ):
        for idx, (x, y) in enumerate(coords):
            xi, yi = int(x), int(y)
            cv2.circle(img_bgr, (xi, yi), radius, color_bgr, thickness=-1)
            if draw_indices:
                # 1-based rank index
                label = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                # Position to the top-right of the point
                tx = xi + radius + 3
                ty = yi - radius - 3
                # Ensure within image bounds
                tx = max(0, min(tx, img_bgr.shape[1] - tw - 1))
                ty = max(th + 1, min(ty, img_bgr.shape[0] - 1))
                # Draw background rectangle for contrast
                cv2.rectangle(
                    img_bgr,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + baseline + 2),
                    (0, 0, 0),
                    thickness=-1,
                )
                # Draw text
                cv2.putText(
                    img_bgr,
                    label,
                    (tx, ty),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
        return img_bgr

    # Organize outputs under .../viz_dir/<method>/<clip>
    viz_root = Path(viz_dir) / method_name / clip_name
    viz_root.mkdir(parents=True, exist_ok=True)

    images_dir = Path(preprocessed_root) / clip_name / images_subdir

    for result in results_splat:
        timestep = result["timestep"]
        frame_number = timestep * cfg.eval.annotation_stride
        frame_path = images_dir / f"frame_{frame_number:06d}.png"
        base_img = cv2.imread(str(frame_path))

        # draw point
        coords = result["predicted"]
        if coords is not None and isinstance(coords, list) and len(coords) == 2:
            # Expect a single point [x, y]; wrap it for _draw_points which expects list of points
            if all(isinstance(c, (int, float)) and c is not None for c in coords):
                img = base_img.copy()
                img = _draw_points(
                    img, [coords], color_bgr=(255, 0, 0), radius=5, draw_indices=True
                )
                out_name = f"{frame_number:06d}_{result['id']}_{_sanitize_filename(result['query'])}.png"
                cv2.imwrite(str(viz_root / out_name), img)
