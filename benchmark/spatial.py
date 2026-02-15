import gc
from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import List, Dict, Any, Optional, Callable
import cv2
import re
import json

from benchmark.graph_utils import get_coord_transformations
from benchmark.serialization_utils import sanitize_tool_calls
from llm.qwen_utils import (
    model_inputs,
    QWEN_CONSTANTS,
    get_patch_hw,
    prompt_graph_agent,
    ask_qwen_about_image,
    ask_qwen_about_image_custom,
    qwen3_cat_to_deepstack_multiple,
    get_patched_qwen3,
)
from llm.tools import GraphTools
from autoencoder.model_qwen import QwenAutoencoder
from scene.dataset_readers import CameraInfo, readColmapSceneInfo
from scene.cameras import Camera
from PIL import Image
from qwen_vl_utils import process_vision_info

from loguru import logger


def find_image_path(images_dir: Path, frame_number: int) -> Optional[Path]:
    """Find image file for a given frame number, supporting both JPG and PNG formats.
    
    Args:
        images_dir: Directory containing images
        frame_number: Frame number (0-indexed)
    
    Returns:
        Path to image file if found, None otherwise
    """
    frame_stem = f"frame_{frame_number:06d}"
    jpg_path = images_dir / f"{frame_stem}.jpg"
    png_path = images_dir / f"{frame_stem}.png"
    
    if jpg_path.exists():
        return jpg_path
    elif png_path.exists():
        return png_path
    else:
        return None


def get_patched_qwen_for_spatial_grounding(
    size: str = "32B",
    use_fp8: bool = False,
):
    model, processor = get_patched_qwen3(
        size=size,
        use_fp8=use_fp8,
        attn_implementation="eager",
    )

    # Enable attention output in model config
    model.config.output_attentions = True
    model.model.config.output_attentions = True
    model.model.language_model.config.output_attentions = True

    return model, processor


def extract_text_to_vision_attention(
    model,
    processor,
    vision_features,
    layers: List[int],
    prompt: str,
    substring: str,
    system_prompt: str,
    qwen_version: str = "qwen25",
):
    """Extract attention scores from substring query tokens to vision tokens across layers.

    Args:
        model: Qwen VL model (Qwen2.5 or Qwen3)
        processor: Qwen VL processor
        vision_features: Vision feature tensor.
            For qwen25: shape (N, hidden_dim)
            For qwen3: shape (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
        layers (List[int]): List of layers to extract attention scores from
        prompt (str): Prompt to use for the query
        substring (str): Substring to extract attention scores from
        qwen_version (str): Either "qwen25" or "qwen3"

    Returns:
        Dict[str, Any]:
            - scores: torch.Tensor of shape (num_layers, num_query_tokens, num_vision_tokens)
            - tokens: List[str] of all decoded tokens for input sequence
            - query_token_indices: List[int] indices into tokens corresponding to substring span
            - vision_token_indices: List[int] indices into tokens corresponding to vision placeholders
    """
    assert substring in prompt

    # Build a message with an image placeholder and the full prompt
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Handle Qwen3 concatenated features: split into main + deepstack
    if qwen_version == "qwen3":
        main_features, deepstack_features = qwen3_cat_to_deepstack_multiple(
            [vision_features]
        )
        # Prepare inputs using main features for correct token count
        inputs = model_inputs(
            messages, main_features, processor, qwen_version=qwen_version
        ).to(model.device)
    else:
        main_features = [vision_features]
        deepstack_features = None
        inputs = model_inputs(
            messages, main_features, processor, qwen_version=qwen_version
        ).to(model.device)

    # Identify vision token positions via <|image_pad|> before forward (needed for custom pos ids)
    input_ids = inputs.input_ids[0]
    image_pad_token = processor.tokenizer.encode(
        "<|image_pad|>", add_special_tokens=False
    )[0]
    image_pad_positions = (input_ids == image_pad_token).nonzero(as_tuple=True)[0]
    if image_pad_positions.numel() == 0:
        raise ValueError("No <|image_pad|> tokens found in input sequence.")
    vision_token_indices = image_pad_positions.tolist()

    with torch.no_grad():
        # Not generating any output tokens here
        model_kwargs = dict(
            output_attentions=True,
            return_dict=True,
            custom_patch_features=main_features,
        )
        if qwen_version == "qwen3" and deepstack_features is not None:
            model_kwargs["custom_deepstack_features"] = deepstack_features
        outputs = model(
            **inputs,
            **model_kwargs,
        )

    all_attn = list(outputs.attentions)

    # vision_token_indices already computed above

    # Map substring to token indices by scanning decoded tokens and locating overlap
    tokens = [processor.tokenizer.decode([tid]) for tid in input_ids.tolist()]
    concatenated = "".join(tokens)
    start_char = concatenated.lower().find(substring.lower())
    end_char = start_char + len(substring)

    query_token_indices = []
    cursor = 0
    for idx, tok in enumerate(tokens):
        token_start = cursor
        token_end = cursor + len(tok)
        if token_start < end_char and token_end > start_char:
            query_token_indices.append(idx)
        cursor = token_end
        if cursor > end_char and len(query_token_indices) > 0:
            break

    # Collect attention: [L, Q, V], averaged across heads
    num_layers = len(layers)
    num_queries = len(query_token_indices)
    num_vision = len(vision_token_indices)
    scores = torch.empty((num_layers, num_queries, num_vision), dtype=torch.float32)

    q_idx = torch.tensor(query_token_indices, device=all_attn[0].device)
    v_idx = torch.tensor(vision_token_indices, device=all_attn[0].device)

    for out_pos, layer_idx in enumerate(layers):
        layer_attn = all_attn[layer_idx]  # [1, heads, seq, seq]
        # select queries and vision columns, average heads -> [Q, V]
        attn_q_to_v = layer_attn[0, :, q_idx, :][:, :, v_idx].mean(
            dim=0
        )  # mean over heads
        scores[out_pos] = attn_q_to_v.detach().to(torch.float32).cpu()

    return {
        "scores": scores,
        "tokens": tokens,
        "query_token_indices": query_token_indices,
        "vision_token_indices": vision_token_indices,
    }


def extract_text_to_image_attention(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    image: Image.Image,
    layers: List[int],
    prompt: str,
    substring: str,
    system_prompt: str,
):
    """Extract attention scores from substring query tokens to image vision tokens across layers.

    Uses the actual image input, not custom patch features. Vision tokens are identified via
    <|image_pad|> positions in the tokenized sequence.
    """
    assert substring in prompt

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                # Pass actual image so apply_chat_template computes correct token count
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Build text with correct image token count
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Extract images using process_vision_info to ensure correct format
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    all_attn = list(outputs.attentions)

    # Identify vision token positions via <|image_pad|>
    input_ids = inputs.input_ids[0]
    image_pad_token = processor.tokenizer.encode(
        "<|image_pad|>", add_special_tokens=False
    )[0]
    image_pad_positions = (input_ids == image_pad_token).nonzero(as_tuple=True)[0]
    if image_pad_positions.numel() == 0:
        raise ValueError("No <|image_pad|> tokens found in input sequence.")
    vision_token_indices = image_pad_positions.tolist()

    # Map substring to token indices
    tokens = [processor.tokenizer.decode([tid]) for tid in input_ids.tolist()]
    concatenated = "".join(tokens)
    start_char = concatenated.lower().find(substring.lower())
    end_char = start_char + len(substring)

    query_token_indices = []
    cursor = 0
    for idx, tok in enumerate(tokens):
        token_start = cursor
        token_end = cursor + len(tok)
        if token_start < end_char and token_end > start_char:
            query_token_indices.append(idx)
        cursor = token_end
        if cursor > end_char and len(query_token_indices) > 0:
            break

    # Collect attention: [L, Q, V], averaged across heads
    num_layers = len(layers)
    num_queries = len(query_token_indices)
    num_vision = len(vision_token_indices)
    scores = torch.empty((num_layers, num_queries, num_vision), dtype=torch.float32)

    q_idx = torch.tensor(query_token_indices, device=all_attn[0].device)
    v_idx = torch.tensor(vision_token_indices, device=all_attn[0].device)

    for out_pos, layer_idx in enumerate(layers):
        layer_attn = all_attn[layer_idx]  # [1, heads, seq, seq]
        attn_q_to_v = layer_attn[0, :, q_idx, :][:, :, v_idx].mean(dim=0)
        scores[out_pos] = attn_q_to_v.detach().to(torch.float32).cpu()

    return {
        "scores": scores,
        "tokens": tokens,
        "query_token_indices": query_token_indices,
        "vision_token_indices": vision_token_indices,
    }


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


def get_proj_matrix_from_timestep(
    timestep: int, train_cameras: list, frame: str
) -> torch.Tensor:
    # Get the camera parameters for the timestep
    camera_info = train_cameras[timestep]
    assert isinstance(camera_info, CameraInfo), (
        "camera_info must be a CameraInfo object"
    )

    # Instantiate a Camera object from the camera info
    image = camera_info.image
    R = camera_info.R
    T = camera_info.T
    FovX = camera_info.FovX
    FovY = camera_info.FovY
    time = camera_info.time
    mask = camera_info.mask
    camera = Camera(
        colmap_id=timestep,
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image=image,
        gt_alpha_mask=None,
        image_name=f"{frame}",
        uid=timestep,
        data_device=torch.device("cuda"),
        time=time,
        mask=mask,
    )

    # Get projection matrix from camera object
    # full_proj_transform includes the world to cam as well, seems to be correct
    # projection_matrix = camera.projection_matrix
    full_proj_matrix = camera.full_proj_transform
    return full_proj_matrix, camera.image_width, camera.image_height


def splat_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    ts_feats,
    pos_t,
    layers,
    top_k,
    frame_number,
    clip_name: str,
    train_cameras,
    prompt_template: str,
    system_prompt: str,
    point_n2o: Callable[[np.ndarray], np.ndarray],
    qwen_version: str = "qwen25",
):
    outputs = []
    frame_idx = int(frame_number)
    frame_name = f"frame_{frame_idx:06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        frame_idx, train_cameras, frame_name
    )

    feats_tensor = torch.tensor(ts_feats, device=model.device)
    for query in queries_list:
        substring = query["query"]
        prompt = prompt_template.format(substring=substring)
        if feats_tensor.shape[0] == 0:
            out_item = {"query": substring, "predictions": {}}
            for layer in layers:
                out_item["predictions"][layer] = {
                    "scores": [],
                    "pixel_coords": [],
                    "positions": [],
                }
            outputs.append(out_item)
            continue
        attn_out = extract_text_to_vision_attention(
            model=model,
            processor=processor,
            vision_features=feats_tensor,
            layers=layers,
            prompt=prompt,
            substring=substring,
            system_prompt=system_prompt,
            qwen_version=qwen_version,
        )
        attn_scores = attn_out["scores"]

        out_item = {"query": substring, "predictions": {}}
        for layer_idx, layer in enumerate(layers):
            layer_scores = attn_scores[layer_idx]
            layer_scores = layer_scores.mean(dim=0)
            top_scores, top_indices = layer_scores.topk(k=top_k, sorted=True)
            top_scores = top_scores.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()

            # Directly index positions by attention-ranked indices
            top_positions = pos_t[top_indices]

            # Convert back to original coordinates for projection
            top_positions_original = point_n2o(top_positions)
            top_pixels = project_3d_to_2d(
                top_positions_original, proj_matrix, img_width, img_height
            )

            out_item["predictions"][layer] = {
                "scores": top_scores.tolist(),
                "pixel_coords": top_pixels.tolist(),
                "positions": top_positions.tolist(),
            }
        outputs.append(out_item)
    return outputs


def splat_feat_queries(
    model,
    processor,
    splat_feats,
    splat_indices,
    positions,
    clip_gt,
    clip: DictConfig,
    cfg: DictConfig,
):
    # load cameras
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    # Compute coordinate transformations and normalize positions
    point_o2n, point_n2o, _, _ = get_coord_transformations(positions)
    positions_normalized = point_o2n(positions)

    results = {}
    splat_system_prompt = cfg.eval.spatial.splat_system_prompt

    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        ts_feats = splat_feats[t]
        pos_t = positions_normalized[t][splat_indices]

        results[timestep] = {"objects": [], "actions": []}

        layers = cfg.eval.spatial.layers
        top_k = cfg.eval.spatial.top_k_scores
        frame_number = timestep_queries["frame_number"]
        prompt_template = cfg.eval.spatial.splat_prompt_template

        results[timestep]["objects"] = splat_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            ts_feats=ts_feats,
            pos_t=pos_t,
            layers=layers,
            top_k=top_k,
            frame_number=frame_number,
            clip_name=clip.name,
            train_cameras=train_cameras,
            prompt_template=prompt_template,
            system_prompt=splat_system_prompt,
            point_n2o=point_n2o,
            qwen_version=cfg.eval.qwen_version,
        )

        results[timestep]["actions"] = splat_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            ts_feats=ts_feats,
            pos_t=pos_t,
            layers=layers,
            top_k=top_k,
            frame_number=frame_number,
            clip_name=clip.name,
            train_cameras=train_cameras,
            prompt_template=prompt_template,
            system_prompt=splat_system_prompt,
            point_n2o=point_n2o,
            qwen_version=cfg.eval.qwen_version,
        )

        # Clear VRAM after each timestep to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _parse_point_from_json(text: str) -> Optional[List[float]]:
    """Extract a single 3D point [x, y, z] from a JSON object in the given text.

    The model is instructed to return pure JSON, but we make this robust by:
      1) locating the first JSON object in the text, attempting json.loads
      2) falling back to regex-based triple float extraction if needed
    """

    # Try to find a JSON object in the text
    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            obj = json.loads(candidate)
            # Accept either {"x":..,"y":..,"z":..} or {"point":{"x":..,"y":..,"z":..}}
            if isinstance(obj, dict):
                if all(k in obj for k in ("x", "y", "z")):
                    return [float(obj["x"]), float(obj["y"]), float(obj["z"])]
                if "point" in obj and isinstance(obj["point"], dict):
                    point = obj["point"]
                    if all(k in point for k in ("x", "y", "z")):
                        return [float(point["x"]), float(point["y"]), float(point["z"])]
    except Exception:
        pass

    return None


def _pixels_to_qwen3_coords(
    pixels: np.ndarray, img_width: int, img_height: int
) -> np.ndarray:
    """Convert pixel coordinates to Qwen3's normalized [0, 1000] coordinate system.
    
    Args:
        pixels: Array of shape (N, 2) with [x, y] pixel coordinates
        img_width: Original image width in pixels
        img_height: Original image height in pixels
    
    Returns:
        Array of shape (N, 2) with [x, y] in [0, 1000] range
    """
    pixels_array = np.array(pixels)
    if pixels_array.ndim == 1:
        pixels_array = pixels_array.reshape(1, -1)
    
    # Normalize to [0, 1] then scale to [0, 1000]
    normalized = pixels_array.copy()
    normalized[:, 0] = (pixels_array[:, 0] / img_width) * 1000.0
    normalized[:, 1] = (pixels_array[:, 1] / img_height) * 1000.0
    
    return normalized


def _qwen3_coords_to_pixels(
    coords: np.ndarray, img_width: int, img_height: int
) -> np.ndarray:
    """Convert Qwen3's normalized [0, 1000] coordinates back to pixel coordinates.
    
    Args:
        coords: Array of shape (N, 2) with [x, y] in [0, 1000] range
        img_width: Original image width in pixels
        img_height: Original image height in pixels
    
    Returns:
        Array of shape (N, 2) with [x, y] pixel coordinates
    """
    coords_array = np.array(coords)
    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)
    
    # Scale from [0, 1000] to [0, 1] then denormalize to pixel coordinates
    pixels = coords_array.copy()
    pixels[:, 0] = (coords_array[:, 0] / 1000.0) * img_width
    pixels[:, 1] = (coords_array[:, 1] / 1000.0) * img_height
    
    return pixels


def _parse_pixel_from_json(
    text: str,
    qwen_version: str = "qwen25",
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Optional[List[float]]:
    """Extract a single 2D pixel [x, y] from a JSON object in the text.

    Accepts either {"x":..,"y":..} or {"u":..,"v":..}. Falls back to first two floats.
    
    Args:
        text: Response text containing JSON coordinates
        qwen_version: Either "qwen25" or "qwen3"
        img_width: Image width in pixels (required for qwen3)
        img_height: Image height in pixels (required for qwen3)
    
    Returns:
        Pixel coordinates [x, y] in original image space
    """
    import json as _json

    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            obj = _json.loads(candidate)
            if isinstance(obj, dict):
                coords = None
                if all(k in obj for k in ("x", "y")):
                    coords = [float(obj["x"]), float(obj["y"])]
                elif all(k in obj for k in ("u", "v")):
                    coords = [float(obj["u"]), float(obj["v"])]
                
                if coords is not None:
                    # For Qwen3, coordinates are in [0, 1000] and need to be scaled back
                    if qwen_version == "qwen3":
                        if img_width is None or img_height is None:
                            raise ValueError(
                                "img_width and img_height required for qwen3 coordinate conversion"
                            )
                        coords_array = _qwen3_coords_to_pixels(
                            np.array(coords), img_width, img_height
                        )
                        return coords_array.flatten().tolist()
                    return coords
    except Exception:
        pass

    return None


def graph_agent_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    node_feats_npz,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    tools: Dict[str, Any],
    timestep_idx: int,
    frame_number: int,
    train_cameras,
    system_prompt: str,
    prompt_template: str,
    point_n2o: Callable[[np.ndarray], np.ndarray],
    max_iterations: int = 10,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    graph_tools: Optional[Any] = None,
    tool_viz_dir: Optional[Path] = None,
    query_type: str = "objects",
):
    """Use prompt_graph_agent with tools to predict a 3D point per query.

    Returns items with a single mock layer key "0" like static_graph baseline.
    """
    outputs = []

    # Precompute projection for this frame
    frame_name = f"frame_{int(frame_number):06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        int(frame_number), train_cameras, frame_name
    )

    for query_idx, query in enumerate(queries_list):
        substring = query["query"]
        question = prompt_template.format(substring=substring)
        
        # Start recording if tool visualization is enabled
        if tool_viz_dir is not None and graph_tools is not None:
            # Sanitize query text for filename
            sanitized_query = re.sub(r'[^\w\s-]', '', substring)  # Remove special chars
            sanitized_query = re.sub(r'\s+', '_', sanitized_query)  # Replace whitespace with _
            sanitized_query = sanitized_query[:50]  # Limit length
            rrd_filename = f"t{timestep_idx:03d}_{query_type}_{query_idx:02d}_{sanitized_query}.rrd"
            rrd_file = tool_viz_dir / rrd_filename
            graph_tools.start_recording(str(rrd_file))

        # Call Qwen agent with the graph at this specific timestep
        agent_result = prompt_graph_agent(
            question=question,
            node_feats=node_feats_npz,
            initial_timestep_idx=int(timestep_idx),
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            tools=tools,
            qwen_version="qwen3",  # Required for agentic mode
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            tool_call_limits=tool_call_limits,
        )

        response = agent_result["final_answer"]

        point3d = _parse_point_from_json(response)
        sanitized_tool_calls = sanitize_tool_calls(agent_result.get("tool_calls", []))
        message_history = agent_result.get("message_history", [])
        if point3d is None:
            # Stop recording if tool visualization is enabled (failed prediction)
            if tool_viz_dir is not None and graph_tools is not None:
                graph_tools.stop_recording()
            
            # 3D failure: fall back to 2D corner pixel (0,0), omit 3D position to avoid skew
            out_item = {
                "query": substring,
                "predictions": {},
                "raw_response": response,
                "tool_calls": sanitized_tool_calls,
                "message_history": message_history,
            }
            out_item["predictions"]["0"] = {
                "pixel_coords": [[0.0, 0.0]],
                "positions": [],
            }
            outputs.append(out_item)
            continue

        # Project to pixel coords - convert normalized point back to original coords first
        pos_arr = np.array(point3d, dtype=np.float32).reshape(1, 3)
        pos_arr_original = point_n2o(pos_arr)
        
        # Log final prediction to rerun before stopping recording
        if tool_viz_dir is not None and graph_tools is not None:
            graph_tools.log_final_prediction(
                position=pos_arr_original,
                timestep_idx=timestep_idx,
                label=substring,
            )
            graph_tools.stop_recording()
        pixels = project_3d_to_2d(pos_arr_original, proj_matrix, img_width, img_height)

        out_item = {
            "query": substring,
            "predictions": {},
            "raw_response": response,
            "tool_calls": sanitized_tool_calls,
            "message_history": message_history,
        }
        # Single mock layer key for graph agent baseline
        out_item["predictions"]["0"] = {
            "pixel_coords": pixels.tolist(),
            "positions": pos_arr.tolist(),
        }
        outputs.append(out_item)

    return outputs


def graph_agent_feat_queries(
    *,
    model,
    processor,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run graph agent with tools across all queries for a clip.

    Uses prompt_graph_agent which requires qwen3 for agentic tool use.
    Loads static graph artifacts and additional data needed for tools.
    """
    graph_dir = Path(graph_dir)

    # Required static graph artifacts
    node_feats_npz_path = graph_dir / "c_qwen_feats.npz"
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"
    positions_path = graph_dir / "positions.npy"
    clusters_path = graph_dir / "clusters.npy"
    patch_latents_path = graph_dir / "patch_latents_through_time.npy"
    adjacency_path = graph_dir / "graph.npy"
    bhattacharyya_path = graph_dir / "bhattacharyya_coeffs.npy"

    node_feats_npz = np.load(node_feats_npz_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    patch_latents_through_time = np.load(patch_latents_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)

    # Compute coordinate transformations (GraphTools will handle normalization internally)
    point_o2n, point_n2o, _, distance_o2n = get_coord_transformations(positions)
    node_centers_norm = point_o2n(node_centers)
    node_centroids_norm = point_o2n(node_centroids)
    node_extents_norm = distance_o2n(node_extents)

    # Load autoencoder for inspect_highres_node_at_time tool
    if cfg.eval.spatial.graph_agent_use_global_autoencoder:
        autoencoder_path = Path(cfg.preprocessed_root) / cfg.eval.spatial.graph_agent_global_autoencoder_checkpoint_dir / "best_ckpt.pth"
    else:
        clip_dir = Path(cfg.preprocessed_root) / clip.name
        autoencoder_path = clip_dir / cfg.eval.spatial.graph_agent_autoencoder_checkpoint_subdir / "best_ckpt.pth"
    autoencoder = QwenAutoencoder(
        input_dim=cfg.eval.spatial.graph_agent_autoencoder_full_dim,
        latent_dim=cfg.eval.spatial.graph_agent_autoencoder_latent_dim,
    ).to(model.device)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, map_location=model.device)
    )
    autoencoder.eval()

    # Create GraphTools instance for tool management
    graph_tools = GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        qwen_feats=node_feats_npz,
        patch_latents_through_time=patch_latents_through_time,
        autoencoder=autoencoder,
    )

    # Setup tool visualization directory if configured
    tool_viz_enabled = cfg.eval.spatial.tool_viz_dir is not None
    tool_viz_dir = None
    if tool_viz_enabled:
        tool_viz_dir = Path(cfg.eval.spatial.tool_viz_dir) / clip.name
        tool_viz_dir.mkdir(parents=True, exist_ok=True)

    # Parse graph_agent_tools config (objects with name and max_calls)
    tool_names = []
    tool_call_limits = {}
    for tool_entry in cfg.eval.spatial.graph_agent_tools:
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

    # Cameras for projection
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    system_prompt = cfg.eval.spatial.graph_agent_system_prompt
    prompt_template = cfg.eval.spatial.graph_agent_prompt_template
    max_iterations = cfg.eval.spatial.graph_agent_max_iterations

    results: Dict[str, Any] = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        frame_number = int(timestep_queries["frame_number"])  # local idx

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = graph_agent_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            node_centers=node_centers_norm,
            node_centroids=node_centroids_norm,
            node_extents=node_extents_norm,
            tools=tools,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            point_n2o=point_n2o,
            max_iterations=max_iterations,
            tool_call_limits=tool_call_limits,
            graph_tools=graph_tools,
            tool_viz_dir=tool_viz_dir,
            query_type="objects",
        )

        results[timestep]["actions"] = graph_agent_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            node_centers=node_centers_norm,
            node_centroids=node_centroids_norm,
            node_extents=node_extents_norm,
            tools=tools,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            point_n2o=point_n2o,
            max_iterations=max_iterations,
            tool_call_limits=tool_call_limits,
            graph_tools=graph_tools,
            tool_viz_dir=tool_viz_dir,
            query_type="actions",
        )

        # Clear VRAM after each timestep to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up autoencoder before returning
    del autoencoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _patch_indices_to_pixel_centers(
    patch_indices: np.ndarray, img_h: int, img_w: int, qwen_version: str = "qwen25"
) -> np.ndarray:
    """Map flat patch indices (row-major) to pixel centers in the original image.

    Returns array of shape (N, 2) with (x, y) in pixel coordinates.
    """
    effective_patch_size = QWEN_CONSTANTS[qwen_version]["effective_patch_size"]
    ph, pw = get_patch_hw(img_h, img_w, qwen_version)
    cols = patch_indices % pw
    rows = patch_indices // pw
    xs = (cols + 0.5) * effective_patch_size
    ys = (rows + 0.5) * effective_patch_size
    xs = np.clip(xs, 0, img_w - 1)
    ys = np.clip(ys, 0, img_h - 1)
    return np.stack([xs, ys], axis=-1)


def frame_direct_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    image: Image.Image,
    prompt_template: str,
    system_prompt: str,
    qwen_version: str = "qwen25",
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        question = prompt_template.format(substring=substring)

        # Custom implementation to test feature extraction and pos encodings (can zero out if needed)
        # This should reproduce the same results as vanilla ask_qwen_about_image with pos encodings enabled
        # It should be possible to swap this against ask_qwen_about_image
        response = ask_qwen_about_image_custom(
            image=image,
            prompt=question,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )

        px = _parse_pixel_from_json(
            response,
            qwen_version=qwen_version,
            img_width=image.width,
            img_height=image.height,
        )
        if px is None:
            px = [0.0, 0.0]

        out_item = {"query": substring, "predictions": {}, "raw_response": response}
        out_item["predictions"]["0"] = {
            "pixel_coords": [px],
        }
        outputs.append(out_item)
    return outputs


def frame_direct_feat_queries(
    *,
    model,
    processor,
    preprocessed_root: Path | str,
    images_subdir: str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run direct Qwen prompting on the frame to return a single pixel per query."""
    results: Dict[str, Any] = {}
    images_dir = Path(preprocessed_root) / clip.name / images_subdir

    prompt_template = cfg.eval.spatial.frame_direct_prompt_template
    system_prompt = cfg.eval.spatial.frame_direct_system_prompt

    for timestep, timestep_queries in clip_gt.items():
        frame_number = int(timestep_queries["frame_number"])  # local idx
        frame_path = find_image_path(images_dir, frame_number)
        if frame_path is None:
            continue
        image = Image.open(frame_path).convert("RGB")

        results[timestep] = {"objects": [], "actions": []}

        logger.info(f"Evaluating frame_direct for objects at timestep: {timestep}")
        results[timestep]["objects"] = frame_direct_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            image=image,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            qwen_version=cfg.eval.qwen_version,
        )

        logger.info(f"Evaluating frame_direct for actions at timestep: {timestep}")
        results[timestep]["actions"] = frame_direct_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            image=image,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            qwen_version=cfg.eval.qwen_version,
        )

        # Clear VRAM after each timestep to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def dump_spatial_prediction_visualizations(
    *,
    results_splat: Dict[str, Any],
    clip_name: str,
    preprocessed_root: Path | str,
    images_subdir: str,
    gt_data: Dict[str, Any],
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

    for timestep, group_preds in results_splat.items():
        timestep_str = str(timestep)
        if timestep_str not in gt_data:
            continue
        frame_number = int(gt_data[timestep_str]["frame_number"])  # type: ignore[index]
        # Use frame_number directly from GT (assumed local zero-based index)
        frame_path = find_image_path(images_dir, frame_number)
        if frame_path is None:
            continue
        base_img = cv2.imread(str(frame_path))
        if base_img is None:
            continue

        # Draw for objects and actions separately
        for group_name, color in (("objects", (255, 0, 0)), ("actions", (0, 0, 255))):
            items = group_preds.get(group_name, [])
            for item in items:
                query = item.get("query", group_name)
                preds_by_layer = item.get("predictions", {})
                for layer_key, pred in preds_by_layer.items():
                    layer_str = str(layer_key)
                    coords = pred.get("pixel_coords", [])
                    if not coords:
                        continue
                    img = base_img.copy()
                    img = _draw_points(
                        img, coords, color_bgr=color, radius=5, draw_indices=True
                    )
                    out_name = f"{frame_number:06d}_L{layer_str}_{group_name}_{_sanitize_filename(query)}.jpg"
                    cv2.imwrite(str(viz_root / out_name), img)
