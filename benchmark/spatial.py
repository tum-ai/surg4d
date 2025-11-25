from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import List, Dict, Any, Optional
import cv2
import re
import os

import qwen_vl
from scene.dataset_readers import CameraInfo, readColmapSceneInfo
from scene.cameras import Camera
from PIL import Image
from autoencoder.model_qwen import QwenAutoencoder


def get_patched_qwen_for_spatial_grounding(
    use_bnb_4bit: bool = False, use_bnb_8bit: bool = False
):
    model, processor = qwen_vl.get_patched_qwen(
        use_bnb_4bit=use_bnb_4bit,
        use_bnb_8bit=use_bnb_8bit,
        attn_implementation="eager",
    )

    # Enable attention output in model config
    model.config.output_attentions = True
    model.model.config.output_attentions = True
    model.model.language_model.config.output_attentions = True

    return model, processor


def extract_text_to_vision_attention(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    vision_features,
    layers: List[int],
    prompt: str,
    substring: str,
    system_prompt: str,
    *,
    vision_patch_positions: Optional[torch.Tensor] = None,
    vision_patch_grid_thw: Optional[torch.Tensor] = None,
    zero_vision_positional_encodings: bool = False,
):
    """Extract attention scores from substring query tokens to vision tokens across layers.

    Args:
        model (Qwen2_5_VLForConditionalGeneration): Qwen2.5-VL model
        processor (Qwen2_5_VLProcessor): Qwen2.5-VL processor
        vision_features (_type_): _description_
        layers (List[int]): List of layers to extract attention scores from
        prompt (str): Prompt to use for the query
        substring (str): Substring to extract attention scores from

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

    # Prepare inputs using a mock image sized to match the number of patch features
    inputs = qwen_vl.model_inputs(messages, [vision_features], processor).to(
        model.device
    )

    if vision_patch_grid_thw is not None:
        inputs["image_grid_thw"] = vision_patch_grid_thw.to(
            model.device, dtype=inputs["image_grid_thw"].dtype
        )

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
            custom_patch_features=[vision_features],
        )
        # Optional positional encoding control
        if zero_vision_positional_encodings:
            # Override all vision token positional ids with zeros (t=h=w=0)
            zeros_pos = torch.zeros(
                (len(vision_token_indices), 3), dtype=torch.long, device=model.device
            )
            model_kwargs["custom_vision_patch_positions"] = zeros_pos
            model_kwargs["custom_vision_token_indices"] = [vision_token_indices]
        elif vision_patch_positions is not None:
            model_kwargs["custom_vision_patch_positions"] = vision_patch_positions
            model_kwargs["custom_vision_token_indices"] = [vision_token_indices]

        # Debug: print applied positional ids for a small window around the middle
        if bool(int(os.getenv("DEBUG_VISION_POS_ENC", "0"))):
            count = max(1, int(os.getenv("DEBUG_VISION_POS_ENC_COUNT", "5")))
            n = len(vision_token_indices)
            if n > 0:
                mid = n // 2
                half = max(0, (count - 1) // 2)
                start = max(0, mid - half)
                end = min(n, start + count)
                # ensure we have exactly up to `count` indices if possible
                if end - start < count:
                    start = max(0, end - count)
                debug_abs_tok_idx = vision_token_indices[start:end]
                model_kwargs["debug_print_vision_positional_encodings"] = True
                model_kwargs["debug_positional_token_indices"] = debug_abs_tok_idx
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
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Build text and inputs using real image
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[[image]], padding=True, return_tensors="pt"
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
    use_frame_patch_grid: bool = False,
    zero_vision_positional_encodings: bool = False,
):
    outputs = []
    frame_idx = int(frame_number)
    frame_name = f"frame_{frame_idx:06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        frame_idx, train_cameras, frame_name
    )

    patch_positions_tensor: Optional[torch.Tensor] = None
    grid_thw_tensor: Optional[torch.Tensor] = None
    if use_frame_patch_grid and ts_feats.shape[0] > 0:
        pixels = project_3d_to_2d(pos_t, proj_matrix, img_width, img_height)
        pix_x = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, img_width - 1)
        pix_y = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, img_height - 1)

        ph, pw = _qwen25_patch_grid(img_height, img_width)
        EFFECTIVE_PATCH_SIZE = qwen_vl.EFFECTIVE_PATCH_SIZE
        cols = np.clip((pix_x // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, pw - 1)
        rows = np.clip((pix_y // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, ph - 1)
        patch_thw = np.stack([np.zeros_like(rows), rows, cols], axis=-1)

        order = np.lexsort((np.arange(patch_thw.shape[0]), patch_thw[:, 2], patch_thw[:, 1]))
        patch_thw = patch_thw[order]
        ts_feats = ts_feats[order]
        pos_t = pos_t[order]

        patch_positions_tensor = torch.as_tensor(patch_thw, device=model.device)
        grid_thw_tensor = torch.tensor(
            [[1, ph, pw]], dtype=torch.long, device=model.device
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
            vision_patch_positions=patch_positions_tensor,
            vision_patch_grid_thw=grid_thw_tensor,
            zero_vision_positional_encodings=zero_vision_positional_encodings,
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

            top_pixels = project_3d_to_2d(
                top_positions, proj_matrix, img_width, img_height
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

    results = {}
    splat_system_prompt = cfg.eval.spatial.splat_system_prompt
    use_frame_grid = bool(getattr(cfg.eval.spatial, "use_frame_patch_grid_for_splat", False))
    zero_posenc = bool(getattr(cfg.eval.spatial, "splat_zero_positional_encodings", False))

    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        ts_feats = splat_feats[t]
        pos_t = positions[t][splat_indices]

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
            use_frame_patch_grid=use_frame_grid,
            zero_vision_positional_encodings=zero_posenc,
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
            use_frame_patch_grid=use_frame_grid,
            zero_vision_positional_encodings=zero_posenc,
        )

    return results


def _parse_point_from_json(text: str) -> Optional[List[float]]:
    """Extract a single 3D point [x, y, z] from a JSON object in the given text.

    The model is instructed to return pure JSON, but we make this robust by:
      1) locating the first JSON object in the text, attempting json.loads
      2) falling back to regex-based triple float extraction if needed
    """
    import json as _json
    import re as _re

    # Try to find a JSON object in the text
    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            obj = _json.loads(candidate)
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

    # Fallback: extract first three floats
    nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 3:
        try:
            return [float(nums[0]), float(nums[1]), float(nums[2])]
        except Exception:
            return None
    return None


def _parse_pixel_from_json(text: str) -> Optional[List[float]]:
    """Extract a single 2D pixel [x, y] from a JSON object in the text.

    Accepts either {"x":..,"y":..} or {"u":..,"v":..}. Falls back to first two floats.
    """
    import json as _json
    import re as _re

    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            obj = _json.loads(candidate)
            if isinstance(obj, dict):
                if all(k in obj for k in ("x", "y")):
                    return [float(obj["x"]), float(obj["y"])]
                if all(k in obj for k in ("u", "v")):
                    return [float(obj["u"]), float(obj["v"])]
    except Exception:
        pass

    nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 2:
        try:
            return [float(nums[0]), float(nums[1])]
        except Exception:
            return None
    return None


def _format_only_substring(template: str, substring: str) -> str:
    """Safely format only the {substring} placeholder, escaping all other braces.

    This allows examples like {"x": 0.1} in the prompt without triggering str.format
    KeyErrors. Usage: question = _format_only_substring(tmpl, substring)
    """
    # First escape all braces
    safe = template.replace("{", "{{").replace("}", "}}")
    # Then unescape the placeholder we want to actually format
    safe = safe.replace("{{substring}}", "{substring}")
    return safe.format(substring=substring)


def static_graph_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    node_feats_npz,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    timestep_idx: int,
    frame_number: int,
    train_cameras,
    system_prompt: str,
    prompt_template: str,
):
    """Predict a single 3D point per query via Qwen prompted with a static graph.

    Returns same structure as splat_predict_query_list but without scores and with
    exactly one point per mock layer.
    """
    outputs = []

    # Precompute projection for this frame
    frame_name = f"frame_{int(frame_number):06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        int(frame_number), train_cameras, frame_name
    )

    for query in queries_list:
        substring = query["query"]
        question = _format_only_substring(prompt_template, substring)

        # Call Qwen with the graph at this specific timestep
        response = qwen_vl.prompt_with_graph_at_timestep(
            question=question,
            node_feats=node_feats_npz,
            timestep_idx=int(timestep_idx),
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )

        point3d = _parse_point_from_json(response)
        if point3d is None:
            # 3D failure: fall back to 2D corner pixel (0,0), omit 3D position to avoid skew
            out_item = {"query": substring, "predictions": {}, "raw_response": response}
            out_item["predictions"]["0"] = {
                "pixel_coords": [[0.0, 0.0]],
                "positions": [],
            }
            outputs.append(out_item)
            continue

        # Project to pixel coords
        pos_arr = np.array(point3d, dtype=np.float32).reshape(1, 3)
        pixels = project_3d_to_2d(pos_arr, proj_matrix, img_width, img_height)

        out_item = {"query": substring, "predictions": {}, "raw_response": response}
        # Single mock layer key for static baseline
        out_item["predictions"]["0"] = {
            "pixel_coords": pixels.tolist(),
            "positions": pos_arr.tolist(),
        }
        outputs.append(out_item)

    return outputs


def static_graph_feat_queries(
    *,
    model,
    processor,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run static-graph prompting baseline across all queries for a clip.

    Loads static graph artifacts and returns grouped results per timestep
    with the same structure as splat_feat_queries (sans scores).
    """
    graph_dir = Path(graph_dir)

    # Required static graph artifacts
    node_feats_npz_path = graph_dir / "c_qwen_feats.npz"
    adjacency_path = graph_dir / "graph.npy"
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"

    node_feats_npz = np.load(node_feats_npz_path)
    adjacency_matrices = np.load(adjacency_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)

    # Cameras for projection
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    system_prompt = cfg.eval.spatial.static_graph_system_prompt
    prompt_template = cfg.eval.spatial.static_graph_prompt_template

    results: Dict[str, Any] = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        frame_number = int(timestep_queries["frame_number"])  # local idx

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = static_graph_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        )

        results[timestep]["actions"] = static_graph_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        )

    return results


def splat_graph_predict_query_list(
    queries_list,
    *,
    model_spatial,
    processor_spatial,
    ts_feats: np.ndarray,
    pos_t: np.ndarray,
    attn_layer: int,
    timestep_idx: int,
    frame_number: int,
    train_cameras,
    node_feats_npz,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model,
    processor,
    system_prompt_splat_attn: str,
    static_graph_system_prompt: str,
    prompt_template_attn: str,
    prompt_template_graph: str,
    max_proposals: int | None = None,
    include_scores_in_context: bool = False,
    use_frame_patch_grid: bool = False,
    zero_vision_positional_encodings: bool = False,
):
    """Use SPLAT attention to propose 3D points, then refine via static-graph prompting.

    Returns items with a single mock layer key "0" like static_graph baseline.
    """
    outputs = []

    # Precompute projection for this frame
    frame_name = f"frame_{int(frame_number):06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        int(frame_number), train_cameras, frame_name
    )

    patch_positions_tensor: Optional[torch.Tensor] = None
    grid_thw_tensor: Optional[torch.Tensor] = None
    if use_frame_patch_grid and ts_feats.shape[0] > 0:
        pixels = project_3d_to_2d(pos_t, proj_matrix, img_width, img_height)
        pix_x = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, img_width - 1)
        pix_y = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, img_height - 1)

        ph, pw = _qwen25_patch_grid(img_height, img_width)
        EFFECTIVE_PATCH_SIZE = qwen_vl.EFFECTIVE_PATCH_SIZE
        cols = np.clip((pix_x // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, pw - 1)
        rows = np.clip((pix_y // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, ph - 1)
        patch_thw = np.stack([np.zeros_like(rows), rows, cols], axis=-1)

        order = np.lexsort((np.arange(patch_thw.shape[0]), patch_thw[:, 2], patch_thw[:, 1]))
        patch_thw = patch_thw[order]
        ts_feats = ts_feats[order]
        pos_t = pos_t[order]

        patch_positions_tensor = torch.as_tensor(patch_thw, device=model_spatial.device)
        grid_thw_tensor = torch.tensor(
            [[1, ph, pw]], dtype=torch.long, device=model_spatial.device
        )

    feats_tensor = torch.tensor(ts_feats, device=model_spatial.device)

    for query in queries_list:
        substring = query["query"]
        prompt_attn = prompt_template_attn.format(substring=substring)
        if feats_tensor.shape[0] == 0:
            proposals_list = []
        else:
            attn_out = extract_text_to_vision_attention(
                model=model_spatial,
                processor=processor_spatial,
                vision_features=feats_tensor,
                layers=[attn_layer],
                prompt=prompt_attn,
                substring=substring,
                system_prompt=system_prompt_splat_attn,
                vision_patch_positions=patch_positions_tensor,
                vision_patch_grid_thw=grid_thw_tensor,
                zero_vision_positional_encodings=zero_vision_positional_encodings,
            )
            attn_scores = attn_out["scores"]  # [1, Q, V]

            # Collect proposals across layers (average over query tokens)
            proposals_list = []
            layer_scores = attn_scores[0].mean(dim=0)  # [V], mean over query tokens only
            k = max_proposals if max_proposals is not None else layer_scores.shape[-1]
            top_scores, top_indices = layer_scores.topk(k=k, sorted=True)
            top_indices_np = top_indices.detach().cpu().numpy()
            top_scores_np = top_scores.detach().cpu().numpy()
            for rank_idx, (orig_i, score_val) in enumerate(
                zip(top_indices_np, top_scores_np)
            ):
                xyz = pos_t[orig_i]
                proposals_list.append(
                    {
                        "xyz": xyz,
                        "score": float(score_val),
                        "layer": int(attn_layer),
                        "rank": int(rank_idx + 1),
                    }
                )

        # Build question for static-graph step, embedding proposals textually
        question = _format_only_substring(prompt_template_graph, substring)
        if include_scores_in_context:
            lines = [
                "Candidate 3D proposals (x, y, z) with attention scores:",
                *[
                    f"- {p['xyz'][0]:.4f}, {p['xyz'][1]:.4f}, {p['xyz'][2]:.4f}"
                    # f" (score={p['score']:.4f}, layer={p['layer']}, rank={p['rank']})"
                    f" (score={p['score']:.4f}, rank={p['rank']})"
                    for p in proposals_list
                ],
            ]
        else:
            lines = [
                "Candidate 3D proposals (x, y, z):",
                *[
                    f"- {p['xyz'][0]:.4f}, {p['xyz'][1]:.4f}, {p['xyz'][2]:.4f}"
                    for p in proposals_list
                ],
            ]
        question = question + "\n\n" + "\n".join(lines)

        response = qwen_vl.prompt_with_graph_at_timestep(
            question=question,
            node_feats=node_feats_npz,
            timestep_idx=int(timestep_idx),
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            system_prompt=static_graph_system_prompt,
        )

        point3d = _parse_point_from_json(response)
        if point3d is None:
            # 3D failure: fall back to 2D corner pixel (0,0), omit 3D position to avoid skew
            out_item = {"query": substring, "predictions": {}, "raw_response": response}
            out_item["predictions"]["0"] = {
                "pixel_coords": [[0.0, 0.0]],
                "positions": [],
            }
            outputs.append(out_item)
            continue

        pos_arr = np.array(point3d, dtype=np.float32).reshape(1, 3)
        pixels = project_3d_to_2d(pos_arr, proj_matrix, img_width, img_height)

        out_item = {"query": substring, "predictions": {}, "raw_response": response}
        out_item["predictions"]["0"] = {
            "pixel_coords": pixels.tolist(),
            "positions": pos_arr.tolist(),
        }
        outputs.append(out_item)

    return outputs


def splat_graph_feat_queries(
    *,
    model_spatial,
    processor_spatial,
    model,
    processor,
    splat_feats: np.ndarray,
    splat_indices: np.ndarray,
    positions: np.ndarray,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run SPLAT->proposals + static-graph refinement across all queries for a clip."""
    graph_dir = Path(graph_dir)

    # Load static graph artifacts
    node_feats_npz_path = graph_dir / "c_qwen_feats.npz"
    adjacency_path = graph_dir / "graph.npy"
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"

    node_feats_npz = np.load(node_feats_npz_path)
    adjacency_matrices = np.load(adjacency_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)

    # Cameras for projection
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    system_prompt_splat_attn = cfg.eval.spatial.splat_system_prompt
    static_graph_system_prompt = cfg.eval.spatial.splat_graph_system_prompt
    prompt_template_attn = cfg.eval.spatial.splat_prompt_template
    prompt_template_graph = cfg.eval.spatial.splat_graph_prompt_template
    attn_layer = cfg.eval.spatial.splat_graph_attn_layer
    max_props = cfg.eval.spatial.splat_graph_max_proposals
    include_scores = cfg.eval.spatial.splat_graph_include_scores_in_context

    results: Dict[str, Any] = {}
    use_frame_grid = bool(getattr(cfg.eval.spatial, "use_frame_patch_grid_for_splat_graph", False))
    zero_posenc = bool(getattr(cfg.eval.spatial, "splat_graph_zero_positional_encodings", False))
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        ts_feats = splat_feats[t]
        pos_t = positions[t][splat_indices]
        frame_number = int(timestep_queries["frame_number"])  # local idx

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = splat_graph_predict_query_list(
            timestep_queries.get("objects", []),
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            ts_feats=ts_feats,
            pos_t=pos_t,
            attn_layer=attn_layer,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            system_prompt_splat_attn=system_prompt_splat_attn,
            static_graph_system_prompt=static_graph_system_prompt,
            prompt_template_attn=prompt_template_attn,
            prompt_template_graph=prompt_template_graph,
            max_proposals=max_props,
            include_scores_in_context=include_scores,
            use_frame_patch_grid=use_frame_grid,
            zero_vision_positional_encodings=zero_posenc,
        )

        results[timestep]["actions"] = splat_graph_predict_query_list(
            timestep_queries.get("actions", []),
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            ts_feats=ts_feats,
            pos_t=pos_t,
            attn_layer=attn_layer,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            system_prompt_splat_attn=system_prompt_splat_attn,
            static_graph_system_prompt=static_graph_system_prompt,
            prompt_template_attn=prompt_template_attn,
            prompt_template_graph=prompt_template_graph,
            max_proposals=max_props,
            include_scores_in_context=include_scores,
            use_frame_patch_grid=use_frame_grid,
            zero_vision_positional_encodings=zero_posenc,
        )

    return results


def _qwen25_patch_grid(im_height: int, im_width: int) -> tuple[int, int]:
    """Compute Qwen2.5-VL patch grid (H, W) for a given image size.

    Mirrors qwen_vl.get_patch_segmasks grid math.
    """
    PATCH_SIZE = qwen_vl.PATCH_SIZE
    EFFECTIVE_PATCH_SIZE = qwen_vl.EFFECTIVE_PATCH_SIZE
    patches_height = im_height // EFFECTIVE_PATCH_SIZE + (
        (im_height // PATCH_SIZE) % 4 == 3
    )
    patches_width = im_width // EFFECTIVE_PATCH_SIZE + (
        (im_width // PATCH_SIZE) % 4 == 3
    )
    return int(patches_height), int(patches_width)


def _patch_indices_to_pixel_centers(
    patch_indices: np.ndarray, img_h: int, img_w: int
) -> np.ndarray:
    """Map flat patch indices (row-major) to pixel centers in the original image.

    Returns array of shape (N, 2) with (x, y) in pixel coordinates.
    """
    EFFECTIVE_PATCH_SIZE = qwen_vl.EFFECTIVE_PATCH_SIZE
    ph, pw = _qwen25_patch_grid(img_h, img_w)
    cols = patch_indices % pw
    rows = patch_indices // pw
    xs = (cols + 0.5) * EFFECTIVE_PATCH_SIZE
    ys = (rows + 0.5) * EFFECTIVE_PATCH_SIZE
    xs = np.clip(xs, 0, img_w - 1)
    ys = np.clip(ys, 0, img_h - 1)
    return np.stack([xs, ys], axis=-1)


def _compute_gaussian_patch_assignment_thw(
    *,
    pos_t: np.ndarray,
    train_cameras,
    frame_number: int,
) -> np.ndarray:
    """Return per-gaussian [t,h,w] indices on the frame's Qwen patch grid (no filtering).

    t is fixed to 0. h,w are computed by projecting to pixels, clamping to the frame,
    then mapping to Qwen's EFFECTIVE_PATCH_SIZE grid and bounding within [0..H-1],[0..W-1].
    """
    local_idx = int(frame_number)
    frame_name = f"frame_{local_idx:06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        local_idx, train_cameras, frame_name
    )

    pixels = project_3d_to_2d(pos_t, proj_matrix, img_width, img_height)
    pix_x = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, img_width - 1)
    pix_y = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, img_height - 1)

    EFFECTIVE_PATCH_SIZE = qwen_vl.EFFECTIVE_PATCH_SIZE
    ph, pw = _qwen25_patch_grid(img_height, img_width)

    cols = np.clip((pix_x // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, pw - 1)
    rows = np.clip((pix_y // EFFECTIVE_PATCH_SIZE).astype(np.int64), 0, ph - 1)
    t = np.zeros_like(rows)
    return np.stack([t, rows, cols], axis=-1)


def frame_attn_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    max_proposals: int,
    image: Image.Image,
    layers: List[int],
    prompt_template: str,
    system_prompt: str,
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        prompt = prompt_template.format(substring=substring)
        attn_out = extract_text_to_image_attention(
            model=model,
            processor=processor,
            image=image,
            layers=layers,
            prompt=prompt,
            substring=substring,
            system_prompt=system_prompt,
        )
        attn_scores = attn_out["scores"]  # [L, Q, V]

        out_item = {"query": substring, "predictions": {}}
        for layer_idx, layer in enumerate(layers):
            layer_scores = attn_scores[layer_idx]  # [Q, V]
            layer_scores = layer_scores.mean(dim=0)  # [V]
            top_scores, top_indices = layer_scores.topk(k=max_proposals, sorted=True)
            top_scores = top_scores.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()

            # Map patch indices to pixel centers
            pixels = _patch_indices_to_pixel_centers(
                top_indices, image.height, image.width
            )

            out_item["predictions"][layer] = {
                "scores": top_scores.tolist(),
                "pixel_coords": pixels.tolist(),
            }
        outputs.append(out_item)
    return outputs


def frame_attn_refine_predict_query_list(
    queries_list,
    *,
    model_spatial,
    processor_spatial,
    model,
    processor,
    image: Image.Image,
    attn_layer: int,
    top_k: int,
    prompt_template_attn: str,
    system_prompt_attn: str,
    prompt_template_refine: str,
    system_prompt_refine: str,
    include_scores_in_context: bool = False,
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        prompt_attn = prompt_template_attn.format(substring=substring)
        attn_out = extract_text_to_image_attention(
            model=model_spatial,
            processor=processor_spatial,
            image=image,
            layers=[attn_layer],
            prompt=prompt_attn,
            substring=substring,
            system_prompt=system_prompt_attn,
        )
        attn_scores = attn_out["scores"]  # [1, Q, V]
        layer_scores = attn_scores[0].mean(dim=0)  # [V]
        top_scores, top_indices = layer_scores.topk(k=top_k, sorted=True)
        top_indices_np = top_indices.detach().cpu().numpy()
        top_scores_np = top_scores.detach().cpu().numpy()

        # Map patch indices to pixel centers
        pixels = _patch_indices_to_pixel_centers(
            top_indices_np, image.height, image.width
        )

        # Prepare refine prompt with proposals
        question = _format_only_substring(prompt_template_refine, substring)
        if pixels.shape[0] > 0:
            if include_scores_in_context:
                lines = [
                    "Candidate pixel proposals (x, y) with attention scores:",
                    *[
                        f"- {float(p[0]):.1f}, {float(p[1]):.1f} (score={float(s):.4f}, rank={i + 1})"
                        for i, (p, s) in enumerate(zip(pixels, top_scores_np))
                    ],
                ]
            else:
                lines = [
                    "Candidate pixel proposals (x, y):",
                    *[f"- {float(p[0]):.1f}, {float(p[1]):.1f}" for p in pixels],
                ]
            question = question + "\n\n" + "\n".join(lines)

        # Ask normal Qwen with the image
        response = qwen_vl.ask_qwen_about_image(
            image=image,
            prompt=question,
            model=model,
            processor=processor,
            system_prompt=system_prompt_refine,
        )

        px = _parse_pixel_from_json(response)
        if px is None:
            # fallback to the top-1 proposal
            if pixels.shape[0] > 0:
                px = [float(pixels[0, 0]), float(pixels[0, 1])]
            else:
                px = [0.0, 0.0]

        out_item = {"query": substring, "predictions": {}, "raw_response": response}
        out_item["predictions"]["0"] = {
            "pixel_coords": [px],
        }
        outputs.append(out_item)
    return outputs


def frame_attn_refine_feat_queries(
    *,
    model_spatial,
    processor_spatial,
    model,
    processor,
    preprocessed_root: Path | str,
    images_subdir: str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    results: Dict[str, Any] = {}
    images_dir = Path(preprocessed_root) / clip.name / images_subdir

    attn_layer = cfg.eval.spatial.frame_attn_refine_attn_layer
    prompt_template_attn = cfg.eval.spatial.frame_attn_prompt_template
    system_prompt_attn = cfg.eval.spatial.frame_attn_system_prompt
    prompt_template_refine = cfg.eval.spatial.frame_attn_refine_prompt_template
    system_prompt_refine = cfg.eval.spatial.frame_attn_refine_system_prompt
    include_scores = cfg.eval.spatial.frame_attn_refine_include_scores_in_context

    for timestep, timestep_queries in clip_gt.items():
        frame_number = int(timestep_queries["frame_number"])  # local idx
        frame_path = images_dir / f"frame_{frame_number:06d}.jpg"
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = frame_attn_refine_predict_query_list(
            timestep_queries.get("objects", []),
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            model=model,
            processor=processor,
            image=image,
            attn_layer=attn_layer,
            top_k=cfg.eval.spatial.frame_attn_refine_max_proposals,
            prompt_template_attn=prompt_template_attn,
            system_prompt_attn=system_prompt_attn,
            prompt_template_refine=prompt_template_refine,
            system_prompt_refine=system_prompt_refine,
            include_scores_in_context=include_scores,
        )

        results[timestep]["actions"] = frame_attn_refine_predict_query_list(
            timestep_queries.get("actions", []),
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            model=model,
            processor=processor,
            image=image,
            attn_layer=attn_layer,
            top_k=cfg.eval.spatial.frame_attn_refine_max_proposals,
            prompt_template_attn=prompt_template_attn,
            system_prompt_attn=system_prompt_attn,
            prompt_template_refine=prompt_template_refine,
            system_prompt_refine=system_prompt_refine,
            include_scores_in_context=include_scores,
        )

    return results


def frame_attn_feat_queries(
    *,
    model,
    processor,
    preprocessed_root: Path | str,
    images_subdir: str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run 2D frame attention baseline across all queries for a clip.

    Uses text-to-vision attention on the original frame and maps top-k patches to pixel centers.
    """
    results: Dict[str, Any] = {}
    images_dir = Path(preprocessed_root) / clip.name / images_subdir
    layers = cfg.eval.spatial.layers
    prompt_template = cfg.eval.spatial.frame_attn_prompt_template
    system_prompt = cfg.eval.spatial.frame_attn_system_prompt

    for timestep, timestep_queries in clip_gt.items():
        frame_number = int(timestep_queries["frame_number"])  # local idx
        frame_path = images_dir / f"frame_{frame_number:06d}.jpg"
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = frame_attn_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            image=image,
            layers=layers,
            max_proposals=cfg.eval.spatial.frame_attn_refine_max_proposals,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )

        results[timestep]["actions"] = frame_attn_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            image=image,
            layers=layers,
            max_proposals=cfg.eval.spatial.frame_attn_refine_max_proposals,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )

    return results


def frame_direct_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    image: Image.Image,
    prompt_template: str,
    system_prompt: str,
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        question = _format_only_substring(prompt_template, substring)

        response = qwen_vl.ask_qwen_about_image(
            image=image,
            prompt=question,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )

        px = _parse_pixel_from_json(response)
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
        frame_path = images_dir / f"frame_{frame_number:06d}.jpg"
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = frame_direct_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            image=image,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )

        results[timestep]["actions"] = frame_direct_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            image=image,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )

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
        frame_path = images_dir / f"frame_{frame_number:06d}.jpg"
        if not frame_path.exists():
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
