from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import List

import qwen_vl
from scene.dataset_readers import CameraInfo, readColmapSceneInfo
from scene.cameras import Camera


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
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using a mock image sized to match the number of patch features
    inputs = qwen_vl.model_inputs(messages, [vision_features], processor).to(
        model.device
    )

    with torch.no_grad():
        # Not generating any output tokens here
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            custom_patch_features=[vision_features],
        )

    all_attn = list(outputs.attentions)

    # Identify vision token positions via <|image_pad|> placeholders
    input_ids = inputs.input_ids[0]
    image_pad_token = processor.tokenizer.encode(
        "<|image_pad|>", add_special_tokens=False
    )[0]
    image_pad_positions = (input_ids == image_pad_token).nonzero(as_tuple=True)[0]
    if image_pad_positions.numel() == 0:
        raise ValueError("No <|image_pad|> tokens found in input sequence.")
    vision_token_indices = image_pad_positions.tolist()

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


def project_3d_to_2d(
    positions: np.ndarray, proj_matrix: torch.Tensor, img_width: int, img_height: int
) -> np.ndarray:
    # Expecting positions to be (N, 3)
    assert positions.shape[1] == 3, "Positions must be (N, 3)"

    # Conver to homogeneous coordinates
    positions = torch.tensor(positions, device=proj_matrix.device, dtype=proj_matrix.dtype)
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
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        prompt = prompt_template.format(substring=substring)
        attn_out = extract_text_to_vision_attention(
            model=model,
            processor=processor,
            vision_features=torch.tensor(ts_feats, device=model.device),
            layers=layers,
            prompt=prompt,
            substring=substring,
        )
        attn_scores = attn_out["scores"]

        out_item = {"query": substring, "predictions": {}}
        for layer_idx, layer in enumerate(layers):
            layer_scores = attn_scores[layer_idx]
            layer_scores = layer_scores.mean(dim=0)
            top_scores, top_indices = layer_scores.topk(k=top_k, sorted=True)
            top_scores = top_scores.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()

            top_positions = pos_t[top_indices]

            frame_name = f"{clip_name}_{frame_number:06d}.jpg"
            proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
                frame_number, train_cameras, frame_name
            )
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

    # positions are (T, N, 3); we'll subset per-timestep inside the loop

    results = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        ts_feats = splat_feats[t]
        # Subset positions for this timestep and the chosen splat indices
        pos_t = positions[t][splat_indices]

        # Prepare grouped containers matching GT schema
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
        )

    return results