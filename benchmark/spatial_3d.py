"""
Spatial query evaluation using 3D Gaussian-to-grid positional encoding.

This module provides graph_agent_3d_feat_queries which uses custom positional encodings
based on Gaussian spatial positions, enabling spatial queries over 3D Gaussian splatting data.
No tools are used - the model directly processes Gaussian features with spatial positional encoding.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from loguru import logger

from benchmark.spatial import (
    get_coord_transformations,
    get_proj_matrix_from_timestep,
    project_3d_to_2d,
)
from typing import List, Dict, Any

from llm.qwen_utils import (
    model_inputs,
    qwen3_cat_to_deepstack_multiple,
    _set_generation_seed,
    NEW_TOKEN_LIMIT,
    THINKING_TOKEN_LIMIT,
    ThinkingTokenBudgetProcessor,
)
from transformers import LogitsProcessorList

import rerun as rr


def compute_gaussian_grid_thw(
    gaussian_means: np.ndarray,  # (N, 3)
    patch_size: float = 0.01,
) -> torch.Tensor:
    """Compute image_grid_thw from Gaussian means.

    Args:
        gaussian_means: (N, 3) array with [x, y, z]
        patch_size: Size of each grid patch

    Returns:
        image_grid_thw: (1, 3) tensor with [t, h, w] in BEFORE spatial merge space
    """
    # Drop z axis
    xy = gaussian_means[:, :2]  # (N, 2)

    # Compute extent
    min_x, min_y = xy.min(axis=0)
    max_x, max_y = xy.max(axis=0)
    logger.info(
        f"Scene min/max: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}"
    )

    extent_x = max_x - min_x
    extent_y = max_y - min_y
    logger.info(f"Scene extent: extent_x={extent_x}, extent_y={extent_y}")

    # Compute grid dimensions (BEFORE spatial merge)
    # The model will apply spatial_merge_size internally in get_rope_index
    grid_w = int(np.ceil(extent_x / patch_size)) + 1
    grid_h = int(np.ceil(extent_y / patch_size)) + 1

    # image_grid_thw uses dimensions BEFORE spatial merge
    # get_rope_index will apply spatial_merge_size internally
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    logger.info(f"Image grid thw: {image_grid_thw}")

    return image_grid_thw, min_x, min_y


def create_gaussian_to_grid_mapping(
    gaussian_means: np.ndarray,  # (N, 3) world coordinates
    image_grid_thw: torch.Tensor,  # (1, 3) with [t, h, w]
    patch_size: float = 0.01,
) -> torch.Tensor:
    """Map Gaussian means to grid positions. Also sort the gaussians according to their grid positions.

    Args:
        gaussian_means: (N, 3) array with [x, y, z]
        image_grid_thw: (1, 3) tensor with [t, h, w] grid dimensions (BEFORE merge)
        patch_size: Size of each grid patch

    Returns:
        gaussian_to_grid_mapping: (N, 2) tensor with [h_idx, w_idx] for each Gaussian
        sorted_indices: (N,) array with original indices of sorted Gaussians
            Indices are in BEFORE spatial merge space (raw grid space)
    """
    # Extract grid dimensions
    _, grid_h, grid_w = image_grid_thw[0].tolist()

    # Drop z axis and compute grid indices
    xy = gaussian_means[:, :2]  # (N, 2)

    # Compute extent to get origin offset
    min_x, min_y = xy.min(axis=0)
    max_x, max_y = xy.max(axis=0)

    # Map to grid indices (0-indexed)
    # x -> w (width), y -> h (height)
    w_indices = ((xy[:, 0] - min_x) / patch_size).astype(np.int64)
    h_indices = ((xy[:, 1] - min_y) / patch_size).astype(np.int64)

    # Check that all indices end up within the grid
    assert (w_indices >= 0).all() and (w_indices < grid_w).all()
    assert (h_indices >= 0).all() and (h_indices < grid_h).all()

    # Stack into (N, 2) with [h_idx, w_idx]
    mapping = torch.tensor(np.stack([h_indices, w_indices], axis=1), dtype=torch.long)

    # Sort the gaussian means
    sorted_indices = np.lexsort((w_indices, h_indices))

    return mapping, sorted_indices


def sample_gaussians(
    gaussian_means: np.ndarray,
    gaussian_features: Optional[torch.Tensor] = None,
    sample_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, Optional[torch.Tensor], np.ndarray]:
    """Sample a subset of Gaussians for efficiency.

    Args:
        gaussian_means: (N, 3) array with Gaussian means
        gaussian_features: Optional (N, D) array with Gaussian features
        sample_ratio: Fraction of Gaussians to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_means, sampled_features, indices) where:
            sampled_means: (M, 3) array with sampled Gaussian means
            sampled_features: (M, D) array with sampled features, or None
            indices: (M,) array with original indices of sampled Gaussians
    """

    num_gaussians = len(gaussian_means)
    num_samples = int(num_gaussians * sample_ratio)

    rng = np.random.RandomState(seed)
    indices = rng.choice(num_gaussians, size=num_samples, replace=False)
    indices = np.sort(indices)  # Keep sorted for consistency

    sampled_means = gaussian_means[indices]

    if gaussian_features is not None:
        # gaussian_features is a torch.Tensor on whatever device the caller chose.
        # We index it using torch to avoid unnecessary CPU transfers.
        idx_tensor = torch.from_numpy(indices).to(gaussian_features.device)
        sampled_features = gaussian_features.index_select(0, idx_tensor)
        return sampled_means, sampled_features, indices

    return sampled_means, None, indices


def generate_with_vision_features_3d(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    image_grid_thw: torch.Tensor,
    gaussian_to_grid_mapping: torch.Tensor,
    gaussians_per_image: List[int],
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
    zero_positional_encodings: bool = False,
) -> str:
    """Generate text from vision features with 3D Gaussian-to-grid positional encoding.

    Args:
        messages: Chat messages with image placeholders
        vision_features: List of vision feature tensors.
            Each tensor is (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
        model: CustomQwen3VLForConditionalGeneration3D model instance
        processor: Qwen3VLProcessor instance
        image_grid_thw: (num_images, 3) tensor with [t, h, w] grid dimensions (BEFORE merge)
        gaussian_to_grid_mapping: (total_gaussians, 2) tensor with [h_idx, w_idx] for each Gaussian
        gaussians_per_image: List of number of Gaussians per image
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for deterministic sampling
        max_thinking_tokens: Maximum tokens for thinking phase (Qwen3 only).
            If None, no limit is applied. If 0, thinking is disabled immediately.
        zero_positional_encodings: Whether to zero out h and w for positional encodings

    Returns:
        Generated text response
    """
    # Build logits processor for thinking budget (Qwen3 only)
    logits_processor = None
    if max_thinking_tokens is not None:
        thinking_processor = ThinkingTokenBudgetProcessor(
            processor.tokenizer, max_thinking_tokens=max_thinking_tokens
        )
        logits_processor = LogitsProcessorList([thinking_processor])

    main_features, deepstack_features = qwen3_cat_to_deepstack_multiple(vision_features)

    logger.info("=" * 80)
    logger.info("PREPROCESSING INPUTS")
    logger.info("=" * 80)
    
    # Preprocess and generate
    image_token_id = model.model.config.image_token_id
    logger.info(f"image_token_id: {image_token_id}")

    image_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    logger.info(f"image_pad_token_id: {image_pad_token_id}")

    vision_start_token_id = model.model.config.vision_start_token_id
    logger.info(f"vision_start_token_id: {vision_start_token_id}")
    
    vision_end_token_id = model.model.config.vision_end_token_id
    logger.info(f"vision_end_token_id: {vision_end_token_id}")
    
    # Log messages before processing
    logger.info(f"Messages structure:")
    for i, msg in enumerate(messages):
        logger.info(f"  Message {i}: role={msg['role']}")
        for j, content_item in enumerate(msg.get("content", [])):
            if content_item.get("type") == "text":
                text_preview = content_item.get("text", "")[:200]
                logger.info(f"    Content {j}: type=text, preview='{text_preview}...' (total {len(content_item.get('text', ''))} chars)")
            else:
                logger.info(f"    Content {j}: type={content_item.get('type')}")
    
    inputs = model_inputs(
        messages, main_features, processor, 
        image_grid_thw=image_grid_thw,
        gaussians_per_image=gaussians_per_image,
        image_token_id=image_token_id,
        image_pad_token_id=image_pad_token_id,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
    ).to(model.device)

    # Decode and log input_ids
    input_ids = inputs["input_ids"]
    logger.info(f"input_ids shape: {input_ids.shape}")
    
    # Decode full input sequence
    decoded_input = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    torch.set_printoptions(threshold=10000)
    print(f"Decoded input: {decoded_input}")
    
    # Count tokens
    num_tokens = input_ids.shape[1]
    logger.info(f"Total input tokens: {num_tokens}")
    
    # Check model config limits
    model_max_length = getattr(model.config, "max_position_embeddings", None)
    if model_max_length:
        logger.info(f"Model max_position_embeddings: {model_max_length}")
        if num_tokens > model_max_length:
            logger.warning(f"⚠️ INPUT EXCEEDS MODEL MAX LENGTH! {num_tokens} > {model_max_length}")
    
    # Log generation parameters
    _set_generation_seed(seed)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "custom_patch_features": main_features,
        "custom_deepstack_features": deepstack_features,
        "zero_image_hw": zero_positional_encodings,
        "gaussian_to_grid_mapping": gaussian_to_grid_mapping.to(model.device),
        "gaussians_per_image": gaussians_per_image,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor
    
    logger.info("=" * 80)
    logger.info("GENERATION PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"max_new_tokens: {max_new_tokens}")
    logger.info(f"max_thinking_tokens: {max_thinking_tokens}")
    logger.info(f"seed: {seed}")
    logger.info(f"zero_positional_encodings: {zero_positional_encodings}")
    logger.info(f"gaussians_per_image: {gaussians_per_image}")
    logger.info(f"image_grid_thw: {image_grid_thw}")
    logger.info(f"gaussian_to_grid_mapping shape: {gaussian_to_grid_mapping.shape}")
    logger.info(f"main_features: {[f.shape for f in main_features]}")
    logger.info(f"deepstack_features: {[f.shape for f in deepstack_features] if deepstack_features else None}")
    
    logger.info("=" * 80)
    logger.info("CALLING model.generate()")
    logger.info("=" * 80)

    # Will go into forward pass based on the image grid thw before spatial merge and the correct input id sequence with enough placeholders for the selected Gaussian tokens
    generated_ids = model.generate(**inputs, **generate_kwargs)
    
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    logger.info(f"Generated length: {generated_ids.shape[1]} tokens")
    logger.info(f"New tokens generated: {generated_ids.shape[1] - inputs['input_ids'].shape[1]}")
    logger.info(f"max_new_tokens limit: {max_new_tokens}")
    
    # Check if we hit the limit
    new_tokens = generated_ids.shape[1] - inputs['input_ids'].shape[1]
    if new_tokens >= max_new_tokens:
        logger.warning(f"⚠️ HIT max_new_tokens LIMIT! Generated {new_tokens} tokens (limit: {max_new_tokens})")
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    # Decode with and without special tokens to see what's happening
    output_text_raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    logger.info(f"Raw response (with special tokens): {output_text_raw[0]}")
    logger.info(f"Cleaned response (no special tokens): {output_text[0]}")
    logger.info(f"Response length: {len(output_text[0])} chars")
    
    return output_text[0]


def filter_gaussian_tokens(gaussian_to_grid_mapping: np.ndarray, depth: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Filter Gaussians based on depth and opacity.

    Args:
        gaussian_to_grid_mapping: (N, 2) array of [h_idx, w_idx] for each Gaussian
        depth: (N,) array of depth values for each Gaussian
        opacity: (N,) array of opacity values for each Gaussian

    Returns:
        filtered_indices: (M,) array of indices of the filtered Gaussians
    """

    # TODO: if we keep using this, it must go into hydra configs
    filtered_depth = depth.copy()
    filtered_depth[opacity < 0.8] = np.inf

    # Get the unique mappings
    _, unique_start_indices, unique_counts = np.unique(gaussian_to_grid_mapping, axis=0, return_index=True, return_counts=True)
    
    final_indices = []
    for unique_start_index, unique_count in zip(unique_start_indices, unique_counts):
        print(f"unique_start_index: {unique_start_index}, unique_count: {unique_count}")
        relevant_depths = filtered_depth[unique_start_index:unique_start_index+unique_count]
        print(f"relevant_depths: {relevant_depths}")

        # If all depths are infinity, we take the shallowest depth from the original depth
        if np.all(relevant_depths == np.inf):
            shallowest_depth_index = np.argmin(depth[unique_start_index:unique_start_index+unique_count])
        else:
            shallowest_depth_index = np.argmin(relevant_depths)

        print(f"shallowest_depth: {relevant_depths[shallowest_depth_index]}")
        final_indices.append(unique_start_index + shallowest_depth_index)

    print(f"final_indices: {final_indices}")

    return np.array(final_indices)


# Entry point to the 3D logic of custom grid etc.
def graph_agent_3d_feat_queries(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run spatial queries with 3D Gaussian-to-grid positional encoding.

    Uses custom positional encodings based on Gaussian spatial positions, enabling
    spatial queries over 3D Gaussian splatting data with many-to-one grid mapping.
    No tools are used - the model directly processes Gaussian features.

    Args:
        model: CustomQwen3VLForConditionalGeneration3D model instance
        processor: Qwen3VLProcessor instance
        graph_dir: Path to graph directory containing Gaussian data
        clip_gt: Ground truth queries per timestep
        clip: Clip configuration
        cfg: Evaluation configuration

    Returns:
        Dictionary of results per timestep
    """
    graph_dir = Path(graph_dir)

    # Load Gaussian positions (means)
    positions_path = graph_dir / "positions.npy"
    if not positions_path.exists():
        raise FileNotFoundError(
            f"Gaussian positions not found at {positions_path}. "
            "This function requires positions.npy with Gaussian means."
        )
    positions = np.load(positions_path)  # Shape: (timesteps, N, 3)
    logger.info(f"Loaded positions of shape: {positions.shape}")

    # Load patch latents (need to decode with autoencoder to get full Qwen features)
    # Unlike graph agent which uses pre-decoded cluster features, we decode on-demand per timestep
    patch_latents_path = graph_dir / "patch_latents_through_time.npy"
    if not patch_latents_path.exists():
        raise FileNotFoundError(
            f"Patch latents not found at {patch_latents_path}. "
            "This function requires patch_latents_through_time.npy."
        )
    patch_latents = np.load(patch_latents_path)  # Shape: (timesteps, N, lang_dim)
    logger.info(f"Loaded patch latents of shape: {patch_latents.shape}")

    # Load per-gaussian RGB colors (same as used for rerun RGB logging in extract_graphs.py)
    colors_path = graph_dir / "colors_rgb.npy"
    gaussian_colors = np.load(colors_path)  # (N, 3) uint8 in [0, 255]
    logger.info(f"Loaded gaussian colors of shape: {gaussian_colors.shape}")

    # Load opacities through time
    opacities_path = graph_dir / "opacities_through_time.npy"
    if not opacities_path.exists():
        raise FileNotFoundError(
            f"Opacities not found at {opacities_path}. "
            "This function requires opacities_through_time.npy."
        )
    opacities = np.load(opacities_path)  # Shape: (timesteps, gaussians)
    logger.info(f"Loaded opacities of shape: {opacities.shape}")

    # Load autoencoder to decode latents to full Qwen features (on-demand per timestep)
    # Use same config as graph_extraction (not graph_agent config)
    if cfg.graph_extraction.use_global_autoencoder:
        logger.info(f"Using global autoencoder at {cfg.graph_extraction.global_autoencoder_checkpoint_dir}")
        autoencoder_path = (
            Path(cfg.preprocessed_root)
            / cfg.graph_extraction.global_autoencoder_checkpoint_dir
            / "best_ckpt.pth"
        )
    else:
        logger.info(f"Using local autoencoder at {clip.name}/{cfg.graph_extraction.checkpoint_subdir}")
        clip_dir = Path(cfg.preprocessed_root) / clip.name
        autoencoder_path = (
            clip_dir
            / cfg.graph_extraction.checkpoint_subdir
            / "best_ckpt.pth"
        )
    from autoencoder.model_qwen import QwenAutoencoder

    autoencoder = QwenAutoencoder(
        input_dim=cfg.graph_extraction.full_dim,
        latent_dim=cfg.graph_extraction.latent_dim,
    ).to(model.device)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, map_location=model.device)
    )
    autoencoder.eval()

    # Decode batch size from config
    decode_batch_size = cfg.graph_extraction.decode_batch_size

    logger.info("Computing coordinate transformations...")
    # Compute coordinate transformations
    point_o2n, point_n2o, _, _ = get_coord_transformations(positions)

    # Cameras for projection
    from scene.dataset_readers import readColmapSceneInfo

    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    system_prompt = cfg.eval.spatial.graph_agent_system_prompt
    prompt_template = cfg.eval.spatial.graph_agent_prompt_template

    # Configuration for 3D Gaussian-to-grid mapping
    patch_size = cfg.eval.spatial.gaussian_patch_size
    gaussian_sample_ratio = cfg.eval.spatial.gaussian_sample_ratio

    # Init rerun logging
    rr.init("custom positional encodings")
    rr.save(graph_dir / "custom_positional_encoding.rrd")

    results: Dict[str, Any] = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        frame_number = int(timestep_queries["frame_number"])
        logger.info(f"Processing timestep {timestep} with frame number {frame_number}")

        # Get Gaussian means for this timestep
        gaussian_means_t = positions[t]  # (N, 3) in world coordinates
        logger.info(f"Gaussian means for timestep {timestep} have shape: {gaussian_means_t.shape}")
        logger.info(f"First 5 Gaussian means: {gaussian_means_t[:5]}")

        # Apply perspective projection to get pixel coordinates for each Gaussian
        frame_name = f"frame_{int(frame_number):06d}.jpg"
        proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
            int(frame_number), train_cameras, frame_name
        )
        gaussian_means_t_projected = project_3d_to_2d(gaussian_means_t, proj_matrix, img_width, img_height)
        # Make homogeneous, downstream will discard z anyways
        gaussian_means_t_projected = np.hstack((gaussian_means_t_projected, np.ones((gaussian_means_t_projected.shape[0], 1))))
        logger.info(f"First 5 Projected Gaussian means (homogeneous): {gaussian_means_t_projected[:5]}")
        

        # Decode latents for this timestep on-demand (much faster than decoding all timesteps)
        logger.info(f"Decoding patch latents for timestep {timestep}...")
        patch_latents_t = patch_latents[t]  # (N, lang_dim)
        logger.info(f"Patch latents shape before decoding: {patch_latents_t.shape}")
        
        # Decode in batches
        decoded_t_list = []
        with torch.no_grad():
            for i in range(0, len(patch_latents_t), decode_batch_size):
                batch = torch.tensor(
                    patch_latents_t[i : i + decode_batch_size],
                    device=model.device,
                    dtype=torch.float32,
                )
                decoded_batch = autoencoder.decode(batch)  # (batch_size, full_dim)
                decoded_t_list.append(decoded_batch.detach().cpu().numpy())
        
        decoded_main_t = np.concatenate(decoded_t_list, axis=0)  # (N, full_dim)
        logger.info(f"Decoded features for timestep {timestep} have shape: {decoded_main_t.shape}")

        # This is already in the [main | ds0 | ds1 | ds2] format
        gaussian_feats_t = torch.tensor(decoded_main_t, dtype=torch.float32, device=model.device)
        logger.info(f"Gaussian features for timestep {timestep} after accounting for deepstack features have shape: {gaussian_feats_t.shape}")

        logger.info(f"Sampling Gaussians for timestep {timestep}...")
        # Sample Gaussians if needed (deterministic sampling, same for all frames)
        # Keep features on the current device to avoid unnecessary CPU transfers.
        # This is all happening on the 3D Gaussians
        gaussian_means_sampled, gaussian_feats_sampled, sample_indices = sample_gaussians(
            gaussian_means_t,
            gaussian_features=gaussian_feats_t,
            sample_ratio=gaussian_sample_ratio,
            seed=42,  # Fixed seed for consistency across frames
        )

        # Apply same sampling also in 2D on the projected Gaussians
        gaussian_means_sampled_projected = gaussian_means_t_projected[sample_indices]
        logger.info(f"Sampled Gaussian means for timestep {timestep} have shape: {gaussian_means_sampled.shape}")
        logger.info(f"Sampled Gaussian features for timestep {timestep} have shape: {gaussian_feats_sampled.shape}")

        # Subsample colors with the same indices so visualization matches sampled Gaussians
        gaussian_colors_sampled = gaussian_colors[sample_indices]

        logger.info(f"Computing image_grid_thw for timestep {timestep}...")
        # Compute image_grid_thw from Gaussian extents
        image_grid_thw, min_x, min_y = compute_gaussian_grid_thw(
            gaussian_means_sampled_projected, patch_size=patch_size
        )

        logger.info(f"Creating Gaussian-to-grid mapping for timestep {timestep}...")
        # Create Gaussian-to-grid mapping
        gaussian_to_grid_mapping, sorted_indices = create_gaussian_to_grid_mapping(
            gaussian_means_sampled_projected,
            image_grid_thw,
            patch_size=patch_size,
        )

        # Sort everything acoording to the sorted indices, they are a byproduct of the gaussian <-> grid mapping
        # Essentially sorting Gaussians by their mapping to the grid, left to right top to bottom
        gaussian_to_grid_mapping = gaussian_to_grid_mapping[sorted_indices]
        gaussian_means_sampled_sorted = gaussian_means_sampled[sorted_indices]
        gaussian_means_sampled_projected_sorted = gaussian_means_sampled_projected[sorted_indices]
        gaussian_feats_sampled = gaussian_feats_sampled[sorted_indices]
        gaussian_colors_sampled = gaussian_colors_sampled[sorted_indices]

        # Depth from orginial means
        depth = gaussian_means_sampled_sorted[:, 2]
        # Opacities
        opacity = opacities[t][sorted_indices]

        filtered_indices = filter_gaussian_tokens(gaussian_to_grid_mapping, depth, opacity)
        # Get the filtered values
        gaussian_to_grid_mapping = gaussian_to_grid_mapping[filtered_indices]
        gaussian_means_sampled_sorted = gaussian_means_sampled_sorted[filtered_indices]
        gaussian_means_sampled_projected_sorted = gaussian_means_sampled_projected_sorted[filtered_indices]
        gaussian_feats_sampled = gaussian_feats_sampled[filtered_indices]
        gaussian_colors_sampled = gaussian_colors_sampled[filtered_indices]

        # Visualization for debugging
        # Rerun visualization: log sampled projected colored Gaussian positions in XY (z=0)
        sampled_means_xy = np.column_stack(
            (gaussian_means_sampled_projected_sorted[:, 0], gaussian_means_sampled_projected_sorted[:, 1], gaussian_means_sampled_projected_sorted[:, 2])
        )
        rr.set_time_sequence("frame", t)
        rr.log(
            "world/sampled_gaussians",
            rr.Points3D(
                positions=sampled_means_xy,
                colors=gaussian_colors_sampled,
                radii=4.0,
            ),
        )
        # For all points on the grid, put a red point in rerun
        grid_positions = torch.tensor([[i * patch_size + min_x, j * patch_size + min_y, 0] for i in range(image_grid_thw[0, 2]) for j in range(image_grid_thw[0, 1])])
        rr.log(
            "world/grid_points",
            rr.Points3D(
                positions=grid_positions,
                colors=torch.tensor([255, 0, 0]),
                radii=1.0,
            ),
        )

        # Number of Gaussians (after sampling)
        num_gaussians = len(gaussian_means_sampled_sorted)
        gaussians_per_image = [num_gaussians]

        results[timestep] = {"objects": [], "actions": []}
        results[timestep]["objects"] = graph_agent_3d_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            gaussian_features=[gaussian_feats_sampled],
            gaussian_means=gaussian_means_sampled_sorted,
            gaussian_means_projected=gaussian_means_sampled_projected_sorted,
            train_cameras=train_cameras,
            frame_number=frame_number,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            point_o2n=point_o2n,
            point_n2o=point_n2o,
            image_grid_thw=image_grid_thw,
            gaussian_to_grid_mapping=gaussian_to_grid_mapping,
            gaussians_per_image=gaussians_per_image,
        )
        results[timestep]["actions"] = graph_agent_3d_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            gaussian_features=[gaussian_feats_sampled],
            gaussian_means=gaussian_means_sampled_sorted,
            gaussian_means_projected=gaussian_means_sampled_projected_sorted,
            train_cameras=train_cameras,
            frame_number=frame_number,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            point_o2n=point_o2n,
            point_n2o=point_n2o,
            image_grid_thw=image_grid_thw,
            gaussian_to_grid_mapping=gaussian_to_grid_mapping,
            gaussians_per_image=gaussians_per_image,
        )

        # Clear VRAM after each timestep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up autoencoder
    del autoencoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _qwen3_coords_to_world_xy(
    coords: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> np.ndarray:
    """Convert Qwen3's [0, 1000] coordinates to XY coordinates.

    Args:
        coords: Array of shape (N, 2) with [x, y] in [0, 1000] range
        min_x, max_x: X
        min_y, max_y: Y

        Returns:
        Array of shape (N, 2) with [x, y]
        Coordinate system: +x is right, +y is down (right-handed)
    """
    coords_array = np.array(coords)
    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)

    # Scale from [0, 1000] to [0, 1] then map to world coordinates
    world_coords = coords_array.copy()
    world_coords[:, 0] = min_x + (coords_array[:, 0] / 1000.0) * (max_x - min_x)
    world_coords[:, 1] = min_y + (coords_array[:, 1] / 1000.0) * (max_y - min_y)

    return world_coords


def _parse_coords_from_json_3d(
    text: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> np.ndarray:
    """Extract 2D coordinates [x, y] from JSON in text and convert to world coordinates.

    The model outputs coordinates in [0, 1000] range which we map to world coordinates
    based on the Gaussian extent.

    Args:
        text: Response text containing JSON coordinates
        min_x, max_x: X extent
        min_y, max_y: Y extent

        Returns:
        World coordinates [x, y]
    """
    import json as _json
    
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    candidate = text[first_brace : last_brace + 1]
    print(f"candidate: {candidate}")
    obj = _json.loads(candidate)
    coords = np.array([float(obj["x"]), float(obj["y"])])
    world_coords = _qwen3_coords_to_world_xy(
        coords.reshape(1, -1), min_x, max_x, min_y, max_y
    )
    return world_coords.flatten()


def graph_agent_3d_predict_query_list(
    queries_list,
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    gaussian_features: List[torch.Tensor],
    gaussian_means: np.ndarray,
    gaussian_means_projected: np.ndarray,
    train_cameras,
    frame_number: int,
    system_prompt: str,
    prompt_template: str,
    point_o2n,
    point_n2o,
    image_grid_thw: torch.Tensor,
    gaussian_to_grid_mapping: torch.Tensor,
    gaussians_per_image: List[int],
):
    """Query model with 3D Gaussian-to-grid positional encoding.

    Directly queries the model with Gaussian features and spatial positional encoding,
    no tools involved. Parses response coordinates from [0, 1000] range to world coordinates.
    """
    outputs = []

    # Get grid extents from token positions; important to go back from Qwen3's [0, 1000] range to desired range
    # Coordinate system: +x is right, +y is down, +z is away from camera (right-handed)
    xy = gaussian_means_projected[:, :2]  # (N, 2)
    min_x, min_y = xy.min(axis=0)
    max_x, max_y = xy.max(axis=0)

    # Precompute projection for this frame; need this to project located 3D position back to 2D image space where we have ground truth for eval later
    frame_name = f"frame_{int(frame_number):06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        int(frame_number), train_cameras, frame_name
    )

    for query_idx, query in enumerate(queries_list):
        substring = query["query"]
        question = prompt_template.format(substring=substring)

        logger.info("=" * 80)
        logger.info(f"QUERY #{query_idx + 1}: {substring}")
        logger.info(f"Question: {question}")
        logger.info("=" * 80)

        # Build messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": question},
                ],
            },
        ]

        logger.info(f"System prompt length: {len(system_prompt)} chars")
        logger.info(f"User question length: {len(question)} chars")
        logger.info(f"Number of vision features: {len(gaussian_features)}")
        logger.info(f"Gaussians per image: {gaussians_per_image}")

        # Generate response with 3D positional encoding
        response = generate_with_vision_features_3d(
            messages=messages,
            vision_features=gaussian_features,
            model=model,
            processor=processor,
            image_grid_thw=image_grid_thw,
            gaussian_to_grid_mapping=gaussian_to_grid_mapping,
            gaussians_per_image=gaussians_per_image,
            zero_positional_encodings=False,  # Use spatial encodings
        )

        # Parse response to extract coordinates from response and map back to desired range defined by token extent
        world_xy = _parse_coords_from_json_3d(
            response, min_x, max_x, min_y, max_y
        )

        # Coordinates are now in the projected plane; locate nearest projected Gaussian token
        # Compute distances in XY plane only
        query_xy = world_xy.reshape(1, 2)  # (1, 2)
        gaussian_xy = gaussian_means_projected[:, :2]  # (N, 2)
        
        # Compute Euclidean distances in XY plane
        distances_xy = np.linalg.norm(gaussian_xy - query_xy, axis=1)  # (N,)
        nearest_idx = np.argmin(distances_xy)

        # Use index to retrieve original Gaussian mean in 3D space
        world_pos = gaussian_means[nearest_idx]
        print(f"predicted world position: {world_pos}")

        # Project back to 2D image for proper evaluation later
        world_pos_reshaped = world_pos.reshape(1, 3)
        pixels = project_3d_to_2d(
            world_pos_reshaped, proj_matrix, img_width, img_height
        )

        # TODO: not sure why we would call this "normalized", just converting between m and cm?
        world_pos_normalized = point_o2n(world_pos_reshaped)

        out_item = {
            "query": substring,
            "predictions": {},
            "raw_response": response,
        }
        out_item["predictions"]["0"] = {
            "pixel_coords": pixels.tolist(),
            "positions": world_pos_normalized.tolist(),
        }
        outputs.append(out_item)
        
        logger.info(f"Response: {response}")
        logger.info(f"Parsed world_xy: {world_xy}")
        logger.info(f"Final world_pos: {world_pos}")

    return outputs
