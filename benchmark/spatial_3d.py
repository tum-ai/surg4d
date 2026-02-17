"""
Spatial query evaluation using 3D Gaussian-to-grid positional encoding.

This module provides graph_agent_3d_feat_queries_general which uses custom positional encodings
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
from transformers.video_utils import VideoMetadata

import rerun as rr


def compute_gaussian_grid_thw(
    gaussian_means: np.ndarray,  # (N, 3)
    img_height: int,
    img_width: int,
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

    # TODO: have to make a decision here; either replicating actual image plane and patches (when patch size is set to 16) or do it free-formm
    #   Free-form including out-of-view stuff and independent of image would be great, but will lead to inconsistencies across frames, "empty" grid cells, etc.
    #   Therefore, using the simplified version to exclude empty patches and irregular amounts of tokens per image as a source of error

    # # Compute grid dimensions (BEFORE spatial merge)
    # # The model will apply spatial_merge_size internally in get_rope_index
    # grid_w = int(np.ceil(extent_x / patch_size)) + 1
    # grid_h = int(np.ceil(extent_y / patch_size)) + 1

    # Simplification for now, replicating image plane and its extent
    grid_w = int(np.ceil(img_width / patch_size))
    grid_h = int(np.ceil(img_height / patch_size))
    min_x = 0
    min_y = 0
    max_x = img_width - 1
    max_y = img_height - 1

    # image_grid_thw uses dimensions BEFORE spatial merge
    # get_rope_index will later apply spatial_merge_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    logger.info(f"Image grid thw: {image_grid_thw}")

    return image_grid_thw, min_x, min_y


def create_gaussian_to_grid_mapping(
    gaussian_means: np.ndarray,  # (N, 3) world coordinates
    image_grid_thw: torch.Tensor,  # (1, 3) with [t, h, w]
    patch_size: float = 0.01,
    # Those are optionally provided when extent is intentionally manipulated, e.g., to conform with original image extents after projection
    min_x = None,
    min_y = None,
    max_x = None,
    max_y = None
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

    # Compute extent to get origin offset; might be provided externally if we want to force a different extent (e.g., excluding out of view, following image plane extent, etc.)
    if min_x is None and min_y is None and max_x is None and max_y is None:
        min_x, min_y = xy.min(axis=0)
        max_x, max_y = xy.max(axis=0)

    # Map to grid indices (0-indexed)
    # x -> w (width), y -> h (height)
    w_indices = np.floor((xy[:, 0] - min_x) / patch_size)
    h_indices = np.floor((xy[:, 1] - min_y) / patch_size)

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
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
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
        video_grid_thw: (num_frames, 3) tensor with [t, h, w] per-frame dims (BEFORE merge)
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
    video_token_id = model.model.config.video_token_id
    logger.info(f"video_token_id: {video_token_id}")

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
        video_grid_thw=video_grid_thw,
        gaussians_per_image=gaussians_per_image,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
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
    logger.info(f"video_grid_thw: {video_grid_thw}")
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


def filter_gaussian_tokens(gaussian_to_grid_mapping: np.ndarray, depth: np.ndarray, opacity: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Filter Gaussians based on depth, opacity, and grid assignment.

    Args:
        gaussian_to_grid_mapping: (N, 2) array of [h_idx, w_idx] for each Gaussian
        depth: (N,) array of depth values for each Gaussian
        opacity: (N,) array of opacity values for each Gaussian
        grid_h: height of the grid
        grid_w: width of the grid

    Returns:
        filtered_indices: (M,) array of indices of the filtered Gaussians
    """

    # TODO: if we keep using this, threshold must go into hydra configs
    filtered_depth = depth.copy()
    filtered_depth[opacity < 0.8] = np.inf

    # Get the unique mappings to grid indices
    _, unique_start_indices, unique_counts = np.unique(gaussian_to_grid_mapping, axis=0, return_index=True, return_counts=True)
    
    final_indices = []
    for unique_start_index, unique_count in zip(unique_start_indices, unique_counts):
        # Check the mapping and potentially skip if it is outside of the grid; want to exclude those Gaussians at the moment
        current_mapping = gaussian_to_grid_mapping[unique_start_index:unique_start_index+unique_count]
        if (current_mapping[:, 0] < 0).any() or (current_mapping[:, 0] >= grid_h).any() or (current_mapping[:, 1] < 0).any() or (current_mapping[:, 1] >= grid_w).any():
            continue

        relevant_depths = filtered_depth[unique_start_index:unique_start_index+unique_count]

        # If all depths are infinity, we take the shallowest depth from the original depth
        if np.all(relevant_depths == np.inf):
            shallowest_depth_index = np.argmin(depth[unique_start_index:unique_start_index+unique_count])
        else:
            shallowest_depth_index = np.argmin(relevant_depths)

        final_indices.append(unique_start_index + shallowest_depth_index)

    return np.array(final_indices)


# TODO: may want to rename the file or move this and its submethods to another file, makes no sense to be in a specific "spatial" file
# General interface for Gaussian token queries (spatial, temporal, ...)
def graph_agent_3d_feat_queries_general(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    graph_dir: Path | str,
    clip: DictConfig,
    cfg: DictConfig,
    queries: List[str],
    timesteps: List[tuple[int, int]],
    frame_numbers: List[tuple[int, int]],
    system_prompts: List[str],
    prompt_template: str,
    query_metadata: List[Dict[str, Any]],
):
    """Run queries with 3D Gaussian-to-grid positional encoding.

    Uses custom positional encodings based on Gaussian spatial positions, enabling
    spatial queries over 3D Gaussian splatting data with many-to-one grid mapping.
    No tools are used - the model directly processes Gaussian features.

    Args:
        model: CustomQwen3VLForConditionalGeneration3D model instance
        processor: Qwen3VLProcessor instance
        graph_dir: Path to graph directory containing Gaussian data
        clip: Clip configuration
        cfg: Evaluation configuration
        queries: List of query strings. If prompt_template is "{substring}", these can be full prompts.
        timesteps: List of inclusive timestep ranges (start, end) per query
        frame_numbers: List of inclusive frame-number ranges (start, end) per query
        system_prompts: System prompts for the queries
        prompt_template: Prompt template used as prompt_template.format(substring=query)
        query_metadata: Metadata per query for downstream postprocessing

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

    # Configuration for 3D Gaussian-to-grid mapping
    # TODO: this does not just concern spatial, should be in general eval config
    patch_size = cfg.eval.spatial.gaussian_patch_size
    gaussian_sample_ratio = cfg.eval.spatial.gaussian_sample_ratio

    # Init rerun logging
    rr.init("custom positional encodings")
    rr.save(graph_dir / "custom_positional_encoding.rrd")

    assert len(queries) == len(timesteps) == len(frame_numbers) == len(query_metadata)
    results: List[Dict[str, Any]] = []
    for query_idx, query in enumerate(queries):
        t1, t2 = int(timesteps[query_idx][0]), int(timesteps[query_idx][1])
        if t1 == t2:
            video_mode = False
        else:
            video_mode = True
        logger.info(f"Processing query {query_idx} in video mode: {video_mode}")

        f1, f2 = int(frame_numbers[query_idx][0]), int(frame_numbers[query_idx][1])
        logger.info(f"Processing timesteps {t1} to {t2} with corresponding frame numbers {f1} to {f2}")

        if video_mode:
            # TODO: should go into hydra
            processor.video_processor.video_metadata = VideoMetadata(
                total_num_frames=(t2 - t1 + 1) * 2,
                fps=6.25 * 2,
                frames_indices=list(range(t1,  (t2 * 2) + 1)),
            )

        # Aggregate all relevant info for downstream inference
        gaussian_means_all = []
        gaussian_means_projected_all = []
        gaussian_features_all = []
        gaussian_to_grid_mapping_all = []
        frame_grid_thw_all = []

        # TODO: later, we may want to experiment with a fixed view, fixed projection plane, and fixed image grid
        # image_grid_thw_cached = None
        # proj_matrix_cached = None
        # img_width_cached = None
        # img_height_cached = None

        # TODO: the way we deal with frame numbers here is hacky; if it works it must become part of hydra config
        frame_numbers_step = 4
        for t, frame_number in zip(range(t1, t2 + 1), range(f1, f2 + 1, frame_numbers_step)):

            # Get Gaussian means for this timestep
            gaussian_means_t = positions[t]  # (N, 3) in world coordinates
            # Apply perspective projection to get pixel coordinates for each Gaussian
            frame_name = f"frame_{int(frame_number):06d}.jpg"

            # TODO: decide on approach here, fixed projection plane or dynamic; currently working with dynamic for various reasons
            # # working with a fixed image plane by fixing to the first view
            # if proj_matrix_cached is None:
            #     proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
            #         int(frame_number), train_cameras, frame_name
            #     )
            #     proj_matrix_cached = proj_matrix
            #     img_width_cached = img_width
            #     img_height_cached = img_height
            # else:
            #     proj_matrix = proj_matrix_cached
            #     img_width = img_width_cached
            #     img_height = img_height_cached
            proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
                int(frame_number), train_cameras, frame_name
            )

            gaussian_means_t_projected = project_3d_to_2d(gaussian_means_t, proj_matrix, img_width, img_height)
            # Make homogeneous
            gaussian_means_t_projected = np.hstack((gaussian_means_t_projected, np.ones((gaussian_means_t_projected.shape[0], 1))))
        
            # TODO: this should happen much further below! decoding only what we need, not all Gaussians
            # Decode latents for this timestep on-demand (much faster than decoding all timesteps)
            logger.info(f"Decoding patch latents for timestep {t}...")
            patch_latents_t = patch_latents[t]  # (N, lang_dim)
            logger.info(f"Patch latents shape before decoding: {patch_latents_t.shape}")
            # Decode in batches
            decoded_t_list = []
            with torch.no_grad():
                for batch_start in range(0, len(patch_latents_t), decode_batch_size):
                    batch = torch.tensor(
                        patch_latents_t[batch_start : batch_start + decode_batch_size],
                        device=model.device,
                        dtype=torch.float32,
                    )
                    decoded_batch = autoencoder.decode(batch)  # (batch_size, full_dim)
                    decoded_t_list.append(decoded_batch.detach().cpu().numpy())
            decoded_main_t = np.concatenate(decoded_t_list, axis=0)  # (N, full_dim)
            logger.info(f"Decoded features for timestep {t} have shape: {decoded_main_t.shape}")
            # This is already in the [main | ds0 | ds1 | ds2] format
            gaussian_feats_t = torch.tensor(decoded_main_t, dtype=torch.float32, device=model.device)
            logger.info(f"Gaussian features for timestep {t} after accounting for deepstack features have shape: {gaussian_feats_t.shape}")

            logger.info(f"Sampling Gaussians for timestep {t}...")
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
            logger.info(f"Sampled Gaussian means for timestep {t} have shape: {gaussian_means_sampled.shape}")
            logger.info(f"Sampled Gaussian features for timestep {t} have shape: {gaussian_feats_sampled.shape}")

            # Subsample colors with the same indices so visualization matches sampled Gaussians
            gaussian_colors_sampled = gaussian_colors[sample_indices]

            # TODO: decide on approach here, again fixed grid vs potentially dynamic (depends on which extents we choose etc.)
            # if image_grid_thw_cached is None:
            #     logger.info(f"Computing image_grid_thw for timestep {t}...")
            #     # Compute image_grid_thw from Gaussian extents
            #     # TODO: making this simplification with the exact image plane; not happy about this but should fix the tokens inconsistencies for now
            #     # TODO: if we find a better solution, may want to remove the reliance on the actual image height and width
            #     image_grid_thw, min_x, min_y = compute_gaussian_grid_thw(
            #         gaussian_means_sampled_projected, img_height, img_width, patch_size=patch_size
            #     )
            #     image_grid_thw_cached = image_grid_thw
            # else:
            #     logger.info("Reusing existing image_grid_thw...")
            #     image_grid_thw = image_grid_thw_cached
            #     # TODO: computing extent, should only be used in debug vis below
            #     min_x, min_y = gaussian_means_sampled_projected[:, :2].min(axis=0)
            image_grid_thw, min_x, min_y = compute_gaussian_grid_thw(
                gaussian_means_sampled_projected, img_height, img_width, patch_size=patch_size
            )

            logger.info(f"Creating Gaussian-to-grid mapping for timestep {t}...")
            logger.info(f"Warning ⚠️: currently enforcing image extent in projection plane")
            # Create Gaussian-to-grid mapping
            gaussian_to_grid_mapping, sorted_indices = create_gaussian_to_grid_mapping(
                gaussian_means_sampled_projected,
                image_grid_thw,
                patch_size=patch_size,
                # TODO: only using this when forcing it to the image extent
                min_x=0,
                min_y=0,
                max_x=img_width - 1,
                max_y=img_height - 1,
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

            # Further filtering (depth, opacities, grid assignment, ...)
            filtered_indices = filter_gaussian_tokens(gaussian_to_grid_mapping, depth, opacity, image_grid_thw[0, 1], image_grid_thw[0, 2])
            # Get the filtered values
            gaussian_to_grid_mapping = gaussian_to_grid_mapping[filtered_indices]
            gaussian_means_sampled_sorted = gaussian_means_sampled_sorted[filtered_indices]
            gaussian_means_sampled_projected_sorted = gaussian_means_sampled_projected_sorted[filtered_indices]
            gaussian_feats_sampled = gaussian_feats_sampled[filtered_indices]
            gaussian_colors_sampled = gaussian_colors_sampled[filtered_indices]

            # TODO: feature decoding should happen here, once filtering is done

            gaussian_means_all.append(gaussian_means_sampled_sorted)
            gaussian_means_projected_all.append(gaussian_means_sampled_projected_sorted)
            gaussian_features_all.append(gaussian_feats_sampled)
            gaussian_to_grid_mapping_all.append(gaussian_to_grid_mapping)
            frame_grid_thw_all.append(image_grid_thw)

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
            grid_positions = torch.tensor([[i * patch_size + min_x, j * patch_size + min_y, 0] for i in range(image_grid_thw[0, 2] + 1) for j in range(image_grid_thw[0, 1] + 1)])
            rr.log(
                "world/grid_points",
                rr.Points3D(
                    positions=grid_positions,
                    colors=torch.tensor([255, 0, 0]),
                    radii=1.0,
                ),
            )

        outputs = []

        gaussian_features = gaussian_features_all
        gaussians_per_image = [int(m.shape[0]) for m in gaussian_to_grid_mapping_all]
        gaussian_to_grid_mapping = torch.cat(gaussian_to_grid_mapping_all, dim=0)
        if video_mode:
            video_grid_thw = torch.cat(frame_grid_thw_all, dim=0)
            image_grid_thw_for_call = None
        else:
            video_grid_thw = None
            image_grid_thw_for_call = frame_grid_thw_all[0]

        # TODO: prompt template is pretty useless at the moment, may want to get rid of that later on
        substring = query # Query from current iter, assuming this was passed correctly
        question = prompt_template.format(substring=substring)

        # Build messages
        if video_mode:
            # Do not matter that much, we are interfering after the video processor anyways, manipulating the input id sequence to match the Gaussian tokens
            video_placeholders = [
                f"frame_{int(frame_idx):06d}.jpg" for frame_idx in range(f1, f2 + 1)
            ]
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompts[query_idx]}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_placeholders},
                        {"type": "text", "text": question},
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompts[query_idx]}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": question},
                    ],
                },
            ]

        logger.info(f"image grid before generate_with_vision_features_3d: {image_grid_thw_for_call}")
        logger.info(f"video grid before generate_with_vision_features_3d: {video_grid_thw}")

        # Generate response with 3D positional encoding
        response = generate_with_vision_features_3d(
            messages=messages,
            vision_features=gaussian_features,
            model=model,
            processor=processor,
            image_grid_thw=image_grid_thw_for_call,
            video_grid_thw=video_grid_thw if video_mode else None,
            gaussian_to_grid_mapping=gaussian_to_grid_mapping,
            gaussians_per_image=gaussians_per_image,
            zero_positional_encodings=False,
        )

        result_item = {
            "raw_response": response,
        }

        # Generally, postprocessing happens in task specific code, but we do need access to internal representations to convert back to pixels for spatial tasks
        if query_metadata[query_idx]["task"] == "spatial":
            result_item["gaussian_means"] = gaussian_means_sampled_sorted
            result_item["gaussian_means_projected"] = gaussian_means_sampled_projected_sorted

        results.append(result_item)

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
