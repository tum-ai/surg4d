import torch
import math
import json
import re
import time
import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.generation import LogitsProcessorList
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.video_utils import VideoMetadata
from typing import Dict, List, Literal, Optional, Union, Any, Callable, Tuple
from functools import lru_cache
from qwen_vl_utils import process_vision_info

from .patched_qwen import (
    PatchedQwen3VLForConditionalGeneration,
)
from .thinking_budget_processor import ThinkingTokenBudgetProcessor
from .tools import IMAGE_PLACEHOLDER

from loguru import logger

# Qwen vision encoder constants by version
# Note: Both Qwen2.5 and Qwen3 use smart_resize() which resizes (not crops/pads)
# images to dimensions divisible by (patch_size × spatial_merge).
# The main difference: Qwen2.5 has an extra-patch quirk when (dim // patch_size) % 4 == 3.
QWEN_CONSTANTS = {
    "qwen3": {
        "patch_size": 16,
        "spatial_merge": 2,
        "effective_patch_size": 32,  # 16 * 2
        "num_deepstack_layers": 3,
        "temporal_patch_size": 2
    },
}

QWEN_VERSIONS = tuple(QWEN_CONSTANTS.keys())

# Make sure they are not equal, if thinking budget is used up, model cannot respond anymore, need some delta
THINKING_TOKEN_LIMIT = 1000
NEW_TOKEN_LIMIT = 1500


def timestep_to_seconds_str(timestep: int, fps: float) -> str:
    """Convert timestep index to Qwen3 temporal format.

    Args:
        timestep: Integer timestep index
        fps: Frames per second

    Returns:
        Formatted string like "<3.0 seconds>"
    """
    seconds = timestep / fps
    return f'time="<{seconds:.1f} seconds>"'


def qwen3_deepstack_to_cat(
    main_feats: torch.Tensor,
    deepstack_feats: List[torch.Tensor],
) -> torch.Tensor:
    """Concatenate main features with deepstack features for storage/autoencoder.

    Args:
        main_feats: Main vision features, shape (N, hidden_dim)
        deepstack_feats: List of 3 deepstack feature tensors, each (N, hidden_dim)

    Returns:
        Concatenated tensor of shape (N, hidden_dim * 4)
        Layout: [main | d0 | d1 | d2]
    """
    num_deepstack = QWEN_CONSTANTS["qwen3"]["num_deepstack_layers"]
    assert len(deepstack_feats) == num_deepstack, (
        f"Expected {num_deepstack} deepstack tensors, got {len(deepstack_feats)}"
    )
    return torch.cat([main_feats, *deepstack_feats], dim=-1)


def qwen3_cat_to_deepstack(
    concat_feats: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Split concatenated features back into main + deepstack for inference.

    Args:
        concat_feats: Concatenated tensor of shape (N, hidden_dim * 4)
                      Layout: [main | d0 | d1 | d2]

    Returns:
        Tuple of (main_feats, deepstack_feats) where:
            main_feats: shape (N, hidden_dim)
            deepstack_feats: list of 3 tensors, each (N, hidden_dim)
    """
    num_deepstack = QWEN_CONSTANTS["qwen3"]["num_deepstack_layers"]
    chunks = concat_feats.chunk(num_deepstack + 1, dim=-1)
    main_feats = chunks[0]
    deepstack_feats = list(chunks[1:])
    return main_feats, deepstack_feats


def qwen3_cat_to_deepstack_multiple(
    vision_features: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Prepare Qwen3 vision features for multiple images.

    Takes a list of concatenated feature tensors (one per image) and splits them
    into the format expected by the Qwen3 model forward pass.

    Args:
        vision_features: List of tensors, each (N_i, hidden_dim * 4) containing
                        [main | ds0 | ds1 | ds2] for each image

    Returns:
        Tuple of (main_features, deepstack_features) where:
            main_features: List of tensors (one per image), each (N_i, hidden_dim)
            deepstack_features: List of 3 tensors (one per layer), each containing
                               ALL visual tokens from ALL images concatenated
    """
    main_features = []
    all_deepstack = [[] for _ in range(QWEN_CONSTANTS["qwen3"]["num_deepstack_layers"])]

    for feat in vision_features:
        main, deepstack_list = qwen3_cat_to_deepstack(feat)
        main_features.append(main)
        for i, ds in enumerate(deepstack_list):
            all_deepstack[i].append(ds)

    # Concatenate deepstack features across all images per layer
    deepstack_features = [torch.cat(ds_list, dim=0) for ds_list in all_deepstack]

    return main_features, deepstack_features


def get_patched_qwen3(
    size: Literal["8B", "32B"] = "8B",
    use_fp8: bool = False,
    attn_implementation: str = "sdpa",  # "flash_attention_2" or "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    repetition_penalty: float = None,
    compile: bool = False,
):
    """Get a patched Qwen3VL model/processor that supports raw patch features.

    Uses inheritance-based patching via __class__ swapping after from_pretrained.
    Parameters allow enabling weight quantization and optimized attention without editing Transformers.
    """
    model_path = f"Qwen/Qwen3-VL-{size.upper()}-Thinking"
    if use_fp8:
        model_path = model_path + "-FP8"

    fp_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if max_memory is not None:
        fp_kwargs["max_memory"] = max_memory

    model = PatchedQwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **fp_kwargs,
    )

    processor = Qwen3VLProcessor.from_pretrained(model_path)
    # Prefer new cache format for memory-efficient caches
    model.generation_config.return_legacy_cache = False
    if repetition_penalty is not None:
        model.generation_config.repetition_penalty = repetition_penalty
    model.eval()
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
    return model, processor


def qwen_encode_image(
    image: Image.Image,
    model,
    processor,
):
    """Encode an image through a Qwen vision encoder.

    Args:
        image: PIL Image to encode
        model: Qwen model (Qwen3-VL)
        processor: Corresponding processor

    Returns:
        Concatenated tensor of shape (N, hidden_dim * 4) containing
        [main | deepstack0 | deepstack1 | deepstack2]
    """
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)

    with torch.no_grad():
        main_embeds_tuple, deepstack_embeds = model.get_image_features(
            pixel_values, image_grid_thw
        )
        main_feats = torch.cat(main_embeds_tuple, dim=0)
        return qwen3_deepstack_to_cat(main_feats, deepstack_embeds)


def _set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ask_qwen_about_image(
    image: Image.Image,
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    system_prompt: str,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: int = THINKING_TOKEN_LIMIT,
    seed: int = 42,
):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(  # type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    _set_generation_seed(seed)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
    }

    # thinking token limit processor
    logits_processor = None
    if max_thinking_tokens is not None:
        thinking_processor = ThinkingTokenBudgetProcessor(
            processor.tokenizer, max_thinking_tokens=max_thinking_tokens
        )
        logits_processor = LogitsProcessorList([thinking_processor])
        generate_kwargs["logits_processor"] = logits_processor

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def ask_qwen_about_image_custom(
    image: Image.Image,
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    system_prompt: str,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: int = THINKING_TOKEN_LIMIT,
    seed: int = 42,
):
    """Custom version that uses extracted features with custom_patch_features path.
    
    This method:
    1. Extracts features from the image using qwen_encode_image (vision encoder)
    2. Uses the custom_patch_features path (like graph queries)
    3. Keeps positional encodings (zero_image_hw=False)
    4. Uses the real image_grid_thw from processor
    
    This should reproduce the same results as ask_qwen_about_image (vanilla),
    allowing us to validate that the custom features mechanism works correctly.
    """
    # Extract features from image (using vision encoder)
    # This gives us the features that would normally be computed internally
    image_features = qwen_encode_image(image, model, processor)
    # image_features shape: (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # Apply chat template and process with real image
    # We use the real image so processor inserts correct placeholder tokens
    # and computes the correct image_grid_thw
    text = processor.apply_chat_template(  # type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],  # Use real image for proper tokenization
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Split features into main + deepstack
    main_features, deepstack_features = qwen3_cat_to_deepstack_multiple([image_features])
    
    # Generate with custom features; note how this is passing the features and the possibility to zero out pos encodings
    # This is handled by the custom forward pass (and rope index computation) of the custom model behind this
    # Again, the purpose here is to have more or less the default 2D Qwen3-VL behavior while testing the feature extraction and custom positional encodings
    # The behavior (when not zeroing out pos encodings) should be identical to the standard Qwen3-VL behavior
    _set_generation_seed(seed)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "custom_patch_features": main_features,
        "custom_deepstack_features": deepstack_features,
        "zero_image_hw": False,  # Keep positional encodings (don't zero out h, w)
    }
    
    # thinking token limit processor
    logits_processor = None
    if max_thinking_tokens is not None:
        thinking_processor = ThinkingTokenBudgetProcessor(
            processor.tokenizer, max_thinking_tokens=max_thinking_tokens
        )
        logits_processor = LogitsProcessorList([thinking_processor])
        generate_kwargs["logits_processor"] = logits_processor
    
    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


@lru_cache(maxsize=1000)
def closest_factor_pair(n: int) -> tuple[int, int]:
    """Find width, height factors for n tokens."""
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


# This function takes in the **patch features** of an image and a prompt
def ask_qwen_about_image_features(
    image_features: torch.Tensor,
    prompt: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
    zero_positional_encodings: bool = True,
):
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
    return generate_with_vision_features(
        messages,
        [image_features],
        model,
        processor,
        max_new_tokens=max_new_tokens,
        seed=seed,
        max_thinking_tokens=max_thinking_tokens,
        zero_positional_encodings=zero_positional_encodings,
    )


def adjust_input_ids_for_gaussians(
    inputs: Dict[str, torch.Tensor],
    processor: Qwen3VLProcessor,
    gaussians_per_image: List[int],
    image_token_id: int,
    video_token_id: int,
    image_pad_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Adjust input_ids to have the correct number of visual tokens per frame/image.
    
    Uses the same logic as get_placeholder_mask: finds all <image> tokens directly,
    groups consecutive ones into blocks (one per image), and replaces each block
    with the correct number of <image> tokens to match num_gaussians.
    
    Args:
        inputs: Dictionary from processor containing input_ids, attention_mask, etc.
        processor: Qwen3VLProcessor instance
        gaussians_per_image: List of number of Gaussians per image
        image_token_id: Token ID for <image> token
        video_token_id: Token ID for <video> token
    
    Returns:
        Modified inputs dict with adjusted input_ids and attention_mask
    """

    input_ids = inputs["input_ids"].clone()  # (B, L)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    logger.info(f"input_ids of shape: {input_ids.shape}")
    logger.info(f"attention_mask of shape: {attention_mask.shape}")
    logger.info(f"image_token_id: {image_token_id}")
    logger.info(f"video_token_id: {video_token_id}")

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert len(gaussians_per_image) > 0, "gaussians_per_image must not be empty"
    
    for b in range(batch_size):
        seq = input_ids[b]  # (L,)
        
        # Find all visual token positions (same logic as get_placeholder_mask)
        visual_mask = (seq == image_token_id) | (seq == video_token_id)
        visual_positions = visual_mask.nonzero(as_tuple=True)[0]

        # Do the same counting for the other provided token ids
        vision_start_mask = seq == vision_start_token_id
        vision_start_positions = vision_start_mask.nonzero(as_tuple=True)[0]
        vision_end_mask = seq == vision_end_token_id
        vision_end_positions = vision_end_mask.nonzero(as_tuple=True)[0]

        logger.info(f"Found total of {len(vision_start_positions)} vision start tokens")
        logger.info(f"Found total of {len(vision_end_positions)} vision end tokens")

        image_pad_mask = seq == image_pad_token_id
        image_pad_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        logger.info(f"Found total of {len(image_pad_positions)} image pad tokens")

        if len(visual_positions) == 0:
            logger.info("No visual tokens found, nothing to adjust")
            continue
        else:
            logger.info(f"Found total of {len(visual_positions)} visual tokens")
        
        # Group consecutive visual tokens into blocks (one block per image/frame)
        blocks = []
        block_start = visual_positions[0].item()
        current_token_id = int(seq[block_start].item())
        for i in range(1, len(visual_positions)):
            cur_pos = visual_positions[i].item()
            prev_pos = visual_positions[i - 1].item()
            cur_token_id = int(seq[cur_pos].item())
            if cur_pos != prev_pos + 1 or cur_token_id != current_token_id:
                blocks.append((block_start, prev_pos + 1, current_token_id))
                block_start = cur_pos
                current_token_id = cur_token_id
        blocks.append((block_start, visual_positions[-1].item() + 1, current_token_id))
        
        assert len(blocks) == len(gaussians_per_image), (
            f"Found {len(blocks)} image token blocks but {len(gaussians_per_image)} Gaussians per image specified"
        )
        
        # Build new sequence: before first block, then each adjusted block, then after last block
        new_seq_parts = []
        prev_end = 0
        
        for block_idx, (block_start, block_end, block_token_id) in enumerate(blocks):
            # Add tokens before this block
            if block_start > prev_end:
                new_seq_parts.append(seq[prev_end:block_start])
            
            # Add correct number of visual tokens for this block
            target_tokens = gaussians_per_image[block_idx]
            current_tokens = block_end - block_start
            
            if target_tokens != current_tokens:
                new_visual_tokens = torch.full(
                    (target_tokens,),
                    block_token_id,
                    dtype=seq.dtype,
                    device=seq.device
                )
                new_seq_parts.append(new_visual_tokens)
            else:
                new_seq_parts.append(seq[block_start:block_end])
            
            prev_end = block_end
        
        # Add remaining tokens after last block
        if prev_end < len(seq):
            new_seq_parts.append(seq[prev_end:])
        
        # Concatenate all parts
        new_seq = torch.cat(new_seq_parts, dim=0)
        
        input_ids = new_seq.unsqueeze(0).to(input_ids.device)
        attention_mask = torch.ones_like(input_ids)
    
    inputs["input_ids"] = input_ids
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask

    logger.info(f"final shape of input_ids: {input_ids.shape}")
    logger.info(f"final shape of attention_mask: {attention_mask.shape}")
    
    return inputs


def model_inputs(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    processor: Qwen3VLProcessor,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    image_pad_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    gaussians_per_image: Optional[List[int]] = None,
    tools: List[Dict[str, Any]] = [],
    **kwargs,
):
    """Prepare model inputs from messages and vision features.
    
    Args:
        messages: Chat messages with image placeholders
        vision_features: List of vision feature tensors
        processor: Qwen3VLProcessor instance
        image_token_id: Token ID for <image> token
        image_pad_token_id: Token ID for <image_pad> token
        vision_start_token_id: Token ID for <vision_start> token
        vision_end_token_id: Token ID for <vision_end> token
        image_grid_thw: Grid dimensions (BEFORE spatial merge)
        gaussians_per_image: List of number of Gaussians per image.
            If provided, adjusts input_ids to have the correct number of <image> tokens.
        tools: Optional tools for chat template
    """

    # create actual text template from messages dict
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    
    logger.info("=" * 80)
    logger.info("model_inputs: CHAT TEMPLATE OUTPUT")
    logger.info("=" * 80)
    logger.info(f"Generated text length: {len(text)} chars")
    logger.info(f"Generated text (first 1000 chars): {text[:1000]}")
    logger.info(f"Generated text (last 1000 chars): {text[-1000:]}")
    if len(text) > 2000:
        logger.warning(f"⚠️ TEXT IS VERY LONG: {len(text)} chars - may be truncated!")

    # create mock images such that their size corresponds to
    # the correct number of tokens after tokenizing and spatial merging
    # (we cannot just overwrite image_grid_thw because
    # input ids already contains placeholder vision tokens)
    effective_patch_size = QWEN_CONSTANTS["qwen3"]["effective_patch_size"]
    temporal_patch_size = QWEN_CONSTANTS["qwen3"]["temporal_patch_size"]
    if image_grid_thw is not None:
        assert image_grid_thw.shape[0] == len(vision_features)
    if video_grid_thw is not None:
        assert video_grid_thw.shape[0] == len(vision_features)
    mock_images: List[Image.Image] = []
    for i, feat in enumerate(vision_features):
        if image_grid_thw is None and video_grid_thw is None:
            n = int(feat.shape[0])
            w, h = closest_factor_pair(n)
            img_w = w * effective_patch_size
            img_h = h * effective_patch_size
        elif image_grid_thw is not None:
            height = int(image_grid_thw[i, 1].item())
            width = int(image_grid_thw[i, 2].item())
            img_w = width * effective_patch_size
            img_h = height * effective_patch_size
        else:
            height = int(video_grid_thw[i, 1].item())
            width = int(video_grid_thw[i, 2].item())
            logger.info(f"Desired video grid thw spatial resolution: {height}, {width})")
            img_w = width * effective_patch_size
            img_h = height * effective_patch_size
            logger.info(f"Video grid spatial resolution after accounting for effective patch size: ({img_w}, {img_h})")
        logger.info(f"Mock image {i}: ({img_w}, {img_h}) pixels, features: {feat.shape[0]}")
        mock_img = Image.new("RGB", (img_w, img_h), color="red")
        mock_images.append(mock_img)

    has_video = any(
        part.get("type") == "video"
        for msg in messages
        if msg["role"] == "user"
        for part in msg.get("content", [])
    )
    if has_video:
        # Preserve per-frame shape information by passing a frame list for one video.
        # TODO: this is accounting for temporal patch size, either use the Qwen constant or move this to hydra
        video_input = [list(mock_images) + list(mock_images)]
        logger.info(
            f"video_input structure: {len(video_input)} video(s), "
            f"{len(video_input[0]) if video_input else 0} frame(s) in first video"
        )
        logger.info(f"number of images: {len(gaussians_per_image)}")
        inputs = processor(
            text=[text],
            videos=video_input,
            padding=False,
            return_tensors="pt",
            do_sample_frames=False,
            do_resize=False,
        )
    else:
        # This will compute the image grid thw based on (mock) image dims and account for patch size only (no spatial merge yet)
        # That means the returned image grid thw will be the desired grid size divided by patch size but before spatial merge, still "too large"
        # Later, a spatial feature merge would be performed (we do not do this on Gaussian tokens) -> grid is divided by spatial merge factor in rope index computation in model forward pass
        # -> yields the actual LLM grid size we want
        # In a sense, we want image grid thw after all this -> above we multiply by effective patch size = patch size * spatial merge factor
        # -> processor divides by patch size, and rope index computation divides by spatial merge -> reverted to desired grid size for final inference
        inputs = processor(
            text=text,
            images=mock_images,
            padding=False,
            return_tensors="pt",
            do_resize=False,
        )
    
    logger.info(f"After processor: input_ids shape: {inputs['input_ids'].shape}")
    logger.info(f"After processor: attention_mask shape: {inputs['attention_mask'].shape if 'attention_mask' in inputs else None}")

    # Decode all input ids after processor and print without skipping special tokens
    input_ids = inputs["input_ids"]
    logger.info(f"input_ids shape: {input_ids.shape}")
    torch.set_printoptions(threshold=10000)
    decoded_input = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    logger.info(f"Decoded input: {decoded_input}")
    
    # Adjust input_ids to allow for passing the Gaussian tokens; processor will assume a 1:1 mapping between tokens derived from llm grid and actual tokens
    # In our case however, we have a desired grid size but we might have a many : 1 mapping between actual tokens and llm grid cells
    # This is still distinct from the actual assignment of tokens to grid cells for pos enc, which is handled in the rope index computation
    # Note: this is called already with the filtered, final selection of gaussian tokens!
    if gaussians_per_image is not None:
        assert image_token_id is not None
        assert video_token_id is not None
        assert image_pad_token_id is not None
        assert vision_start_token_id is not None
        assert vision_end_token_id is not None
        assert len(gaussians_per_image) == len(vision_features), (
            f"gaussians_per_image length ({len(gaussians_per_image)}) must match "
            f"vision_features length ({len(vision_features)})"
        )
        logger.info(f"Adjusting input_ids for gaussians_per_image: {gaussians_per_image}")
        inputs = adjust_input_ids_for_gaussians(
            inputs,
            processor,
            gaussians_per_image,
            image_token_id,
            video_token_id,
            image_pad_token_id,
            vision_start_token_id,
            vision_end_token_id,
        )
        logger.info(f"After adjustment: input_ids shape: {inputs['input_ids'].shape}")

    return inputs


def generate_with_vision_features(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
    zero_positional_encodings: bool = True,
):
    """Generate text from vision features.

    Args:
        messages: Chat messages with image placeholders
        vision_features: List of vision feature tensors.
            Each tensor is (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
        model: Qwen model (patched version)
        processor: Qwen processor
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

    # preprocess and generate
    inputs = model_inputs(
        messages,
        main_features,
        processor,
        image_token_id=model.model.config.image_token_id,
        video_token_id=model.model.config.video_token_id,
        image_pad_token_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        vision_start_token_id=model.model.config.vision_start_token_id,
        vision_end_token_id=model.model.config.vision_end_token_id,
    ).to(model.device)

    _set_generation_seed(seed)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "custom_patch_features": main_features,
        "custom_deepstack_features": deepstack_features,
        "zero_image_hw": zero_positional_encodings,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor
    generated_ids = model.generate(**inputs, **generate_kwargs)

    # remove prefix tokens (model input) and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_text[0]
    return response


def _parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool calls from Qwen's response.

    Qwen uses the format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>
    """
    tool_calls = []
    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    return tool_calls


def _extract_final_answer(response: str) -> str:
    """Extract the final answer from the response (text outside tool calls)."""
    # Remove tool call blocks
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
    return cleaned.strip()


def build_tool_response_message(
    tool_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    """Build a user message containing tool responses with interleaved text and images.

    Takes a list of tool result records and constructs a message in the format expected
    by Qwen's chat template. Each tool result is wrapped in <tool_response> tags.
    Vision features are inserted at positions marked by IMAGE_PLACEHOLDER in the text.

    Args:
        tool_results: List of tool result records, each containing:
            - "tool_name" (str): Name of the tool that was called
            - "arguments" (dict): Arguments passed to the tool
            - "result" (dict): Tool return value with:
                - "text" (str): Text content, may contain IMAGE_PLACEHOLDER markers
                - "vision_features" (List[torch.Tensor], optional): Feature tensors to insert
                  at IMAGE_PLACEHOLDER positions. Each tensor is (N, hidden_dim * 4) in
                  concatenated format [main | d0 | d1 | d2].

    Returns:
        Tuple of (message, vision_features) where:
            - message: Dict with "role": "user" and "content" list containing interleaved
              {"type": "text", "text": ...} and {"type": "image", "image": None} entries
            - vision_features: List of all vision feature tensors from all tools,
              in the order they appear in the message (matching image placeholder order)

    Example:
        Tool returns: {"text": '{"node": "<image/>"}', "vision_features": [tensor]}

        Generated content list:
        [
            {"type": "text", "text": '<tool_response>\\n{"name": "my_tool", "content": "{\\"node\\": \\"'},
            {"type": "image", "image": None},
            {"type": "text", "text": '\\"}"}\\n</tool_response>'},
        ]

    Raises:
        AssertionError: If number of IMAGE_PLACEHOLDER markers doesn't match
            number of vision_features for any tool.
    """
    content = []
    all_vision_features = []

    for record in tool_results:
        result = record["result"]
        text_content = result.get("text", "")
        tool_response_text = (
            f"<tool_response>\n"
            f'{{"name": "{record["tool_name"]}", "content": {json.dumps(text_content)}}}\n'
            f"</tool_response>"
        )

        if "vision_features" in result:
            tool_features = result["vision_features"]
            if not isinstance(tool_features, list):
                tool_features = [tool_features]

            # Split text by IMAGE_PLACEHOLDER and interleave with image placeholders
            parts = tool_response_text.split(IMAGE_PLACEHOLDER)
            n_markers = len(parts) - 1
            assert n_markers == len(tool_features), (
                f"Tool {record['tool_name']} returned {len(tool_features)} vision_features "
                f"but text contains {n_markers} IMAGE_PLACEHOLDER markers"
            )

            # Build interleaved content: text, image, text, image, ..., text
            # Empty text parts (from markers at start/end) are skipped
            for i, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(tool_features):
                    content.append({"type": "image", "image": None})

            all_vision_features.extend(tool_features)
        else:
            # No vision features - just add the text
            content.append({"type": "text", "text": tool_response_text})

    message = {"role": "user", "content": content}
    return message, all_vision_features


def _filter_tensors_for_debug(obj: Any) -> Any:
    """Recursively filter out tensors and numpy arrays from objects for debugging."""
    if isinstance(obj, (torch.Tensor, np.ndarray, np.generic)):
        return None  # Skip tensors/arrays
    elif isinstance(obj, dict):
        filtered = {}
        for k, v in obj.items():
            filtered_val = _filter_tensors_for_debug(v)
            if filtered_val is not None:
                filtered[k] = filtered_val
        return filtered if filtered else None
    elif isinstance(obj, (list, tuple)):
        filtered = [_filter_tensors_for_debug(item) for item in obj]
        filtered = [item for item in filtered if item is not None]
        return filtered if filtered else None
    else:
        return obj


def _format_message_trace_for_debug(
    current_messages: List[Dict[str, Any]],
    tool_call_history: List[Dict[str, Any]],
    iteration: int,
) -> str:
    """Format message trace and tool calls for debugging output."""
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"EXCEPTION DURING AGENT GENERATION - Message Trace (iteration {iteration})"
    )
    lines.append("=" * 80)
    lines.append("\n--- MESSAGE HISTORY ---\n")

    for i, msg in enumerate(current_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        lines.append(f"\n[{i}] Role: {role}")

        if isinstance(content, list):
            for j, item in enumerate(content):
                item_type = item.get("type", "unknown")
                if item_type == "text":
                    text = item.get("text", "")
                    # Truncate very long text
                    if len(text) > 500:
                        text = text[:500] + "... [truncated]"
                    lines.append(f"  Content[{j}]: text = {repr(text)}")
                elif item_type == "image":
                    lines.append(f"  Content[{j}]: image (vision feature)")
                else:
                    lines.append(f"  Content[{j}]: {item_type} = {str(item)[:200]}")
        else:
            lines.append(f"  Content: {str(content)[:500]}")

    lines.append("\n--- TOOL CALL HISTORY ---\n")
    for i, tool_call in enumerate(tool_call_history):
        tool_name = tool_call.get("tool_name", "unknown")
        arguments = tool_call.get("arguments", {})
        result = tool_call.get("result", {})

        lines.append(f"\n[{i}] Tool: {tool_name}")

        # Filter out tensors before serializing
        filtered_args = _filter_tensors_for_debug(arguments)
        if filtered_args:
            try:
                args_str = json.dumps(filtered_args, indent=2)
                if len(args_str) > 1000:
                    args_str = args_str[:1000] + "... [truncated]"
                lines.append(f"  Arguments: {args_str}")
            except Exception:
                lines.append("  Arguments: <error serializing arguments>")
        else:
            lines.append("  Arguments: (filtered - contained only tensors/arrays)")

        filtered_result = _filter_tensors_for_debug(result)
        if filtered_result:
            try:
                result_str = json.dumps(filtered_result, indent=2)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... [truncated]"
                lines.append(f"  Result: {result_str}")
            except Exception:
                lines.append("  Result: <error serializing result>")
        else:
            lines.append("  Result: (filtered - contained only tensors/arrays)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# TODO: Pretty sure this would no longer work, adjust if needed in the end or remove
def generate_with_vision_features_agentic(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    max_iterations: int = 10,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
    zero_positional_encodings: bool = True,
) -> Dict[str, Any]:
    """Generate with vision features in an agentic loop, executing tools until done.

    Args:
        messages: Chat messages (same format as generate_with_vision_features)
        vision_features: List of vision feature tensors.
            For qwen25: each tensor is (N, hidden_dim)
            For qwen3: each tensor is (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
        model: The Qwen model
        processor: The Qwen processor
        tools: Dict mapping tool_name -> (callable, json_spec)
               json_spec should be in OpenAI function calling format:
               {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
               Tools must return a dict with:
                   - "text" (str): Text content of the result (required). Use IMAGE_PLACEHOLDER
                     markers to indicate where vision features should be inserted.
                   - "vision_features" (List[torch.Tensor], optional): List of feature tensors,
                     each (N, hidden_dim * 4) in concatenated format [main | d0 | d1 | d2].
                     Must have exactly as many tensors as IMAGE_PLACEHOLDER markers in text.
        max_iterations: Maximum number of tool-calling iterations
        tool_call_limits: Optional dict mapping tool_name -> max_calls (int or None for infinite).
            If None, all tools have infinite calls. If a tool is not in the dict, it defaults to infinite.
        verbose: If True, prints message and tool results at each iteration.
        seed: Random seed for deterministic sampling
        max_new_tokens: Maximum number of new tokens to generate
        max_thinking_tokens: Maximum tokens for thinking phase per iteration (Qwen3 only).
            If None, no limit is applied. If 0, thinking is disabled immediately.
            A new processor is created each iteration since it has internal state.
        zero_positional_encodings: Whether to zero out h and w for positional encodings
    Returns:
        Dict with keys:
            - "final_answer" (str): The extracted final answer from the model's last response.
            - "message_history" (List[Dict]): Complete conversation history. Each message has:
                - "role": "system" | "user" | "assistant"
                - "content": List of {"type": "text"|"image", "text": str} dicts
            - "tool_calls" (List[Dict]): All tool calls made. Each entry has:
                - "tool_name" (str): Name of the tool called
                - "arguments" (dict): Arguments passed to the tool
                - "result" (Any): Result returned by the tool (or error message string)
            - "tok_per_sec" (float): Tokens generated per second across all iterations.
            - "total_generation_time" (float): Total time spent in model.generate() calls (seconds).
            - "total_time" (float): Total wall time for the entire agentic loop (seconds).
    """
    fn_start_time = time.time()

    # Extract tool specs for the model
    tool_specs = [spec for _, spec in tools.values()]

    # Copy messages to avoid mutating the original
    current_messages = [msg.copy() for msg in messages]
    for i, msg in enumerate(current_messages):
        if isinstance(msg.get("content"), list):
            current_messages[i]["content"] = msg["content"].copy()

    tool_call_history = []

    # Initialize tool call limits tracking
    # Track remaining calls (None means infinite, int means remaining count)
    remaining_calls: Dict[str, Optional[int]] = {}
    if tool_call_limits is not None:
        for tool_name in tools.keys():
            if tool_name in tool_call_limits:
                limit = tool_call_limits[tool_name]
                remaining_calls[tool_name] = limit  # None for infinite, int for limit
            else:
                remaining_calls[tool_name] = None  # Default to infinite
    else:
        # No limits specified - all tools have infinite calls
        for tool_name in tools.keys():
            remaining_calls[tool_name] = None

    # Track all vision features (initial + those added by tools)
    # Keep in concatenated format, convert to deepstack before each generation
    all_vision_features = list(vision_features)

    # Track generation timing and tokens for tok/s calculation
    total_generation_time = 0.0
    total_generated_tokens = 0

    for iteration in range(max_iterations):
        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] --- Iteration {iteration} ---", flush=True)

        # Convert accumulated features to deepstack format for this iteration
        main_features, deepstack_features = qwen3_cat_to_deepstack_multiple(
            all_vision_features
        )

        # Generate response with tools
        inputs = model_inputs(
            current_messages,
            main_features,
            processor,
            tools=tool_specs,
        ).to(model.device)

        # Build logits processor for thinking budget (new each iteration due to internal state)
        logits_processor = None
        if max_thinking_tokens is not None:
            thinking_processor = ThinkingTokenBudgetProcessor(
                processor.tokenizer, max_thinking_tokens=max_thinking_tokens
            )
            logits_processor = LogitsProcessorList([thinking_processor])

        try:
            _set_generation_seed(seed + iteration)
            gen_start_time = time.time()
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "custom_patch_features": main_features,
                "custom_deepstack_features": deepstack_features,
                "zero_image_hw": zero_positional_encodings,
            }
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = logits_processor
            generated_ids = model.generate(**inputs, **generate_kwargs)
            gen_end_time = time.time()
            total_generation_time += gen_end_time - gen_start_time
        except Exception:
            # Print message trace for any exception during generation
            trace_output = _format_message_trace_for_debug(
                current_messages, tool_call_history, iteration
            )
            print("\n" + trace_output + "\n", flush=True)
            # Re-raise the exception
            raise

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        total_generated_tokens += sum(len(ids) for ids in generated_ids_trimmed)
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [Assistant Response]:\n{response}\n", flush=True)

        # Parse tool calls
        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
            # No tool calls - we have the final answer
            # Add final assistant response to message history
            current_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )
            final_answer = _extract_final_answer(response)
            tok_per_sec = (
                total_generated_tokens / total_generation_time
                if total_generation_time > 0
                else 0.0
            )
            total_time = time.time() - fn_start_time
            return {
                "final_answer": final_answer,
                "message_history": current_messages,
                "tool_calls": tool_call_history,
                "tok_per_sec": tok_per_sec,
                "total_generation_time": total_generation_time,
                "total_time": total_time,
            }

        # Add assistant message with the response
        current_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )

        # Execute each tool call and collect results
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if verbose:
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] [Tool Call]: {tool_name}({json.dumps(_filter_tensors_for_debug(arguments))})",
                    flush=True,
                )

            if tool_name not in tools:
                result = {"text": json.dumps({"error": f"Unknown tool '{tool_name}'"})}
            else:
                # Check remaining calls before executing
                remaining = remaining_calls.get(tool_name, None)

                if remaining is not None and remaining <= 0:
                    # No calls left
                    result = {
                        "text": json.dumps(
                            {
                                "error": f"No tool calls remaining for '{tool_name}'. Call limit exceeded.",
                                "tool_name": tool_name,
                                "remaining_calls": 0,
                            }
                        )
                    }
                else:
                    # Execute the tool
                    callable_fn, _ = tools[tool_name]
                    try:
                        result = callable_fn(**arguments)

                        # Decrement remaining calls if not infinite
                        if remaining is not None:
                            remaining_calls[tool_name] = remaining - 1
                            remaining_after = remaining - 1
                        else:
                            remaining_after = None

                        # Add remaining calls info to the result
                        result_data = json.loads(result["text"])
                        result_data["remaining_calls"] = (
                            remaining_after
                            if remaining_after is not None
                            else "infinite"
                        )
                        result["text"] = json.dumps(result_data)
                    except Exception as e:
                        result = {
                            "text": json.dumps(
                                {"error": f"Error executing tool: {str(e)}"}
                            )
                        }
                        # Still decrement on error to prevent infinite retries
                        if remaining is not None:
                            remaining_calls[tool_name] = max(0, remaining - 1)
                            remaining_after_error = remaining_calls[tool_name]
                        else:
                            remaining_after_error = None
                        result_data = json.loads(result["text"])
                        result_data["remaining_calls"] = (
                            remaining_after_error
                            if remaining_after_error is not None
                            else "infinite"
                        )
                        result["text"] = json.dumps(result_data)

            if verbose:
                # Filter for logging
                log_result = _filter_tensors_for_debug(result)
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] [Tool Result]: {json.dumps(log_result)}\n",
                    flush=True,
                )

            tool_call_record = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
            tool_call_history.append(tool_call_record)
            tool_results.append(tool_call_record)

        # Build tool response message and collect vision features
        tool_response_message, new_features = build_tool_response_message(tool_results)
        current_messages.append(tool_response_message)
        all_vision_features.extend(new_features)

    # Max iterations reached - try to extract any answer from the last response
    final_answer = _extract_final_answer(response)
    tok_per_sec = (
        total_generated_tokens / total_generation_time
        if total_generation_time > 0
        else 0.0
    )
    total_time = time.time() - fn_start_time
    return {
        "final_answer": final_answer,
        "message_history": current_messages,
        "tool_calls": tool_call_history,
        "tok_per_sec": tok_per_sec,
        "total_generation_time": total_generation_time,
        "total_time": total_time,
    }


def prompt_graph_agent(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    initial_timestep_idx: int,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    system_prompt: str = None,
    max_iterations: int = 20,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
    max_new_tokens: int = 8192,
    max_thinking_tokens: Optional[int] = None,
    zero_positional_encodings: bool = True,
):
    """
    node_feats: np.lib.npyio.NpzFile - npz file containing node features for each timestep
    timestep_idx: int - index of the timestep to use for the node features
    adjacency_matrices: np.ndarray - adjacency matrices through time - weights are bhattacharyya coefficients (timesteps, n_clusters, n_clusters)
    node_centers: np.ndarray - cluster centers through time (timesteps, n_clusters, 3)
    node_centroids: np.ndarray - cluster centroids through time (timesteps, n_clusters, 3)
    node_extents: np.ndarray - cluster extents through time (timesteps, n_clusters, 3)
    model: Qwen2_5_VLForConditionalGeneration - model to use
    processor: Qwen2_5_VLProcessor - processor to use
    system_prompt: str - system prompt to use
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]] - tools to use
    verbose: If True, prints message and tool results at each iteration.
    max_new_tokens: int - maximum number of new tokens to generate
    max_thinking_tokens: int - maximum tokens for thinking phase per iteration (Qwen3 only).
        If None, no limit is applied. If 0, thinking is disabled immediately.
    zero_positional_encodings: Whether to zero out h and w for positional encodings
    """
    assert tools is not None and len(tools) > 0, (
        "tools are required for graph agentic prompting"
    )
    assert len(node_centers) == len(node_centroids) == len(node_extents), (
        "timestep mismatch"
    )

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats = [node_feats[idx] for idx in node_feat_indices]
    node_feats = [i[initial_timestep_idx] for i in node_feats]
    centroids = node_centroids[initial_timestep_idx]
    extents = node_extents[initial_timestep_idx]
    centers = node_centers[initial_timestep_idx]

    # Build JSON structure with IMAGE_PLACEHOLDER markers for nodes
    nodes_data = []
    for n in range(centroids.shape[0]):
        nodes_data.append(
            {
                "node_id": int(n),
                "rough_image": IMAGE_PLACEHOLDER,
                "centroid": {
                    "x": round(float(centroids[n][0]), 2),
                    "y": round(float(centroids[n][1]), 2),
                    "z": round(float(centroids[n][2]), 2),
                },
                "bbox_center": {
                    "x": round(float(centers[n][0]), 2),
                    "y": round(float(centers[n][1]), 2),
                    "z": round(float(centers[n][2]), 2),
                },
                "bbox_extent": {
                    "x": round(float(extents[n][0]), 2),
                    "y": round(float(extents[n][1]), 2),
                    "z": round(float(extents[n][2]), 2),
                },
            }
        )

    graph_data = {
        "timestep": int(initial_timestep_idx),
        "nodes": nodes_data,
    }

    # Serialize to JSON and split by IMAGE_PLACEHOLDER to interleave images
    graph_json = json.dumps(graph_data, indent=2)
    graph_parts = graph_json.split(IMAGE_PLACEHOLDER)

    # Build interleaved content: text, image, text, image, ..., text
    graph_content = []
    for i, part in enumerate(graph_parts):
        if part:
            graph_content.append({"type": "text", "text": part})
        if i < len(nodes_data):
            graph_content.append({"type": "image", "image": None})

    # Add tool call limits information to the prompt (as JSON)
    tool_limits_content = []
    if tool_call_limits is not None:
        tool_limits_data = {}
        for tool_name in tools.keys():
            limit = tool_call_limits.get(tool_name, None)
            tool_limits_data[tool_name] = "infinite" if limit is None else limit

        tool_limits_json = json.dumps({"tool_call_limits": tool_limits_data}, indent=2)
        tool_limits_content.append({"type": "text", "text": tool_limits_json + "\n\n"})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                *graph_content,
                {"type": "text", "text": "\n\n"},
                *tool_limits_content,
                {"type": "text", "text": question},
            ],
        },
    ]

    return generate_with_vision_features_agentic(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats],
        model=model,
        processor=processor,
        tools=tools,
        max_iterations=max_iterations,
        tool_call_limits=tool_call_limits,
        verbose=verbose,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_thinking_tokens=max_thinking_tokens,
        zero_positional_encodings=zero_positional_encodings,
    )


def prompt_with_video_frames(
    question: str,
    image_paths: List[Any],
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    system_prompt: str = None,
    fps: float = None,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: int = THINKING_TOKEN_LIMIT,
) -> str:
    """Prompt model with video frames (list of images).

    Args:
        question: Question to ask about the video
        image_paths: List of image file paths (as strings or Path objects)
        model: Qwen VL model
        processor: Qwen VL processor
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        max_thinking_tokens: Maximum tokens for thinking phase (Qwen3 only).
            If None, no limit is applied. If 0, thinking is disabled immediately.
        fps: Optional frames per second for video metadata
        max_new_tokens: Maximum tokens to generate
        max_thinking_tokens: Maximum tokens for thinking phase (Qwen3 only).
            If None, no limit is applied. If 0, thinking is disabled immediately.
    Returns:
        Model response text
    """
    # Convert paths to strings
    image_paths_str = [str(p) for p in image_paths]

    # Build messages with video content
    content = []
    video_content = {"type": "video", "video": image_paths_str}
    if fps is not None:
        # raw_fps: actual framerate used in video_metadata
        # sample_fps: sampling rate passed to processor for frame selection
        video_content["raw_fps"] = fps
        video_content["sample_fps"] = fps
    content.append(video_content)
    content.append({"type": "text", "text": question})

    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info to extract images and videos
    # Note: raw_fps and sample_fps in video_content create metadata internally in qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages)

    # Create VideoMetadata for explicit fps specification
    # This ensures the model knows the correct temporal spacing between frames
    video_metadata = None
    if fps is not None and video_inputs is not None:
        num_frames = len(image_paths)
        video_metadata = [
            VideoMetadata(
                total_num_frames=num_frames,
                fps=fps,
                frames_indices=list(range(num_frames)),
            )
        ]

    # Prepare inputs
    # CRITICAL: Set do_sample_frames=False to prevent processor from resampling our pre-selected frames
    # Pass video_metadata explicitly to ensure model gets correct fps
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # thinking token limit processor
    logits_processor = None
    if max_thinking_tokens is not None:
        thinking_processor = ThinkingTokenBudgetProcessor(
            processor.tokenizer, max_thinking_tokens=max_thinking_tokens
        )
        logits_processor = LogitsProcessorList([thinking_processor])

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    # Generate
    _set_generation_seed(seed)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def crop_patch_features(patch_feat: torch.Tensor, cw, ch, cx1, cx2, cy1, cy2):
    return patch_feat.reshape(ch, cw, -1)[cy1 : cy2 + 1, cx1 : cx2 + 1].flatten(
        end_dim=1
    )


def get_patch_hw(im_height: int, im_width: int) -> Tuple[int, int]:
    """Get patchgrid dimensions for given image dimensions.

    Uses smart_resize logic: round(dim / factor) * factor to get resized dimensions,
    then divide by factor to get patch count.

    Args:
        im_height: Image height in pixels
        im_width: Image width in pixels

    Returns:
        Tuple of (patches_height, patches_width)
    """
    factor = QWEN_CONSTANTS["qwen3"]["effective_patch_size"]
    resized_h, resized_w = smart_resize(
        im_height, im_width, factor=factor, min_pixels=1, max_pixels=10**9
    )
    return resized_h // factor, resized_w // factor


def get_patch_segmasks(
    im_height: int, im_width: int
) -> torch.Tensor:
    """Generate an instance segmentation mask where each instance corresponds to one vision encoder patch.

    Args:
        im_height: Image height in pixels
        im_width: Image width in pixels

    Returns:
        Tensor of shape (resized_H, resized_W) with patch indices at each pixel
    """
    factor = QWEN_CONSTANTS["qwen3"]["effective_patch_size"]
    resized_h, resized_w = smart_resize(
        im_height, im_width, factor=factor, min_pixels=1, max_pixels=10**9
    )
    rowcol = torch.stack(
        torch.meshgrid(
            torch.arange(resized_h),
            torch.arange(resized_w),
            indexing="ij",
        )
    )
    patch_coords = torch.floor_divide(rowcol, factor)
    patches_per_row = resized_w // factor
    return patch_coords[0] * patches_per_row + patch_coords[1]
