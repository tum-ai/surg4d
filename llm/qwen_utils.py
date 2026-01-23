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
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.video_utils import VideoMetadata
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from functools import lru_cache
from transformers.utils.quantization_config import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from .patched_qwen import (
    PatchedQwen2_5_VLForConditionalGeneration,
    PatchedQwen3VLForConditionalGeneration,
)
from .tools import IMAGE_PLACEHOLDER

# Qwen vision encoder constants by version
# Note: Both Qwen2.5 and Qwen3 use smart_resize() which resizes (not crops/pads)
# images to dimensions divisible by (patch_size × spatial_merge).
# The main difference: Qwen2.5 has an extra-patch quirk when (dim // patch_size) % 4 == 3.
QWEN_CONSTANTS = {
    "qwen25": {
        "patch_size": 14,
        "spatial_merge": 2,
        "effective_patch_size": 28,  # 14 * 2
        "num_deepstack_layers": 0,
    },
    "qwen3": {
        "patch_size": 16,
        "spatial_merge": 2,
        "effective_patch_size": 32,  # 16 * 2
        "num_deepstack_layers": 3,
    },
}

QWEN_VERSIONS = tuple(QWEN_CONSTANTS.keys())


def timestep_to_seconds_str(timestep: int, fps: float) -> str:
    """Convert timestep index to Qwen3 temporal format.
    
    Args:
        timestep: Integer timestep index
        fps: Frames per second
        
    Returns:
        Formatted string like "<3.0 seconds>"
    """
    seconds = timestep / fps
    return f"time=\"<{seconds:.1f} seconds>\""


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


def qwen3_format_multiple_deepstack_features(
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


def get_patched_qwen25(
    use_bnb_4bit: bool = False,
    use_bnb_8bit: bool = False,
    attn_implementation: str = "sdpa",  # "flash_attention_2" or "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
):
    """Get a patched Qwen2_5_VL model/processor that supports raw patch features.

    Uses inheritance-based patching via __class__ swapping after from_pretrained.
    Parameters allow enabling weight quantization and optimized attention without editing Transformers.
    """
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    quantization_config = None
    if use_bnb_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif use_bnb_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    fp_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        fp_kwargs["quantization_config"] = quantization_config
    if max_memory is not None:
        fp_kwargs["max_memory"] = max_memory

    model = PatchedQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **fp_kwargs,
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    # Prefer new cache format for memory-efficient caches
    model.generation_config.return_legacy_cache = False
    model.eval()
    return model, processor


def get_patched_qwen3(
    use_bnb_4bit: bool = False,
    use_bnb_8bit: bool = False,
    attn_implementation: str = "sdpa",  # "flash_attention_2" or "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    repetition_penalty: float = None,
):
    """Get a patched Qwen3VL model/processor that supports raw patch features.

    Uses inheritance-based patching via __class__ swapping after from_pretrained.
    Parameters allow enabling weight quantization and optimized attention without editing Transformers.
    """
    model_path = "Qwen/Qwen3-VL-8B-Thinking"

    quantization_config = None
    if use_bnb_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif use_bnb_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    fp_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        fp_kwargs["quantization_config"] = quantization_config
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
    return model, processor


def get_patched_qwen(
    qwen_version: str,
    use_bnb_4bit: bool = False,
    use_bnb_8bit: bool = False,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
):
    """Get a patched Qwen model/processor based on version string.

    Args:
        qwen_version: Either "qwen25" or "qwen3"
        Other args: Same as get_patched_qwen25/get_patched_qwen3

    Returns:
        Tuple of (model, processor)
    """
    if qwen_version == "qwen25":
        return get_patched_qwen25(
            use_bnb_4bit=use_bnb_4bit,
            use_bnb_8bit=use_bnb_8bit,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
    elif qwen_version == "qwen3":
        return get_patched_qwen3(
            use_bnb_4bit=use_bnb_4bit,
            use_bnb_8bit=use_bnb_8bit,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
    else:
        raise ValueError(
            f"Unknown qwen_version: {qwen_version}. Must be one of {QWEN_VERSIONS}"
        )


def qwen_encode_image(
    image: Image.Image,
    model,
    processor,
    qwen_version: str,
):
    """Encode an image through a Qwen vision encoder.

    Args:
        image: PIL Image to encode
        model: Qwen model (either Qwen2.5-VL or Qwen3-VL)
        processor: Corresponding processor
        qwen_version: Either "qwen25" or "qwen3"

    Returns:
        For qwen25: Tensor of shape (N, hidden_dim)
        For qwen3: Concatenated tensor of shape (N, hidden_dim * 4) containing
                   [main | deepstack0 | deepstack1 | deepstack2]
    """
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)

    with torch.no_grad():
        if qwen_version == "qwen25":
            feats = model.visual(pixel_values, image_grid_thw)
            return feats
        elif qwen_version == "qwen3":
            # Qwen3 visual returns (main_embeds, deepstack_embeds_list)
            main_embeds_tuple, deepstack_embeds = model.get_image_features(
                pixel_values, image_grid_thw
            )
            main_feats = torch.cat(main_embeds_tuple, dim=0)
            # Concatenate into single tensor for consistent storage format
            return qwen3_deepstack_to_cat(main_feats, deepstack_embeds)
        else:
            raise ValueError(
                f"Unknown qwen_version: {qwen_version}. Must be one of {QWEN_VERSIONS}"
            )


# This function takes in an RGB image  and a prompt
def _set_generation_seed(seed: int) -> None:
    """Set random seed for deterministic generation without enabling torch deterministic mode.

    This seeds the RNG for sampling-based generation while avoiding the performance
    and compatibility issues of torch.backends.cudnn.deterministic.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ask_qwen_about_image(
    image: Image.Image,
    prompt: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
    max_tokens: int = 5012,
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
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
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
    max_tokens: int = 5012,
    seed: int = 42,
    qwen_version: str = "qwen25",
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
        messages, [image_features], model, processor, qwen_version=qwen_version, max_tokens=max_tokens, seed=seed
    )


def model_inputs(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    tools: List[Dict[str, Any]] = [],
):
    """Prepare model inputs from messages and vision features."""
    # make sure number of messages and vision feature sets match
    n_msg_images = sum(
        1
        for msg in messages
        if msg["role"] == "user"
        for part in msg.get("content", [])
        if part.get("type") == "image"
    )
    assert n_msg_images == len(vision_features), (
        "number of messages and vision features must match"
    )

    # create actual text template from messages dict
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )

    # create mock images such that their size corresponds to
    # the correct number of tokens after tokenizing and spatial merging
    # (we cannot just overwrite image_grid_thw because
    # input ids already contains placeholder vision tokens)
    effective_patch_size = QWEN_CONSTANTS[qwen_version]["effective_patch_size"]
    mock_images: List[Image.Image] = []
    for feat in vision_features:
        n = int(feat.shape[0])
        w, h = closest_factor_pair(n)
        img_w = w * effective_patch_size
        img_h = h * effective_patch_size
        mock_img = Image.new("RGB", (img_w, img_h), color="red")
        mock_images.append(mock_img)

    inputs = processor(
        text=text,
        images=mock_images,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )

    return inputs


def generate_with_vision_features(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    max_tokens: int = 5012,
    seed: int = 42,
):
    """Generate text from vision features.

    Args:
        messages: Chat messages with image placeholders
        vision_features: List of vision feature tensors.
            For qwen25: each tensor is (N, hidden_dim)
            For qwen3: each tensor is (N, hidden_dim * 4) containing [main | d0 | d1 | d2]
        model: Qwen model (patched version)
        processor: Qwen processor
        qwen_version: Either "qwen25" or "qwen3"
        max_tokens: Maximum tokens to generate
        seed: Random seed for deterministic sampling

    Returns:
        Generated text response
    """
    # For Qwen3, we need to split concatenated features into main + deepstack
    if qwen_version == "qwen3":
        main_features, deepstack_features = qwen3_format_multiple_deepstack_features(
            vision_features
        )

        # preprocess and generate
        inputs = model_inputs(
            messages, main_features, processor, qwen_version=qwen_version
        ).to(model.device)

        _set_generation_seed(seed)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            custom_patch_features=main_features,
            custom_deepstack_features=deepstack_features,
        )
    else:
        # Qwen2.5 path
        inputs = model_inputs(
            messages, vision_features, processor, qwen_version=qwen_version
        ).to(model.device)

        _set_generation_seed(seed)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            custom_patch_features=vision_features,
        )

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
    lines.append(f"EXCEPTION DURING AGENT GENERATION - Message Trace (iteration {iteration})")
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


def generate_with_vision_features_agentic(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    qwen_version: str = "qwen3",
    max_tokens: int = 5012,
    max_iterations: int = 10,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
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
        qwen_version: Either "qwen25" or "qwen3"
        max_tokens: Maximum tokens per generation
        max_iterations: Maximum number of tool-calling iterations
        tool_call_limits: Optional dict mapping tool_name -> max_calls (int or None for infinite).
            If None, all tools have infinite calls. If a tool is not in the dict, it defaults to infinite.
        verbose: If True, prints message and tool results at each iteration.
        seed: Random seed for deterministic sampling

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
    """
    assert qwen_version == "qwen3", (
        "qwen3 is the only supported version for agentic mode"
    )

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

    for iteration in range(max_iterations):
        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] --- Iteration {iteration} ---", flush=True)

        # Convert accumulated features to deepstack format for this iteration
        main_features, deepstack_features = qwen3_format_multiple_deepstack_features(
            all_vision_features
        )

        # Generate response with tools
        inputs = model_inputs(
            current_messages,
            main_features,
            processor,
            qwen_version=qwen_version,
            tools=tool_specs,
        ).to(model.device)

        try:
            _set_generation_seed(seed + iteration)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                custom_patch_features=main_features,
                custom_deepstack_features=deepstack_features,
            )
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
            return {
                "final_answer": final_answer,
                "message_history": current_messages,
                "tool_calls": tool_call_history,
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
                print(f"[{timestamp}] [Tool Call]: {tool_name}({json.dumps(_filter_tensors_for_debug(arguments))})", flush=True)

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
                            remaining_after if remaining_after is not None else "infinite"
                        )
                        result["text"] = json.dumps(result_data)
                    except Exception as e:
                        result = {"text": json.dumps({"error": f"Error executing tool: {str(e)}"})}
                        # Still decrement on error to prevent infinite retries
                        if remaining is not None:
                            remaining_calls[tool_name] = max(0, remaining - 1)
                            remaining_after_error = remaining_calls[tool_name]
                        else:
                            remaining_after_error = None
                        result_data = json.loads(result["text"])
                        result_data["remaining_calls"] = (
                            remaining_after_error if remaining_after_error is not None else "infinite"
                        )
                        result["text"] = json.dumps(result_data)

            if verbose:
                # Filter for logging
                log_result = _filter_tensors_for_debug(result)
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] [Tool Result]: {json.dumps(log_result)}\n", flush=True)

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
    return {
        "final_answer": final_answer,
        "message_history": current_messages,
        "tool_calls": tool_call_history,
    }


def prompt_with_graph_at_timestep(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    timestep_idx: int,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    system_prompt: str = None,
    seed: int = 42,
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
    """
    assert (
        len(adjacency_matrices)
        == len(node_centers)
        == len(node_centroids)
        == len(node_extents)
    ), "timestep mismatch"

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats = [node_feats[idx] for idx in node_feat_indices]
    node_feats = [i[timestep_idx] for i in node_feats]
    A = adjacency_matrices[timestep_idx]
    centroids = node_centroids[timestep_idx]
    extents = node_extents[timestep_idx]

    graph_content = []
    graph_content.append(
        {
            "type": "text",
            "text": "<spatial-graph>\n",
        }
    )
    for n in range(A.shape[0]):
        graph_content.extend(
            [
                {
                    "type": "text",
                    "text": f'<node id="{n}">\n',
                },
                {
                    "type": "text",
                    "text": "<descriptor>",
                },
                {
                    "type": "image",
                    "image": None,
                },
                {
                    "type": "text",
                    "text": "</descriptor>\n",
                },
                {
                    "type": "text",
                    "text": f'<centroid x="{centroids[n][0]:.2f}" y="{centroids[n][1]:.2f}" z="{centroids[n][2]:.2f}"/>\n',
                },
                {
                    "type": "text",
                    "text": f'<extent x="{extents[n][0]:.2f}" y="{extents[n][1]:.2f}" z="{extents[n][2]:.2f}"/>\n',
                },
                {
                    "type": "text",
                    "text": "</node>\n",
                },
            ]
        )
    for n in range(A.shape[0]):
        for m in range(A.shape[1]):
            if A[n, m] > 0:
                graph_content.append(
                    {
                        "type": "text",
                        "text": f'<edge from="{n}" to="{m}" overlap_score="{A[n, m]:.2f}" centroid_distance="{np.linalg.norm(centroids[n] - centroids[m]):.2f}"/>\n',
                    }
                )
    graph_content.append(
        {
            "type": "text",
            "text": "</spatial-graph>\n",
        }
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                *graph_content,
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
            ],
        },
    ]

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats],
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        max_tokens=5012,
        seed=seed,
    )


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
    qwen_version: str = "qwen3",
    system_prompt: str = None,
    max_iterations: int = 20,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
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
    """
    assert qwen_version == "qwen3", "qwen3 is required for graph agentic prompting"
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
        nodes_data.append({
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
        })

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
        qwen_version=qwen_version,
        max_tokens=5012,
        max_iterations=max_iterations,
        tool_call_limits=tool_call_limits,
        verbose=verbose,
        seed=seed,
    )


def prompt_with_static_graph(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    node_feats_timestep_idx: int,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    system_prompt: str = None,
    fps: float = None,
    seed: int = 42,
):
    """
    node_feats: np.lib.npyio.NpzFile - npz file containing node features for each timestep
    node_feats_timestep_idx: int - index of the timestep to use for the node features
    adjacency_matrices: np.ndarray - adjacency matrices through time - weights are bhattacharyya coefficients (timesteps, n_clusters, n_clusters)
    node_centers: np.ndarray - cluster centers through time (timesteps, n_clusters, 3)
    node_centroids: np.ndarray - cluster centroids through time (timesteps, n_clusters, 3)
    node_extents: np.ndarray - cluster extents through time (timesteps, n_clusters, 3)
    model: Qwen2_5_VLForConditionalGeneration - model to use
    processor: Qwen2_5_VLProcessor - processor to use
    system_prompt: str - system prompt to use
    fps: Optional frames per second. If provided, uses seconds format instead of timestep.
    """
    assert (
        len(adjacency_matrices)
        == len(node_centers)
        == len(node_centroids)
        == len(node_extents)
    ), "timestep mismatch"

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats = [node_feats[idx] for idx in node_feat_indices]
    node_feats = [i[node_feats_timestep_idx] for i in node_feats]
    object_content = []
    for i in range(len(node_feats)):
        object_content.extend(
            [
                {
                    "type": "text",
                    "text": f'<object id="{i}">',
                },
                {
                    "type": "image",
                    "image": None,
                },
                {
                    "type": "text",
                    "text": "</object>\n",
                },
            ]
        )

    graph_content = []
    for t in range(len(adjacency_matrices)):
        A = adjacency_matrices[t]
        
        # Format time reference
        if fps is not None:
            time_attr = f'timestep="{t}" {timestep_to_seconds_str(t, fps)}'
        else:
            time_attr = f'timestep="{t}"'
        
        graph_content.append(
            {
                "type": "text",
                "text": f'<spatial-graph {time_attr}>\n',
            }
        )
        for n in range(A.shape[0]):
            graph_content.extend(
                [
                    {
                        "type": "text",
                        "text": f'<node object-id="{n}">\n',
                    },
                    {
                        "type": "text",
                        "text": f'<centroid x="{node_centroids[t][n][0]:.2f}" y="{node_centroids[t][n][1]:.2f}" z="{node_centroids[t][n][2]:.2f}"/>\n',
                    },
                    {
                        "type": "text",
                        "text": f'<extent x="{node_extents[t][n][0]:.2f}" y="{node_extents[t][n][1]:.2f}" z="{node_extents[t][n][2]:.2f}"/>\n',
                    },
                    {
                        "type": "text",
                        "text": "</node>\n",
                    },
                ]
            )
        for n in range(A.shape[0]):
            for m in range(A.shape[1]):
                if A[n, m] > 0:
                    graph_content.append(
                        {
                            "type": "text",
                            "text": f'<edge from="{n}" to="{m}" overlap_score="{A[n, m]:.2f}" centroid_distance="{np.linalg.norm(node_centroids[t][n] - node_centroids[t][m]):.2f}"/>\n',
                        }
                    )
        graph_content.append(
            {
                "type": "text",
                "text": "</spatial-graph>\n",
            }
        )
    # TODO: adapt question and system_prompt
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<scene-graph>\n"},
                {"type": "text", "text": "<objects>\n"},
                *object_content,
                {"type": "text", "text": "</objects>\n"},
                {"type": "text", "text": "<spatial-graphs>\n"},
                *graph_content,
                {"type": "text", "text": "</spatial-graphs>\n"},
                {"type": "text", "text": "</scene-graph>\n"},
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
            ],
        },
    ]

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats],
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        max_tokens=5012,
        seed=seed,
    )


def prompt_with_dynamic_graph(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    system_prompt: str = None,
    fps: float = None,
    seed: int = 42,
):
    assert (
        len(adjacency_matrices)
        == len(node_centers)
        == len(node_centroids)
        == len(node_extents)
    ), "timestep mismatch"

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats = [node_feats[idx] for idx in node_feat_indices]

    graph_content = []
    for t in range(len(adjacency_matrices)):
        A = adjacency_matrices[t]
        
        # Format time reference
        if fps is not None:
            time_attr = f'timestep="{t}" {timestep_to_seconds_str(t, fps)}'
        else:
            time_attr = f'timestep="{t}"'
        
        graph_content.append(
            {
                "type": "text",
                "text": f'<spatial-graph {time_attr}>\n',
            }
        )
        for n in range(A.shape[0]):
            graph_content.extend(
                [
                    {
                        "type": "text",
                        "text": f'<node id="{n}">\n',
                    },
                    {
                        "type": "text",
                        "text": "<descriptor>",
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "text",
                        "text": "</descriptor>\n",
                    },
                    {
                        "type": "text",
                        "text": f'<center x="{node_centers[t][n][0]:.2f}" y="{node_centers[t][n][1]:.2f}" z="{node_centers[t][n][2]:.2f}"/>\n',
                    },
                    {
                        "type": "text",
                        "text": f'<centroid x="{node_centroids[t][n][0]:.2f}" y="{node_centroids[t][n][1]:.2f}" z="{node_centroids[t][n][2]:.2f}"/>\n',
                    },
                    {
                        "type": "text",
                        "text": f'<extent x="{node_extents[t][n][0]:.2f}" y="{node_extents[t][n][1]:.2f}" z="{node_extents[t][n][2]:.2f}"/>\n',
                    },
                    {
                        "type": "text",
                        "text": "</node>\n",
                    },
                ]
            )
        for n in range(A.shape[0]):
            for m in range(A.shape[1]):
                if A[n, m] > 0:
                    centroid_dist = float(
                        np.linalg.norm(node_centroids[t][n] - node_centroids[t][m])
                    )
                    graph_content.append(
                        {
                            "type": "text",
                            "text": f'<edge from="{n}" to="{m}" overlap_score="{A[n, m]:.2f}" centroid_distance="{centroid_dist:.2f}"/>\n',
                        }
                    )
        graph_content.append(
            {
                "type": "text",
                "text": "</spatial-graph>\n",
            }
        )
    
    # TODO: adapt question and system_prompt
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<scene-graph>\n"},
                {"type": "text", "text": "<spatial-graphs>\n"},
                *graph_content,
                {"type": "text", "text": "</spatial-graphs>\n"},
                {"type": "text", "text": "</scene-graph>\n"},
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
            ],
        },
    ]

    feature_list = []
    for n in range(len(node_feats)):
        for t in range(node_feats[n].shape[0]):
            feature_list.append(torch.Tensor(node_feats[n][t]))

    return generate_with_vision_features(
        messages=messages,
        vision_features=feature_list,
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        max_tokens=5012,
        seed=seed,
    )


def prompt_with_descriptors_at_timestep(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    timestep_idx: int,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    system_prompt: str = None,
    seed: int = 42,
):
    """
    Ablation of prompt_with_graph_at_timestep: only pass cluster descriptor images
    for a single timestep, without any graph structure (no nodes/edges/geometry).

    node_feats: np.lib.npyio.NpzFile - npz file containing node features through time
    timestep_idx: int - index of the timestep to use for the node features
    """
    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats_list = [node_feats[idx] for idx in node_feat_indices]
    node_feats_t = [i[timestep_idx] for i in node_feats_list]

    # Create a descriptor-only prompt with images, no graph structure
    descriptor_content = []
    descriptor_content.append(
        {"type": "text", "text": f'<descriptors t="{timestep_idx}">\n'}
    )
    for i in range(len(node_feats_t)):
        descriptor_content.extend(
            [
                {"type": "text", "text": f'<descriptor id="{i}">'},
                {"type": "image", "image": None},
                {"type": "text", "text": "</descriptor>\n"},
            ]
        )
    descriptor_content.append({"type": "text", "text": "</descriptors>\n"})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                *descriptor_content,
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
            ],
        },
    ]

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats_t],
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        max_tokens=5012,
        seed=seed,
    )


def prompt_with_dynamic_descriptors(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen25",
    system_prompt: str = None,
    fps: float = None,
    seed: int = 42,
):
    """
    Ablation of prompt_with_dynamic_graph: pass only descriptor images separated by
    timestep, omitting all graph structure (no nodes/edges/geometry).
    """
    assert (
        len(adjacency_matrices)
        == len(node_centers)
        == len(node_centroids)
        == len(node_extents)
    ), "timestep mismatch"

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats_list = [node_feats[idx] for idx in node_feat_indices]

    # Build descriptor-only content with timestep separation
    content = []
    for t in range(len(adjacency_matrices)):
        # Format time reference
        if fps is not None:
            time_attr = f'timestep="{t}" {timestep_to_seconds_str(t, fps)}'
        else:
            time_attr = f'timestep="{t}"'
        
        content.append({"type": "text", "text": f'<descriptors {time_attr}>\n'})
        # number of clusters = len(node_feats_list)
        for n in range(len(node_feats_list)):
            content.extend(
                [
                    {"type": "text", "text": f'<descriptor id="{n}">'},
                    {"type": "image", "image": None},
                    {"type": "text", "text": "</descriptor>\n"},
                ]
            )
        content.append({"type": "text", "text": "</descriptors>\n"})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<descriptors>\n"},
                *content,
                {"type": "text", "text": "</descriptors>\n"},
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
            ],
        },
    ]

    # Keep the same feature ordering pattern as prompt_with_dynamic_graph
    feature_list = []
    for n in range(len(node_feats_list)):
        for t in range(node_feats_list[n].shape[0]):
            feature_list.append(torch.Tensor(node_feats_list[n][t]))

    return generate_with_vision_features(
        messages=messages,
        vision_features=feature_list,
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        max_tokens=5012,
        seed=seed,
    )


def prompt_with_video_frames(
    question: str,
    image_paths: List[Any],
    model: Union[Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration],
    processor: Union[Qwen2_5_VLProcessor, Qwen3VLProcessor],
    qwen_version: str = "qwen3",
    system_prompt: str = None,
    max_tokens: int = 5012,
    fps: float = None,
    seed: int = 42,
) -> str:
    """Prompt model with video frames (list of images).
    
    Args:
        question: Question to ask about the video
        image_paths: List of image file paths (as strings or Path objects)
        model: Qwen VL model
        processor: Qwen VL processor
        qwen_version: Either "qwen25" or "qwen3"
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        fps: Optional frames per second for video metadata
        
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
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
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
                frames_indices=list(range(num_frames))
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
    
    # Generate
    _set_generation_seed(seed)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


def crop_patch_features(patch_feat: torch.Tensor, cw, ch, cx1, cx2, cy1, cy2):
    return patch_feat.reshape(ch, cw, -1)[cy1 : cy2 + 1, cx1 : cx2 + 1].flatten(
        end_dim=1
    )


def get_patch_hw(im_height: int, im_width: int, qwen_version: str) -> Tuple[int, int]:
    """Get patchgrid dimensions for given image dimensions.

    Uses smart_resize logic: round(dim / factor) * factor to get resized dimensions,
    then divide by factor to get patch count.

    Args:
        im_height: Image height in pixels
        im_width: Image width in pixels
        qwen_version: Either "qwen25" or "qwen3"

    Returns:
        Tuple of (patches_height, patches_width)
    """
    factor = QWEN_CONSTANTS[qwen_version]["effective_patch_size"]
    resized_h, resized_w = smart_resize(
        im_height, im_width, factor=factor, min_pixels=1, max_pixels=10**9
    )
    return resized_h // factor, resized_w // factor


def get_patch_segmasks(
    im_height: int, im_width: int, qwen_version: str
) -> torch.Tensor:
    """Generate an instance segmentation mask where each instance corresponds to one vision encoder patch.

    Args:
        im_height: Image height in pixels
        im_width: Image width in pixels
        qwen_version: Either "qwen25" or "qwen3"

    Returns:
        Tensor of shape (resized_H, resized_W) with patch indices at each pixel
    """
    factor = QWEN_CONSTANTS[qwen_version]["effective_patch_size"]
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
