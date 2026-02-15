"""
Utility functions for loading and using Qwen3-VL models with 3D Gaussian-to-grid positional encoding.

Provides model loading function that returns CustomQwen3VLForConditionalGeneration3D
with support for many-to-one Gaussian-to-grid mapping.
"""

import torch
from typing import Any, Dict, Literal, Optional, Union
from transformers import Qwen3VLProcessor

from .patched_qwen_3d import (
    CustomQwen3VLForConditionalGeneration3D,
)


def get_custom_qwen3_3d(
    size: Literal["8B", "32B"] = "8B",
    use_fp8: bool = False,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    repetition_penalty: float = None,
    compile: bool = False,
):
    """Get a custom Qwen3VL model with 3D Gaussian-to-grid positional encoding support.

    Loads the base Qwen3-VL model and swaps the class to CustomQwen3VLForConditionalGeneration3D
    which supports many-to-one Gaussian-to-grid mapping for 3D spatial queries.

    Args:
        size: Model size ("8B" or "32B")
        use_fp8: Whether to use FP8 quantized version
        attn_implementation: Attention implementation ("sdpa" or "flash_attention_2")
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        max_memory: Maximum memory per device
        repetition_penalty: Repetition penalty for generation
        compile: Whether to compile the model with torch.compile

    Returns:
        Tuple of (model, processor) where:
            model: CustomQwen3VLForConditionalGeneration3D instance
            processor: Qwen3VLProcessor instance
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

    # Load base model using from_pretrained
    # We load as the base class first, then swap to our custom class
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **fp_kwargs,
    )

    # Swap to our custom class that supports 3D positional encoding
    model.__class__ = CustomQwen3VLForConditionalGeneration3D
    # Re-initialize to ensure the inner model is also swapped
    # (CustomQwen3VLForConditionalGeneration3D.__init__ does this, but we need to call it)
    # Actually, since we're swapping the class, we need to ensure the inner model is also swapped
    # The __init__ method in CustomQwen3VLForConditionalGeneration3D does: self.model.__class__ = CustomQwen3VLModel3D
    # But we're swapping after __init__, so we need to do it manually
    from .patched_qwen_3d import CustomQwen3VLModel3D
    model.model.__class__ = CustomQwen3VLModel3D

    processor = Qwen3VLProcessor.from_pretrained(model_path)
    
    # Configure generation settings
    model.generation_config.return_legacy_cache = False
    if repetition_penalty is not None:
        model.generation_config.repetition_penalty = repetition_penalty
    
    model.eval()
    
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    return model, processor

