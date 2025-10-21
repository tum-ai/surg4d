from typing import Any, Dict, Optional
from abc import ABC

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from ..qwen_vl_qwen3 import generate_with_vision_features_qwen3


class LocalMLLMInterfaceQwen3(ABC):
    """Minimal local interface for Qwen3-VL using Auto classes and patch-level features.

    Keep this separate from the Qwen2.5 interface so both can coexist.
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        use_bnb_4bit: bool = False,
        use_bnb_8bit: bool = False,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_bnb_4bit = use_bnb_4bit
        self.use_bnb_8bit = use_bnb_8bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        from_kwargs: Dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.use_bnb_4bit:
            from_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_bnb_8bit:
            from_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **from_kwargs
        )
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = self.temperature
        self.model.eval()

    def query_with_features(
        self,
        image_features: torch.Tensor,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": question},
                ],
            }
        )
        return generate_with_vision_features_qwen3(
            messages, [image_features], self.model, self.processor, self.max_tokens
        )
