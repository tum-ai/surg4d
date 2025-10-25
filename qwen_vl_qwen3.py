import math
from types import MethodType
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch.nn.functional as F


# Conservative defaults matching common Qwen VL configs
PATCH_SIZE = 14
SPATIAL_MERGE = 2
EFFECTIVE_PATCH_SIZE = PATCH_SIZE * SPATIAL_MERGE


def _maybe_make_bnb_config(
    use_bnb_4bit: bool, use_bnb_8bit: bool, torch_dtype: torch.dtype
) -> Optional[BitsAndBytesConfig]:
    if use_bnb_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    if use_bnb_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_patched_qwen3(
    model_path: str = "Qwen/Qwen3-VL-7B-Instruct",
    use_bnb_4bit: bool = False,
    use_bnb_8bit: bool = False,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Load Qwen3-VL and install a light-weight patch to inject precomputed patch features.

    This mirrors the Qwen2.5 approach but uses Auto classes with trust_remote_code=True.
    The patch does NOT rewrite model internals; it wraps input-prep and forwards to allow
    passing `custom_patch_features` through generation, then injects them at the image
    placeholder positions.
    """

    bnb_config = _maybe_make_bnb_config(use_bnb_4bit, use_bnb_8bit, torch_dtype)

    from_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        from_kwargs["quantization_config"] = bnb_config
    if max_memory is not None:
        from_kwargs["max_memory"] = max_memory

    # attn_implementation is accepted by many recent models; try it, fallback if needed
    # Strictly use the official Qwen3-VL class (no fallbacks)
    try:
        from_kwargs["attn_implementation"] = attn_implementation
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **from_kwargs
        )
    except TypeError:
        from_kwargs.pop("attn_implementation", None)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **from_kwargs
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Prefer new cache format if available
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.return_legacy_cache = False
        except Exception:
            pass

    # 1) Patch prepare_inputs_for_generation to thread through `custom_patch_features`.
    orig_prepare = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_wrapper(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        custom_patch_features = kwargs.pop("custom_patch_features", None)
        model_inputs = orig_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        return model_inputs

    model.prepare_inputs_for_generation = MethodType(
        prepare_inputs_for_generation_wrapper, model
    )

    # 2) Patch the inner multimodal model forward to inject features at placeholders
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "forward"):
        # As a fallback, try to patch the top-level forward (less ideal)
        inner = model

    orig_forward = inner.forward

    def patched_forward(self, *args, **kwargs):
        # Only intercept when custom features provided; otherwise delegate
        custom_patch_features = kwargs.pop("custom_patch_features", None)
        if custom_patch_features is None:
            # Fallback: check attribute set by top-level wrapper
            custom_patch_features = getattr(self, "_custom_patch_features", None)
        if custom_patch_features is None:
            return orig_forward(*args, **kwargs)

        # Extract common inputs; rely on kwargs as modern models pass by name
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)

        # 2.1 Ensure we have token embeddings to write into
        if inputs_embeds is None:
            if input_ids is None:
                # If neither provided, we cannot build inputs; delegate to orig
                return orig_forward(*args, **kwargs)
            if hasattr(self, "get_input_embeddings"):
                inputs_embeds = self.get_input_embeddings()(input_ids)
            elif hasattr(model, "get_input_embeddings"):
                inputs_embeds = model.get_input_embeddings()(input_ids)
            else:
                return orig_forward(*args, **kwargs)

        # 2.2 Determine image features to inject (use custom if provided, else fallback)
        if custom_patch_features is not None:
            image_embeds = custom_patch_features
        else:
            # Fallback: compute from pixel_values when not provided (rare here)
            image_embeds = None
            for fn_name in (
                "get_image_features",
                "get_vision_features",
                "compute_image_features",
            ):
                fn = getattr(self, fn_name, None)
                if fn is not None and pixel_values is not None:
                    image_embeds = fn(pixel_values, image_grid_thw)
                    break
            if image_embeds is None:
                return orig_forward(*args, **kwargs)

        # Concatenate list of image features if model returns per-image list
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # 2.3 Locate placeholder mask expected by the model (first without features),
        # then adapt our features to that token count and scatter visual tokens.
        image_mask = None
        expected_tokens = None
        for fn_name in (
            "get_placeholder_mask",
            "compute_placeholder_mask",
            "mask_image_placeholders",
        ):
            fn = getattr(self, fn_name, None)
            if fn is not None:
                try:
                    tmp = fn(input_ids, inputs_embeds=inputs_embeds)
                except TypeError:
                    tmp = fn(input_ids)
                if isinstance(tmp, tuple):
                    image_mask = tmp[0]
                else:
                    image_mask = tmp
                if image_mask is not None:
                    expected_tokens = int(image_mask.sum().item())
                break

        if image_mask is None:
            # Cannot safely inject; delegate
            return orig_forward(*args, **kwargs)

        # Resize features to expected token count if needed
        if expected_tokens is not None:
            K, D = image_embeds.shape
            if K != expected_tokens:
                # Infer input grid (h_in, w_in) and desired (h_out, w_out)
                h_in, w_in = closest_factor_pair(K)
                h_out, w_out = closest_factor_pair(expected_tokens)
                grid_in = (
                    image_embeds.reshape(h_in, w_in, D).permute(2, 0, 1).unsqueeze(0)
                )
                grid_out = F.interpolate(
                    grid_in, size=(h_out, w_out), mode="bilinear", align_corners=False
                )
                image_embeds = (
                    grid_out.squeeze(0).permute(1, 2, 0).reshape(expected_tokens, D)
                )

        # Recompute mask using features (some impls validate length)
        for fn_name in (
            "get_placeholder_mask",
            "compute_placeholder_mask",
            "mask_image_placeholders",
        ):
            fn = getattr(self, fn_name, None)
            if fn is not None:
                try:
                    res = fn(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        image_features=image_embeds,
                    )
                except TypeError:
                    res = fn(input_ids, inputs_embeds=inputs_embeds)
                if isinstance(res, tuple):
                    image_mask = res[0]
                else:
                    image_mask = res
                break

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 2.4 Call original forward with inputs_embeds and no pixel_values
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = inputs_embeds
        kwargs["pixel_values"] = None

        # Some models validate attention-mask length; keep alignment if needed
        if (
            attention_mask is not None
            and kwargs.get("input_ids", None) is None
            and inputs_embeds is not None
        ):
            # Preserve original attention mask; if model expects alignment, let it handle internally
            kwargs["attention_mask"] = attention_mask

        return orig_forward(*args, **kwargs)

    inner.forward = MethodType(patched_forward, inner)

    # Also patch the top-level forward to accept arbitrary kwargs and stash custom features
    orig_top_forward = model.forward

    def top_forward(self, *args, **kwargs):
        cpf = kwargs.pop("custom_patch_features", None)
        if cpf is not None and hasattr(self, "model") and self.model is not None:
            setattr(self.model, "_custom_patch_features", cpf)
        return orig_top_forward(*args, **kwargs)

    model.forward = MethodType(top_forward, model)

    model.eval()
    return model, processor


def qwen3_encode_image(
    image: Image.Image, model: torch.nn.Module, processor: Any
) -> torch.Tensor:
    """Extract patch features from an image using Qwen3 vision tower.

    Tries `model.visual(...)` first, then `model.model.visual(...)`. If unavailable,
    attempts standard processor path and raises if the tower is not exposed.
    """
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")  # type: ignore[attr-defined]
    pixel_values = image_inputs["pixel_values"].to(model.device)
    image_grid_thw = image_inputs.get("image_grid_thw", None)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(model.device)

    def _coerce_to_2d(feat_out: Any) -> torch.Tensor:
        # Accept nested tuple/list/dict/ModelOutput/tensor and return (num_patches, hidden_dim)
        def collect_tensors(obj: Any, acc: list[torch.Tensor]):
            if isinstance(obj, torch.Tensor):
                acc.append(obj)
                return
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    collect_tensors(it, acc)
                return
            if isinstance(obj, dict):
                # Prefer common keys first
                for key in (
                    "image_features",
                    "visual_features",
                    "last_hidden_state",
                    "embeddings",
                    "features",
                ):
                    if key in obj and isinstance(obj[key], torch.Tensor):
                        acc.append(obj[key])
                for v in obj.values():
                    collect_tensors(v, acc)
                return
            # ModelOutput-like: iterate attributes
            for attr in (
                "image_features",
                "visual_features",
                "last_hidden_state",
                "embeddings",
                "features",
            ):
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if isinstance(val, torch.Tensor):
                        acc.append(val)
            # Fallback: scan dir for tensor attrs (best-effort)
            for name in dir(obj):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(obj, name)
                except Exception:
                    continue
                if isinstance(val, torch.Tensor):
                    acc.append(val)

        tensors: list[torch.Tensor] = []
        collect_tensors(feat_out, tensors)
        if not tensors:
            raise RuntimeError(
                "Could not extract tensor features from vision encoder output."
            )
        candidate = tensors[0]
        # Ensure 2D: (num_patches, hidden_dim)
        if candidate.ndim == 3:
            # assume (batch, seq, dim)
            if candidate.shape[0] == 1:
                candidate = candidate[0]
            else:
                candidate = candidate.reshape(-1, candidate.shape[-1])
        elif candidate.ndim == 4:
            # flatten spatial dims if any (batch, h, w, dim)
            b, h, w, d = candidate.shape
            candidate = candidate.reshape(b * h * w, d)
        elif candidate.ndim == 2:
            pass
        else:
            candidate = (
                candidate.flatten(1) if candidate.ndim > 1 else candidate.unsqueeze(0)
            )
        return candidate

    with torch.no_grad():
        # Prefer model's image feature API first (matches token grid expected by placeholders)
        inner = getattr(model, "model", model)
        for fn_name in (
            "get_image_features",
            "get_vision_features",
            "encode_image",
            "forward_vision",
        ):
            fn = getattr(inner, fn_name, None)
            if callable(fn):
                try:
                    out = fn(pixel_values, image_grid_thw)
                except TypeError:
                    out = fn(pixel_values)
                return _coerce_to_2d(out)

        # Fallback: direct visual() on top-level or inner model
        for owner in (model, getattr(model, "model", None)):
            if owner is None:
                continue
            visual = getattr(owner, "visual", None)
            if callable(visual):
                try:
                    out = visual(pixel_values, image_grid_thw)
                except TypeError:
                    out = visual(pixel_values)
                return _coerce_to_2d(out)

        raise RuntimeError(
            "Qwen3 model does not expose a recognized vision feature API (get_image_features/visual/etc.)."
        )


def patches_to_2d(patch_features: torch.Tensor, src_image: Image.Image) -> torch.Tensor:
    patches_width = src_image.width // EFFECTIVE_PATCH_SIZE
    patches_height = src_image.height // EFFECTIVE_PATCH_SIZE
    return patch_features.reshape(patches_height, patches_width, -1)


def patches_to_2d_from_hw(
    patch_features: torch.Tensor, img_height: int, img_width: int
) -> torch.Tensor:
    patches_width = img_width // EFFECTIVE_PATCH_SIZE
    patches_height = img_height // EFFECTIVE_PATCH_SIZE
    return patch_features.reshape(patches_height, patches_width, -1)


@lru_cache(maxsize=1000)
def closest_factor_pair(n: int) -> Tuple[int, int]:
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            return a, n // a
    raise ValueError("feature patches do not correspond to a rectangular layout")


def _build_model_inputs(
    messages: List[Dict[str, Any]], vision_features: List[torch.Tensor], processor: Any
) -> Dict[str, torch.Tensor]:
    # Validate image placeholders count
    n_msg_images = sum(
        1
        for msg in messages
        if msg.get("role") == "user"
        for part in msg.get("content", [])
        if part.get("type") == "image"
    )
    assert n_msg_images == len(vision_features), (
        "number of messages and vision features must match"
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Create mock images to let the processor compute image_grid_thw
    mock_images: List[Image.Image] = []
    for feats in vision_features:
        w, h = closest_factor_pair(int(feats.shape[0]))
        mock_img = Image.new(
            "RGB", (EFFECTIVE_PATCH_SIZE * w, EFFECTIVE_PATCH_SIZE * h), color="red"
        )
        mock_images.append(mock_img)

    inputs = processor(
        text=[text], images=[mock_images], padding=True, return_tensors="pt"
    )
    return inputs


def _resize_feature_grid_to(
    enc: Dict[str, torch.Tensor], feats: torch.Tensor
) -> torch.Tensor:
    """Resize feature grid to match model's expected image token grid.

    enc: output dict from processor containing image_grid_thw
    feats: tensor of shape (K, D)
    returns: tensor of shape (H*W, D) where H*W matches enc grid
    """
    grid = enc.get("image_grid_thw", None)
    if grid is None:
        return feats
    # Expect shape (num_images, 3): [T,H,W]; for images T is usually 1
    g = grid[0].tolist()
    if len(g) != 3:
        return feats
    _, H, W = g
    K, D = feats.shape
    if H * W == K:
        return feats
    # infer input grid from K
    in_h, in_w = closest_factor_pair(K)
    grid_in = (
        feats.reshape(in_h, in_w, D).permute(2, 0, 1).unsqueeze(0)
    )  # [1,D,in_h,in_w]
    grid_out = F.interpolate(grid_in, size=(H, W), mode="bilinear", align_corners=False)
    out = grid_out.squeeze(0).permute(1, 2, 0).reshape(H * W, D)
    return out


def get_patch_segmasks(im_height: int, im_width: int) -> torch.Tensor:
    """Generate an instance segmentation mask for Qwen3-VL patches.

    Each instance corresponds to one Qwen3-VL vision encoder patch.
    Returns a tensor of shape (H, W) where each pixel value is the patch index it belongs to.
    """
    # Qwen3 uses the same patch size conventions as Qwen2.5
    patches_height = im_height // EFFECTIVE_PATCH_SIZE + (
        (im_height // PATCH_SIZE) % 4 == 3
    )
    patches_width = im_width // EFFECTIVE_PATCH_SIZE + (
        (im_width // PATCH_SIZE) % 4 == 3
    )

    rowcol = torch.stack(
        torch.meshgrid(
            torch.arange(
                patches_height * EFFECTIVE_PATCH_SIZE
                + ((im_height // PATCH_SIZE) % 4 == 3) * PATCH_SIZE
            ),
            torch.arange(
                patches_width * EFFECTIVE_PATCH_SIZE
                + ((im_width // PATCH_SIZE) % 4 == 3) * PATCH_SIZE
            ),
            indexing="ij",
        )
    )
    patch_coords = torch.floor_divide(rowcol, EFFECTIVE_PATCH_SIZE)
    return patch_coords[0] * patches_width + patch_coords[1]


def generate_with_vision_features_qwen3(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: torch.nn.Module,
    processor: Any,
    max_tokens: int = 128,
) -> str:
    enc = _build_model_inputs(messages, vision_features, processor)
    # Adjust custom features to expected token grid
    adjusted = [
        _resize_feature_grid_to(enc, vf.to(model.device)) for vf in vision_features
    ]
    inputs: Dict[str, torch.Tensor] = {}
    if "input_ids" in enc:
        inputs["input_ids"] = enc["input_ids"].to(model.device)
    if "attention_mask" in enc:
        inputs["attention_mask"] = enc["attention_mask"].to(model.device)
    if "position_ids" in enc:
        inputs["position_ids"] = enc["position_ids"].to(model.device)
    # Stash custom features directly on the inner model to bypass generate() kwargs validation
    if hasattr(model, "model") and model.model is not None:
        setattr(model.model, "_custom_patch_features", adjusted)
    else:
        setattr(model, "_custom_patch_features", adjusted)
    with torch.no_grad():
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
