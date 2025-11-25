import torch
import math
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import Dict, List, Optional, Union, Any
from types import MethodType
from functools import lru_cache
from transformers.utils.quantization_config import BitsAndBytesConfig
import json

from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack

PATCH_SIZE = 14
SPATIAL_MERGE = 2
EFFECTIVE_PATCH_SIZE = PATCH_SIZE * SPATIAL_MERGE


# original function from transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# this is a modified version to work with raw patch features that can be used for monkey patching
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Track whether custom vision features are being used and capture mask safely
    image_mask = None
    using_custom_vision_features = False

    if pixel_values is not None:
        if "custom_patch_features" in kwargs:
            image_embeds = kwargs["custom_patch_features"]
            using_custom_vision_features = True
        else:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros(
                    (batch_size, seq_length), device=inputs_embeds.device
                )
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    # Optional: override vision token position ids with custom patch assignments
    # Expect kwargs["custom_vision_patch_positions"] as (num_vision_tokens, 3 [t,h,w]) per batch item
    # and kwargs["custom_vision_token_indices"] as List[List[int]] token indices for vision placeholders
    custom_patch_pos = kwargs.pop("custom_vision_patch_positions", None)
    custom_tok_idx = kwargs.pop("custom_vision_token_indices", None)
    if custom_patch_pos is not None and custom_tok_idx is not None and position_ids is not None:
        # Ensure tensor on correct device/dtype
        if not torch.is_tensor(custom_patch_pos):
            custom_patch_pos = torch.as_tensor(custom_patch_pos, device=inputs_embeds.device)
        custom_patch_pos = custom_patch_pos.to(device=inputs_embeds.device, dtype=position_ids.dtype)

        # Handle batch = 1 common case; support lists per batch
        if isinstance(custom_tok_idx, (list, tuple)) and len(custom_tok_idx) > 0 and isinstance(custom_tok_idx[0], (list, tuple)):
            token_indices_per_batch = custom_tok_idx
        else:
            token_indices_per_batch = [custom_tok_idx]

        batch_size = position_ids.shape[1]
        # If a single positions tensor is provided, reuse for all batch items
        per_batch_positions = [custom_patch_pos for _ in range(batch_size)]

        for b in range(min(batch_size, len(token_indices_per_batch))):
            tok_idx = token_indices_per_batch[b]
            if tok_idx is None or len(tok_idx) == 0:
                continue
            # Expect shape (N, 3), slice if longer than tok count
            pos_b = per_batch_positions[b]
            if pos_b.dim() == 2 and pos_b.shape[-1] == 3:
                # Align length
                n = min(len(tok_idx), pos_b.shape[0])
                assign = pos_b[:n].T  # (3, n)
                # Overwrite t/h/w indices for the selected token positions
                position_ids[:, b, tok_idx[:n]] = assign

    # Always zero out positional encodings for any custom vision tokens
    # When using custom patch features, relative patch geometry should not affect prompts
    if using_custom_vision_features and (position_ids is not None):
        # If we have the image_mask from placeholder computation, derive token indices
        if image_mask is not None:
            token_mask = image_mask.any(dim=-1)  # (batch, seq)
            if token_mask.ndim == 2:
                for b in range(token_mask.shape[0]):
                    tok_idx_b = torch.nonzero(token_mask[b], as_tuple=False).squeeze(-1)
                    if tok_idx_b.numel() > 0:
                        position_ids[:, b, tok_idx_b] = 0
        # Also support explicit token indices if provided
        if custom_tok_idx is not None:
            if isinstance(custom_tok_idx, (list, tuple)) and len(custom_tok_idx) > 0 and isinstance(custom_tok_idx[0], (list, tuple)):
                token_indices_per_batch = custom_tok_idx
            else:
                token_indices_per_batch = [custom_tok_idx]
            batch_size = position_ids.shape[1]
            for b in range(min(batch_size, len(token_indices_per_batch))):
                tok_idx = token_indices_per_batch[b]
                if tok_idx is None or len(tok_idx) == 0:
                    continue
                position_ids[:, b, tok_idx] = 0

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def square_crop_image(image: Image.Image, xslide=0, yslide=0):
    side_length = min(image.width, image.height)
    wm, hm, rd = image.width // 2, image.height // 2, side_length // 2
    return image.crop(
        (wm - rd - xslide, hm - rd - yslide, wm + rd - xslide, hm + rd - yslide)
    )


def get_patched_qwen(
    use_bnb_4bit: bool = False,
    use_bnb_8bit: bool = False,
    attn_implementation: str = "sdpa",  # "flash_attention_2" or "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
):
    """Get a monkey-patched Qwen2_5_VL model/processor that supports raw patch features.

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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **fp_kwargs,
    )
    model.model.forward = MethodType(forward, model.model)
    # Also patch prepare_inputs_for_generation to accept and forward custom_patch_features through generation
    orig_prepare = model.prepare_inputs_for_generation

    # input preparation needs to access custom_patch_features
    # so transformers doesn't complain about unsupported attributes
    def prepare_inputs_for_generation_wrapper(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        custom_patch_features=None,
        **kwargs,
    ):
        # During decoding steps, GenerationMixin provides only the last token in input_ids
        # but the full attention_mask. Align mask length to avoid shape mismatches
        # inside Qwen2_5_VL get_rope_index when indexing with the mask.
        if attention_mask is not None and input_ids is not None:
            # If lengths differ, slice mask to align with provided input_ids tokens
            if attention_mask.shape[-1] != input_ids.shape[-1]:
                attention_mask = attention_mask[..., -input_ids.shape[-1] :]
        model_inputs = orig_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        return model_inputs

    model.prepare_inputs_for_generation = MethodType(
        prepare_inputs_for_generation_wrapper, model
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    # Prefer new cache format for memory-efficient caches
    model.generation_config.return_legacy_cache = False
    model.eval()
    return model, processor


def qwen_encode_image(
    image: Image.Image,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    image_inputs = processor.image_processor(  # type:ignore
        images=[image], return_tensors="pt"
    )
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)
    with torch.no_grad():
        feats = model.visual(pixel_values, image_grid_thw)
        return feats


def patches_to_2d(patch_features: torch.Tensor, src_image: Image.Image):
    patches_width, patches_height = (
        src_image.width // EFFECTIVE_PATCH_SIZE,
        src_image.height // EFFECTIVE_PATCH_SIZE,
    )
    return patch_features.reshape(patches_height, patches_width, -1)


def patches_to_2d_from_hw(
    patch_features: torch.Tensor, img_height: int, img_width: int
):
    patches_width, patches_height = (
        img_width // EFFECTIVE_PATCH_SIZE,
        img_height // EFFECTIVE_PATCH_SIZE,
    )
    return patch_features.reshape(patches_height, patches_width, -1)


def patch_cds(src_image: Image.Image):
    effective_patch_size = PATCH_SIZE * SPATIAL_MERGE
    patches_width, patches_height = (
        src_image.width // effective_patch_size,
        src_image.height // effective_patch_size,
    )
    patch_coords_list = []
    for i in range(patches_height):
        for j in range(patches_width):
            y1 = i * effective_patch_size
            y2 = min((i + 1) * effective_patch_size, src_image.height)
            x1 = j * effective_patch_size
            x2 = min((j + 1) * effective_patch_size, src_image.width)
            patch_coords_list.append((y1, y2, x1, x2))
    cds: torch.Tensor = torch.as_tensor(patch_coords_list)
    cds: torch.Tensor = cds.reshape(patches_height, patches_width, 2, 2)
    return cds


# This function takes in an RGB image  and a prompt
def ask_qwen_about_image(
    image: Image.Image,
    prompt: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
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
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
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
def closest_factor_pair(n) -> tuple[int, int]:
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            return a, n // a
    raise Exception(
        "the given feature patches don't correspond to a nice rectangular size in pixels"
    )


# This function takes in the **patch features** of an image and a prompt
def ask_qwen_about_image_features(
    image_features: torch.Tensor,
    prompt: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
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
        messages, [image_features], model, processor, 128
    )


def model_inputs(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    processor: Qwen2_5_VLProcessor,
):
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
        messages, tokenize=False, add_generation_prompt=True
    )

    # create mock images so the processor precomputes grid; features are passed as-is
    mock_images: List[Image.Image] = []
    for i in range(len(vision_features)):
        n = int(vision_features[i].shape[0])
        w, h = closest_factor_pair(n)
        assert w * h == n
        mock_img = Image.new(
            "RGB", (EFFECTIVE_PATCH_SIZE * w, EFFECTIVE_PATCH_SIZE * h), color="red"
        )
        mock_images.append(mock_img)

    inputs = processor(
        text=[text],
        images=[mock_images],
        padding=True,
        return_tensors="pt",
    )
    return inputs


def generate_with_vision_features(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    max_tokens: int = 128,
):
    # preprocess and generate
    inputs = model_inputs(messages, vision_features, processor).to(model.device)
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

def prompt_with_graph_at_timestep(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    timestep_idx: int,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = None,
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
            "text": '<spatial-graph>\n',
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

    # TODO: adapt question and system_prompt
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<scene-graph>\n"},
                *graph_content,
                {"type": "text", "text": "</scene-graph>\n"},
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
                {"type": "text", "text": "\nYour response:\n"},
            ],
        },
    ]

    with open("qwen_messages.json", "w") as fp:
        json.dump(messages, fp)

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats],
        model=model,
        processor=processor,
        # TODO: increase this?
        max_tokens=5012,
    )

def prompt_with_static_graph(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    node_feats_timestep_idx: int,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = None,
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
        graph_content.append(
            {
                "type": "text",
                "text": f'<spatial-graph t="{t}">\n',
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
                {"type": "text", "text": "\nYour response:\n"},
            ],
        },
    ]

    with open("qwen_messages.json", "w") as fp:
        json.dump(messages, fp)

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats],
        model=model,
        processor=processor,
        # TODO: increase this?
        max_tokens=5012,
    )


def prompt_with_dynamic_graph(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str,
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
        graph_content.append(
            {
                "type": "text",
                "text": f'<spatial-graph t="{t}">\n',
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
                    centroid_dist = float(np.linalg.norm(node_centroids[t][n] - node_centroids[t][m]))
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
                {"type": "text", "text": "\nYour response:\n"},
            ],
        },
    ]

    with open("qwen_messages.json", "w") as fp:
        json.dump(messages, fp)

    feature_list = []
    for n in range(len(node_feats)):
        for t in range(node_feats[n].shape[0]):
            feature_list.append(torch.Tensor(node_feats[n][t]))

    # TODO include l2 distances on edges as well
    return generate_with_vision_features(
        messages=messages,
        vision_features=feature_list,
        model=model,
        processor=processor,
        # TODO: increase this?
        max_tokens=5012,
    )


def prompt_with_descriptors_at_timestep(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    timestep_idx: int,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = None,
):
    """
    Ablation of prompt_with_graph_at_timestep: only pass cluster descriptor images
    for a single timestep, without any graph structure (no nodes/edges/geometry).

    node_feats: np.lib.npyio.NpzFile - npz file containing node features through time
    timestep_idx: int - index of the timestep to use for the node features
    """
    # keep signature parity; inputs other than node_feats/timestep are unused here
    _ = adjacency_matrices, node_centers, node_centroids, node_extents

    # node feat indices correspond to cluster ids
    node_feat_indices = sorted(list(node_feats.keys()), key=lambda x: int(x))
    node_feats_list = [node_feats[idx] for idx in node_feat_indices]
    node_feats_t = [i[timestep_idx] for i in node_feats_list]

    # Create a descriptor-only prompt with images, no graph structure
    descriptor_content = []
    descriptor_content.append({"type": "text", "text": f'<descriptors t="{timestep_idx}">\n'})
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
                {"type": "text", "text": "<descriptors>\n"},
                *descriptor_content,
                {"type": "text", "text": "</descriptors>\n"},
                {"type": "text", "text": "<prompt>"},
                {"type": "text", "text": question},
                {"type": "text", "text": "</prompt>\n"},
                {"type": "text", "text": "\nYour response:\n"},
            ],
        },
    ]

    with open("qwen_messages.json", "w") as fp:
        json.dump(messages, fp)

    return generate_with_vision_features(
        messages=messages,
        vision_features=[torch.Tensor(f) for f in node_feats_t],
        model=model,
        processor=processor,
        max_tokens=5012,
    )


def prompt_with_dynamic_descriptors(
    question: str,
    node_feats: np.lib.npyio.NpzFile,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str,
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
        content.append({"type": "text", "text": f'<descriptors t="{t}">\n'})
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
                {"type": "text", "text": "\nYour response:\n"},
            ],
        },
    ]

    with open("qwen_messages.json", "w") as fp:
        json.dump(messages, fp)

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
        max_tokens=5012,
    )

def crop_patch_features(patch_feat: torch.Tensor, cw, ch, cx1, cx2, cy1, cy2):
    return patch_feat.reshape(ch, cw, -1)[cy1 : cy2 + 1, cx1 : cx2 + 1].flatten(
        end_dim=1
    )


def get_patch_segmasks(im_height, im_width):
    """generate an instance segmentation mask
    where each instance corresponds to one qwen 2.5 vl vision encoder patch"""
    # raw_patches_height, raw_patches_width = (
    #     im_height // PATCH_SIZE,
    #     im_width // PATCH_SIZE,
    # )
    # patches_width = torch.ceil(torch.div(raw_patches_width, SPATIAL_MERGE))
    # rowcol = torch.stack(
    #     torch.meshgrid(
    #         torch.arange(raw_patches_height * PATCH_SIZE),
    #         torch.arange(raw_patches_width * PATCH_SIZE),
    #         indexing="ij",
    #     )
    # )
    # patch_coords = torch.floor_divide(rowcol, EFFECTIVE_PATCH_SIZE)
    # return patch_coords[0] * patches_width + patch_coords[1]
    patches_height = im_height // EFFECTIVE_PATCH_SIZE + (
        (im_height // PATCH_SIZE) % 4 == 3
    )  # cursed behavior
    patches_width = im_width // EFFECTIVE_PATCH_SIZE + (
        (im_width // PATCH_SIZE) % 4 == 3
    )  # -,,-
    rowcol = torch.stack(
        torch.meshgrid(
            torch.arange(
                patches_height * EFFECTIVE_PATCH_SIZE
                + ((im_height // PATCH_SIZE) % 4 == 3) * PATCH_SIZE
            ),
            torch.arange(patches_width * EFFECTIVE_PATCH_SIZE)
            + ((im_width // PATCH_SIZE) % 4 == 3) * PATCH_SIZE,
            indexing="ij",
        )
    )
    patch_coords = torch.floor_divide(rowcol, EFFECTIVE_PATCH_SIZE)
    return patch_coords[0] * patches_width + patch_coords[1]


def center_crop_patch_size_multiple(img: Image.Image):
    """center crop a PIL image to a multiple of qwen 2.5 vl vision encoder patch size"""
    side_cropping_vertical = img.height % PATCH_SIZE
    side_cropping_horizontal = img.width % PATCH_SIZE
    crop_top, crop_bottom = (
        np.floor(side_cropping_vertical / 2),
        np.ceil(side_cropping_vertical / 2),
    )
    crop_left, crop_right = (
        np.floor(side_cropping_horizontal / 2),
        np.ceil(side_cropping_horizontal / 2),
    )
    return img.crop(
        (crop_left, crop_top, img.width - crop_right, img.height - crop_bottom)
    )
