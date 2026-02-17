"""
Custom Qwen VL classes for 3D spatial queries with many-to-one Gaussian-to-grid positional encoding.

Extends base Qwen3VLModel to support:
1. Custom patch features (bypassing vision encoder)
2. Many-to-one Gaussian-to-grid positional encoding for 3D spatial queries

"""

import torch
from typing import Optional, Union
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

from loguru import logger


class CustomQwen3VLModel3D(Qwen3VLModel):
    """Qwen3VLModel with support for custom patch features and many-to-one Gaussian-to-grid positional encoding.

    Extends base Qwen3VLModel to handle:
    - Custom patch features (bypassing vision encoder)
    - Many-to-one Gaussian-to-grid mapping for 3D spatial queries
    - Custom positional encodings based on Gaussian spatial positions
    """

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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        """Forward pass with support for custom_patch_features and gaussian_to_grid_mapping.

        Additional kwargs:
            custom_patch_features: List of pre-computed main vision features (one tensor per image)
            custom_deepstack_features: List of 3 tensors for deepstack layers
            gaussian_to_grid_mapping: Optional tensor mapping Gaussian tokens to grid positions.
                Shape: (num_gaussians, 2) with [h_idx, w_idx] for each Gaussian.
            zero_image_hw: Whether to zero out h, w positional encodings
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            # logger.info("=" * 80)
            # logger.info("FORWARD PASS: Getting input embeddings")
            # logger.info("=" * 80)
            # logger.info(f"input_ids shape: {input_ids.shape if input_ids is not None else None}")
            
            # # Decode input_ids to see what text the model sees
            # if input_ids is not None:
            #     try:
            #         decoded_text = self.language_model.embed_tokens.weight.device
            #         # Get tokenizer from processor if available, otherwise try to decode
            #         # For now, just log the shape and first/last tokens
            #         logger.info(f"input_ids first 10 tokens: {input_ids[0, :10].tolist() if input_ids.shape[1] > 10 else input_ids[0].tolist()}")
            #         logger.info(f"input_ids last 10 tokens: {input_ids[0, -10:].tolist() if input_ids.shape[1] > 10 else input_ids[0].tolist()}")
            #     except Exception as e:
            #         logger.warning(f"Could not decode input_ids: {e}")
            
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # logger.info(f"inputs_embeds shape: {inputs_embeds.shape}")

        image_mask = None
        video_mask = None
        deepstack_visual_embeds = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        custom_patch_features = kwargs.get("custom_patch_features", None)
        custom_deepstack_features = kwargs.get("custom_deepstack_features", None)

        if custom_patch_features is None:
            raise ValueError(
                "custom_patch_features must be provided when using CustomQwen3VLModel3D."
            )

        # Determine whether this is the prefill forward (full sequence, no cache)
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        is_prefill = prefill_compiled_stage or prefill_noncompiled_stage or self.rope_deltas is None

        if is_prefill:
            # Concatenate per-image feature tensors into a single (N_total, hidden_dim) tensor
            if isinstance(custom_patch_features, (list, tuple)):
                custom_patch_features = torch.cat(custom_patch_features, dim=0)

            # Sanity-check feature dimension against language-model hidden size
            custom_feat_dim = custom_patch_features.shape[-1]
            model_hidden_size = inputs_embeds.shape[-1]
            if custom_feat_dim != model_hidden_size:
                raise ValueError(
                    f"Custom patch feature dim ({custom_feat_dim}) does not match model hidden size ({model_hidden_size})."
                )

            image_token_id = self.config.image_token_id
            video_token_id = self.config.video_token_id
            n_image_tokens = (input_ids == image_token_id).sum()
            n_video_tokens = (input_ids == video_token_id).sum()
            assert not (n_image_tokens > 0 and n_video_tokens > 0), (
                "Mixed image and video placeholders are not supported with custom_patch_features."
            )

            if n_video_tokens > 0:
                video_embeds = custom_patch_features.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_video_embeds = custom_deepstack_features

                logger.info(f"[prefill] inputs_embeds of shape: {inputs_embeds.shape}")
                logger.info(f"[prefill] video_embeds of shape: {video_embeds.shape}")
                if deepstack_video_embeds is not None:
                    logger.info(f"[prefill] deepstack_video_embeds (list of {len(deepstack_video_embeds)} tensors)")

                special_video_mask = input_ids == video_token_id
                logger.info(f"[prefill] n_video_tokens: {special_video_mask.sum()}")
                special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                num_el_embeds = inputs_embeds[special_video_mask].numel()
                num_el_features = video_embeds.numel()
                logger.info(f"[prefill] num_el_embeds: {num_el_embeds}")
                logger.info(f"[prefill] num_el_features: {num_el_features}")
                assert (
                    num_el_embeds == num_el_features
                ), "Video features and video tokens do not match in prefill forward"

                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            else:
                deepstack_image_embeds = custom_deepstack_features
                image_embeds = custom_patch_features.to(inputs_embeds.device, inputs_embeds.dtype)

                logger.info(f"[prefill] inputs_embeds of shape: {inputs_embeds.shape}")
                logger.info(f"[prefill] image_embeds of shape: {image_embeds.shape}")
                if deepstack_image_embeds is not None:
                    logger.info(f"[prefill] deepstack_image_embeds (list of {len(deepstack_image_embeds)} tensors)")

                special_image_mask = input_ids == image_token_id
                logger.info(f"[prefill] n_image_tokens: {special_image_mask.sum()}")
                special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                num_el_embeds = inputs_embeds[special_image_mask].numel()
                num_el_features = image_embeds.numel()
                logger.info(f"[prefill] num_el_embeds: {num_el_embeds}")
                logger.info(f"[prefill] num_el_features: {num_el_features}")
                assert (
                    num_el_embeds == num_el_features
                ), "Image features and image tokens do not match in prefill forward"

                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Build visual_pos_masks and deepstack_visual_embeds
        visual_pos_masks = None
        if image_mask is not None and video_mask is not None:
            image_mask_2d = image_mask[..., 0]
            video_mask_2d = video_mask[..., 0]
            visual_pos_masks = image_mask_2d | video_mask_2d
            if (
                deepstack_image_embeds is not None
                and deepstack_video_embeds is not None
            ):
                deepstack_visual_embeds = []
                image_mask_joint = image_mask_2d[visual_pos_masks]
                video_mask_joint = video_mask_2d[visual_pos_masks]
                for img_embed, vid_embed in zip(
                    deepstack_image_embeds, deepstack_video_embeds
                ):
                    embed_joint = img_embed.new_zeros(
                        visual_pos_masks.sum(), img_embed.shape[-1]
                    ).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask_2d = image_mask[..., 0]
            visual_pos_masks = image_mask_2d
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask_2d = video_mask[..., 0]
            visual_pos_masks = video_mask_2d
            deepstack_visual_embeds = deepstack_video_embeds

        # Position ID computation with support for gaussian_to_grid_mapping
        if position_ids is None:
            # logger.info("position_ids is None, computing position_ids")

            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask.get("full_attention")
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                zero_image_hw = kwargs.get("zero_image_hw", False)
                gaussian_to_grid_mapping = kwargs.get("gaussian_to_grid_mapping", None)
                gaussians_per_image = kwargs.get("gaussians_per_image", None)
                logger.info("Computing our custom rope indices!")
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                    zero_image_hw=zero_image_hw,
                    gaussian_to_grid_mapping=gaussian_to_grid_mapping,
                    gaussians_per_image=gaussians_per_image,
                )
                self.rope_deltas = rope_deltas
                # Store the full position_ids for decoding stage to preserve custom encodings
                self._cached_position_ids = position_ids
            else:
                # Decoding stage: preserve custom positional encodings from prefill
                # logger.info("NOT RECOMPUTING our custom rope indices!")

                batch_size, seq_length, _ = inputs_embeds.shape
                # logger.info(f"Seq length: {seq_length}")

                # Only create sequential IDs for newly generated tokens
                if hasattr(self, '_cached_position_ids') and self._cached_position_ids is not None:
                    # logger.info("Using cached position_ids")
                    cached_pos_ids = self._cached_position_ids  # (3, batch, cached_seq_len)
                    cached_seq_len = cached_pos_ids.shape[2]
                    # logger.info(f"Shape of cached position ids: {cached_pos_ids.shape}")
                    # logger.info(f"Cached seq len: {cached_seq_len}")
                    
                    # Keep cached positions for input, add sequential for new
                    # After prefill, this is the sequence length of the previous prefill tokens
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
 
                    # Create sequential positions for new tokens; those are relative starting at 0, no offset (=delta) applied yet
                    # Delta is necessary to bring it in global context of the entire sequence again, including prefill
                    new_token_count = seq_length
                    new_positions = torch.arange(new_token_count, device=inputs_embeds.device)
                    new_positions = new_positions.view(1, -1).expand(batch_size, -1) # should be (batch, new_token_count)
                    if cache_position is not None:
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    new_positions = new_positions.add(delta) # still (batch, new_token_count)
                    new_positions = new_positions.unsqueeze(0).expand(3, -1, -1) # now (3, batch, new_token_count)

                    # In this decoding phase, only new token position ids matter, the rest is handled already!!
                    position_ids = new_positions
                    
                    # # Only for debugging
                    # position_ids_printable = position_ids[:, 0, -200:].transpose(0, 1)
                    # logger.info(f"Position ids: {position_ids_printable}")
                else:
                    logger.info("Warning ⚠️ No cached position_ids, using default sequential behavior")
                    # Fallback: no cache, use default sequential behavior
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        else:
            logger.info("Warning ⚠️ position_ids is not None, NOT COMPUTING position_ids")

        # # Log final inputs to language model
        # logger.info("=" * 80)
        # logger.info("CALLING language_model.forward()")
        # logger.info("=" * 80)
        # logger.info(f"inputs_embeds shape: {inputs_embeds.shape}")
        # logger.info(f"position_ids shape: {position_ids.shape if position_ids is not None else None}")
        # logger.info(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        # logger.info(f"visual_pos_masks shape: {visual_pos_masks.shape if visual_pos_masks is not None else None}")
        # logger.info(f"deepstack_visual_embeds: {[e.shape for e in deepstack_visual_embeds] if deepstack_visual_embeds else None}")
        # logger.info(f"past_key_values: {past_key_values is not None}")
        # if past_key_values is not None:
        #     logger.info(f"past_key_values seq_length: {past_key_values.get_seq_length()}")
        # logger.info(f"cache_position: {cache_position}")
        
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        
        # logger.info(f"language_model output shape: {outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else 'N/A'}")

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,   # BEFORE merge
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        zero_image_hw: bool = False,
        gaussian_to_grid_mapping: Optional[torch.LongTensor] = None,  # (total_gaussians, 2) in BEFORE-merge space
        gaussians_per_image: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size

        # Just ids to match in the sequence
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids # shape should be (batch, seq_len)
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones( # (3, batch, seq_len) where 3 is for t h w
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids): # loops over the batches
                # Only keeping input_ids where attention_mask is 1
                input_ids = input_ids[attention_mask[i] == 1] # shape should be (seq_len)
                image_nums, video_nums = 0, 0

                # Finds vision start tokens, then checks the next ones to distinguish whether it's an image or a video
                # Count amount of images and videos in the input ids
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                assert gaussian_to_grid_mapping is not None
                assert gaussians_per_image is not None
                assert len(gaussians_per_image) == int(image_nums + video_nums), (
                    f"gaussians_per_image has {len(gaussians_per_image)} entries, "
                    f"but sequence has {int(image_nums + video_nums)} visual blocks."
                )

                input_tokens = input_ids.tolist() # list of len seq_len
                llm_pos_ids_list: list = []
                st = 0
                gaussian_cursor = 0
                visual_block_index = 0
                # Counters to keep track of how many images and videos are still missing
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    # Yields next index where image token appears (starting from st) or seq_len + 1 if done with images in that sequence
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                        logger.info(f"ed_image {ed_image}")
                    else:
                        ed_image = len(input_tokens) + 1

                    # Same for videos now as for images above
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    
                    # If the image appears before the video, retrieve t, h, w from the image grid
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        logger.info(f"ed {ed}")
                    # Otherwise, get same info from the video grid
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    
                    # Have to create this separately for each image /video because there is no guarantee that they all have the same format
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        # By this divison, we finally get back to the desired (LLM) grid size we set from the very beginning
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    logger.info(f"LLM grid is t: {llm_grid_t}, h: {llm_grid_h}, w: {llm_grid_w}")
                    # Text length must be the difference from beginning of image/video to start of this "image/video block"
                    text_len = ed - st
                    logger.info(f"text_len {text_len} before image")
                    # Looks back at prevously computed positions, takes max + 1 to keep moving
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # text tokens are easy -> first relative +1 offsets starting at 0, expanding to 3D grid (t, h, w are all the sam for text), then add the init st_idx
                    # again, note that st_idx is NOT the same as sequence length, for the very reason that image grids do not just lead to +1 increments
                    llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

                    # Note: below is the standard implemenation for 1:1 mapping between image tokens and grid
                    # # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    # # This whole block ensure that we put together a correct "intinerary" through the whole grid from top left to bottom right
                    # t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    # h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    # w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    # Ours: gaussian to grid mapping already contains the h, w indices, t can be set to torch.ones and concatenated
                    num_gaussians_this_block = int(gaussians_per_image[visual_block_index])
                    visual_block_index += 1
                    mapping_this_block = gaussian_to_grid_mapping[
                        gaussian_cursor : gaussian_cursor + num_gaussians_this_block
                    ]
                    gaussian_cursor += num_gaussians_this_block

                    t_index = torch.ones(
                        num_gaussians_this_block,
                        device=input_ids.device,
                        dtype=torch.long,
                    ).flatten()
                    h_index = mapping_this_block[..., 0].flatten().long()
                    w_index = mapping_this_block[..., 1].flatten().long()
                    logger.info(f"Number of gaussians: {num_gaussians_this_block}")
                    logger.info(f"max h_index rope: {h_index.max()}, max w_index rope: {w_index.max()}")

                    # Standard implementation:
                    # # Offset local image grid positions by the previous positions/previous "context": st_idx (previous max position + 1) and text len before that image or video in the current "processing block"
                    # llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    # Ours:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    
                    
                    # Standard: This keeps track of where we are in the input tokens; next search for image / video will start from there
                    # st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                    # Ours: we are not moving ahead based on grid size, but based on number of gaussians!
                    st = ed + num_gaussians_this_block
                    logger.info(f"st {st} after image")

                # Check if we are done with the input tokens; if there are some left, they must be text tokens
                if st < len(input_tokens):
                    # Keeping track of where we are in tokens (done by st) != tracking position ids (done by st_idx) because one image tokens does not just increase positions +1 but moves in a grid
                    # Therefore, this keeps track of where the next positional id is to continue on!
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    logger.info(f"remaining text len {text_len}")
                    llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

                # llm_pos_ids_list is a list of (3, seq_len) tensors, cat them together to get (3, total_seq_len)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                # Set the position ids for thw (...) for this batch (i) where attention is 1
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
                # Essentially gives an offset to what would follow from previous input ids sequence length + 1 (position ids may be shorter) -> that difference is stored in there
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            # # Debug stuff
            # # Make sure logger outputs the full tensor and does not truncate
            # torch.set_printoptions(threshold=10000)
            # # Position ids is (3, batch, seq_len); take first batch, always get the t w h pairs and iterate through them over the whole sequence
            # printable_position_ids = position_ids[:, 0, -200:].transpose(0, 1)
            # # Attention mask is (batch, seq_len)
            # printable_attention_mask = attention_mask[0, -200:]
            # logger.info(f"Position ids: {printable_position_ids}")
            # logger.info(f"Attention mask: {printable_attention_mask}")
            # # Check out the rope deltas
            # print(f"Mrope position deltas: {mrope_position_deltas}")

            return position_ids, mrope_position_deltas
        # TODO: believe we should never get to this case? always have input ids and a grid
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas 
       



class CustomQwen3VLForConditionalGeneration3D(Qwen3VLForConditionalGeneration):
    """Qwen3VLForConditionalGeneration with support for 3D Gaussian-to-grid positional encoding.

    Extends base Qwen3VLForConditionalGeneration to use CustomQwen3VLModel3D which supports
    many-to-one Gaussian-to-grid mapping for 3D spatial queries.
    """

    def __init__(self, config):
        super().__init__(config)
        # Swap inner model to use our 3D custom version
        self.model.__class__ = CustomQwen3VLModel3D

    def prepare_inputs_for_generation(
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
        custom_patch_features=None,
        custom_deepstack_features=None,
        zero_image_hw=None,
        gaussian_to_grid_mapping=None,
        gaussians_per_image=None,
        **kwargs,
    ):
        """Prepare inputs for generation, forwarding 3D-specific parameters.
        
        This is called internally by model.generate() during the generation loop.
        The gaussian_to_grid_mapping and gaussians_per_image parameters come from
        the caller of model.generate()
        """
        model_inputs = super().prepare_inputs_for_generation(
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
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        if custom_deepstack_features is not None:
            model_inputs["custom_deepstack_features"] = custom_deepstack_features
        if zero_image_hw is not None:
            model_inputs["zero_image_hw"] = zero_image_hw
        if gaussian_to_grid_mapping is not None:
            model_inputs["gaussian_to_grid_mapping"] = gaussian_to_grid_mapping
        if gaussians_per_image is not None:
            model_inputs["gaussians_per_image"] = gaussians_per_image
        return model_inputs
