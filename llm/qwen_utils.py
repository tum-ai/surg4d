import torch
import json
import re
import time
import numpy as np
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.generation import LogitsProcessorList
from transformers.video_utils import VideoMetadata
from typing import Dict, List, Literal, Optional, Union, Any, Callable, Tuple
from qwen_vl_utils import process_vision_info

from .thinking_budget_processor import ThinkingTokenBudgetProcessor
from .tools import IMAGE_PLACEHOLDER

THINKING_TOKEN_LIMIT = 8000
NEW_TOKEN_LIMIT = 10000


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



def get_qwen3(
    size: Literal["8B", "32B"] = "8B",
    use_fp8: bool = False,
    attn_implementation: str = "sdpa",  # "flash_attention_2" or "sdpa"
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    repetition_penalty: float = None,
    compile: bool = False,
):
    """Get a Qwen3VL model/processor.

    Parameters allow enabling weight quantization and optimized attention.
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

    model = Qwen3VLForConditionalGeneration.from_pretrained(
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



def _set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_with_image(
    image: Image.Image,
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
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
) -> Tuple[Dict[str, Any], List[Image.Image]]:
    """Build a user message containing tool responses with interleaved text and images.

    Takes a list of tool result records and constructs a message in the format expected
    by Qwen's chat template. Each tool result is wrapped in <tool_response> tags.
    Tool images are inserted at positions marked by IMAGE_PLACEHOLDER in the text.

    Args:
        tool_results: List of tool result records, each containing:
            - "tool_name" (str): Name of the tool that was called
            - "arguments" (dict): Arguments passed to the tool
            - "result" (dict): Tool return value with:
                - "text" (str): Text content, may contain IMAGE_PLACEHOLDER markers
                                - "images" (List[PIL.Image], optional): Raw images to insert at
                                    IMAGE_PLACEHOLDER positions.

        Returns:
                                Tuple of (message, images) where:
            - message: Dict with "role": "user" and "content" list containing interleaved
              {"type": "text", "text": ...} and {"type": "image", "image": None} entries
                                                - images: List of raw PIL images from all tools, in the order they
                                                        appear in the message
    """
    content = []
    all_images = []

    for record in tool_results:
        result = record["result"]
        text_content = result.get("text", "")
        tool_response_text = (
            f"<tool_response>\n"
            f'{{"name": "{record["tool_name"]}", "content": {json.dumps(text_content)}}}\n'
            f"</tool_response>"
        )

        tool_images = result.get("images", [])
        if not isinstance(tool_images, list):
            tool_images = [tool_images]

        if tool_images:
            n_tool_images = len(tool_images)

            # Split text by IMAGE_PLACEHOLDER and interleave with image placeholders
            parts = tool_response_text.split(IMAGE_PLACEHOLDER)
            n_markers = len(parts) - 1
            assert n_markers == n_tool_images, (
                f"Tool {record['tool_name']} returned {n_tool_images} images "
                f"but text contains {n_markers} IMAGE_PLACEHOLDER markers"
            )

            # Build interleaved content: text, image, text, image, ..., text
            # Empty text parts (from markers at start/end) are skipped
            for i, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(tool_images):
                    content.append({"type": "image", "image": None})

            all_images.extend(tool_images)
        else:
            # No images - just add the text
            content.append({"type": "text", "text": tool_response_text})

    message = {"role": "user", "content": content}
    return message, all_images


def _filter_tensors_for_debug(obj: Any) -> Any:
    """Recursively filter out tensors and numpy arrays from objects for debugging."""
    if isinstance(obj, (torch.Tensor, np.ndarray, np.generic)):
        return None  # Skip tensors/arrays
    elif isinstance(obj, Image.Image):
        return f"<PIL.Image size={obj.size}>"
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
                    lines.append(f"  Content[{j}]: image")
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


def generate_agentic(
    messages: List[Dict[str, Any]],
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    max_iterations: int = 10,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
) -> Dict[str, Any]:
    """Generate in an agentic loop, executing tools until done.

    Args:
        messages: Chat messages in Qwen chat-template format
        model: The Qwen model
        processor: The Qwen processor
        tools: Dict mapping tool_name -> (callable, json_spec)
               json_spec should be in OpenAI function calling format:
               {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
               Tools must return a dict with:
                   - "text" (str): Text content of the result (required). Use IMAGE_PLACEHOLDER
                                         markers to indicate where images should be inserted.
                                     - "images" (List[PIL.Image], optional): Images for IMAGE_PLACEHOLDER markers.
        max_iterations: Maximum number of tool-calling iterations
        tool_call_limits: Optional dict mapping tool_name -> max_calls (int or None for infinite).
            If None, all tools have infinite calls. If a tool is not in the dict, it defaults to infinite.
        verbose: If True, prints message and tool results at each iteration.
        seed: Random seed for deterministic sampling
        max_new_tokens: Maximum number of new tokens to generate
        max_thinking_tokens: Maximum tokens for thinking phase per iteration (Qwen3 only).
            If None, no limit is applied. If 0, thinking is disabled immediately.
            A new processor is created each iteration since it has internal state.
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

    # Track raw images added by tools
    all_raw_images: List[Image.Image] = []

    # Track generation timing and tokens for tok/s calculation
    total_generation_time = 0.0
    total_generated_tokens = 0

    for iteration in range(max_iterations):
        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] --- Iteration {iteration} ---", flush=True)

        # Generate response with tools
        text = processor.apply_chat_template(  # type: ignore
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tool_specs,
        )
        inputs = processor(
            text=text,
            images=all_raw_images if all_raw_images else None,
            padding=True,
            return_tensors="pt",
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

        # Build tool response message and collect images
        tool_response_message, new_images = build_tool_response_message(tool_results)
        current_messages.append(tool_response_message)
        all_raw_images.extend(new_images)

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


def prompt_graph_agent_with_semantic_labels(
    question: str,
    initial_timestep_idx: int,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    node_semantic_labels: dict[int, str],
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    system_prompt: str = None,
    max_iterations: int = 20,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
    max_new_tokens: int = 8192,
    max_thinking_tokens: Optional[int] = None,
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
    """
    assert tools is not None and len(tools) > 0, (
        "tools are required for graph agentic prompting"
    )
    assert len(node_centers) == len(node_centroids) == len(node_extents), (
        "timestep mismatch"
    )

    # node feat indices correspond to cluster ids
    centroids = node_centroids[initial_timestep_idx]
    extents = node_extents[initial_timestep_idx]
    centers = node_centers[initial_timestep_idx]

    # Build JSON structure (no image placeholders - semantic labels only)
    nodes_data = []
    for n in range(centroids.shape[0]):
        nodes_data.append(
            {
                "node_id": int(n),
                "semantic_label": node_semantic_labels[str(n)],
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

    # Serialize to JSON (no image placeholders in this version)
    graph_json = json.dumps(graph_data, indent=2)
    graph_content = [{"type": "text", "text": graph_json}]

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

    return generate_agentic(
        messages=messages,
        model=model,
        processor=processor,
        tools=tools,
        max_iterations=max_iterations,
        tool_call_limits=tool_call_limits,
        verbose=verbose,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_thinking_tokens=max_thinking_tokens,
    )


def prompt_with_video(
    question: str,
    image_paths: List[Any],
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
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