import json
import inspect
import os
import re
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLProcessor

from .tools import IMAGE_PLACEHOLDER

THINKING_TOKEN_LIMIT = 8000
NEW_TOKEN_LIMIT = 10000


@dataclass
class VLLMQwen3Model:
    llm: Any
    model_path: str



def timestep_to_seconds_str(timestep: int, fps: float) -> str:
    seconds = timestep / fps
    return f'time="<{seconds:.1f} seconds>"'



def _build_model_path(size: Literal["8B", "32B"], use_fp8: bool) -> str:
    model_path = f"Qwen/Qwen3-VL-{size.upper()}-Thinking"
    if use_fp8:
        model_path = model_path + "-FP8"
    return model_path


def _prepend_env_path(var_name: str, path_value: str) -> None:
    existing = os.environ.get(var_name, "")
    os.environ[var_name] = f"{path_value}:{existing}" if existing else path_value


def _configure_runtime_build_env() -> None:
    python_include = sysconfig.get_paths()["include"]
    env_lib = str(Path(sys.prefix) / "lib")
    env_lib_stubs = str(Path(sys.prefix) / "lib" / "stubs")
    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")

    _prepend_env_path("LD_LIBRARY_PATH", torch_lib)
    _prepend_env_path("LIBRARY_PATH", env_lib_stubs)
    _prepend_env_path("LIBRARY_PATH", env_lib)
    _prepend_env_path("C_INCLUDE_PATH", python_include)
    _prepend_env_path("CPATH", python_include)


def _patch_video_metadata_for_vllm() -> None:
    from transformers import video_utils

    video_metadata_cls = video_utils.VideoMetadata
    if getattr(video_metadata_cls, "_vllm_video_metadata_compat", False):
        return

    init_signature = inspect.signature(video_metadata_cls.__init__)
    if "total_num_frames" not in init_signature.parameters:
        return

    original_init = video_metadata_cls.__init__
    valid_param_names = set(init_signature.parameters.keys())

    def patched_init(self, *args, **kwargs):
        if "num_frames" in kwargs and "total_num_frames" not in kwargs:
            kwargs["total_num_frames"] = kwargs.pop("num_frames")

        if "sample_frames" in kwargs and "total_num_frames" not in kwargs:
            sample_frames = kwargs.get("sample_frames")
            if isinstance(sample_frames, (list, tuple)):
                kwargs["total_num_frames"] = len(sample_frames)

        if "frames_indices" in kwargs and "total_num_frames" not in kwargs:
            frames_indices = kwargs.get("frames_indices")
            if isinstance(frames_indices, (list, tuple)) and len(frames_indices) > 0:
                kwargs["total_num_frames"] = int(max(frames_indices)) + 1

        if "duration" in kwargs and "fps" in kwargs and "total_num_frames" not in kwargs:
            duration = kwargs.get("duration")
            fps = kwargs.get("fps")
            if duration is not None and fps is not None:
                kwargs["total_num_frames"] = max(1, int(round(float(duration) * float(fps))))

        if "total_num_frames" not in kwargs:
            kwargs["total_num_frames"] = 1

        kwargs["total_num_frames"] = max(1, int(kwargs["total_num_frames"]))

        unexpected_keys = [k for k in kwargs.keys() if k not in valid_param_names]
        for key in unexpected_keys:
            kwargs.pop(key)

        return original_init(self, *args, **kwargs)

    video_metadata_cls.__init__ = patched_init
    video_metadata_cls._vllm_video_metadata_compat = True



def get_qwen3(
    size: Literal["8B", "32B"] = "8B",
    use_fp8: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    worker_multiproc_method: Literal["spawn", "fork"] = "spawn",
    execute_model_timeout_seconds: int = 120,
    gpu_memory_utilization: float = 0.9,
):
    model_path = _build_model_path(size=size, use_fp8=use_fp8)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = worker_multiproc_method
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(execute_model_timeout_seconds)
    _configure_runtime_build_env() # need to do this before import
    _patch_video_metadata_for_vllm()
    from vllm import LLM

    dtype_str = "auto" if use_fp8 else ("bfloat16" if torch_dtype == torch.bfloat16 else "float16")

    llm = LLM(
        model=model_path,
        dtype=dtype_str,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    processor = Qwen3VLProcessor.from_pretrained(model_path)
    model = VLLMQwen3Model(
        llm=llm,
        model_path=model_path,
    )
    return model, processor



def _set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)



def _sampling_params(
    seed: int,
    max_new_tokens: int,
) -> Any:
    from vllm import SamplingParams

    kwargs: Dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_new_tokens,
        "seed": seed,
    }
    return SamplingParams(**kwargs)



def _extract_text_from_request_output(request_output: Any) -> str:
    return request_output.outputs[0].text



def _extract_generated_token_count(request_output: Any) -> int:
    return len(request_output.outputs[0].token_ids)



def _generate_text(
    model: VLLMQwen3Model,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    multi_modal_data: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, float]:
    params = _sampling_params(
        seed=seed,
        max_new_tokens=max_new_tokens,
    )

    llm_input: Dict[str, Any] = {"prompt": prompt}
    if multi_modal_data is not None:
        llm_input["multi_modal_data"] = multi_modal_data

    start = time.time()
    outputs = model.llm.generate([llm_input], sampling_params=params)
    end = time.time()

    output = outputs[0]
    text = _extract_text_from_request_output(output)
    num_tokens = _extract_generated_token_count(output)
    return text, num_tokens, end - start



def prompt_with_image(
    image: Image.Image,
    prompt: str,
    model: VLLMQwen3Model,
    processor: Qwen3VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: int = THINKING_TOKEN_LIMIT,
    seed: int = 42,
):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    _ = max_thinking_tokens

    _set_generation_seed(seed)
    output_text, _, _ = _generate_text(
        model=model,
        prompt=text_prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        multi_modal_data={"image": image},
    )
    return output_text



def _parse_tool_calls(response: str) -> List[Dict[str, Any]]:
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
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
    return cleaned.strip()



def build_tool_response_message(
    tool_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Image.Image]]:
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
            parts = tool_response_text.split(IMAGE_PLACEHOLDER)
            n_markers = len(parts) - 1
            assert n_markers == n_tool_images, (
                f"Tool {record['tool_name']} returned {n_tool_images} images "
                f"but text contains {n_markers} IMAGE_PLACEHOLDER markers"
            )

            for i, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(tool_images):
                    content.append({"type": "image", "image": None})

            all_images.extend(tool_images)
        else:
            content.append({"type": "text", "text": tool_response_text})

    message = {"role": "user", "content": content}
    return message, all_images



def _filter_tensors_for_debug(obj: Any) -> Any:
    if isinstance(obj, (torch.Tensor, np.ndarray, np.generic)):
        return None
    elif isinstance(obj, Image.Image):
        return f"<PIL.Image size={obj.size}>"
    elif isinstance(obj, dict):
        filtered = {}
        for key, value in obj.items():
            filtered_val = _filter_tensors_for_debug(value)
            if filtered_val is not None:
                filtered[key] = filtered_val
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
    model: VLLMQwen3Model,
    processor: Qwen3VLProcessor,
    tools: Dict[str, Tuple[Callable, Dict[str, Any]]],
    max_iterations: int = 10,
    tool_call_limits: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: Optional[int] = THINKING_TOKEN_LIMIT,
) -> Dict[str, Any]:
    fn_start_time = time.time()
    _ = max_thinking_tokens
    tool_specs = [spec for _, spec in tools.values()]

    current_messages = [msg.copy() for msg in messages]
    for i, msg in enumerate(current_messages):
        if isinstance(msg.get("content"), list):
            current_messages[i]["content"] = msg["content"].copy()

    tool_call_history = []

    remaining_calls: Dict[str, Optional[int]] = {}
    if tool_call_limits is not None:
        for tool_name in tools.keys():
            if tool_name in tool_call_limits:
                limit = tool_call_limits[tool_name]
                remaining_calls[tool_name] = limit
            else:
                remaining_calls[tool_name] = None
    else:
        for tool_name in tools.keys():
            remaining_calls[tool_name] = None

    all_raw_images: List[Image.Image] = []

    total_generation_time = 0.0
    total_generated_tokens = 0

    response = ""
    for iteration in range(max_iterations):
        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] --- Iteration {iteration} ---", flush=True)

        text_prompt = processor.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tool_specs,
        )

        try:
            _set_generation_seed(seed + iteration)
            response, generated_tokens, generation_time = _generate_text(
                model=model,
                prompt=text_prompt,
                max_new_tokens=max_new_tokens,
                seed=seed + iteration,
                multi_modal_data={"image": all_raw_images} if all_raw_images else None,
            )
            total_generation_time += generation_time
            total_generated_tokens += generated_tokens
        except Exception:
            trace_output = _format_message_trace_for_debug(
                current_messages,
                tool_call_history,
                iteration,
            )
            print("\n" + trace_output + "\n", flush=True)
            raise

        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [Assistant Response]:\n{response}\n", flush=True)

        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
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

        current_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )

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
                remaining = remaining_calls.get(tool_name, None)

                if remaining is not None and remaining <= 0:
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
                    callable_fn, _ = tools[tool_name]
                    try:
                        result = callable_fn(**arguments)

                        if remaining is not None:
                            remaining_calls[tool_name] = remaining - 1
                            remaining_after = remaining - 1
                        else:
                            remaining_after = None

                        result_data = json.loads(result["text"])
                        result_data["remaining_calls"] = (
                            remaining_after if remaining_after is not None else "infinite"
                        )
                        result["text"] = json.dumps(result_data)
                    except Exception as exc:
                        result = {
                            "text": json.dumps(
                                {"error": f"Error executing tool: {str(exc)}"}
                            )
                        }
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

        tool_response_message, new_images = build_tool_response_message(tool_results)
        current_messages.append(tool_response_message)
        all_raw_images.extend(new_images)

    final_answer = _extract_final_answer(response)
    tok_per_sec = (
        total_generated_tokens / total_generation_time if total_generation_time > 0 else 0.0
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
    model: VLLMQwen3Model,
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
    assert tools is not None and len(tools) > 0, (
        "tools are required for graph agentic prompting"
    )
    assert len(node_centers) == len(node_centroids) == len(node_extents), (
        "timestep mismatch"
    )

    centroids = node_centroids[initial_timestep_idx]
    extents = node_extents[initial_timestep_idx]
    centers = node_centers[initial_timestep_idx]

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

    graph_json = json.dumps(graph_data, indent=2)
    graph_content = [{"type": "text", "text": graph_json}]

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



def _load_video_frames(image_paths: List[Any]) -> List[np.ndarray]:
    frames = []
    for image_path in image_paths:
        with Image.open(Path(image_path)) as frame_image:
            frames.append(np.array(frame_image.convert("RGB")))
    return frames



def prompt_with_video(
    question: str,
    image_paths: List[Any],
    model: VLLMQwen3Model,
    processor: Qwen3VLProcessor,
    system_prompt: str = None,
    fps: float = None,
    seed: int = 42,
    max_new_tokens: int = NEW_TOKEN_LIMIT,
    max_thinking_tokens: int = THINKING_TOKEN_LIMIT,
) -> str:
    image_paths_str = [str(p) for p in image_paths]
    _ = max_thinking_tokens

    content = []
    video_content: Dict[str, Any] = {"type": "video", "video": image_paths_str}
    if fps is not None:
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

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    video_frames = _load_video_frames(image_paths=image_paths)
    multi_modal_video: Any = video_frames
    if fps is not None:
        first_frame = video_frames[0]
        frame_height = int(first_frame.shape[0])
        frame_width = int(first_frame.shape[1])
        n_frames = len(video_frames)
        multi_modal_video = [
            (
                video_frames,
                {
                    "fps": fps,
                    "total_num_frames": n_frames,
                    "frames_indices": list(range(n_frames)),
                    "width": frame_width,
                    "height": frame_height,
                    "duration": float(n_frames) / float(fps),
                },
            )
        ]

    _set_generation_seed(seed)
    output_text, _, _ = _generate_text(
        model=model,
        prompt=text_prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        multi_modal_data={"video": multi_modal_video},
    )
    return output_text
