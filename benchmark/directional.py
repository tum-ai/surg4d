import gc
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig

from benchmark.graph_utils import get_coord_transformations
from benchmark.serialization_utils import parse_json, sanitize_tool_calls
from llm.qwen_utils import (
    prompt_graph_agent_with_semantic_labels,
    prompt_with_video,
)
from llm.tools import GraphTools


def _parse_axis_class(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if numeric_value not in (-1.0, 0.0, 1.0):
        return None
    return int(numeric_value)


def multiframe_directional_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path, # mock
    annotations: List[Dict[str, Any]],
    clip: DictConfig, # mock
    cfg: DictConfig,
    use_semantic_labels: bool = False, # mock
    semantic_method_name: str = "", # mock
) -> List[Dict[str, Any]]:
    sampled_frames = video_frames[::cfg.eval.annotation_stride]
    effective_fps = cfg.eval.video_fps / cfg.eval.annotation_stride

    system_prompt = cfg.eval.directional.multiframe_system_prompt
    prompt_template = cfg.eval.directional.multiframe_prompt_template

    results = []
    for query_anno in annotations:
        query_id = query_anno["id"]
        method_name = "multiframe"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Running [{query_id}] with method [{method_name}]")
        query = query_anno["query"]
        temporal_range = query_anno["range"]

        selected_frames = sampled_frames[temporal_range[0] : temporal_range[1] + 1]

        prompt = prompt_template.format(
            question=query,
        )

        response = prompt_with_video(
            question=prompt,
            image_paths=selected_frames,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            fps=effective_fps,
        )

        json_data = parse_json(response)
        if json_data is None or "x" not in json_data or "y" not in json_data or "z" not in json_data:
            prediction = None
        else:
            x_class = _parse_axis_class(json_data["x"])
            y_class = _parse_axis_class(json_data["y"])
            z_class = _parse_axis_class(json_data["z"])
            if x_class is None or y_class is None or z_class is None:
                prediction = None
            else:
                prediction = {
                    "x": x_class,
                    "y": y_class,
                    "z": z_class,
                }

        results.append(
            {
                "id": query_id,
                "query": query,
                "range": temporal_range,
                "predicted": prediction,
                "raw_response": response,
                "message_history": [],
                "tool_calls": [],
            }
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def graph_agent_directional_queries(
    model,
    processor,
    graph_path: Path,
    annotations: List[Dict[str, Any]],
    clip: DictConfig,
    cfg: DictConfig,
    video_frames: List[Path],
    use_semantic_labels: bool = True,
    semantic_method_name: str = "graph_agent_semantics",
) -> List[Dict[str, Any]]:
    """Run directional graph-agent queries with tools.

    Mirrors temporal/spatial graph-agent benchmark style for directional labels.
    """
    centers_path = graph_path / "c_centers.npy"
    centroids_path = graph_path / "c_centroids.npy"
    extents_path = graph_path / "c_extents.npy"
    positions_path = graph_path / "positions.npy"
    clusters_path = graph_path / "clusters.npy"
    adjacency_path = graph_path / "graph.npy"
    bhattacharyya_path = graph_path / "bhattacharyya_coeffs.npy"

    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)

    if use_semantic_labels:
        semantic_labels_path = graph_path / "cluster_semantics.json"
        with open(semantic_labels_path, "r") as f:
            node_semantic_labels = json.load(f)

    point_o2n, _, distance_o2n, _ = get_coord_transformations(positions)

    if semantic_method_name == "graph_agent_semantics_vision":
        max_iterations = cfg.eval.directional.graph_agent_semantics_vision_max_iterations
        tool_config = cfg.eval.directional.graph_agent_semantics_vision_tools
        system_prompt = cfg.eval.directional.graph_agent_semantics_vision_system_prompt
        prompt_template = cfg.eval.directional.graph_agent_semantics_vision_prompt_template
    elif semantic_method_name == "graph_agent_semantics":
        max_iterations = cfg.eval.directional.graph_agent_semantics_max_iterations
        tool_config = cfg.eval.directional.graph_agent_semantics_tools
        system_prompt = cfg.eval.directional.graph_agent_semantics_system_prompt
        prompt_template = cfg.eval.directional.graph_agent_semantics_prompt_template

    graph_tools = GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        video_frames=video_frames,
        annotation_stride=cfg.eval.annotation_stride,
    )

    tool_viz_enabled = cfg.eval.directional.tool_viz_dir is not None
    tool_viz_dir = None
    if tool_viz_enabled:
        tool_viz_dir = Path(cfg.eval.directional.tool_viz_dir) / clip.name
        tool_viz_dir.mkdir(parents=True, exist_ok=True)

    tool_names = []
    tool_call_limits = {}
    for tool_entry in tool_config:
        tool_name = tool_entry.name
        tool_names.append(tool_name)
        max_calls = tool_entry.max_calls
        if max_calls is not None:
            tool_call_limits[tool_name] = max_calls

    tools = graph_tools.get_tools_by_name(tool_names)
    if len(tool_call_limits) == 0:
        tool_call_limits = None

    results = []
    for query_anno in annotations:
        query_id = query_anno["id"]
        method_name = semantic_method_name if use_semantic_labels else "graph_agent"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Running [{query_id}] with method [{method_name}]")
        query = query_anno["query"]
        temporal_range = query_anno["range"]

        if tool_viz_enabled:
            sanitized_question = re.sub(r"[^\w\s-]", "", query)
            sanitized_question = re.sub(r"\s+", "_", sanitized_question)
            sanitized_question = sanitized_question[:50]
            rrd_file = tool_viz_dir / f"{method_name}_{query_id}_{sanitized_question}.rrd"
            graph_tools.start_recording(str(rrd_file))

        num_ts = graph_tools.adjacency.shape[0]
        prompt = prompt_template.format(
            question=query,
            range_start=temporal_range[0],
            range_end=temporal_range[1],
            num_frames=num_ts,
            last_frame=num_ts - 1,
        )

        initial_timestep_idx = temporal_range[0]
        agent_result = prompt_graph_agent_with_semantic_labels(
            question=prompt,
            initial_timestep_idx=initial_timestep_idx,
            node_centers=point_o2n(node_centers),
            node_centroids=point_o2n(node_centroids),
            node_extents=distance_o2n(node_extents),
            node_semantic_labels=node_semantic_labels,
            model=model,
            processor=processor,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            tool_call_limits=tool_call_limits,
        )

        if tool_viz_enabled:
            graph_tools.stop_recording()

        json_data = parse_json(agent_result["final_answer"])
        if json_data is None or "x" not in json_data or "y" not in json_data or "z" not in json_data:
            prediction = None
        else:
            x_class = _parse_axis_class(json_data["x"])
            y_class = _parse_axis_class(json_data["y"])
            z_class = _parse_axis_class(json_data["z"])
            if x_class is None or y_class is None or z_class is None:
                prediction = None
            else:
                prediction = {
                    "x": x_class,
                    "y": y_class,
                    "z": z_class,
                }

        results.append(
            {
                "id": query_id,
                "query": query,
                "range": temporal_range,
                "predicted": prediction,
                "raw_response": agent_result["final_answer"],
                "message_history": agent_result["message_history"],
                "tool_calls": sanitize_tool_calls(agent_result.get("tool_calls", [])),
            }
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
