"""
Temporal action localization evaluation for surgical videos.

Supports temporal query types:
- Action Onset: When does X start?
- Action Offset: When does X end?
- Action Duration: During which timesteps does X happen?
- Multiple Event Ordering: Which happens first?
- Count/Frequency: How many times does X happen?
"""

import gc
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import torch
from omegaconf import DictConfig
from llm.qwen_utils import (
    prompt_with_dynamic_graph,
    prompt_with_dynamic_descriptors,
    prompt_with_video_frames,
    prompt_graph_agent,
)
from llm.tools import GraphTools
from autoencoder.model_qwen import QwenAutoencoder
from benchmark.serialization_utils import sanitize_tool_calls


def load_video_frames(video_dir: Path, images_subdir: str) -> "Tuple[List[Path], int]":
    """Load video frames from directory.
    
    Args:
        video_dir: Root directory containing video data
        images_subdir: Subdirectory containing frame images
        
    Returns:
        Tuple of (sorted frame paths, num_frames)
    """
    images_dir = video_dir / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    frames = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    num_frames = len(frames)
    
    if num_frames == 0:
        raise FileNotFoundError(f"No .jpg or .png images found in {images_dir}")
    
    return frames, num_frames


def seconds_to_timestep(seconds: float, num_timesteps: int, fps: float) -> int:
    """Convert seconds to nearest timestep using nearest neighbor rounding.
    
    Args:
        seconds: Time in seconds
        num_timesteps: Total number of timesteps (frames)
        fps: Frames per second
        
    Returns:
        Nearest timestep (integer frame index)
    """
    # Convert seconds to frame index and round to nearest integer
    frame_idx = int(round(seconds * fps))
    # Clamp to valid range
    return max(0, min(frame_idx, num_timesteps - 1))


def load_graph_data(graph_path: Path) -> Dict[str, Any]:
    """Load precomputed 4D graph data.
    
    Args:
        graph_path: Path to graph directory
        
    Returns:
        Dict with graph data
    """
    qwen_feat_file = graph_path / "c_qwen_feats.npz"
    qwen_feats_dict = np.load(qwen_feat_file)
    adjacency_matrices = np.load(graph_path / "graph.npy")
    centers = np.load(graph_path / "c_centers.npy")
    centroids = np.load(graph_path / "c_centroids.npy")
    extents = np.load(graph_path / "c_extents.npy")
    
    return {
        'node_feats': qwen_feats_dict,
        'adjacency_matrices': adjacency_matrices,
        'node_centers': centers,
        'node_centroids': centroids,
        'node_extents': extents
    }


def get_num_timesteps_from_graph(graph_path: Path) -> int:
    """Get number of timesteps from graph data.
    
    Args:
        graph_path: Path to graph directory
        
    Returns:
        Number of timesteps
    """
    graph_data = load_graph_data(graph_path)
    return int(graph_data['adjacency_matrices'].shape[0])


def build_prompt(
    template: str,
    question: str,
    answer_format: str,
    num_frames: int,
    last_frame: int
) -> str:
    """Fill in prompt template with query-specific info.
    
    Args:
        template: Prompt template string with placeholders
        question: The question to answer
        answer_format: Expected answer format
        num_frames: Number of frames/timesteps
        last_frame: Last frame index (num_frames - 1)
        
    Returns:
        Formatted prompt string
    """
    return template.format(
        num_frames=num_frames,
        last_frame=last_frame,
        question=question,
        answer_format=answer_format,
    )




def query_with_graph(
    graph_path: Path,
    prompt: str,
    model,
    processor,
    qwen_version: str,
    system_prompt: str,
    fps: float = None,
) -> str:
    """Query model with scene graph through time.
    
    Args:
        graph_path: Path to graph directory
        prompt: Text prompt
        model: Qwen VL model
        processor: Qwen VL processor
        qwen_version: Either "qwen25" or "qwen3"
        system_prompt: System prompt
        fps: Optional frames per second for seconds-based timestamps
        
    Returns:
        Model response text
    """
    graph_data = load_graph_data(graph_path)
    
    response = prompt_with_dynamic_graph(
        question=prompt,
        node_feats=graph_data['node_feats'],
        adjacency_matrices=graph_data['adjacency_matrices'],
        node_centers=graph_data['node_centers'],
        node_centroids=graph_data['node_centroids'],
        node_extents=graph_data['node_extents'],
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        system_prompt=system_prompt,
        fps=fps,
    )
    
    return response


def query_with_descriptors(
    graph_path: Path,
    prompt: str,
    model,
    processor,
    qwen_version: str,
    system_prompt: str,
    fps: float = None,
) -> str:
    """Query model with descriptor features only (no graph structure).
    
    Args:
        graph_path: Path to graph directory
        prompt: Text prompt
        model: Qwen VL model
        processor: Qwen VL processor
        qwen_version: Either "qwen25" or "qwen3"
        system_prompt: System prompt
        fps: Optional frames per second for seconds-based timestamps
        
    Returns:
        Model response text
    """
    graph_data = load_graph_data(graph_path)

    response = prompt_with_dynamic_descriptors(
        question=prompt,
        node_feats=graph_data['node_feats'],
        adjacency_matrices=graph_data['adjacency_matrices'],
        node_centers=graph_data['node_centers'],
        node_centroids=graph_data['node_centroids'],
        node_extents=graph_data['node_extents'],
        model=model,
        processor=processor,
        qwen_version=qwen_version,
        system_prompt=system_prompt,
        fps=fps,
    )

    return response


def parse_single_frame(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for action_onset or action_offset.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'timestep' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'timestep' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner

            # Support both "timestep" (old format) and "second" (new Qwen3 format)
            if 'timestep' in data:
                timestep_val = data['timestep']
                if isinstance(timestep_val, (int, float)):
                    return {'timestep': int(timestep_val)}
            elif 'second' in data:
                second_val = data['second']
                if isinstance(second_val, (int, float)):
                    return {'second': float(second_val)}
        return None
    except Exception:
        return None


def parse_frame_ranges(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for action_duration.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'ranges' or 'second_ranges' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'ranges' not in data and 'second_ranges' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            # Support both "ranges" (old format) and "second_ranges" (new Qwen3 format)
            if 'ranges' in data:
                ranges = []
                for r in data['ranges']:
                    if isinstance(r, list) and len(r) == 2:
                        ranges.append([int(r[0]), int(r[1])])
                if ranges:
                    return {'ranges': ranges}
            elif 'second_ranges' in data:
                second_ranges = []
                for r in data['second_ranges']:
                    if isinstance(r, list) and len(r) == 2:
                        second_ranges.append([float(r[0]), float(r[1])])
                if second_ranges:
                    return {'second_ranges': second_ranges}
        return None
    except Exception:
        return None


def parse_ordered_events(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for multiple_event_ordering.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'events' key or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'events' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            if 'events' in data and isinstance(data['events'], list):
                events = []
                for event in data['events']:
                    if isinstance(event, dict) and 'order' in event and 'frame_range' in event:
                        events.append({
                            'order': int(event['order']),
                            'description': event.get('description', ''),
                            'frame_range': [int(event['frame_range'][0]), int(event['frame_range'][1])]
                        })
                if events:
                    events.sort(key=lambda x: x['order'])
                    return {'events': events}
        return None
    except Exception:
        return None


def parse_count_and_ranges(response: str, query_type: str) -> Optional[Dict]:
    """Parse response for count_frequency.
    
    Args:
        response: Model response text
        query_type: Type of query (for unwrapping nested JSON)
        
    Returns:
        Dict with 'count' and 'occurrences' keys or None
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if 'count' not in data and isinstance(data, dict):
                inner = data.get(query_type)
                if isinstance(inner, dict):
                    data = inner
                    
            if 'count' in data:
                occurrences = []
                if 'occurrences' in data and isinstance(data['occurrences'], list):
                    for occ in data['occurrences']:
                        if isinstance(occ, list) and len(occ) >= 2:
                            occurrences.append([int(occ[0]), int(occ[1])])
                return {
                    'count': int(data['count']),
                    'occurrences': occurrences
                }
        return None
    except Exception:
        return None


def multiframe_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
) -> List[Dict]:
    """Run multiframe (video-only) temporal queries.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        video_frames: List of video frame paths (full set)
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    # Load graph to determine stride for frame sampling
    num_ts = get_num_timesteps_from_graph(graph_path)
    stride = max(1, round(len(video_frames) / num_ts))
    selected_frames = video_frames[::stride]
    
    # Calculate effective FPS based on stride
    # Original video is at video_fps, but we sample every stride frames
    # For 20 graph timesteps from 25 fps video: effective_fps = 25 / 4 = 6.25
    effective_fps = cfg.eval.video_fps / stride
    
    parser_map = {
        'action_onset': parse_single_frame,
        'action_offset': parse_single_frame,
        'action_duration': parse_frame_ranges,
        'multiple_event_ordering': parse_ordered_events,
        'count_frequency': parse_count_and_ranges
    }
    
    results = []
    for query_anno in annotations:
        query_type = query_anno['query_type']
        
        # Get system prompt and template using new flat structure
        system_prompt_key = f"multiframe_{query_type}_system_prompt"
        template_key = f"multiframe_{query_type}_prompt_template"
        
        system_prompt = cfg.eval.temporal[system_prompt_key]
        template = cfg.eval.temporal[template_key]
        
        # Build prompt
        prompt = build_prompt(
            template=template,
            question=query_anno['question'],
            answer_format=query_anno['answer_format'],
            num_frames=len(selected_frames),
            last_frame=len(selected_frames) - 1,
        )
        
        # Query model
        response = prompt_with_video_frames(
            question=prompt,
            image_paths=selected_frames,
            model=model,
            processor=processor,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            max_tokens=2048,
            fps=effective_fps,
        )
        
        # Parse response
        predicted = parser_map[query_type](response, query_type)
        
        # Convert seconds to timesteps if using Qwen3 format
        if predicted and 'second' in predicted:
            # Convert single second to timestep
            predicted['timestep'] = seconds_to_timestep(predicted['second'], num_ts, effective_fps)
        elif predicted and 'second_ranges' in predicted:
            # Convert second ranges to timestep ranges
            timestep_ranges = []
            for start_sec, end_sec in predicted['second_ranges']:
                start_timestep = seconds_to_timestep(start_sec, num_ts, effective_fps)
                end_timestep = seconds_to_timestep(end_sec, num_ts, effective_fps)
                timestep_ranges.append([start_timestep, end_timestep])
            predicted['ranges'] = timestep_ranges
        
        # Build message history for consistency with graph_agent
        # (multiframe doesn't support tools, so this is just the initial query and response)
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'query_id': query_anno['query_id'],
            'query_type': query_type,
            'question': query_anno['question'],
            'predicted': predicted,
            'raw_response': response,
            'message_history': message_history,
        })
        
        # Clear memory after each query to prevent OOM with video inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def multiframe_graph_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
) -> List[Dict]:
    """Run multiframe+graph temporal queries.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        video_frames: List of video frame paths (not used - graph provides visual info)
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    # Get number of timesteps from graph
    num_ts = get_num_timesteps_from_graph(graph_path)
    
    # Calculate effective FPS based on stride (same logic as multiframe_queries)
    stride = max(1, round(len(video_frames) / num_ts))
    effective_fps = cfg.eval.video_fps / stride
    
    parser_map = {
        'action_onset': parse_single_frame,
        'action_offset': parse_single_frame,
        'action_duration': parse_frame_ranges,
        'multiple_event_ordering': parse_ordered_events,
        'count_frequency': parse_count_and_ranges
    }
    
    results = []
    for query_anno in annotations:
        query_type = query_anno['query_type']
        
        # Get system prompt and template using new flat structure
        system_prompt_key = f"multiframe_graph_{query_type}_system_prompt"
        template_key = f"multiframe_graph_{query_type}_prompt_template"
        
        system_prompt = cfg.eval.temporal[system_prompt_key]
        template = cfg.eval.temporal[template_key]
        
        # Build prompt
        prompt = build_prompt(
            template=template,
            question=query_anno['question'],
            answer_format=query_anno['answer_format'],
            num_frames=num_ts,
            last_frame=num_ts - 1,
        )
        
        # Query model with graph
        response = query_with_graph(
            graph_path=graph_path,
            prompt=prompt,
            model=model,
            processor=processor,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            fps=effective_fps,
        )
        
        # Parse response
        predicted = parser_map[query_type](response, query_type)
        
        # Build message history for consistency
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'query_id': query_anno['query_id'],
            'query_type': query_type,
            'question': query_anno['question'],
            'predicted': predicted,
            'raw_response': response,
            'message_history': message_history,
        })
        
        # Clear memory after each query to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def multiframe_descriptors_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
) -> List[Dict]:
    """Run multiframe+descriptors temporal queries.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        video_frames: List of video frame paths (not used - descriptors provide visual info)
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    # Get number of timesteps from graph
    num_ts = get_num_timesteps_from_graph(graph_path)
    
    # Calculate effective FPS based on stride (same logic as multiframe_queries)
    stride = max(1, round(len(video_frames) / num_ts))
    effective_fps = cfg.eval.video_fps / stride
    
    parser_map = {
        'action_onset': parse_single_frame,
        'action_offset': parse_single_frame,
        'action_duration': parse_frame_ranges,
        'multiple_event_ordering': parse_ordered_events,
        'count_frequency': parse_count_and_ranges
    }
    
    results = []
    for query_anno in annotations:
        query_type = query_anno['query_type']
        
        # Get system prompt and template using new flat structure
        system_prompt_key = f"multiframe_descriptors_{query_type}_system_prompt"
        template_key = f"multiframe_descriptors_{query_type}_prompt_template"
        
        system_prompt = cfg.eval.temporal[system_prompt_key]
        template = cfg.eval.temporal[template_key]
        
        # Build prompt
        prompt = build_prompt(
            template=template,
            question=query_anno['question'],
            answer_format=query_anno['answer_format'],
            num_frames=num_ts,
            last_frame=num_ts - 1,
        )
        
        # Query model with descriptors
        response = query_with_descriptors(
            graph_path=graph_path,
            prompt=prompt,
            model=model,
            processor=processor,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            fps=effective_fps,
        )
        
        # Parse response
        predicted = parser_map[query_type](response, query_type)
        
        # Build message history for consistency
        message_history = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        results.append({
            'query_id': query_anno['query_id'],
            'query_type': query_type,
            'question': query_anno['question'],
            'predicted': predicted,
            'raw_response': response,
            'message_history': message_history,
        })
        
        # Clear memory after each query to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def graph_agent_queries(
    model,
    processor,
    video_frames: List[Path],
    graph_path: Path,
    annotations: List[Dict],
    clip: "DictConfig",
    cfg: "DictConfig",
) -> List[Dict]:
    """Run graph agent temporal queries with tools.
    
    Uses prompt_graph_agent with agentic tool use (requires qwen3).
    Provides initial rough nodes at timestep 0, then agent uses temporal tools to explore.
    Loads graph artifacts and creates GraphTools instance with temporal reasoning tools.
    
    Args:
        model: Qwen VL model (must be qwen3)
        processor: Qwen VL processor
        video_frames: List of video frame paths (not used - graph provides info)
        graph_path: Path to graph directory
        annotations: List of query annotations
        clip: Clip config
        cfg: Full hydra config
        
    Returns:
        List of result dicts
    """
    assert cfg.eval.qwen_version == "qwen3", "graph_agent requires qwen3 for agentic tool use"
    
    # Load required graph artifacts
    node_feats_npz_path = graph_path / "c_qwen_feats.npz"
    centers_path = graph_path / "c_centers.npy"
    centroids_path = graph_path / "c_centroids.npy"
    extents_path = graph_path / "c_extents.npy"
    positions_path = graph_path / "positions.npy"
    clusters_path = graph_path / "clusters.npy"
    patch_latents_path = graph_path / "patch_latents_through_time.npy"
    adjacency_path = graph_path / "graph.npy"
    bhattacharyya_path = graph_path / "bhattacharyya_coeffs.npy"
    
    node_feats_npz = np.load(node_feats_npz_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    patch_latents_through_time = np.load(patch_latents_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)
    
    # Get number of timesteps
    num_ts = adjacency.shape[0]
    
    # Calculate effective FPS based on stride (same logic as multiframe_queries)
    # The graph has num_ts timesteps sampled from len(video_frames) video frames
    stride = max(1, round(len(video_frames) / num_ts))
    effective_fps = cfg.eval.video_fps / stride
    
    # Load autoencoder for highres inspection tools (following spatial pattern)
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    autoencoder_path = clip_dir / cfg.eval.temporal.graph_agent_autoencoder_checkpoint_subdir / "best_ckpt.pth"
    
    autoencoder = QwenAutoencoder(
        input_dim=cfg.eval.temporal.graph_agent_autoencoder_full_dim,
        latent_dim=cfg.eval.temporal.graph_agent_autoencoder_latent_dim,
    ).to(model.device)
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, map_location=model.device)
    )
    autoencoder.eval()
    
    # Create GraphTools instance for tool management with fps for seconds-based timestamps
    graph_tools = GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        qwen_feats=node_feats_npz,
        patch_latents_through_time=patch_latents_through_time,
        autoencoder=autoencoder,
    )
    
    # Setup tool visualization directory if configured
    tool_viz_enabled = cfg.eval.temporal.tool_viz_dir is not None
    tool_viz_dir = None
    if tool_viz_enabled:
        tool_viz_dir = Path(cfg.eval.temporal.tool_viz_dir) / clip.name
        tool_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse graph_agent_tools config (list of objects with name and max_calls)
    tool_names = []
    tool_call_limits = {}
    tool_config = cfg.eval.temporal.graph_agent_tools
    for tool_entry in tool_config:
        tool_name = tool_entry['name']
        tool_names.append(tool_name)
        max_calls = tool_entry.get('max_calls')
        if max_calls is not None:
            tool_call_limits[tool_name] = max_calls
    
    # Get the specific tools needed for graph agent
    tools = graph_tools.get_tools_by_name(tool_names)
    
    # Convert to None if no limits specified
    if len(tool_call_limits) == 0:
        tool_call_limits = None
    
    # Parser map for response formats
    parser_map = {
        'action_onset': parse_single_frame,
        'action_offset': parse_single_frame,
        'action_duration': parse_frame_ranges,
        'multiple_event_ordering': parse_ordered_events,
        'count_frequency': parse_count_and_ranges
    }
    
    # Process each query
    results = []
    for query_anno in annotations:
        query_type = query_anno['query_type']
        query_id = query_anno['query_id']
        question_text = query_anno['question']
        
        # Start recording if tool visualization is enabled
        if tool_viz_enabled:
            # Sanitize question text for filename
            sanitized_question = re.sub(r'[^\w\s-]', '', question_text)  # Remove special chars
            sanitized_question = re.sub(r'\s+', '_', sanitized_question)  # Replace whitespace with _
            sanitized_question = sanitized_question[:50]  # Limit length
            rrd_file = tool_viz_dir / f"{query_id}_{sanitized_question}.rrd"
            graph_tools.start_recording(str(rrd_file))
        
        system_prompt_key = f"graph_agent_{query_type}_system_prompt"
        template_key = f"graph_agent_{query_type}_prompt_template"
        
        system_prompt = cfg.eval.temporal[system_prompt_key]
        template = cfg.eval.temporal[template_key]
        
        # Build prompt
        prompt = build_prompt(
            template=template,
            question=query_anno['question'],
            answer_format=query_anno['answer_format'],
            num_frames=num_ts,
            last_frame=num_ts - 1,
        )
        
        # Query model with graph and tools
        # Use prompt_graph_agent which gives initial rough nodes at timestep 0
        # The agent then uses temporal tools to explore across time
        agent_result = prompt_graph_agent(
            question=prompt,
            node_feats=node_feats_npz,
            initial_timestep_idx=0,  # Start at first timestep, agent explores temporally
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            tools=tools,
            qwen_version=cfg.eval.qwen_version,
            system_prompt=system_prompt,
            max_iterations=cfg.eval.temporal.graph_agent_max_iterations,
            tool_call_limits=tool_call_limits,
        )
        
        # Stop recording if tool visualization is enabled
        if tool_viz_enabled:
            graph_tools.stop_recording()
        
        # Extract response (agent_result is a dict when tools are used)
        if isinstance(agent_result, dict):
            response = agent_result.get("final_answer", str(agent_result))
            tool_calls = sanitize_tool_calls(agent_result.get("tool_calls", []))
            message_history = agent_result.get("message_history", [])
        else:
            response = agent_result
            tool_calls = []
            message_history = []
        
        # Parse response
        predicted = parser_map[query_type](response, query_type)
        
        # Convert seconds to timesteps if using seconds format (same logic as multiframe)
        if predicted and 'second' in predicted:
            # Convert single second to timestep
            predicted['timestep'] = seconds_to_timestep(predicted['second'], num_ts, effective_fps)
        elif predicted and 'second_ranges' in predicted:
            # Convert second ranges to timestep ranges
            timestep_ranges = []
            for start_sec, end_sec in predicted['second_ranges']:
                start_timestep = seconds_to_timestep(start_sec, num_ts, effective_fps)
                end_timestep = seconds_to_timestep(end_sec, num_ts, effective_fps)
                timestep_ranges.append([start_timestep, end_timestep])
            predicted['ranges'] = timestep_ranges
        
        results.append({
            'query_id': query_anno['query_id'],
            'query_type': query_type,
            'question': query_anno['question'],
            'predicted': predicted,
            'raw_response': response,
            'tool_calls': tool_calls,
            'message_history': message_history,
        })
        
        # Clear memory after each query to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clean up autoencoder
    del autoencoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

