import gc
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import random
from tqdm import tqdm
import hydra
import numpy as np
import torch
from loguru import logger

# Standard patched Qwen3 model
from llm.qwen_utils import get_patched_qwen3

# Custom Qwen3 model with 3D positional encoding support
from llm.qwen_utils_3d import get_custom_qwen3_3d

from benchmark.temporal import (
    load_video_frames,
    multiframe_queries,
)
from benchmark.spatial import (
    dump_spatial_prediction_visualizations,
    frame_direct_feat_queries,
)
# General interface to the custom "3D" model
from benchmark.spatial_3d import graph_agent_3d_feat_queries_general

# Needed for postprocessing of model response to task specific outputs
from benchmark.spatial_3d import _parse_coords_from_json_3d
from benchmark.temporal import (
    parse_single_frame,
    parse_frame_ranges,
    seconds_to_timestep,
    get_num_timesteps_from_graph,
)


def _build_spatial_general_query_inputs(
    gt_data: dict,
    cfg: DictConfig,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]], list[str], str, list[dict]]:
    queries: list[str] = []
    timesteps: list[tuple[int, int]] = []
    frame_numbers: list[tuple[int, int]] = []
    query_metadata: list[dict] = []
    prompt_template = cfg.eval.spatial.graph_agent_prompt_template
    system_prompt = cfg.eval.spatial.graph_agent_system_prompt
    # For compatibility with temporal where the system prompts can change, we also have one per query here
    system_prompts: list[str] = []

    for timestep_key, timestep_item in gt_data.items():
        timestep_idx = int(timestep_key)
        frame_number = int(timestep_item["frame_number"])


        for query_obj in timestep_item.get("objects", []):
            queries.append(query_obj["query"])
            timesteps.append((timestep_idx, timestep_idx))
            frame_numbers.append((frame_number, frame_number))
            query_metadata.append(
                {
                    "task": "spatial",
                    "group": "objects",
                    "timestep": timestep_idx,
                    "frame_number": frame_number,
                    "query_gt": query_obj,
                }
            )
            system_prompts.append(system_prompt)

        for query_obj in timestep_item.get("actions", []):
            queries.append(query_obj["query"])
            timesteps.append((timestep_idx, timestep_idx))
            frame_numbers.append((frame_number, frame_number))
            query_metadata.append(
                {
                    "task": "spatial",
                    "group": "actions",
                    "timestep": timestep_idx,
                    "frame_number": frame_number,
                    "query_gt": query_obj,
                }
            )
            system_prompts.append(system_prompt)

    return queries, timesteps, frame_numbers, system_prompts, prompt_template, query_metadata


def _temporal_query_range_from_ground_truth(ground_truth: dict) -> tuple[int, int]:
    if "frame" in ground_truth:
        frame = int(ground_truth["frame"])
        return frame, frame
    if "ranges" in ground_truth:
        starts = [int(r[0]) for r in ground_truth["ranges"]]
        ends = [int(r[1]) for r in ground_truth["ranges"]]
        return min(starts), max(ends)
    raise ValueError(f"Unsupported temporal ground_truth format: {ground_truth}")


def _build_temporal_general_query_inputs(
    temporal_data: dict,
    cfg: DictConfig,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]], list[str], str, list[dict]]:
    queries: list[str] = []
    timesteps: list[tuple[int, int]] = []
    frame_numbers: list[tuple[int, int]] = []
    system_prompts: list[str] = []
    query_metadata: list[dict] = []
    prompt_template = "{substring}"

    # TODO: this might be needed later
    # num_frames = int(temporal_data["clip_info"]["num_frames"])
    # last_frame = num_frames - 1

    for query_anno in temporal_data["annotations"]:
        query_type = query_anno["query_type"]
        system_prompt_key = f"graph_agent_{query_type}_system_prompt"
        template_key = f"graph_agent_{query_type}_prompt_template"
        system_prompt = cfg.eval.temporal[system_prompt_key]
        template = cfg.eval.temporal[template_key]

        # TODO: the num_frames and last_frame are not used at the moment but might be needed laters
        # prompt = template.format(
        #     question=query_anno["question"],
        #     num_frames=num_frames,
        #     last_frame=last_frame,
        # )
        # Instead, for now we have a simpler prompt with just the question on the whole sequence
        prompt = template.format(question=query_anno["question"])
        queries.append(prompt)

        # TODO: fix this later, hardcoding this to be the whole clip for now
        #   This might become range dependent
        # TODO: once we have ranges, this will need to be adjusted
        #   Atm, hardcoding this to the number of timesteps for all clips
        t1 = 0
        t2 = 19
        timesteps.append((t1, t2))

        # TODO: Timesteps (also how queries are defined and splat / graph is saved) are at a 4x lower temporal resolution than original frames
        #   Need to properly formulate this and deal with it through hydra config, should not be hardcoded
        #   0 corresponds to 0, 1 corresponds to 4, ..., 19 corresponds to 76
        #   This assumes that we also take the first of the 4 frames that would map from frame numbers to timesteps
        f1 = t1 * 4
        f2 = t2 * 4
        frame_numbers.append((f1, f2))

        system_prompts.append(system_prompt)
        query_metadata.append(
            {
                "task": "temporal",
                "query_id": query_anno["query_id"],
                "query_type": query_type,
                "question": query_anno["question"],
                "answer_format": query_anno.get("answer_format"),
                "ground_truth": query_anno["ground_truth"],
            }
        )

    return queries, timesteps, frame_numbers, system_prompts, prompt_template, query_metadata


def _format_general_spatial_results_for_legacy(
    general_results: list[dict],
    query_metadata: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    formatted: dict[str, dict[str, list[dict]]] = {}

    for result, meta in zip(general_results, query_metadata):
        timestep_key = str(meta["timestep"])
        group_key = meta["group"]
        if timestep_key not in formatted:
            formatted[timestep_key] = {"objects": [], "actions": []}
        gaussian_means = np.asarray(result["gaussian_means"], dtype=np.float64)
        gaussian_means_projected = np.asarray(result["gaussian_means_projected"], dtype=np.float64)
        xy = gaussian_means_projected[:, :2]
        min_x, min_y = xy.min(axis=0)
        max_x, max_y = xy.max(axis=0)

        # This just converts Qwen3's artificial, hardcoded range back to the projected plane
        world_xy = _parse_coords_from_json_3d(
            result["raw_response"], min_x, max_x, min_y, max_y
        )
        # For position in projection plane, find nearest Gaussian
        query_xy = world_xy.reshape(1, 2)
        distances_xy = np.linalg.norm(gaussian_means_projected[:, :2] - query_xy, axis=1)
        nearest_idx = int(np.argmin(distances_xy))

        # Can get the Gaussians 3D position; technically, would project the 3D position into the image plane to get the pixel coords
        #   however, in the current implementation, the Gaussians are already projected with that transform so it is already a pixel coordinate
        world_pos = gaussian_means[nearest_idx]
        # TODO: if we ever use a different projection plane for the Gaussians, this needs to change!
        pixel_xy = gaussian_means_projected[nearest_idx, :2]

        formatted[timestep_key][group_key].append(
            {
                "query": meta["query_gt"]["query"],
                "predictions": {
                    "0": {
                        "pixel_coords": [[float(pixel_xy[0]), float(pixel_xy[1])]],
                        "positions": [world_pos.tolist()],
                    }
                },
                "raw_response": result["raw_response"],
            }
        )
    return formatted

def evaluate_triplets(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
):
    """Run triplet recognition evaluation for a single clip."""
    if cfg.eval is None or cfg.eval.triplets is None:
        return
    
    from benchmark.triplets import (
        load_triplet_samples,
        single_frame_queries,
        single_frame_mask_overlay_queries,
        multiframe_queries,
        multiframe_mask_overlay_queries,
        graph_agent_single_queries,
        graph_agent_dynamic_queries,
    )
    from benchmark.cholect50_utils import CholecT50Loader
    
    print(f"\n{'=' * 80}")
    print(f"TRIPLETS EVALUATION: {clip.name}")
    print(f"{'=' * 80}")
    
    # Parse video_id and clip_start from clip name (e.g., "video01_00100")
    clip_name = str(clip.name)
    try:
        video_id = int(clip_name.split("_")[0].replace("video", ""))
        clip_start = int(clip_name.split("_")[1])
    except Exception as e:
        print(f"ERROR: Could not parse clip name '{clip_name}': {e}")
        return
    
    # CholecT50 only has annotations for videos 1-50
    if video_id > 50:
        print(f"Skipping: CholecT50 annotations only available for videos 1-50 (got video {video_id})")
        return
    
    # Load GT samples
    video_dir = Path(cfg.preprocessed_root) / clip_name
    cholect50_loader = CholecT50Loader(str(cfg.cholect50_root))
    
    try:
        samples = load_triplet_samples(
            video_dir=video_dir,
            clip_start=clip_start,
            video_id=video_id,
            cholect50_loader=cholect50_loader,
            images_subdir=cfg.eval.paths.images_subdir,
            framerate=cfg.eval.triplets.FRAMERATE,
            num_frames=cfg.eval.triplets.NUM_FRAMES,
            frame_stride=cfg.eval.triplets.frame_stride,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Skipping triplets evaluation for this clip.")
        return
    
    if not samples:
        print("ERROR: No samples found! Check if preprocessed data is available.")
        return
    
    print(f"Loaded {len(samples)} evaluation samples")
    
    # Get graph path if needed
    graph_path = Path(cfg.output_root) / clip_name / cfg.eval.paths.graph_subdir
    
    # Map method names to strategy functions
    method_map = {
        "single_frame": single_frame_queries,
        "single_frame_mask_overlay": single_frame_mask_overlay_queries,
        "multiframe": multiframe_queries,
        "multiframe_mask_overlay": multiframe_mask_overlay_queries,
        "graph_agent_single": graph_agent_single_queries,
        "graph_agent_dynamic": graph_agent_dynamic_queries,
    }
    
    # Collect predictions for all methods
    all_results = {}
    
    for method_name in cfg.eval.triplets.methods:
        if method_name not in method_map:
            print(f"WARNING: Unknown method '{method_name}', skipping")
            continue
        
        print(f"\nRunning method: {method_name}")
        strategy_fn = method_map[method_name]
        
        # Call strategy function (with graph_path for agent methods)
        if "graph_agent" in method_name:
            if not graph_path.exists():
                print(f"  WARNING: Graph not found at {graph_path}, skipping {method_name}")
                continue
            results = strategy_fn(
                model=model,
                processor=processor,
                samples=samples,
                graph_path=graph_path,
                clip=clip,
                cfg=cfg,
            )
        else:
            results = strategy_fn(
                model=model,
                processor=processor,
                samples=samples,
                clip=clip,
                cfg=cfg,
            )
        
        all_results[method_name] = results
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save per-clip predictions for compute_metrics stage
    pred_out_dir = Path(cfg.eval.triplets.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
    
    with pred_out_file.open("w") as f:
        json.dump({"clip": clip.name, "methods": all_results}, f, indent=2)


def evaluate_temporal(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
    method
):
    """Run temporal action localization evaluation for a single clip."""
    if cfg.eval is None or cfg.eval.temporal is None:
        return
    
    graph_path = Path(cfg.output_root) / str(clip.name) / cfg.eval.paths.graph_subdir
    
    # Load temporal annotations
    temporal_labels_root = Path(cfg.eval.temporal.labels_root)
    filename_template = cfg.eval.temporal.labels_filename_template
    temporal_anno_file = temporal_labels_root / filename_template.format(clip_name=str(clip.name))
    
    with open(temporal_anno_file) as f:
        temporal_data = json.load(f)
    
    annotations = temporal_data["annotations"]
    
    # Collect predictions for all methods specified in config
    all_results = {}
    
    # TODO: regardless of method, common preprocessing should be done here
    if method == "multiframe":
        video_dir = Path(cfg.preprocessed_root) / str(clip.name)
        video_frames, _ = load_video_frames(video_dir, cfg.eval.paths.images_subdir)
        results = multiframe_queries(
            model=model,
            processor=processor,
            video_frames=video_frames,
            graph_path=graph_path,
            annotations=annotations,
            clip=clip,
            cfg=cfg,
        )
        all_results[method] = results
    elif method == "graph_agent":
        (
            queries,
            timesteps,
            frame_numbers,
            system_prompts,
            prompt_template,
            query_metadata,
        ) = _build_temporal_general_query_inputs(temporal_data, cfg)
        results = graph_agent_3d_feat_queries_general(
            model=model,
            processor=processor,
            graph_dir=graph_path,
            clip=clip,
            cfg=cfg,
            queries=queries,
            timesteps=timesteps,
            frame_numbers=frame_numbers,
            system_prompts=system_prompts,
            prompt_template=prompt_template,
            query_metadata=query_metadata,
        )     
        all_results[method] = results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parser_map = {
        'action_onset': parse_single_frame,
        'action_duration': parse_frame_ranges,
    }
    # Parse responses
    parsed_results = []

    # Load graph to determine stride for frame sampling
    num_ts = get_num_timesteps_from_graph(graph_path)
    effective_fps = cfg.eval.fps_timesteps

    for query_anno, result in zip(annotations, results):
        response = result["raw_response"]
        predicted = parser_map[query_anno['query_type']](response, query_anno['query_type'])
        
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
        # message_history = [
        #     {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        #     {"role": "user", "content": [{"type": "text", "text": prompt}]},
        #     {"role": "assistant", "content": [{"type": "text", "text": response}]}
        # ]
        
        parsed_results.append({
            'query_id': query_anno['query_id'],
            'query_type': query_anno['query_type'],
            'question': query_anno['question'],
            'predicted': predicted,
            'raw_response': response,
            # 'message_history': message_history,
        })
    
    # Convert to prediction dump format
    preds_by_method: dict[str, list[dict]] = {}
    preds_by_method[method] = parsed_results
    
    # Save per-clip predictions for compute_metrics stage
    pred_out_dir = Path(cfg.eval.temporal.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
    # Check if that file already has content, if so append to it
    if pred_out_file.exists():
        with pred_out_file.open("r") as f:
            existing_data = json.load(f)
        existing_data["methods"].update(preds_by_method)
        with pred_out_file.open("w") as f:
            json.dump(existing_data, f, indent=2)
    else:
        with pred_out_file.open("w") as f:
            json.dump({"clip": str(clip.name), "methods": preds_by_method}, f, indent=2)


def get_timestep_from_frame(frame: str, image_dir: Path) -> int:
    # Convert image_dir to list of image paths and look up position of frame in there
    image_paths = list(Path(image_dir).glob("*.jpg"))
    image_filenames = [path.name for path in image_paths]
    image_filenames.sort()
    timestep = image_filenames.index(frame)
    return timestep


def evaluate_spatial(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
    method
):
    """Run spatial grounding evaluation using frame-based and graph-based methods.

    Methods supported:
      - frame_direct: Direct Qwen prompting on frame to return a pixel
      - graph_agent: Agentic exploration of 3D scene graph with tools

    Graph data (graph_dir) is only needed for graph_agent method.

    Args:
      clip: Clip to evaluate
      cfg: Configuration
      model: Pre-loaded Qwen model
      processor: Pre-loaded Qwen processor
      method: Method to evaluate
    """
    # skip this if no eval config is set
    if cfg.eval is None or cfg.eval.spatial is None:
        return

    # graph data (only needed for graph_agent)
    graph_dir = Path(cfg.output_root) / clip.name / cfg.eval.paths.graph_subdir

    # load gt data
    gt_file = Path(cfg.preprocessed_root) / clip.name / cfg.eval.spatial.gt_filename
    with gt_file.open("r") as f:
        gt_data = json.load(f)

    # compute predictions
    all_results = {}

    def _clear_vram():
        """Clear VRAM cache after each method to prevent OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if method == "frame_direct":
        # Direct Qwen prompting on frame to return a pixel
        logger.info(f"Evaluating frame_direct for clip: {clip.name}")

        all_results["frame_direct"] = frame_direct_feat_queries(
            model=model,
            processor=processor,
            preprocessed_root=Path(cfg.preprocessed_root),
            images_subdir=cfg.eval.paths.images_subdir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )
        
        logger.info(f"Finished evaluating frame_direct for clip: {clip.name}")
        _clear_vram()

    if method == "graph_agent":        
        (
            queries,
            timesteps,
            frame_numbers,
            system_prompts,
            prompt_template,
            query_metadata,
        ) = _build_spatial_general_query_inputs(gt_data, cfg)

        graph_agent_general_results = graph_agent_3d_feat_queries_general(
            model=model,
            processor=processor,
            graph_dir=graph_dir,
            clip=clip,
            cfg=cfg,
            queries=queries,
            timesteps=timesteps,
            frame_numbers=frame_numbers,
            system_prompts=system_prompts,
            prompt_template=prompt_template,
            query_metadata=query_metadata,
        )
        all_results["graph_agent"] = _format_general_spatial_results_for_legacy(
            graph_agent_general_results,
            query_metadata,
        )
        _clear_vram()

    out_dir = Path(cfg.eval.spatial.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / f"{clip.name}.json"
    with open(predictions_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Optional: dump per-(query, layer) visualizations of top-k points on the frame
    if cfg.eval.spatial.dump_visualizations:
        viz_method_names = {
            "frame_direct": "frame_direct",
            "graph_agent": "graph_agent",
        }
        for method_key, viz_name in viz_method_names.items():
            if method_key in all_results:
                dump_spatial_prediction_visualizations(
                    results_splat=all_results[method_key],
                    clip_name=clip.name,
                    preprocessed_root=Path(cfg.preprocessed_root),
                    images_subdir=cfg.eval.paths.images_subdir,
                    gt_data=gt_data,
                    viz_dir=Path(cfg.eval.spatial.visualizations_dir),
                    method_name=viz_name,
                )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Spatial evaluation
    methods_spatial = set(cfg.eval.spatial.methods)
    for method in methods_spatial:
        if method == "frame_direct":
            model, processor = get_patched_qwen3(
                size=cfg.eval.qwen3_size,
                use_fp8=cfg.eval.qwen3_use_fp8,
            )
        elif method == "graph_agent":
            model, processor = get_custom_qwen3_3d(
                size=cfg.eval.qwen3_size,
                use_fp8=cfg.eval.qwen3_use_fp8,
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        for clip in tqdm(cfg.clips, desc="Evaluating clips (spatial)", unit="clip"):
            logger.info(f"Evaluating spatial for clip: {clip.name}")
            evaluate_spatial(
                clip=clip,
                cfg=cfg,
                model=model,
                processor=processor,
                method=method,
            )

    methods_temporal = set(cfg.eval.temporal.methods)
    for method in methods_temporal:
        if method == "multiframe":
            model, processor = get_patched_qwen3(
                size=cfg.eval.qwen3_size,
                use_fp8=cfg.eval.qwen3_use_fp8,
            )
        elif method == "graph_agent":
            model, processor = get_custom_qwen3_3d(
                size=cfg.eval.qwen3_size,
                use_fp8=cfg.eval.qwen3_use_fp8,
            )
            # The processor contains the VideoProcessor (https://huggingface.co/docs/transformers/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor)
            # For our Gaussian tokens, we do not want to "sample" from the video but use the whole thing
            processor.video_processor.do_sample_frames = False
        else:
            raise ValueError(f"Invalid method: {method}")

        for clip in tqdm(cfg.clips, desc="Evaluating clips (temporal)", unit="clip"):
            evaluate_temporal(clip, cfg, model=model, processor=processor, method=method)


if __name__ == "__main__":
    main()
