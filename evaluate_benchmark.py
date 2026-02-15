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
    graph_agent_queries,
)
from benchmark.spatial import (
    dump_spatial_prediction_visualizations,
    frame_direct_feat_queries,
)
from benchmark.spatial_3d import graph_agent_3d_feat_queries

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
):
    """Run temporal action localization evaluation for a single clip."""
    if cfg.eval is None or cfg.eval.temporal is None:
        return
    
    video_dir = Path(cfg.preprocessed_root) / str(clip.name)
    graph_path = Path(cfg.output_root) / str(clip.name) / cfg.eval.paths.graph_subdir
    
    video_frames, _ = load_video_frames(video_dir, cfg.eval.paths.images_subdir)
    
    # Load temporal annotations
    temporal_labels_root = Path(cfg.eval.temporal.labels_root)
    filename_template = cfg.eval.temporal.labels_filename_template
    temporal_anno_file = temporal_labels_root / filename_template.format(clip_name=str(clip.name))
    
    with open(temporal_anno_file) as f:
        temporal_data = json.load(f)
    
    annotations = temporal_data["annotations"]

    # Map method names to strategy functions
    method_map = {
        "multiframe": multiframe_queries,
        "graph_agent": graph_agent_queries,
    }
    
    # Collect predictions for all methods specified in config
    all_results = {}
    
    for method_name in cfg.eval.temporal.methods:
        if method_name not in method_map:
            continue
        
        strategy_fn = method_map[method_name]
        results = strategy_fn(
            model=model,
            processor=processor,
            video_frames=video_frames,
            graph_path=graph_path,
            annotations=annotations,
            clip=clip,
            cfg=cfg,
        )
        all_results[method_name] = results
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to prediction dump format
    preds_by_method: dict[str, list[dict]] = {}
    for method_name, results in all_results.items():
        preds_by_method[method_name] = [
            {
                "query_id": r.get("query_id"),
                "query_type": r.get("query_type"),
                "question": r.get("question"),
                "predicted": r.get("predicted"),
                "raw_response": r.get("raw_response"),
                "tool_calls": r.get("tool_calls"),
                "message_history": r.get("message_history"),
            }
            for r in results
        ]
    
    # Save per-clip predictions for compute_metrics stage
    pred_out_dir = Path(cfg.eval.temporal.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
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
        all_results["graph_agent"] = graph_agent_3d_feat_queries(
            model=model,  # Uses CustomQwen3VLForConditionalGeneration3D (class swapped)
            processor=processor,
            graph_dir=graph_dir,  # Contains Gaussian splat data and features
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
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
            "frame_attn": "frame_attn",
            "frame_attn_refine": "frame_attn_refine",
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
    methods_to_run = set(cfg.eval.spatial.methods)
    for method in methods_to_run:
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

    # TODO: handle triplets and temporal accordingly down the line
    # for clip in tqdm(cfg.clips, desc="Evaluating clips (temporal)", unit="clip"):
    #     evaluate_temporal(clip, cfg, model=model, processor=processor)
    # for clip in tqdm(cfg.clips, desc="Evaluating clips (triplets)", unit="clip"):
    #     evaluate_triplets(clip, cfg, model=model, processor=processor)


if __name__ == "__main__":
    main()
