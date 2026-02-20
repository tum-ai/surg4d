import gc
import json
from pathlib import Path
from omegaconf import DictConfig
import random
from tqdm import tqdm
import hydra
import numpy as np
import torch

from llm.qwen_utils import get_patched_qwen3
from llm.qwen_utils_3d import get_custom_qwen3_3d
from benchmark.temporal import (
    load_video_frames,
    multiframe_queries,
    graph_agent_queries,
)
from benchmark.spatial import (
    dump_spatial_prediction_visualizations,
    frame_direct_feat_queries,
    graph_agent_feat_queries,
)
from benchmark.directional import (
    graph_agent_directional_queries,
    multiframe_directional_queries,
)
from benchmark.spatial_3d import splat_grid_feat_queries, splat_grid_temporal_queries


def evaluate_temporal(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
    model_3d=None,
    processor_3d=None,
):
    """Run temporal action localization evaluation for a single clip.

    Args:
      model: Pre-loaded patched Qwen model (for multiframe / graph_agent)
      processor: Pre-loaded Qwen processor (for multiframe / graph_agent)
      model_3d: Pre-loaded 3D patched Qwen model (for splat_grid)
      processor_3d: Pre-loaded Qwen processor (for splat_grid)
    """
    if cfg.eval is None or cfg.eval.temporal is None:
        return
    
    video_dir = Path(cfg.preprocessed_root) / str(clip.name)
    graph_path = Path(cfg.output_root) / str(clip.name) / cfg.eval.paths.graph_subdir
    
    video_frames, _ = load_video_frames(video_dir, cfg.eval.paths.images_subdir)
    
    # load annotations
    temporal_anno_file = Path(cfg.eval.annotations_root) / "temporal" / f"{clip.name}.json"
    with open(temporal_anno_file) as f:
        temporal_data = json.load(f)
    annotations = temporal_data["annotations"]

    # Methods using the normal model
    method_map = {
        "multiframe": multiframe_queries,
        "graph_agent": graph_agent_queries,
        "graph_agent_semantics": graph_agent_queries,
        "graph_agent_semantics_vision": graph_agent_queries,
    }
    
    # Collect predictions for all methods specified in config
    all_results = {}
    
    for method_name in cfg.eval.temporal.methods:
        if method_name == "splat_grid":
            # splat_grid uses the 3D model, handled separately below
            continue
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
            use_semantic_labels=method_name in {"graph_agent_semantics", "graph_agent_semantics_vision"},
            semantic_method_name=method_name,
        )
        all_results[method_name] = results
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "splat_grid" in cfg.eval.temporal.methods:
        all_results["splat_grid"] = splat_grid_temporal_queries(
            model=model_3d,
            processor=processor_3d,
            graph_path=graph_path,
            annotations=annotations,
            clip=clip,
            cfg=cfg,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to prediction dump format
    preds_by_method: dict[str, list[dict]] = {}
    for method_name, results in all_results.items():
        preds_by_method[method_name] = [
            {
                "id": r.get("id"),
                "type": r.get("type"),
                "query": r.get("query"),
                "predicted": r.get("predicted"),
                "raw_response": r.get("raw_response"),
                "message_history": r.get("message_history"),
                "tool_calls": r.get("tool_calls"),
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
    model_3d=None,
    processor_3d=None,
):
    """Run spatial grounding evaluation using frame-based and graph-based methods.

    Methods supported:
      - frame_direct: Direct Qwen prompting on frame to return a pixel
      - graph_agent: Agentic exploration of 3D scene graph with tools
      - splat_grid: 3D Gaussian-to-grid positional encoding (requires 3D model)

    Args:
      model: Pre-loaded patched Qwen model (for frame_direct / graph_agent)
      processor: Pre-loaded Qwen processor (for frame_direct / graph_agent)
      model_3d: Pre-loaded 3D patched Qwen model (for splat_grid)
      processor_3d: Pre-loaded Qwen processor (for splat_grid)
    """
    # skip this if no eval config is set
    if cfg.eval is None or cfg.eval.spatial is None:
        return

    # which methods to run
    methods_to_run = set(cfg.eval.spatial.methods)

    # graph data (needed for graph_agent and splat_grid)
    graph_dir = Path(cfg.output_root) / clip.name / cfg.eval.paths.graph_subdir

    # load gt data
    gt_file = Path(cfg.preprocessed_root) / clip.name / cfg.eval.spatial.gt_filename
    with gt_file.open("r") as f:
        gt_data = json.load(f)["annotations"]

    # compute predictions
    all_results = {}

    if "frame_direct" in methods_to_run:
        # Direct Qwen prompting on frame to return a pixel
        all_results["frame_direct"] = frame_direct_feat_queries(
            model=model,
            processor=processor,
            preprocessed_root=Path(cfg.preprocessed_root),
            images_subdir=cfg.eval.paths.images_subdir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

    if "graph_agent" in methods_to_run:
        # Graph agent with tools (requires qwen3)
        all_results["graph_agent"] = graph_agent_feat_queries(
            model=model,
            processor=processor,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
            use_semantic_labels=False,
        )

    if "graph_agent_semantics" in methods_to_run:
        all_results["graph_agent_semantics"] = graph_agent_feat_queries(
            model=model,
            processor=processor,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
            use_semantic_labels=True,
            semantic_method_name="graph_agent_semantics",
        )

    if "graph_agent_semantics_vision" in methods_to_run:
        all_results["graph_agent_semantics_vision"] = graph_agent_feat_queries(
            model=model,
            processor=processor,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
            use_semantic_labels=True,
            semantic_method_name="graph_agent_semantics_vision",
        )

    if "splat_grid" in methods_to_run:
        # 3D Gaussian-to-grid positional encoding (requires 3D patched model)
        all_results["splat_grid"] = splat_grid_feat_queries(
            model=model_3d,
            processor=processor_3d,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

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
            "graph_agent_semantics": "graph_agent_semantics",
            "graph_agent_semantics_vision": "graph_agent_semantics_vision",
            "splat_grid": "splat_grid",
        }
        for method_key, viz_name in viz_method_names.items():
            if method_key in all_results:
                dump_spatial_prediction_visualizations(
                    cfg=cfg,
                    results_splat=all_results[method_key],
                    clip_name=clip.name,
                    preprocessed_root=Path(cfg.preprocessed_root),
                    images_subdir=cfg.eval.paths.images_subdir,
                    viz_dir=Path(cfg.eval.spatial.visualizations_dir),
                    method_name=viz_name,
                )


def evaluate_directional(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
):
    if cfg.eval is None or cfg.eval.directional is None:
        return

    methods_to_run = set(cfg.eval.directional.methods)
    graph_path = Path(cfg.output_root) / str(clip.name) / cfg.eval.paths.graph_subdir
    video_dir = Path(cfg.preprocessed_root) / str(clip.name)

    video_frames = None
    if methods_to_run & {"multiframe", "graph_agent", "graph_agent_semantics", "graph_agent_semantics_vision"}:
        video_frames, _ = load_video_frames(video_dir, cfg.eval.paths.images_subdir)

    directional_anno_file = Path(cfg.eval.annotations_root) / "directional" / f"{clip.name}.json"
    with open(directional_anno_file) as f:
        directional_data = json.load(f)
    annotations = directional_data["annotations"]

    method_map = {
        "multiframe": multiframe_directional_queries,
        "graph_agent": graph_agent_directional_queries,
        "graph_agent_semantics": graph_agent_directional_queries,
        "graph_agent_semantics_vision": graph_agent_directional_queries,
    }

    all_results = {}
    for method_name in cfg.eval.directional.methods:
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
            use_semantic_labels=method_name in {"graph_agent_semantics", "graph_agent_semantics_vision"},
            semantic_method_name=method_name,
        )
        all_results[method_name] = results

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds_by_method: dict[str, list[dict]] = {}
    for method_name, results in all_results.items():
        preds_by_method[method_name] = [
            {
                "id": r.get("id"),
                "query": r.get("query"),
                "range": r.get("range"),
                "predicted": r.get("predicted"),
                "raw_response": r.get("raw_response"),
                "message_history": r.get("message_history"),
                "tool_calls": r.get("tool_calls"),
            }
            for r in results
        ]

    pred_out_dir = Path(cfg.eval.directional.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
    with pred_out_file.open("w") as f:
        json.dump({"clip": str(clip.name), "methods": preds_by_method}, f, indent=2)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Determine which models are needed based on active methods
    needs_normal_model = False
    needs_3d_model = False

    if cfg.eval is not None and cfg.eval.temporal is not None:
        temporal_methods = set(cfg.eval.temporal.methods)
        if temporal_methods & {"multiframe", "graph_agent", "graph_agent_semantics", "graph_agent_semantics_vision"}:
            needs_normal_model = True
        if "splat_grid" in temporal_methods:
            needs_3d_model = True

    if cfg.eval is not None and cfg.eval.spatial is not None:
        spatial_methods = set(cfg.eval.spatial.methods)
        if spatial_methods & {"frame_direct", "graph_agent", "graph_agent_semantics", "graph_agent_semantics_vision"}:
            needs_normal_model = True
        if "splat_grid" in spatial_methods:
            needs_3d_model = True

    if cfg.eval is not None and cfg.eval.directional is not None:
        directional_methods = set(cfg.eval.directional.methods)
        if directional_methods & {"multiframe", "graph_agent", "graph_agent_semantics", "graph_agent_semantics_vision"}:
            needs_normal_model = True

    model, processor = None, None
    model_3d, processor_3d = None, None

    if needs_normal_model:
        model, processor = get_patched_qwen3(
            size=cfg.eval.qwen3_size,
            use_fp8=cfg.eval.qwen3_use_fp8,
        )

    if needs_3d_model:
        model_3d, processor_3d = get_custom_qwen3_3d(
            size=cfg.eval.qwen3_size,
            use_fp8=cfg.eval.qwen3_use_fp8,
        )

    for clip in tqdm(cfg.clips, desc="Evaluating clips", unit="clip"):
        evaluate_temporal(
            clip=clip, cfg=cfg,
            model=model, processor=processor,
            model_3d=model_3d, processor_3d=processor_3d,
        )
        evaluate_spatial(
            clip=clip, cfg=cfg,
            model=model, processor=processor,
            model_3d=model_3d, processor_3d=processor_3d,
        )
        evaluate_directional(
            clip=clip,
            cfg=cfg,
            model=model,
            processor=processor,
        )

if __name__ == "__main__":
    main()
