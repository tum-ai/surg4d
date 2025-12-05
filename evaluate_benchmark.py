import gc
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import random
from tqdm import tqdm
import hydra
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from llm.qwen_utils import get_patched_qwen

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.frame_selectors import TripletsFrameSelector
from benchmark.frame_evaluators import TripletsFrameEvaluator
from benchmark.temporal_evaluator import TemporalFrameEvaluator
from benchmark.spatial import (
    get_patched_qwen_for_spatial_grounding,
    splat_feat_queries,
    dump_spatial_prediction_visualizations,
    static_graph_feat_queries,
    frame_attn_feat_queries,
    splat_graph_feat_queries,
    frame_attn_refine_feat_queries,
    frame_direct_feat_queries,
    graph_agent_feat_queries,
)
from rerun_utils import (
    init_and_save_rerun,
    log_spatial_predictions,
)


def _build_benchmark_config(cfg: DictConfig, clip: DictConfig) -> BenchmarkConfig:
    # Infer qwen version and quantization from feature_extraction group if present
    use_4bit = bool(cfg.get("feature_extraction", {}).get("bnb_4bit", False))

    # Paths
    cholect50_root = Path(cfg.cholect50_root)
    preprocessed_root = Path(cfg.preprocessed_root)
    output_root = Path(cfg.output_root)

    clip_name = str(clip.name)
    video_dir = preprocessed_root / clip_name

    graph_dir = output_root / clip_name / cfg.eval.paths.graph_subdir

    # Convert nested eval configs (OmegaConf) to plain dicts
    triplets_cfg = None
    temporal_cfg = None
    spatial_cfg = None
    spatiotemporal_cfg = None

    if cfg.get("eval") is not None:
        if cfg.eval.get("triplets") is not None:
            triplets_cfg = OmegaConf.to_container(cfg.eval.triplets, resolve=True)
        if cfg.eval.get("temporal") is not None:
            temporal_cfg = OmegaConf.to_container(cfg.eval.temporal, resolve=True)
        if cfg.eval.get("spatial") is not None:
            spatial_cfg = OmegaConf.to_container(cfg.eval.spatial, resolve=True)

    bench_cfg = BenchmarkConfig(
        triplets_config=triplets_cfg,
        temporal_config=temporal_cfg,
        spatial_config=spatial_cfg,
        spatiotemporal_config=spatiotemporal_cfg,
        cholect50_root=cholect50_root,
        preprocessed_root=preprocessed_root,
        output_root=output_root,
        results_dir=output_root / "benchmark",
        video_dir=video_dir,
        graph_dir=graph_dir,
        images_subdir=cfg.eval.paths.images_subdir,
        graph_subdir=cfg.eval.paths.graph_subdir,
        use_4bit_quantization=use_4bit,
    )
    return bench_cfg


def evaluate_triplets(
    clip: DictConfig,
    cfg: DictConfig,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    """Run triplet recognition evaluation for a single clip."""
    if cfg.eval is None or cfg.eval.triplets is None:
        return
    bench_cfg = _build_benchmark_config(cfg, clip)
    triplets_cfg = bench_cfg.triplets_config
    assert triplets_cfg is not None, "cfg.eval.triplets must be provided"
    if triplets_cfg is None:
        return

    print(f"\n{'=' * 80}")
    print(f"TRIPLETS EVALUATION: {clip.name}")
    print(f"{'=' * 80}")

    try:
        selector = TripletsFrameSelector(bench_cfg)
        samples = selector.select_sequences()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Skipping triplets evaluation for this clip.")
        print("CholecT50 labels are only available for videos 1-50.")
        return

    if not samples:
        print("ERROR: No samples selected! Check if preprocessed data is available.")
        return

    selector.print_summary(samples)

    evaluator = TripletsFrameEvaluator(bench_cfg, model=model, processor=processor)
    results = evaluator.run_ablation_study(
        samples,
        ablations=bench_cfg.triplets_config["ablations"],  # type: ignore[index]
    )

    # Convert to prediction dump analogous to temporal eval
    preds_by_ablation: dict[str, list[dict]] = {}
    for ablation, payload in results.get("conditions", {}).items():
        ablation_results = payload.get("results", [])
        preds_by_ablation[ablation] = [
            {
                "sample_id": r.get("sample_id"),
                "video_id": r.get("video_id"),
                "clip_start": r.get("clip_start"),
                "end_frame": r.get("end_frame"),
                "second_idx": r.get("second_idx"),
                "predicted": r.get("predicted_triplets", []),
                "raw_response": r.get("response"),
            }
            for r in ablation_results
        ]

    # Save per-clip predictions for compute_metrics stage
    pred_out_dir = Path(cfg.eval.triplets.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
    with pred_out_file.open("w") as f:
        json.dump({"clip": clip.name, "ablations": preds_by_ablation}, f, indent=2)


def evaluate_temporal(
    clip: DictConfig,
    cfg: DictConfig,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    """Run temporal action localization evaluation for a single clip."""
    if cfg.eval is None or cfg.eval.temporal is None:
        return
    bench_cfg = _build_benchmark_config(cfg, clip)
    temporal_cfg = bench_cfg.temporal_config
    assert temporal_cfg is not None, "cfg.eval.temporal must be provided"
    if temporal_cfg is None:
        return

    print(f"\n{'=' * 80}")
    print(f"TEMPORAL EVALUATION: {clip.name}")
    print(f"{'=' * 80}")

    # Resolve temporal annotations file
    # Priority: clip.temporal_eval_file (override) -> eval.temporal.labels_root/template
    temporal_anno_file: Path
    if hasattr(clip, "temporal_eval_file") and clip.temporal_eval_file is not None:
        temporal_anno_file = Path(clip.temporal_eval_file)
    else:
        # Build from Hydra config (no preprocessing required)
        temporal_labels_root = Path(cfg.eval.temporal.get("labels_root", "data/temporal_labels"))
        filename_template = cfg.eval.temporal.get("labels_filename_template", "{clip_name}_temporal.json")
        temporal_anno_file = temporal_labels_root / filename_template.format(clip_name=str(clip.name))

    # Load temporal annotations from resolved path
    if not temporal_anno_file.exists():
        print(f"ERROR: Temporal annotations not found: {temporal_anno_file}")
        print("Skipping temporal evaluation for this clip.")
        print("Provide 'temporal_eval_file' in clip config or place labels under eval.temporal.labels_root.")
        return

    import json

    try:
        with open(temporal_anno_file) as f:
            temporal_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in temporal annotations file: {temporal_anno_file}")
        print(f"JSON error: {e}")
        print("Skipping temporal evaluation for this clip.")
        return
    except Exception as e:
        print(f"ERROR: Could not load temporal annotations: {e}")
        print("Skipping temporal evaluation for this clip.")
        return

    if "annotations" not in temporal_data or not temporal_data["annotations"]:
        print(f"ERROR: No annotations found in {temporal_anno_file}")
        print("Skipping temporal evaluation for this clip.")
        return

    print(
        f"Loaded {len(temporal_data['annotations'])} temporal queries from {temporal_anno_file}"
    )

    # Run temporal evaluation
    evaluator = TemporalFrameEvaluator(bench_cfg, model=model, processor=processor)

    # Check if evaluator was initialized successfully
    if evaluator.num_frames == 0:
        print("Skipping temporal evaluation due to missing data.")
        return

    results = evaluator.run_temporal_benchmark(
        annotations=temporal_data["annotations"],
        ablations=bench_cfg.temporal_config["ablations"],
    )

    # Convert to prediction dump analogous to spatial eval
    preds_by_ablation: dict[str, list[dict]] = {}
    for ablation, payload in results.get("ablations", {}).items():
        ablation_results = payload.get("results", [])
        preds_by_ablation[ablation] = [
            {
                "query_id": r.get("query_id"),
                "query_type": r.get("query_type"),
                "question": r.get("question"),
                "predicted": r.get("predicted"),
                "raw_response": r.get("raw_response"),
            }
            for r in ablation_results
        ]

    # Save per-clip predictions for compute_metrics stage
    pred_out_dir = Path(cfg.eval.temporal.output_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_file = pred_out_dir / f"{clip.name}.json"
    with pred_out_file.open("w") as f:
        json.dump({"clip": clip.name, "ablations": preds_by_ablation}, f, indent=2)


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
    model_spatial: Qwen2_5_VLForConditionalGeneration,
    processor_spatial: Qwen2_5_VLProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    """Compute text-to-vision attention over sampled scene points and log heatmaps to rerun.

    Expects graph extraction outputs to exist under output_root/<clip.name>/<graph_subdir>:
      - visualization.rrd (existing rerun file; we'll append new logs)
      - splat_spatial_grounding_feats.npy (T, N, D)
      - splat_spatial_grounding_indices.npy (N,)
      - positions.npy (M, 3) positions of filtered gaussians (indices refer into this)

    Config (cfg.eval.spatial):
      - graph_subdir: subdirectory name used by graph extraction (defaults to cfg.graph_extraction.graph_output_subdir)
      - layers: list of transformer layers to read attentions from
      - prompt: full prompt string containing the substring
      - substring: substring to select query tokens for
      - timestep: integer index into T for which features to use
      - colormap: optional matplotlib colormap name (default 'jet')

    Args:
      model: Pre-loaded patched qwen model for spatial grounding
      processor: Pre-loaded qwen processor
    """
    # skip this if no eval config is set
    if cfg.eval is None or cfg.eval.spatial is None:
        return

    # splat data
    graph_dir = Path(cfg.output_root) / clip.name / cfg.eval.paths.graph_subdir
    splat_feats = np.load(graph_dir / "splat_spatial_grounding_feats.npy")  #  (T, N, D)
    splat_indices = np.load(
        graph_dir / "splat_spatial_grounding_indices.npy"
    )  # c_id -> (N,)
    positions = np.load(graph_dir / "positions.npy")  # (T, N, 3)

    # load gt data
    gt_file = Path(cfg.preprocessed_root) / clip.name / cfg.eval.spatial.gt_filename
    with gt_file.open("r") as f:
        gt_data = json.load(f)

    # which methods to run
    methods_to_run = set(cfg.eval.spatial.methods)

    # compute predictions
    all_results = {}

    if "splat" in methods_to_run:
        all_results["splat"] = splat_feat_queries(
            model=model_spatial,
            processor=processor_spatial,
            splat_feats=splat_feats,
            splat_indices=splat_indices,
            positions=positions,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

    if "static_graph" in methods_to_run:
        all_results["static_graph"] = static_graph_feat_queries(
            model=model,
            processor=processor,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

    if "splat_graph" in methods_to_run:
        # SPLAT proposals + static graph refinement
        all_results["splat_graph"] = splat_graph_feat_queries(
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            model=model,
            processor=processor,
            splat_feats=splat_feats,
            splat_indices=splat_indices,
            positions=positions,
            graph_dir=graph_dir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

    if "frame_attn" in methods_to_run:
        all_results["frame_attn"] = frame_attn_feat_queries(
            model=model_spatial,
            processor=processor_spatial,
            preprocessed_root=Path(cfg.preprocessed_root),
            images_subdir=cfg.eval.paths.images_subdir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

    if "frame_attn_refine" in methods_to_run:
        # 2D attention proposals + refinement via normal Qwen
        all_results["frame_attn_refine"] = frame_attn_refine_feat_queries(
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
            model=model,
            processor=processor,
            preprocessed_root=Path(cfg.preprocessed_root),
            images_subdir=cfg.eval.paths.images_subdir,
            clip_gt=gt_data,
            clip=clip,
            cfg=cfg,
        )

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
        )
    out_dir = Path(cfg.eval.spatial.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / f"{clip.name}.json"
    with open(predictions_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Optional: dump per-(query, layer) visualizations of top-k points on the frame
    if cfg.eval.spatial.dump_visualizations:
        viz_method_names = {
            "splat": "splat",
            "static_graph": "static",
            "frame_attn": "frame_attn",
            "splat_graph": "splat_graph",
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

    # Initialize rerun sink for spatial visualization
    init_and_save_rerun(graph_dir / "visualization_spatial.rrd")

    # Log results for each method that was run
    for method_key in all_results:
        log_spatial_predictions(
            base_path=method_key,
            clip_name=clip.name,
            positions_through_time=positions,
            results=all_results[method_key],
            cmap_name=cfg.eval.spatial.colormap,
        )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model, processor = get_patched_qwen(
        qwen_version=cfg.eval.qwen_version,
        use_bnb_4bit=cfg.eval.use_bnb_4bit,
        use_bnb_8bit=cfg.eval.use_bnb_8bit,
    )
    model_spatial, processor_spatial = get_patched_qwen_for_spatial_grounding(
        qwen_version=cfg.eval.qwen_version,
        use_bnb_4bit=cfg.eval.use_bnb_4bit,
        use_bnb_8bit=cfg.eval.use_bnb_8bit,
    )

    for clip in tqdm(cfg.clips, desc="Evaluating clips", unit="clip"):
        evaluate_triplets(clip, cfg, model=model, processor=processor)
        evaluate_temporal(clip, cfg, model=model, processor=processor)
        evaluate_spatial(
            clip=clip,
            cfg=cfg,
            model=model,
            processor=processor,
            model_spatial=model_spatial,
            processor_spatial=processor_spatial,
        )

    del model_spatial
    del processor_spatial
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
