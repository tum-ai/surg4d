import gc
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.frame_selectors import TripletsFrameSelector
from benchmark.frame_evaluators import TripletsFrameEvaluator
from benchmark.temporal_evaluator import TemporalFrameEvaluator
from benchmark.spatial import (
    get_patched_qwen_for_spatial_grounding,
    splat_feat_queries,
)
from rerun_utils import (
    init_and_save_rerun,
    log_spatial_predictions,
)
import numpy as np
import torch

import json

 


def _build_benchmark_config(cfg: DictConfig, clip: DictConfig) -> BenchmarkConfig:
    # Infer qwen version and quantization from feature_extraction group if present
    use_qwen3 = bool(cfg.get("feature_extraction", {}).get("use_qwen3", False))
    qwen_version = "qwen3" if use_qwen3 else "qwen2.5"
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
        model_name="qwen",
        qwen_version=qwen_version,
        use_4bit_quantization=use_4bit,
    )
    return bench_cfg


def evaluate_triplets(clip: DictConfig, cfg: DictConfig):
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

    evaluator = TripletsFrameEvaluator(bench_cfg)
    results = evaluator.run_ablation_study(
        samples,
        ablations=bench_cfg.triplets_config["ablations"],  # type: ignore[index]
    )

    # Save to required output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{clip.name}_triplet_ablation_{ts}.json"
    evaluator.save_results(results, out_file)


def evaluate_temporal(clip: DictConfig, cfg: DictConfig):
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

    # Check if clip has temporal annotations specified
    if not hasattr(clip, "temporal_eval_file") or clip.temporal_eval_file is None:
        print(f"No temporal_eval_file specified for clip {clip.name}")
        print("Skipping temporal evaluation for this clip.")
        print(
            "To enable, add 'temporal_eval_file: path/to/annotations.json' to clip config"
        )
        return

    # Load temporal annotations from JSON file specified in clip config
    temporal_anno_file = Path(clip.temporal_eval_file)
    if not temporal_anno_file.exists():
        print(f"ERROR: Temporal annotations not found: {temporal_anno_file}")
        print("Skipping temporal evaluation for this clip.")
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
    evaluator = TemporalFrameEvaluator(bench_cfg)

    # Check if evaluator was initialized successfully
    if evaluator.num_frames == 0:
        print("Skipping temporal evaluation due to missing data.")
        return

    results = evaluator.run_temporal_benchmark(
        annotations=temporal_data["annotations"],
        ablations=bench_cfg.temporal_config["ablations"],
    )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{clip.name}_temporal_{ts}.json"
    evaluator.save_results(results, out_file)


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

    # compute predictions
    results_splat = splat_feat_queries(
        model=model_spatial,
        processor=processor_spatial,
        splat_feats=splat_feats,
        splat_indices=splat_indices,
        positions=positions,
        clip_gt=gt_data,
        clip=clip,
        cfg=cfg,
    )

    # save predictions
    all_results = {
        "splat": results_splat,
    }
    out_dir = Path(cfg.eval.spatial.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / f"{clip.name}.json"
    with open(predictions_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Initialize rerun sink for spatial visualization
    init_and_save_rerun(graph_dir / "visualization_spatial.rrd")

    # Single-call visualization
    log_spatial_predictions(
        base_path="splat",
        clip_name=clip.name,
        positions_through_time=positions,
        results=results_splat,
        cmap_name=cfg.eval.spatial.colormap,
    )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    model_spatial, processor_spatial = get_patched_qwen_for_spatial_grounding(
        use_bnb_4bit=cfg.eval.spatial.use_bnb_4bit,
        use_bnb_8bit=cfg.eval.spatial.use_bnb_8bit,
    )

    for clip in tqdm(cfg.clips, desc="Evaluating clips", unit="clip"):
        evaluate_triplets(clip, cfg)
        evaluate_temporal(clip, cfg)
        evaluate_spatial(clip, cfg, model_spatial, processor_spatial)

    del model_spatial
    del processor_spatial
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
