from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.frame_selectors import TripletsFrameSelector
from benchmark.frame_evaluators import TripletsFrameEvaluator
from benchmark.spatial import (
    get_patched_qwen_for_spatial_grounding,
    extract_text_to_vision_attention,
)
from rerun_utils import (
    init_and_save_rerun,
    log_spatial_grounding_heatmaps,
    log_basic_points,
)
import numpy as np
import torch
# rerun is used via helper functions imported from rerun_utils


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
    graph_dir = output_root / clip_name / "graph"

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
        model_name="qwen",
        qwen_version=qwen_version,
        use_4bit_quantization=use_4bit,
    )
    return bench_cfg


def evaluate_triplets(clip: DictConfig, cfg: DictConfig):
    bench_cfg = _build_benchmark_config(cfg, clip)
    selector = TripletsFrameSelector(bench_cfg)
    samples = selector.select_sequences()
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
    out_file = out_dir / f"{clip.name}_ablation_{ts}.json"
    evaluator.save_results(results, out_file)


def evaluate_temporal(clip: DictConfig, cfg: DictConfig):
    pass


def evaluate_spatial(clip: DictConfig, cfg: DictConfig):
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
    """
    bench_cfg = _build_benchmark_config(cfg, clip)
    spatial_cfg = bench_cfg.spatial_config
    assert spatial_cfg is not None, "cfg.eval.spatial must be provided"

    graph_subdir = spatial_cfg["graph_subdir"]
    graph_dir = bench_cfg.output_root / clip.name / graph_subdir

    # Load required artifacts
    splat_feats_path = graph_dir / "splat_spatial_grounding_feats.npy"
    splat_indices_path = graph_dir / "splat_spatial_grounding_indices.npy"
    positions_path = graph_dir / "positions.npy"
    # Write overlays to a dedicated spatial file (non-destructive)
    rerun_file = graph_dir / "visualization_spatial.rrd"

    assert splat_feats_path.exists(), f"Missing {splat_feats_path}"
    assert splat_indices_path.exists(), f"Missing {splat_indices_path}"
    assert positions_path.exists(), f"Missing {positions_path}"

    splat_feats = np.load(splat_feats_path)  # (T, N, D)
    splat_indices = np.load(splat_indices_path)  # (N,)
    positions = np.load(positions_path)  # (M, 3), filtered gaussians

    T, N, D = splat_feats.shape

    # Config
    layers = list(map(int, spatial_cfg["layers"]))
    prompt = str(spatial_cfg["prompt"])
    substring = str(spatial_cfg["substring"])
    timestep = int(spatial_cfg["timestep"])
    cmap_name = str(spatial_cfg["colormap"])  # required

    assert 0 <= timestep < T, f"Invalid timestep {timestep}; valid range [0,{T-1}]"

    # Prepare vision features for the timestep
    vision_features = torch.tensor(splat_feats[timestep], dtype=torch.float32)

    # Load patched model with attentions enabled
    use_4bit = bench_cfg.use_4bit_quantization
    model, processor = get_patched_qwen_for_spatial_grounding(
        use_bnb_4bit=use_4bit, use_bnb_8bit=False
    )

    # Compute attentions
    attn_out = extract_text_to_vision_attention(
        model=model,
        processor=processor,
        vision_features=vision_features,
        layers=layers,
        prompt=prompt,
        substring=substring,
    )

    scores: torch.Tensor = attn_out["scores"]  # (L, Q, N)
    tokens = attn_out["tokens"]
    query_token_indices = attn_out["query_token_indices"]
    # vision_token_indices = attn_out["vision_token_indices"]  # equals [0..N-1] in our setup

    # Selected positions for the N sampled points
    point_positions = positions[splat_indices]

    # Init rerun sink to the existing file
    init_and_save_rerun(rerun_file)

    # Log a neutral base point cloud as context for overlay
    log_basic_points(
        entity_path="spatial_grounding/base_points",
        positions=point_positions,
        color=[180, 180, 180],
        timestep=timestep,
    )

    log_spatial_grounding_heatmaps(
        base_path="spatial_grounding",
        positions=point_positions,
        layers=layers,
        tokens=tokens,
        query_token_indices=query_token_indices,
        scores=scores.detach().cpu().numpy(),
        cmap_name=cmap_name,
        timestep=timestep,
    )


def evaluate_clip(clip: DictConfig, cfg: DictConfig):
    """Run benchmark evaluations for a single clip using Hydra configs."""
    if cfg.eval.triplets is not None:
        evaluate_triplets(clip, cfg)

    if cfg.eval.temporal is not None:
        evaluate_temporal(clip, cfg)

    if cfg.eval.spatial is not None:
        evaluate_spatial(clip, cfg)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    for clip in tqdm(cfg.clips, desc="Evaluating clips", unit="clip"):
        evaluate_clip(clip, cfg)


if __name__ == "__main__":
    main()
