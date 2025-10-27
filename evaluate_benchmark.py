from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.frame_selectors import TripletsFrameSelector
from benchmark.frame_evaluators import TripletsFrameEvaluator
from benchmark.temporal_evaluator import TemporalFrameEvaluator
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

import json
from scene.dataset_readers import readColmapSceneInfo, CameraInfo
from scene.cameras import Camera
import os

import matplotlib.pyplot as plt
import cv2


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

    # Get path configuration from eval config (with defaults)
    images_subdir = "images"
    graph_subdir = "graph"
    if cfg.get("eval") is not None and cfg.eval.get("paths") is not None:
        images_subdir = cfg.eval.paths.get("images_subdir", "images")
        graph_subdir = cfg.eval.paths.get("graph_subdir", "graph")

    graph_dir = output_root / clip_name / graph_subdir

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
        images_subdir=images_subdir,
        graph_subdir=graph_subdir,
        model_name="qwen",
        qwen_version=qwen_version,
        use_4bit_quantization=use_4bit,
    )
    return bench_cfg


def evaluate_triplets(clip: DictConfig, cfg: DictConfig):
    """Run triplet recognition evaluation for a single clip."""
    bench_cfg = _build_benchmark_config(cfg, clip)
    triplets_cfg = bench_cfg.triplets_config
    assert triplets_cfg is not None, "cfg.eval.triplets must be provided"
    if triplets_cfg is None:
        return

    print(f"\n{'='*80}")
    print(f"TRIPLETS EVALUATION: {clip.name}")
    print(f"{'='*80}")

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
        samples, ablations=bench_cfg.triplets_config["ablations"]  # type: ignore[index]
    )

    # Save to required output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{clip.name}_triplet_ablation_{ts}.json"
    evaluator.save_results(results, out_file)


def evaluate_temporal(clip: DictConfig, cfg: DictConfig):
    """Run temporal action localization evaluation for a single clip."""
    bench_cfg = _build_benchmark_config(cfg, clip)
    temporal_cfg = bench_cfg.temporal_config
    assert temporal_cfg is not None, "cfg.eval.temporal must be provided"
    if temporal_cfg is None:
        return

    print(f"\n{'='*80}")
    print(f"TEMPORAL EVALUATION: {clip.name}")
    print(f"{'='*80}")

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


def get_proj_matrix_from_timestep(
    timestep: int, train_cameras: list, frame: str
) -> torch.Tensor:
    # Get the camera parameters for the timestep
    camera_info = train_cameras[timestep]
    assert isinstance(
        camera_info, CameraInfo
    ), "camera_info must be a CameraInfo object"

    # Instantiate a Camera object from the camera info
    image = camera_info.image
    R = camera_info.R
    T = camera_info.T
    FovX = camera_info.FovX
    FovY = camera_info.FovY
    time = camera_info.time
    mask = camera_info.mask
    camera = Camera(
        colmap_id=timestep,
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image=image,
        gt_alpha_mask=None,
        image_name=f"{frame}",
        uid=timestep,
        data_device=torch.device("cuda"),
        time=time,
        mask=mask,
    )

    # Get projection matrix from camera object
    # full_proj_transform includes the world to cam as well, seems to be correct
    # projection_matrix = camera.projection_matrix
    full_proj_matrix = camera.full_proj_transform
    return full_proj_matrix, camera.image_width, camera.image_height


def project_3d_to_2d(
    positions: np.ndarray, proj_matrix: torch.Tensor, img_width: int, img_height: int
) -> np.ndarray:
    # Expecting positions to be (N, 3)
    assert positions.shape[1] == 3, "Positions must be (N, 3)"

    # Conver to homogeneous coordinates
    ones = torch.ones(
        positions.shape[0], 1, dtype=positions.dtype, device=positions.device
    )
    positions = torch.cat([positions, ones], dim=1)  # (N, 4)

    # Apply full projection transform: world to image space
    # Apparently full_proj_transform is transposed, seems to be correct
    coords = (proj_matrix.T @ positions.T).T  # (N, 4)

    # Perspective division to get NDC (Normalized Device Coordinates)
    w = coords[:, 3]
    ndc = coords[:, :3] / (w.unsqueeze(1) + 1e-7)  # (N, 3) [x, y, z] in [-1, 1]

    # Convert NDC to pixel coordinates
    # NDC: x, y in [-1, 1] → Pixel: u in [0, width], v in [0, height]
    pixels_x = (ndc[:, 0] + 1.0) * 0.5 * img_width
    pixels_y = (ndc[:, 1] + 1.0) * 0.5 * img_height

    pixels = np.stack([pixels_x, pixels_y], axis=-1)  # (N, 2)
    return pixels


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

    Args:
      model: Pre-loaded patched qwen model for spatial grounding
      processor: Pre-loaded qwen processor
    """
    # skip this if no eval config is set
    if cfg.eval is None or cfg.eval.spatial is None:
        return

    bench_cfg = _build_benchmark_config(cfg, clip)
    spatial_cfg = bench_cfg.spatial_config
    assert spatial_cfg is not None, "cfg.eval.spatial must be provided"
    if spatial_cfg is None:
        return

    print(f"\n{'='*80}")
    print(f"SPATIAL EVALUATION: {clip.name}")
    print(f"{'='*80}")

    # Check if clip has spatial evaluation file specified
    if not hasattr(clip, "spatial_eval_file") or clip.spatial_eval_file is None:
        print(f"No spatial_eval_file specified for clip {clip.name}")
        print("Skipping spatial evaluation for this clip.")
        print(
            "To enable, add 'spatial_eval_file: path/to/spatial_prompts.json' to clip config"
        )
        return

    # Use configured graph_subdir or fall back to the one in spatial_cfg if specified
    graph_subdir = spatial_cfg.get("graph_subdir", bench_cfg.graph_subdir)
    graph_dir = bench_cfg.output_root / clip.name / graph_subdir

    # Load required artifacts
    splat_feats_path = graph_dir / "splat_spatial_grounding_feats.npy"
    splat_indices_path = graph_dir / "splat_spatial_grounding_indices.npy"
    positions_path = graph_dir / "positions.npy"
    # Write overlays to a dedicated spatial file (non-destructive)
    rerun_file = graph_dir / "visualization_spatial.rrd"

    # Check that all required files exist before proceeding
    missing_files = []
    if not splat_feats_path.exists():
        missing_files.append(str(splat_feats_path))
    if not splat_indices_path.exists():
        missing_files.append(str(splat_indices_path))
    if not positions_path.exists():
        missing_files.append(str(positions_path))

    if missing_files:
        print(f"ERROR: Missing required spatial grounding files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Skipping spatial evaluation due to missing data.")
        print("Please run spatial grounding feature extraction first.")
        return

    splat_feats = np.load(splat_feats_path)  # (T, N, D)
    splat_indices = np.load(splat_indices_path)  # (N,)
    positions = np.load(positions_path)  # (M, 3), filtered gaussians

    T, N, D = splat_feats.shape

    # Get spatial eval data from file
    spatial_eval_file = Path(clip.spatial_eval_file)
    if not spatial_eval_file.exists():
        print(f"ERROR: Spatial evaluation file not found: {spatial_eval_file}")
        print("Skipping spatial evaluation for this clip.")
        return

    try:
        with open(spatial_eval_file, "r") as f:
            spatial_eval_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in spatial evaluation file: {spatial_eval_file}")
        print(f"JSON error: {e}")
        print("Skipping spatial evaluation for this clip.")
        return
    except Exception as e:
        print(f"ERROR: Could not load spatial evaluation file: {e}")
        print("Skipping spatial evaluation for this clip.")
        return

    if (
        "spatial_prompts" not in spatial_eval_data
        or "spatial_prompts_frames" not in spatial_eval_data
    ):
        print(f"ERROR: Invalid format in spatial evaluation file: {spatial_eval_file}")
        print("Expected keys: 'spatial_prompts' and 'spatial_prompts_frames'")
        print("Skipping spatial evaluation for this clip.")
        return

    spatial_prompts = spatial_eval_data["spatial_prompts"]
    spatial_prompts_frames = spatial_eval_data["spatial_prompts_frames"]

    # Basic spatial eval setup
    layers = list(map(int, spatial_cfg["layers"]))
    cmap_name = str(spatial_cfg["colormap"])  # required

    # Compute relevant dirs from cfg
    colmap_root = os.path.join(cfg.preprocessed_root, clip.name)
    image_dir = os.path.join(cfg.preprocessed_root, clip.name, bench_cfg.images_subdir)

    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"ERROR: Images directory not found: {image_dir}")
        print(
            f"Expected structure: {cfg.preprocessed_root}/{clip.name}/{bench_cfg.images_subdir}/"
        )
        print("Skipping spatial evaluation due to missing data.")
        return

    # Load all camera parameters of the appropriate clip, will need some of them depending to project 3D to 2D
    try:
        scene_info = readColmapSceneInfo(colmap_root, images=None, eval=False)
        train_cameras = scene_info.train_cameras
        test_cameras = scene_info.test_cameras
        assert len(test_cameras) == 0, "Test cameras should be empty"
    except Exception as e:
        print(f"ERROR: Could not load COLMAP scene info from {colmap_root}")
        print(f"Error: {e}")
        print("Skipping spatial evaluation due to missing data.")
        return

    # Load patched model with attentions enabled
    use_4bit = bench_cfg.use_4bit_quantization
    model, processor = get_patched_qwen_for_spatial_grounding(
        use_bnb_4bit=use_4bit, use_bnb_8bit=False
    )

    for prompt, frame in zip(spatial_prompts, spatial_prompts_frames):
        # TODO: experiment with this, might be better to choose more focused substrings
        substring = prompt

        # Convert frame to timestep
        timestep = get_timestep_from_frame(frame, image_dir=image_dir)
        print(f"timestep: {timestep}")

        proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
            timestep, train_cameras, frame
        )

        # Prepare vision features for the timestep
        vision_features = torch.tensor(splat_feats[timestep], dtype=torch.float32)

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
