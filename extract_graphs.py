from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import torch
import argparse
import numpy as np
import torchvision
from pathlib import Path
import logging
import mmcv
import rerun as rr
import random
import hydra
import os
from omegaconf import DictConfig
from tqdm import tqdm
from typing import List, Optional, Tuple
import gc
import copy

from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from gaussian_renderer import render_opacity as gs_render_opacity
from utils.sh_utils import RGB2SH
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import (
    log_points_through_time,
    log_graph_structure_through_time,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gaussian_model(
    clip: DictConfig,
    cfg: DictConfig,
):
    """Load Gaussian model and scene from checkpoint.

    Returns:
        Tuple[GaussianModel, Scene, ModelParams]: gaussians, scene, dataset
    """
    import os

    # Set up environment variables (needed for model architecture)
    os.environ["language_feature_hiddendim"] = str(
        cfg.graph_extraction.language_feature_hiddendim
    )
    os.environ["use_discrete_lang_f"] = cfg.graph_extraction.use_discrete_lang_f

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    model_path = Path(cfg.output_root) / clip.name

    # Create argument parser
    parser = argparse.ArgumentParser()

    # Register parameters
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", choices=["rgb", "lang"], default="rgb")
    parser.add_argument("--novideo", type=int, default=0)
    parser.add_argument("--noimage", type=int, default=0)
    parser.add_argument("--nonpy", type=int, default=0)
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    parser.add_argument("--qwen_autoencoder_ckpt_path", type=str, default=None)
    parser.add_argument("--rgb_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--qwen_model_path", type=str, default=None)
    parser.add_argument("--rgb_load_stage", type=str, default=None)
    parser.add_argument("--clip_load_stage", type=str, default=None)
    parser.add_argument("--qwen_load_stage", type=str, default=None)
    parser.add_argument("--store_verbose", action="store_true")

    # Build command line args
    qwen_ae_path = clip_dir / cfg.graph_extraction.checkpoint_subdir / "best_ckpt.pth"

    cmd_args = [
        "-s",
        str(clip_dir),
        "--model_path",
        str(model_path),
        "--language_features_name",
        cfg.graph_extraction.latent_cat_feat_subdir,
        "--feature_level",
        "0",
        "--configs",
        cfg.graph_extraction.config_path,
        "--load_stage",
        cfg.graph_extraction.load_stage,
        "--iteration",
        str(cfg.graph_extraction.iteration),
        "--qwen_autoencoder_ckpt_path",
        str(qwen_ae_path),
        "--no_dlang",
        "0"
        if cfg.graph_extraction.dynamic_language
        else "1",  # Pass dynamic language flag
    ]

    if cfg.graph_extraction.store_verbose:
        cmd_args.append("--store_verbose")

    # Parse arguments
    args = parser.parse_args(cmd_args)

    # Load and merge config
    if args.configs:
        config = None
        try:
            if hasattr(mmcv, "Config"):
                config = mmcv.Config.fromfile(args.configs)
        except Exception:
            pass

        if config is None:
            try:
                from mmengine.config import Config as MMEngineConfig

                config = MMEngineConfig.fromfile(args.configs)
            except Exception:
                raise ImportError(
                    "Neither mmcv.Config nor mmengine.config.Config is available"
                )
        args = merge_hparams(args, config)

    # Load model
    hyper = hyperparam.extract(args)
    dataset = model_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)  # type:ignore
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        load_stage=args.load_stage,
    )

    return gaussians, scene, dataset, args, pipeline


def filter_gaussians(gaussians: GaussianModel, mask: torch.Tensor):
    """Filter set of gaussians based on a mask.

    Args:
        gaussians (GaussianModel): The gaussian model to filter.
        mask (torch.Tensor): The mask to filter the gaussians. Shape (n_gaussians,)
    """
    for prop in dir(gaussians):
        attribute = getattr(gaussians, prop)
        a_type = type(attribute)
        if a_type == torch.Tensor or a_type == torch.nn.Parameter:
            if attribute.shape[0] == len(mask):
                setattr(gaussians, prop, attribute[mask])
                logger.info(f"Filtered {prop} with shape {attribute.shape}")


def normalize_indep_dim(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def normalize_dep_dim(x):
    return (x - x.mean()) / x.std()


def deform_at_timestep(gaussians: GaussianModel, timestep: float):
    """Extract deformed positions and language features at a specific timestep.

    Args:
        gaussians (GaussianModel): The gaussian model
        timestep (float): The timestep to extract features at

    Returns:
        tuple: (positions, language_patch, language_instance) as numpy arrays
            - positions: (N, 3)
            - language_patch: (N, 3)
            - language_instance: (N, 3)
    """
    with torch.no_grad():
        means3D = gaussians.get_xyz
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        lang = gaussians.get_language_feature

        # Ensure time has the same dtype/device as model tensors
        time = torch.full(
            (means3D.shape[0], 1),
            float(timestep),
            device=means3D.device,
            dtype=means3D.dtype,
        )

        # Single pass through deformation network
        means3D_final, _, _, _, _, lang_final, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )

        positions = means3D_final.detach().cpu().numpy()
        lang_patch = lang_final[:, :3].detach().cpu().numpy()
        lang_instance = lang_final[:, 3:].detach().cpu().numpy()
        lang_patch /= np.linalg.norm(lang_patch, axis=-1, keepdims=True)
        lang_instance /= np.linalg.norm(lang_instance, axis=-1, keepdims=True)

    return positions, lang_patch, lang_instance


def cluster_distances(
    norm_lf_at_timestep: List[np.ndarray],
    positions_at_timestep: List[np.ndarray],
    cfg: DictConfig,
):
    dist_matrix = 0
    for norm_lf in norm_lf_at_timestep:
        d_lf = 1 - (norm_lf @ norm_lf.T)
        d_lf = (d_lf - d_lf.min()) / (d_lf.max() - d_lf.min())
        dist_matrix = dist_matrix + d_lf * cfg.graph_extraction.clustering_weights.instance

    for positions in positions_at_timestep:
        d_pos = pairwise_distances(positions, positions, metric="euclidean")
        d_pos = (d_pos - d_pos.min()) / (d_pos.max() - d_pos.min())
        dist_matrix = dist_matrix + d_pos * cfg.graph_extraction.clustering_weights.position

    return dist_matrix


def cluster_gaussians(gaussians: GaussianModel, cfg: DictConfig):
    deformed_data = [deform_at_timestep(gaussians, t) for t in [0.0, 0.5, 1.0]]
    if cfg.graph_extraction.custom_cluster_metric:
        dist_matrix = cluster_distances(
            norm_lf_at_timestep=[i[2] for i in deformed_data],
            positions_at_timestep=[i[0] for i in deformed_data],
            cfg=cfg,
        )
        clusters = HDBSCAN(
            min_cluster_size=cfg.graph_extraction.cluster_min_size,
            metric="precomputed",
            min_samples=cfg.graph_extraction.cluster_min_samples,
        ).fit_predict(dist_matrix)
    else:
        # position
        pos_through_time = np.concatenate(
            [
                normalize_indep_dim(deform_at_timestep(gaussians, t)[0])
                for t in [0.0, 0.5, 1.0]
            ],
            axis=-1,
        )
        pos_through_time /= 9

        # instance language features
        _, _, instance_features = deform_at_timestep(gaussians, 0.5)
        instance_features = normalize_dep_dim(instance_features)
        instance_features /= 3

        clustering_feats = np.concatenate(
            [
                cfg.graph_extraction.clustering_weights.position * pos_through_time,
                cfg.graph_extraction.clustering_weights.instance * instance_features,
            ],
            axis=1,
        )
        clusters = HDBSCAN(
            min_cluster_size=cfg.graph_extraction.cluster_min_size,
            metric=cfg.graph_extraction.cluster_metric,
            min_samples=cfg.graph_extraction.cluster_min_samples,
        ).fit_predict(clustering_feats)

    return clusters


def filter_clusters(clusters, cfg: DictConfig):
    i = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        if cluster_id >= 0:
            # filter clusters
            if cluster_mask.sum() < cfg.graph_extraction.min_gaussians_per_cluster:
                clusters[cluster_mask] = -1
                logger.info(
                    f"\tFiltered because cluster contained <{cfg.graph_extraction.min_gaussians_per_cluster} gaussians"
                )
                continue

            # restore contiguousness of cluster ids
            clusters[cluster_mask] = i
            i += 1


def set_cluster_colors(gaussians: GaussianModel, clusters: np.ndarray):
    colors = torch.zeros_like(gaussians._features_dc)  # outliers black
    cluster_colors, palette = clusters_to_rgb(clusters)
    sh_dc = RGB2SH(cluster_colors)  # (N,3)
    colors[:, 0, :] = torch.tensor(sh_dc, device=colors.device, dtype=colors.dtype)
    gaussians._features_dc.data = colors  # constant part becomes cluster color
    gaussians._features_rest.data = torch.zeros_like(
        gaussians._features_rest
    )  # higher order coefficients (handle view dependence) become 0

    return palette


def render(
    cam: Camera,
    timestep: float,
    gaussians: GaussianModel,
    pipe: PipelineParams,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
):
    cam.time = timestep
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pkg = gs_render(
        cam,
        gaussians,
        pipe,
        background,
        None,
        stage=args.load_stage,
        cam_type=scene.dataset_type,
        args=args,
    )
    img = torch.clamp(pkg["render"], 0.0, 1.0)
    return img


def render_and_save_all(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
    out: Path,
    cfg: DictConfig,
):
    save_dir = out / "c_renders"
    save_dir.mkdir(parents=True, exist_ok=True)

    # pick random views
    test_cams = scene.getVideoCameras()  # test + train
    random_idx = random.sample(
        range(len(test_cams)), cfg.graph_extraction.n_render_views
    )
    cams = [test_cams[i] for i in random_idx]

    # evenly spaced timesteps
    timesteps = np.linspace(0, 1, cfg.graph_extraction.n_timesteps, dtype=np.float32)

    # render and save
    for i, cam in enumerate(cams):
        cam_dir = save_dir / f"cam_{i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for j, timestep in enumerate(timesteps):
            img = render(cam, timestep, gaussians, pipe, scene, args, dataset)
            torchvision.utils.save_image(img, cam_dir / f"timestep_{j:02d}.png")


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU = |A ∩ B| / |A ∪ B| for boolean masks of same shape."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_mask_coverage(cluster_mask: np.ndarray, instance_mask: np.ndarray) -> float:
    """Coverage = |cluster ∩ instance| / |instance|, clipped at 1.0.

    Both masks are expected to be the same shape as they originate from the same frame.
    """
    inter = np.logical_and(cluster_mask, instance_mask).sum()
    denom = instance_mask.sum()
    if denom == 0:
        return 0.0
    ratio = float(inter) / float(denom)
    return min(ratio, 1.0)


# Instance mask paths are mapped deterministically from frame indices: frame_{idx:06d}.npy


def _render_cluster_binary_mask(
    gaussians: GaussianModel,
    cluster_mask_tensor: torch.Tensor,
    cam: Camera,
    pipe: PipelineParams,
    args: argparse.Namespace,
    dataset: ModelParams,
) -> np.ndarray:
    """Render a single cluster to a binary mask via opacity rendering (>0)."""
    # Use a shallow copy to avoid PyTorch tensor deepcopy limitations while
    # allowing us to swap tensor attributes on the copy only.
    g_copy: GaussianModel = copy.copy(gaussians)
    filter_gaussians(g_copy, cluster_mask_tensor)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pkg = gs_render_opacity(
        cam,
        g_copy,
        pipe,
        background,
        None,
        stage=args.load_stage,
        cam_type=None,
        args=args,
    )
    # render_opacity returns a tensor directly (C,H,W). Not a dict.
    img = torch.clamp(pkg, 0.0, 1.0)
    mask = (img.sum(dim=0) > 0).detach().cpu().numpy()
    return mask


def filter_clusters_by_masks(
    gaussians: GaussianModel,
    clusters: np.ndarray,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
    pipe: PipelineParams,
    cfg: DictConfig,
    clip: DictConfig,
) -> np.ndarray:
    """Filter clusters by comparing rendered cluster masks to instance masks.

    Metric is controlled by cfg.graph_extraction.mask_filtering.metric (coverage|iou).
    Threshold by cfg.graph_extraction.mask_filtering.threshold.
    """
    unique_ids = [cid for cid in np.unique(clusters) if cid >= 0]
    if len(unique_ids) == 0:
        return clusters

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    mask_subdir = cfg.graph_extraction.mask_filtering.mask_subdir
    mask_dir = clip_dir / mask_subdir
    if not mask_dir.exists():
        logger.warning(
            f"Instance mask directory not found: {mask_dir}; skipping mask-based filtering."
        )
        return clusters

    video_cams = scene.getVideoCameras()
    total_views = len(video_cams)
    if total_views == 0:
        logger.warning("No video cameras available; skipping mask-based filtering.")
        return clusters

    n_timesteps = int(cfg.graph_extraction.n_timesteps)
    timesteps = np.linspace(0, 1, n_timesteps, dtype=np.float32)

    cams_and_maskpaths: List[Tuple[Camera, Optional[Path]]] = []
    for t in timesteps:
        idx = int(round(t * max(0, total_views - 1)))
        cam = video_cams[idx]
        # Expect standard naming: frame_{idx:06d}.npy
        mask_path = mask_dir / f"frame_{idx:06d}.npy"
        if not mask_path.exists():
            logger.warning(
                f"Missing instance mask for frame index {idx:06d} at {mask_path}; skipping mask-based filtering."
            )
            return clusters
        cams_and_maskpaths.append((cam, mask_path))

    metric = str(cfg.graph_extraction.mask_filtering.metric).lower()
    threshold = float(cfg.graph_extraction.mask_filtering.threshold)
    min_area = int(cfg.graph_extraction.mask_filtering.min_component_area)

    keep_cluster: dict[int, bool] = {}
    for cid in unique_ids:
        cid_mask = torch.tensor(clusters == cid, device="cuda")
        consistent = True
        matched_instance_id: Optional[int] = None

        for cam, mask_path in cams_and_maskpaths:
            cluster_mask = _render_cluster_binary_mask(
                gaussians, cid_mask, cam, pipe, args, dataset
            )
            if cluster_mask.sum() == 0:
                consistent = False
                break

            # Load instance masks from .npy: expect 2D labelmap or 3D stack
            if not mask_path.exists():
                consistent = False
                break
            arr = np.load(str(mask_path), allow_pickle=True)
            if arr.dtype == object:
                try:
                    obj = arr.item()
                    if isinstance(obj, dict):
                        if "masks" in obj:
                            arr = obj["masks"]
                        elif "instances" in obj:
                            arr = obj["instances"]
                except Exception:
                    pass

            instance_masks: List[np.ndarray] = []
            if arr.ndim == 2:
                labels = arr
                for v in np.unique(labels):
                    if v == 0:
                        continue
                    m = labels == v
                    if m.sum() >= min_area:
                        instance_masks.append(m)
            elif arr.ndim == 3:
                axis = int(np.argmin(arr.shape))
                if axis == 0:
                    it = [arr[i] for i in range(arr.shape[0])]
                elif axis == 2:
                    it = [arr[..., i] for i in range(arr.shape[2])]
                else:
                    it = [arr[:, i, :] for i in range(arr.shape[1])]
                for a in it:
                    m = a > 0
                    if m.sum() >= min_area:
                        instance_masks.append(m)
            if not instance_masks:
                consistent = False
                break

            if metric == "iou":
                scores = [compute_iou(cluster_mask, m) for m in instance_masks]
            else:
                scores = [
                    compute_mask_coverage(cluster_mask, m) for m in instance_masks
                ]

            best_idx = int(np.argmax(scores)) if scores else -1
            best_score = scores[best_idx] if best_idx >= 0 else 0.0
            if best_score < threshold:
                consistent = False
                break

            if matched_instance_id is None:
                matched_instance_id = best_idx
            elif matched_instance_id != best_idx:
                consistent = False
                break

        keep_cluster[cid] = consistent

    # Remap kept clusters to contiguous ids, filtered to -1
    new_id_map: dict[int, int] = {}
    next_id = 0
    for cid in sorted(unique_ids):
        if keep_cluster.get(cid, False):
            new_id_map[cid] = next_id
            next_id += 1
        else:
            new_id_map[cid] = -1

    new_clusters = clusters.copy()
    for i, c in enumerate(new_clusters):
        new_clusters[i] = new_id_map.get(c, -1)

    return new_clusters


def bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2):
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    Sigma1, Sigma2 = np.asarray(Sigma1), np.asarray(Sigma2)

    # Average covariance
    Sigma = 0.5 * (Sigma1 + Sigma2)

    # Cholesky factorization for stability
    L = np.linalg.cholesky(Sigma)
    # Solve for (mu2 - mu1) without explicit inverse
    diff = mu2 - mu1
    sol = np.linalg.solve(L, diff)
    sol = np.linalg.solve(L.T, sol)
    term1 = 0.125 * np.dot(diff, sol)  # (1/8) Δμᵀ Σ⁻¹ Δμ

    # log-determinants via Cholesky
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
    logdet_Sigma1 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma1))))
    logdet_Sigma2 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma2))))
    term2 = 0.5 * (logdet_Sigma - 0.5 * (logdet_Sigma1 + logdet_Sigma2))

    DB = term1 + term2
    return np.exp(-DB)  # Bhattacharyya coefficient


def decode_qwen(lfs: torch.Tensor, cfg: DictConfig, clip: DictConfig) -> np.ndarray:
    BATCH_SIZE = cfg.graph_extraction.decode_batch_size

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    ae_path = clip_dir / cfg.graph_extraction.checkpoint_subdir / "best_ckpt.pth"

    ae = QwenAutoencoder(
        input_dim=cfg.graph_extraction.full_dim,
        latent_dim=cfg.graph_extraction.latent_dim,
    ).to("cuda")
    ae.load_state_dict(torch.load(ae_path, map_location="cuda"))
    ae.eval()

    decoded_lfs = []
    with torch.no_grad():
        for i in range(0, lfs.shape[0], BATCH_SIZE):
            batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
            decoded_lfs.append(ae.decode(batch))
    decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
    decoded_lfs = np.concatenate(decoded_lfs, axis=0)
    return decoded_lfs


def clusterwise_qwen_feats(
    clusters: np.ndarray,
    cfg: DictConfig,
    clip: DictConfig,
    patch_lf: np.ndarray = None,
) -> dict[int, np.ndarray]:
    """Extract qwen features from each cluster by sub-clustering its Qwen features.

    Args:
        gaussians (GaussianModel): Gaussian model.
        clusters (np.ndarray): Cluster ids.
        cfg (DictConfig): Hydra configuration.
        clip (DictConfig): Clip configuration.
        lang_features (np.ndarray): Language features to use. If None, use static features from gaussians.

    Returns:
        Dict[int, np.ndarray]: map of cluster id to torch tensor of shape (n_feats, 3584)
    """
    cluster_feats = {}
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_lfs = patch_lf[cluster_mask]

        indices = np.random.choice(
            np.arange(cluster_lfs.shape[0]),
            np.clip(
                cluster_lfs.shape[0],
                cfg.graph_extraction.min_features_per_cluster,
                cfg.graph_extraction.max_features_per_cluster,
            ),
            replace=True,
        )
        feats = decode_qwen(
            torch.tensor(cluster_lfs[indices], device="cuda", dtype=torch.float32),
            cfg,
            clip,
        )
        cluster_feats[cluster_id] = feats

    return cluster_feats


def properties_through_time(positions_through_time: np.ndarray, clusters: np.ndarray):
    """Compute spatial cluster properties through time.

    Args:
        gaussians (GaussianModel): Gaussian model.
        clusters (np.ndarray): Cluster ids through time. (T, N)

    Returns:
        np.ndarray: Centroid through time. (T, C, 3)
        np.ndarray: Center through time. (T, C, 3)
        np.ndarray: Extent through time. (T, C, 3)
    """
    cluster_ids = np.unique(clusters)

    centroid = np.empty((len(positions_through_time), len(cluster_ids), 3))
    center = np.empty((len(positions_through_time), len(cluster_ids), 3))
    extent = np.empty((len(positions_through_time), len(cluster_ids), 3))
    for t in range(len(positions_through_time)):
        for i in range(len(cluster_ids)):
            pos = positions_through_time[t][clusters == cluster_ids[i]]
            centroid[t, i] = pos.mean(0)
            center[t, i] = (pos.max(0) + pos.min(0)) / 2
            extent[t, i] = pos.max(0) - pos.min(0)

    return centroid, center, extent


def timestep_graph(positions, clusters, cfg: DictConfig):
    n_nodes = len(np.unique(clusters))
    means = np.stack([positions[clusters == i].mean(0) for i in range(n_nodes)])
    covs = np.stack([np.cov(positions[clusters == i].T) for i in range(n_nodes)])

    distances = np.empty((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i, j] = bhattacharyya_coefficient(
                means[i], covs[i], means[j], covs[j]
            )

    A = np.where(distances >= cfg.graph_extraction.graph_edge_threshold, distances, 0)
    return A


def splat_spatial_grounding_feats(
    gaussians: GaussianModel, cfg: DictConfig, timesteps: List[float], clip: DictConfig
):
    """Sample qwen feats throughout the whole scene.

    Args:
        gaussians (GaussianModel): Gaussian model.
        cfg (DictConfig): Hydra configuration.
        timesteps (List[float]): Timesteps to extract features at.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of numpy array of shape (T, n_feats, 3584) and numpy array of shape (n_feats)
            - feats: numpy array of shape (T, n_feats, 3584)
            - indices: numpy array of shape (n_splat_points) into the whole splat
    """
    patch_feats = [deform_at_timestep(gaussians, t)[1] for t in timesteps]
    # indices = np.random.choice(
    #     gaussians.get_xyz.shape[0],
    #     cfg.graph_extraction.spatial_grounding.n_splat_points,
    #     replace=False,
    # )
    # Select top-k most opaque gaussians (deterministic) instead of random sampling
    k = int(cfg.graph_extraction.spatial_grounding.n_splat_points)
    total = int(gaussians.get_xyz.shape[0])
    k = min(k, total)
    # get_opacity is (N,1) or (N,), pick as 1D numpy array
    opac = gaussians.get_opacity.squeeze().detach().cpu().numpy()
    order = np.argsort(opac)  # ascending
    indices = order[-k:][::-1]  # top-k descending

    patch_feats = [i[indices] for i in patch_feats]
    decoded_patch_feats = [
        decode_qwen(torch.tensor(i, device="cuda", dtype=torch.float32), cfg, clip)
        for i in patch_feats
    ]
    return np.stack(decoded_patch_feats), indices


def cluster_spatial_grounding_feats(
    gaussians: GaussianModel,
    cfg: DictConfig,
    timesteps: List[float],
    clip: DictConfig,
    clusters: np.ndarray,
):
    """Extract cluster-level spatial grounding features and their indices.

    Args:
        gaussians (GaussianModel): Gaussian model.
        cfg (DictConfig): Hydra configuration.
        timesteps (List[float]): Timesteps to extract features at.
        clip (DictConfig): Clip configuration.
        clusters (np.ndarray): Cluster ids.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Tuple of dictionaries
        - feat_dict: Map of cluster id to numpy array of shape (T, n_feats, 3584)
        - indices_dict: Map of cluster id to numpy array of shape (n_feats) (ATTENTION: indices are into cluster gaussians, not the whole splat)
    """
    lf_patch_through_time = [deform_at_timestep(gaussians, t)[1] for t in timesteps]
    feat_dict, indices_dict = {}, {}
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        indices = np.random.choice(
            np.arange(cluster_mask.sum()),
            min(
                cluster_mask.sum(),
                cfg.graph_extraction.spatial_grounding.n_cluster_points,
            ),
            replace=False,
        )
        indices_dict[str(cluster_id)] = indices
        cluster_feats = []
        for t in range(len(timesteps)):
            feats = lf_patch_through_time[t][cluster_mask][indices]
            feats = decode_qwen(
                torch.tensor(feats, device="cuda", dtype=torch.float32), cfg, clip
            )
            cluster_feats.append(feats)
        cluster_feats = np.stack(cluster_feats)
        feat_dict[str(cluster_id)] = cluster_feats

    return feat_dict, indices_dict


def extract_graph(clip: DictConfig, cfg: DictConfig):
    """Extract scene graph from trained Gaussian Splatting model.

    Args:
        clip (DictConfig): Clip configuration
        cfg (DictConfig): Full hydra configuration
    """
    # Set deterministic seeds
    random.seed(cfg.graph_extraction.random_seed)
    np.random.seed(cfg.graph_extraction.random_seed)
    torch.manual_seed(cfg.graph_extraction.random_seed)

    # Load model
    gaussians, scene, dataset, args, pipeline = load_gaussian_model(clip, cfg)

    # Gaussian filtering
    opacity_mask = (
        gaussians.get_opacity > cfg.graph_extraction.opacity_threshold
    ).squeeze()
    inst_feat_norm_mask = (
        gaussians.get_language_feature[:, 3:].norm(dim=-1)
        > cfg.graph_extraction.inst_feat_norm_threshold
    ).squeeze()
    filter_gaussians(gaussians, opacity_mask & inst_feat_norm_mask)

    # Cluster, filter clusters, optional mask-based filtering, then drop -1s
    clusters = cluster_gaussians(gaussians, cfg=cfg)
    filter_clusters(clusters, cfg)

    # Optional: mask-based cluster filtering (coverage or IoU)
    mf = getattr(cfg.graph_extraction, "mask_filtering", None)
    if mf and bool(mf.get("enabled", False)):
        logger.info(
            "Applying mask-based cluster filtering (%s, threshold=%.3f)",
            mf.metric,
            mf.threshold,
        )
        clusters = filter_clusters_by_masks(
            gaussians=gaussians,
            clusters=clusters,
            scene=scene,
            args=args,
            dataset=dataset,
            pipe=pipeline,
            cfg=cfg,
            clip=clip,
        )
    cluster_mask = clusters >= 0
    filter_gaussians(gaussians, cluster_mask)
    clusters = clusters[cluster_mask]
    cluster_colors, palette = clusters_to_rgb(clusters)

    # Cluster features
    timesteps = np.linspace(0, 1, cfg.graph_extraction.n_timesteps)
    deformed_data = [deform_at_timestep(gaussians, t) for t in timesteps]
    pos_through_time = np.stack([i[0] for i in deformed_data])
    lf_patch_through_time = np.stack([i[1] for i in deformed_data])
    lf_instance_through_time = np.stack([i[2] for i in deformed_data])

    selected_decoded_lf_through_time = [
        clusterwise_qwen_feats(
            clusters,
            cfg,
            clip,
            patch_lf=lf_patch_through_time[t_idx],
        )
        for t_idx in range(len(timesteps))
    ]

    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(pos_through_time, clusters)

    # Graph
    graphs = np.stack(
        [
            timestep_graph(pos_through_time[i], clusters, cfg)
            for i in range(len(timesteps))
        ]
    )

    # Save outputs
    model_path = Path(cfg.output_root) / clip.name
    out = Path(model_path) / cfg.graph_extraction.graph_output_subdir

    out.mkdir(parents=True, exist_ok=True)
    if cfg.graph_extraction.store_verbose:
        render_and_save_all(
            gaussians, pipeline, scene, args, dataset, out, cfg
        )  # renders of cluster coloring
        store_palette(
            palette, out / "c_palette.png"
        )  # palette of renders (color position corresponds to cluster id)

        # Save language features (either canonical or through time, but not both)
        np.save(
            out / "patch_latents_through_time.npy", lf_patch_through_time
        )  # patch latents through time (timesteps, n_filtered_gaussians, lang_dim)
        np.save(
            out / "instance_latents_through_time.npy", lf_instance_through_time
        )  # instance latents through time (timesteps, n_filtered_gaussians, lang_dim)

        np.save(out / "opacities.npy", gaussians._opacity.detach().cpu().numpy())
        np.save(out / "colors.npy", gaussians.get_features.detach().cpu().numpy())

    cluster_feats_dict = {}
    for cluster_id in selected_decoded_lf_through_time[0].keys():
        cluster_feats_dict[str(cluster_id)] = np.stack(
            [lf_dict[cluster_id] for lf_dict in selected_decoded_lf_through_time]
        )  # shape: (timesteps, n_feats, 3584)
    np.savez(
        out / "c_qwen_feats.npz", **cluster_feats_dict
    )  # qwen features per cluster through time (cluster_id -> (timesteps, n_feats, 3584))

    np.save(
        out / "c_centroids.npy", cluster_pos_through_time
    )  # cluster centroids through time (timesteps, n_clusters, 3)
    np.save(
        out / "c_centers.npy", cluster_center_through_time
    )  # cluster centers through time (timesteps, n_clusters, 3)
    np.save(
        out / "c_extents.npy", cluster_extent_through_time
    )  # cluster extents through time (timesteps, n_clusters, 3)
    np.save(
        out / "graph.npy", graphs
    )  # adjacency matrices through time - weights are bhattacharyya coefficients (timesteps, n_clusters, n_clusters)

    # save stuff for spatial grounding
    splat_feats, splat_indices = splat_spatial_grounding_feats(
        gaussians, cfg, timesteps, clip
    )
    np.save(
        out / "splat_spatial_grounding_feats.npy", splat_feats
    )  # (T, n_splat_points, 3584)
    np.save(
        out / "splat_spatial_grounding_indices.npy", splat_indices
    )  # (n_splat_points,)

    cluster_feats, cluster_indices = cluster_spatial_grounding_feats(
        gaussians, cfg, timesteps, clip, clusters
    )
    np.savez(
        out / "cluster_spatial_grounding_feats.npz", **cluster_feats
    )  # (cluster_id -> (T, n_feats, 3584))
    np.savez(
        out / "cluster_spatial_grounding_indices.npz", **cluster_indices
    )  # (cluster_id -> (n_feats,)) (ATTENTION: indices are into cluster gaussians, not the whole splat)
    np.save(out / "positions.npy", pos_through_time) # (T, n_filtered_gaussians, 3)
    np.save(out / "clusters.npy", clusters)  # (n_filtered_gaussians,)

    # Visualize to rerun
    rr.init("clusters")
    rr.save(out / "visualization.rrd")  # save to file for offline viewing
    log_points_through_time(
        gaussians=gaussians,
        clusters=clusters,
        cluster_colors=cluster_colors,
        timesteps=timesteps,
        pos_through_time=pos_through_time,
        cluster_pos_through_time=cluster_pos_through_time,
        text_queries=None,
        cluster_correspondences=None,
        patch_lf_through_time=lf_patch_through_time,
        instance_lf_through_time=lf_instance_through_time,
    )
    log_graph_structure_through_time(
        cluster_pos_through_time=cluster_pos_through_time,
        graphs_through_time=graphs,
    )

    # Clean up GPU memory - this is critical before evaluate_clip tries to load Qwen
    del gaussians
    del scene
    del dataset
    del args
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """Main graph extraction loop for all clips."""
    # Deterministic Torch/CUDA setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for clip in tqdm(cfg.clips, desc="Extracting graphs", unit="clip"):
        extract_graph(clip, cfg)


if __name__ == "__main__":
    main()
