from sklearn.cluster import HDBSCAN
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
from omegaconf import DictConfig
from tqdm import tqdm

from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
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
    os.environ["language_feature_hiddendim"] = str(cfg.splat.language_feature_hiddendim)
    os.environ["use_discrete_lang_f"] = cfg.splat.use_discrete_lang_f
    
    clip_dir = Path(clip.dir)
    output_root = Path(cfg.get("output_root", "output"))
    model_path = output_root / clip_dir.name
    
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
    qwen_ae_path = clip_dir / cfg.autoencoder.checkpoint_subdir / "best_ckpt.pth"
    
    cmd_args = [
        "-s",
        str(clip_dir),
        "--model_path",
        str(model_path),
        "--language_features_name",
        cfg.autoencoder.latent_cat_feat_subdir,
        "--feature_level",
        "0",
        "--configs",
        cfg.splat.config_path,
        "--load_stage",
        cfg.graph_extraction.load_stage,
        "--iteration",
        str(cfg.graph_extraction.iteration),
        "--qwen_autoencoder_ckpt_path",
        str(qwen_ae_path),
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


def positions_at_timestep(gaussians: GaussianModel, timestep: float):
    with torch.no_grad():
        means3D = gaussians.get_xyz
        # Short-circuit if no gaussians remain after filtering
        if means3D.shape[0] == 0:
            return means3D.detach().cpu().numpy()
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
        # Ensure language deformation is disabled for positional query
        try:
            gaussians._deformation.deformation_net.args.no_dlang = 1
        except Exception:
            pass
        means3D_final, _, _, _, _, _, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )
    return means3D_final.detach().cpu().numpy()


def cluster_gaussians(gaussians: GaussianModel, cfg: DictConfig):
    # position
    pos_through_time = np.concatenate(
        [
            normalize_indep_dim(positions_at_timestep(gaussians, t))
            for t in [0.0, 0.5, 1.0]
        ],
        axis=-1,
    )
    pos_through_time /= 9

    # instance qwen
    instance_features = gaussians.get_language_feature[:, 3:].detach().cpu().numpy()
    instance_features = normalize_dep_dim(instance_features)
    instance_features /= 3

    clustering_feats = np.concatenate([pos_through_time, 2 * instance_features], axis=1)
    clusters = HDBSCAN(
        min_cluster_size=cfg.graph_extraction.cluster_min_size,
        metric=cfg.graph_extraction.cluster_metric
    ).fit_predict(clustering_feats)

    return clusters


def filter_clusters(clusters, gaussians, scene, cfg: DictConfig):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, 0.0))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    i = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        opacity = gaussians.get_opacity[cluster_mask].mean()
        std_pos = pos[cluster_mask].std()
        std_lang = lf[cluster_mask].std()

        logger.info(
            f"Cluster {cluster_id}\tn_points {cluster_mask.sum()}\topacity {opacity:.4f}\tstd_pos {std_pos:.4f}\tstd_lang {std_lang:.4f}"
        )

        if cluster_id >= 0:
            # filter clusters
            if cluster_mask.sum() < cfg.graph_extraction.min_gaussians_per_cluster:
                clusters[cluster_mask] = -1
                logger.info(f"\tFiltered because cluster contained <{cfg.graph_extraction.min_gaussians_per_cluster} gaussians")
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
    random_idx = random.sample(range(len(test_cams)), cfg.graph_extraction.n_render_views)
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
    
    clip_dir = Path(clip.dir)
    ae_path = clip_dir / cfg.autoencoder.checkpoint_subdir / "best_ckpt.pth"

    ae = QwenAutoencoder(
        input_dim=cfg.autoencoder.full_dim,
        latent_dim=cfg.autoencoder.latent_dim,
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


def cluster_qwen_features(
    gaussians: GaussianModel,
    clusters: np.ndarray,
    cfg: DictConfig,
    clip: DictConfig,
) -> dict[int, np.ndarray]:
    """Extract qwen features from each cluster by sub-clustering its Qwen features.
    
    Args:
        gaussians (GaussianModel): Gaussian model.
        clusters (np.ndarray): Cluster ids.
        cfg (DictConfig): Hydra configuration.
        clip (DictConfig): Clip configuration.
        
    Returns:
        Dict[int, np.ndarray]: map of cluster id to torch tensor of shape (n_feats, 3584)
    """
    cluster_feats = {}
    for cluster_id in np.unique(clusters):
        cluster_mask = torch.tensor(clusters) == cluster_id
        cluster_lfs = (
            gaussians.get_language_feature[cluster_mask][:, :3].detach().cpu().numpy()
        )
        random_sample = cluster_lfs[
            np.random.choice(
                np.arange(cluster_lfs.shape[0]),
                np.clip(
                    cluster_lfs.shape[0],
                    a_min=cfg.graph_extraction.min_features_per_cluster,
                    a_max=cfg.graph_extraction.max_features_per_cluster,
                ),
                replace=True,
            )
        ]
        feature_selection = torch.tensor(random_sample, device="cuda")
        cluster_feats[cluster_id] = feature_selection.float()

    feats_per_cluster = {k: decode_qwen(v, cfg, clip) for k, v in cluster_feats.items()}
    return feats_per_cluster


def properties_through_time(positions_through_time, clusters):
    """Compute spatial cluster properties through time.
    
    Args:
        positions_through_time (np.ndarray): Positions through time. (T, N, 3)
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
    opacity_mask = (gaussians._opacity > cfg.graph_extraction.opacity_threshold).squeeze()
    inst_feat_norm_mask = (
        gaussians.get_language_feature[:, :3].norm(dim=-1) > cfg.graph_extraction.inst_feat_norm_threshold
    ).squeeze()
    filter_gaussians(gaussians, opacity_mask & inst_feat_norm_mask)

    # Normalize language features
    gaussians._language_feature = gaussians._language_feature.detach()
    patch_feats = gaussians.get_language_feature[:, :3]
    instance_feats = gaussians.get_language_feature[:, 3:]
    patch_feats = patch_feats / torch.clamp(
        patch_feats.norm(dim=-1, keepdim=True), min=1e-8
    )
    instance_feats = instance_feats / torch.clamp(
        instance_feats.norm(dim=-1, keepdim=True), min=1e-8
    )
    gaussians._language_feature[:, :3] = patch_feats
    gaussians._language_feature[:, 3:] = instance_feats

    # Cluster, filter clusters, filter gaussians that are not in a cluster
    clusters = cluster_gaussians(gaussians, cfg=cfg)
    filter_clusters(clusters, gaussians, scene, cfg)
    cluster_mask = clusters >= 0
    filter_gaussians(gaussians, cluster_mask)
    clusters = clusters[cluster_mask]
    cluster_colors, palette = clusters_to_rgb(clusters)

    # Cluster features
    timesteps = np.linspace(0, 1, cfg.graph_extraction.n_timesteps)
    qwen_per_cluster = cluster_qwen_features(gaussians, clusters, cfg, clip)
    pos_through_time = np.stack(
        [positions_at_timestep(gaussians, t) for t in timesteps]
    )
    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(pos_through_time, clusters)

    # Graph
    graphs = np.stack(
        [timestep_graph(pos_through_time[i], clusters, cfg) for i in range(len(timesteps))]
    )

    # Save outputs
    clip_dir = Path(clip.dir)
    output_root = Path(cfg.get("output_root", "output"))
    model_path = output_root / clip_dir.name
    out = Path(model_path) / "graph"

    out.mkdir(parents=True, exist_ok=True)
    if cfg.graph_extraction.store_verbose:
        render_and_save_all(
            gaussians, pipeline, scene, args, dataset, out, cfg
        )  # renders of cluster coloring
        store_palette(
            palette, out / "c_palette.png"
        )  # palette of renders (color position corresponds to cluster id)
        np.save(
            out / "clusters.npy", clusters
        )  # cluster ids after all filtering (n_filtered_gaussians,)
        np.save(
            out / "patch_latents.npy",
            gaussians.get_language_feature[:, :3].detach().cpu().numpy(),
        )  # patch features (n_filtered_gaussians, 3)
        np.save(
            out / "instance_latents.npy",
            gaussians.get_language_feature[:, 3:].detach().cpu().numpy(),
        )  # qwen features (n_filtered_gaussians, 3)
        np.save(out / "opacities.npy", gaussians._opacity.detach().cpu().numpy())
        np.save(out / "positions.npy", gaussians.get_xyz.detach().cpu().numpy())
        np.save(out / "colors.npy", gaussians.get_features.detach().cpu().numpy())
        
    cluster_feats_out = out / "c_qwen_feats"
    cluster_feats_out.mkdir(parents=True, exist_ok=True)
    for k, v in qwen_per_cluster.items():
        np.save(
            cluster_feats_out / f"{k}.npy", v
        )  # qwen features of each cluster
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
    )
    log_graph_structure_through_time(
        cluster_pos_through_time=cluster_pos_through_time,
        graphs_through_time=graphs,
    )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """Main graph extraction loop for all clips."""
    for clip in tqdm(cfg.clips, desc="Extracting graphs", unit="clip"):
        extract_graph(clip, cfg)


if __name__ == "__main__":
    main()

