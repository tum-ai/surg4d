from typing import List, Optional
from sklearn.cluster import HDBSCAN, KMeans
import torch
import argparse
import numpy as np
import torchvision
from pathlib import Path
import copy
import logging

try:
    import mmcv  # type: ignore
    from mmcv import Config as MMCVConfig  # type: ignore
except Exception:
    mmcv = None  # type: ignore
    MMCVConfig = None  # type: ignore
try:
    from mmengine.config import Config as MMEngineConfig  # type: ignore
except Exception:
    MMEngineConfig = None  # type: ignore
import rerun as rr
import random
import cv2

from eval.openclip_encoder import OpenCLIPNetwork
from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render, render_opacity as gs_render_opacity
from utils.sh_utils import RGB2SH, SH2RGB
from autoencoder.model import Autoencoder
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import (
    log_points_through_time,
    log_graph_structure_through_time,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


N_TOP_FEATS = 9
N_TIMESTEPS = 15


def init_params():
    """Setup parameters similar to the train_eval.sh script"""
    parser = argparse.ArgumentParser()

    # these register parameters to the parser
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    # parser.add_argument("--skip_new_view", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", choices=["rgb", "lang"], default="rgb")
    parser.add_argument("--novideo", type=int, default=0)
    parser.add_argument("--noimage", type=int, default=0)
    parser.add_argument("--nonpy", type=int, default=0)
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--qwen_autoencoder_ckpt_path", type=str, default=None)
    # mask-based cluster filtering
    parser.add_argument("--enable_mask_filter", action="store_true")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--coverage_threshold", type=float, default=0.5)
    # Additional model paths/stages for loading multiple GaussianModels
    parser.add_argument("--rgb_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--qwen_model_path", type=str, default=None)
    parser.add_argument("--rgb_load_stage", type=str, default=None)
    parser.add_argument("--clip_load_stage", type=str, default=None)
    parser.add_argument("--qwen_load_stage", type=str, default=None)
    parser.add_argument("--store_verbose", action="store_true")

    # load config file if specified
    args = parser.parse_args()
    if args.configs:
        if MMCVConfig is not None:
            config = MMCVConfig.fromfile(args.configs)
        elif MMEngineConfig is not None:
            config = MMEngineConfig.fromfile(args.configs)
        else:
            raise ImportError(
                "Neither mmcv.Config nor mmengine.config.Config is available; install mmcv<=1.x or mmengine."
            )
        args = merge_hparams(args, config)

    return args, model_params, pipeline, hyperparam


def load_all_models(
    args: argparse.Namespace,
    model_params: ModelParams,
    pipeline: PipelineParams,
    hyperparam: ModelHiddenParams,
):
    """Load and return three GaussianModels and Scenes: clip, rgb, qwen.

    Returns:
        Tuple[GaussianModel, Scene, GaussianModel, Scene, GaussianModel, Scene]
        In order: (clip_gaussians, clip_scene, rgb_gaussians, rgb_scene, qwen_gaussians, qwen_scene)
    """
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

    return gaussians, scene, dataset


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


def verify_deformed_properties(gaussians: GaussianModel):
    with torch.no_grad():
        means3D = gaussians.get_xyz
        deforms_means3D = False
        scales = gaussians._scaling
        deforms_scales = False
        rotations = gaussians._rotation
        deforms_rotations = False
        opacity = gaussians._opacity
        deforms_opacity = False
        shs = gaussians.get_features
        deforms_shs = False
        lang = gaussians.get_language_feature
        deforms_lang = False

        for timestep in np.linspace(0, 1, N_TIMESTEPS):
            time = torch.full(
                (means3D.shape[0], 1),
                float(timestep),
                device=means3D.device,
                dtype=means3D.dtype,
            )

            (
                means3D_final,
                scales_final,
                rotations_final,
                opacity_final,
                shs_final,
                lang_final,
                _,
            ) = gaussians._deformation(
                means3D, scales, rotations, opacity, shs, lang, time
            )

            if not torch.allclose(means3D_final, means3D):
                deforms_means3D = True
            if not torch.allclose(scales_final, scales):
                deforms_scales = True
            if not torch.allclose(rotations_final, rotations):
                deforms_rotations = True
            if not torch.allclose(opacity_final, opacity):
                deforms_opacity = True
            if not torch.allclose(shs_final, shs):
                deforms_shs = True
            if not torch.allclose(lang_final, lang):
                deforms_lang = True

        print("deforms means3D:", deforms_means3D)
        print("deforms scales:", deforms_scales)
        print("deforms rotations:", deforms_rotations)
        print("deforms opacity:", deforms_opacity)
        print("deforms shs:", deforms_shs)
        print("deforms lang:", deforms_lang)


def positions_at_timestep(gaussians: GaussianModel, timestep: float, scene: Scene):
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


def cluster_gaussians(gaussians: GaussianModel, scene: Scene):
    # position
    pos_through_time = np.concatenate(
        [
            normalize_indep_dim(positions_at_timestep(gaussians, t, scene))
            for t in [0.0, 0.5, 1.0]
        ],
        axis=-1,
    )
    pos_through_time /= 9

    # instance qwen
    instance_features = gaussians.get_language_feature[:, 3:].detach().cpu().numpy()
    instance_features = normalize_dep_dim(instance_features)
    instance_features /= 3

    clustering_feats = np.concatenate(
        [2 * pos_through_time, 1 * instance_features], axis=1
    )  # factor is how the lf's are weighted to the spatial positions
    clusters = HDBSCAN(
        min_cluster_size=250, metric="euclidean", min_samples=30
    ).fit_predict(clustering_feats)

    return clusters


def filter_clusters(clusters, gaussians, scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, 0.0, scene))
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
            # if std_lang < 0.1:
            #     clusters[cluster_mask] = -1
            #     logger.info("\tFiltered because std_lang < 0.1")
            #     continue
            # if opacity < 0.05:  # 0.75 too much
            #     clusters[cluster_mask] = -1
            #     logger.info("\tFiltered because opacity < 0.4")
            #     continue
            if cluster_mask.sum() < 30:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because cluster contained <30 gaussians")
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
):
    save_dir = out / "c_renders"
    save_dir.mkdir(parents=True, exist_ok=True)

    # pick random views
    test_cams = scene.getVideoCameras()  # test + train
    random_idx = random.sample(range(len(test_cams)), args.num_views)
    cams = [test_cams[i] for i in random_idx]

    # evenly spaced timesteps
    timesteps = np.linspace(0, 1, args.num_views, dtype=np.float32)

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


# def decode_clip(lfs: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
#     BATCH_SIZE = 1024

#     ae = Autoencoder(
#         encoder_hidden_dims=[256, 128, 64, 32, 3],
#         decoder_hidden_dims=[16, 32, 64, 128, 256, 512],
#         feature_dim=512,
#     ).to("cuda")
#     ae.load_state_dict(torch.load(args.clip_autoencoder_ckpt_path, map_location="cuda"))
#     ae.eval()

#     decoded_lfs = []
#     with torch.no_grad():
#         for i in range(0, lfs.shape[0], BATCH_SIZE):
#             batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
#             decoded_lfs.append(ae.decode(batch))
#     decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
#     decoded_lfs = np.concatenate(decoded_lfs, axis=0)
#     decoded_lfs = decoded_lfs / np.linalg.norm(decoded_lfs, axis=-1, keepdims=True)
#     return decoded_lfs


def decode_qwen(lfs: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    BATCH_SIZE = 1024

    ae = QwenAutoencoder(
        input_dim=3584,
        latent_dim=3,
    ).to("cuda")
    ae.load_state_dict(torch.load(args.qwen_autoencoder_ckpt_path, map_location="cuda"))
    ae.eval()

    decoded_lfs = []
    with torch.no_grad():
        for i in range(0, lfs.shape[0], BATCH_SIZE):
            batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
            decoded_lfs.append(ae.decode(batch))
    decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
    decoded_lfs = np.concatenate(decoded_lfs, axis=0)
    return decoded_lfs


# def cluster_clip_features(
#     gaussians: GaussianModel, clusters: np.ndarray, args: argparse.Namespace
# ) -> np.ndarray:
#     # get average language feature weighted by opacity
#     weighted_cluster_lfs = []
#     n_nodes = len(np.unique(clusters))
#     opacities = gaussians.get_opacity.detach().cpu().numpy()
#     decoded_lfs = decode_clip(
#         gaussians.get_language_feature, args
#     )  # decode before aggregation, works slightly better but not much
#     for cluster_id in range(n_nodes):
#         cluster_mask = clusters == cluster_id
#         cluster_opacities = opacities[cluster_mask]
#         cluster_lfs = decoded_lfs[cluster_mask]
#         cluster_lf = (cluster_lfs * cluster_opacities).sum(
#             axis=0
#         ) / cluster_opacities.sum()
#         cluster_lf = cluster_lf / np.linalg.norm(cluster_lf)
#         weighted_cluster_lfs.append(cluster_lf)

#     lfs_weighted_centroids = np.stack(weighted_cluster_lfs)

#     return lfs_weighted_centroids


# def cluster_qwen_features(
#     qwen_g: GaussianModel,
#     rgb_g: GaussianModel,
#     clusters: np.ndarray,
#     args: argparse.Namespace,
# ) -> np.ndarray:
#     """Extract top N_TOP_FEATS qwen features by opacity for each cluster.

#     Args:
#         qwen_g (GaussianModel): Gaussian model of qwen features.
#         rgb_g (GaussianModel): Gaussian model of rgb images.
#         clusters (np.ndarray): Cluster ids.
#         args (argparse.Namespace): Arguments.

#     Returns:
#         Dict[int, np.ndarray]: map of cluster id to torch tensor of shape (n_feats, 3584)
#         where n_feats is min(N_TOP_FEATS, n_cluster_gaussians)
#     """
#     n_nodes = len(np.unique(clusters))
#     opacities = rgb_g.get_opacity.squeeze()
#     top_feats_per_cluster = {}
#     for cluster_id in range(n_nodes):
#         cluster_mask = torch.tensor(clusters) == cluster_id
#         cluster_opacities = opacities[cluster_mask]
#         cluster_lfs = qwen_g.get_language_feature[cluster_mask]
#         top_indices = torch.topk(cluster_opacities, min(cluster_opacities.shape[0], N_TOP_FEATS)).indices
#         top_feats_per_cluster[cluster_id] = cluster_lfs[top_indices]
#     top_feats_per_cluster = {
#         k: decode_qwen(v, args)
#         for k, v in top_feats_per_cluster.items()
#     }
#     return top_feats_per_cluster


def cluster_qwen_features(
    gaussians: GaussianModel,
    clusters: np.ndarray,
    args: argparse.Namespace,
    max_f_per_cluster: int = 40,
    min_f_per_cluster: int = 4,
) -> dict[int, np.ndarray]:
    """Extract qwen features from each cluster by sub-clustering its Qwen features.

    Args:
        qwen_g (GaussianModel): Gaussian model of qwen features.
        rgb_g (GaussianModel): Gaussian model of rgb images.
        clusters (np.ndarray): Cluster ids.
        args (argparse.Namespace): Arguments.

    Returns:
        Dict[int, np.ndarray]: map of cluster id to torch tensor of shape (n_feats, 3584)
        where n_feats is min(N_TOP_FEATS, n_cluster_gaussians)
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
                    a_min=min_f_per_cluster,
                    a_max=max_f_per_cluster,
                ),
                replace=True,
            )
        ]
        feature_selection = torch.tensor(random_sample, device="cuda")
        cluster_feats[cluster_id] = feature_selection.float()

    feats_per_cluster = {k: decode_qwen(v, args) for k, v in cluster_feats.items()}
    return feats_per_cluster

    # cluster_feats = {}
    # hdb = HDBSCAN(
    #     min_cluster_size=20, store_centers="centroid"
    # )  # store_centers="medioid" would be slower, but guerantee an observed feature
    # for cluster_id in np.unique(clusters):
    #     cluster_mask = torch.tensor(clusters) == cluster_id
    #     cluster_lfs = qwen_g.get_language_feature[cluster_mask][:, :3].detach().cpu().numpy()
    #     sub_clusters = hdb.fit_predict(
    #         cluster_lfs
    #     )  # -1 for unmatched feats (not belonging to clear cluster) and 0-n for clusters
    #     if len(np.unique(sub_clusters)) == 1:
    #         print(
    #             f"HDBSCAN couldn't subcluster features in cluster {cluster_id} containing {cluster_lfs.shape[0]} gaussians, using KMeans on this cluster instead"
    #         )
    #         kmeans = KMeans(
    #             n_clusters=min(bg_f_per_cluster, cluster_lfs.shape[0]), random_state=42
    #         ).fit(cluster_lfs)
    #         normed_means = kmeans.cluster_centers_ / np.linalg.norm(
    #             kmeans.cluster_centers_, axis=-1, keepdims=True
    #         )
    #         cluster_feats[cluster_id] = torch.as_tensor(normed_means, device="cuda")
    #         continue
    #     # means = []
    #     # for sub_cluster_id in np.unique(sub_clusters)[1:]:
    #     #     feature_mean = cluster_lfs[sub_clusters == sub_cluster_id].mean(axis=0)
    #     #     means.append(feature_mean / np.linalg.norm(feature_mean))
    #     # means = np.stack(means)
    #     means = hdb.centroids_ / np.linalg.norm(
    #         hdb.centroids_
    #     )  # subsitute hdb.medioids_ for medioids

    #     # Using KMeans on the unclustered gaussians for a representative sample

    #     if (sub_clusters == -1).sum() < bg_f_per_cluster:
    #         bg_means = torch.as_tensor(cluster_lfs[sub_clusters == -1], device="cuda")
    #     else:
    #         kmeans = KMeans(n_clusters=bg_f_per_cluster, random_state=42).fit(
    #             cluster_lfs[sub_clusters == -1]
    #         )
    #         bg_means = kmeans.cluster_centers_ / np.linalg.norm(
    #             kmeans.cluster_centers_, axis=-1, keepdims=True
    #         )

    #     feature_selection = torch.cat(
    #         (
    #             torch.as_tensor(
    #                 means, device="cuda"
    #             ),  # attach the means of the feature sub-clusters per cluster
    #             torch.as_tensor(
    #                 bg_means, device="cuda"
    #             ),  # also attach the unclustered features as they contain context information
    #         )
    #     )

    #     # fill up feature selection to at least min_f_per_cluster, if necessary
    #     if feature_selection.shape[0] < min_f_per_cluster:
    #         feature_selection = torch.cat(
    #             (
    #                 feature_selection,
    #                 torch.as_tensor(
    #                     cluster_lfs[
    #                         np.random.randint(
    #                             0,
    #                             cluster_lfs.shape[0],
    #                             min_f_per_cluster - feature_selection.shape[0],
    #                         )
    #                     ],
    #                     device="cuda",
    #                 ),
    #             )
    #         )

    #     cluster_feats[cluster_id] = feature_selection.float()

    # feats_per_cluster = {k: decode_qwen(v, args) for k, v in cluster_feats.items()}
    # return feats_per_cluster


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


def timestep_graph(positions, clusters):
    n_nodes = len(np.unique(clusters))
    means = np.stack([positions[clusters == i].mean(0) for i in range(n_nodes)])
    covs = np.stack([np.cov(positions[clusters == i].T) for i in range(n_nodes)])

    distances = np.empty((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i, j] = bhattacharyya_coefficient(
                means[i], covs[i], means[j], covs[j]
            )

    A = np.where(distances >= 0.05, distances, 0)
    return A


def lerf_relevancies(lfs: np.ndarray, queries: List[str], canonical_corpus: List[str]):
    """Compute LERF relevance scores for a set of language features.

    Args:
        lfs (np.ndarray): Language features to compute relevance scores for. (n_lfs, dim)

    Returns:
        np.ndarray: Relevance scores for the language features. (n_queries, n_lfs)
    """
    ocn = OpenCLIPNetwork(device="cuda", canonical_corpus=canonical_corpus)
    ocn.set_positives(queries)

    lfs = torch.tensor(lfs).to("cuda")

    lerf_relevancies = []
    for i in range(len(queries)):
        r = ocn.get_relevancy(lfs, i)
        r = r[:, 0].detach().cpu().numpy()
        lerf_relevancies.append(r)
    lerf_relevancies = np.stack(lerf_relevancies)  # (n_queries, n_lfs)

    return lerf_relevancies


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of same shape (intersection/union)."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_mask_coverage(cluster_mask: np.ndarray, instance_mask: np.ndarray) -> float:
    """Coverage = |cluster ∩ instance| / |instance|.

    Returns 1.0 when the instance is fully covered by the cluster, even if the
    cluster spills beyond the instance. Use this when you want "100 when a mask
    is covered fully".
    """
    if cluster_mask.shape != instance_mask.shape:
        return 0.0
    inter = np.logical_and(cluster_mask, instance_mask).sum()
    denom = instance_mask.sum()
    if denom == 0:
        return 0.0
    return float(inter) / float(denom)


def _find_mask_root_dir(source_path: str) -> Optional[Path]:
    """Try common directories for watershed masks under source_path."""
    candidates = [
        "instance_masks",
    ]
    for name in candidates:
        p = Path(source_path) / name
        if p.exists():
            return p
    return None


def _split_into_instances(
    binary_mask: np.ndarray, min_area: int = 50
) -> List[np.ndarray]:
    """Split a binary mask into connected-component instances.

    Args:
        binary_mask: HxW boolean or 0/1 array
        min_area: discard tiny components below this pixel area
    """
    if binary_mask.dtype != np.uint8:
        bm = (binary_mask.astype(np.uint8) > 0).astype(np.uint8)
    else:
        bm = (binary_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bm, connectivity=8)
    instances: List[np.ndarray] = []
    for lid in range(1, num):
        comp = labels == lid
        if comp.sum() >= min_area:
            instances.append(comp)
    return instances


def _load_watershed_instance_masks(
    cam: Camera, mask_root: Path, min_component_area: int = 50
) -> List[np.ndarray]:
    """Load instance masks strictly from .npy files in mask_root.

    No PNG support and no directory fallback. If the .npy for this frame is
    not found, return [].
    """
    stems = set([cam.image_name, Path(cam.image_name).stem])
    import re

    m = re.search(r"(\d+)$", Path(cam.image_name).stem)
    if m:
        idx = int(m.group(1))
        stems.update([f"frame_{idx}", f"frame_{idx:06d}", f"{idx:06d}"])

    # Prefer strict frame_######.npy, then other .npy stems
    candidates = [mask_root / f"{s}.npy" for s in stems]
    chosen: Optional[Path] = None
    for pref in [r"^frame_\d{6}\.npy$", r"^frame_\d+\.npy$"]:
        for fp in candidates:
            if fp.exists() and re.match(pref, fp.name):
                chosen = fp
                break
        if chosen is not None:
            break
    if chosen is None:
        for fp in candidates:
            if fp.exists():
                chosen = fp
                break
    if chosen is None:
        return []

    arr = np.load(str(chosen), allow_pickle=True)
    masks: List[np.ndarray] = []
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
    if arr.ndim == 2:
        labels = arr
        for v in np.unique(labels):
            if v == 0:
                continue
            m = labels == v
            if m.any():
                masks.extend(_split_into_instances(m, min_area=min_component_area))
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
            if m.any():
                masks.extend(_split_into_instances(m, min_area=min_component_area))
    return masks


def _npy_mask_exists_for_cam(cam: Camera, mask_root: Path) -> bool:
    import re

    stems = set([cam.image_name, Path(cam.image_name).stem])
    m = re.search(r"(\d+)$", Path(cam.image_name).stem)
    if m:
        idx = int(m.group(1))
        stems.update([f"frame_{idx}", f"frame_{idx:06d}", f"{idx:06d}"])
    for s in stems:
        if (mask_root / f"{s}.npy").exists():
            return True
    return False


def _render_cluster_binary_mask(
    gaussians: GaussianModel,
    cluster_mask_tensor: torch.Tensor,
    cam: Camera,
    pipe: PipelineParams,
    args: argparse.Namespace,
    dataset: ModelParams,
) -> np.ndarray:
    """Render a single cluster to a binary mask via opacity rendering (>0)."""
    # Work on a copy so we don't mutate the original model
    g_copy: GaussianModel = copy.deepcopy(gaussians)
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
    img = torch.clamp(pkg["render"], 0.0, 1.0)
    # To HxW boolean mask; any positive value considered foreground
    mask = (img.sum(dim=0) > 0).detach().cpu().numpy()
    return mask


def filter_clusters_by_mask_render(
    gaussians: GaussianModel,
    clusters: np.ndarray,
    scene: Scene,
    args: argparse.Namespace,
    dataset: ModelParams,
    pipe: PipelineParams,
    mask_dir: Optional[str] = None,
    iou_threshold: float = 0.5,  # TODO: tune this
    num_views: Optional[int] = None,
) -> np.ndarray:
    """Filter clusters by rendering them and comparing against watershed instance masks with IoU.

    Rules:
    - If IoU with all instance masks is below threshold in all views -> discard cluster.
    - If IoU >= threshold with more than one instance mask in any view -> discard cluster (ambiguous).
    - Otherwise keep cluster. Cluster ids are remapped to be contiguous.
    """
    unique_ids = [cid for cid in np.unique(clusters) if cid >= 0]
    if len(unique_ids) == 0:
        return clusters

    # Require explicit mask_dir; no automatic fallback
    if mask_dir is None:
        logger.warning("mask_dir not provided; skipping mask-based filtering.")
        return clusters
    mask_root = Path(mask_dir)
    if not mask_root.exists():
        logger.warning(
            f"Mask directory not found: {mask_root}; skipping mask-based filtering."
        )
        return clusters

    # Select views
    video_cams = scene.getVideoCameras()
    total_views = len(video_cams)
    if total_views == 0:
        logger.warning("No video cameras available; skipping mask-based filtering.")
        return clusters
    # Restrict views strictly to those with .npy masks available
    cams_with_masks = [c for c in video_cams if _npy_mask_exists_for_cam(c, mask_root)]
    if not cams_with_masks:
        logger.warning(
            "No per-frame .npy masks found for any view; skipping mask-based filtering."
        )
        return clusters
    k = args.num_views if num_views is None else num_views
    k = max(1, min(k, len(cams_with_masks)))
    view_indices = random.sample(range(len(cams_with_masks)), k)
    cams = [cams_with_masks[i] for i in view_indices]

    # Decide keep/discard per cluster (allow multiple mask intersections; single good coverage is enough)
    keep_cluster: dict[int, bool] = {}
    for cid in unique_ids:
        cid_mask = torch.tensor(clusters == cid, device="cuda")
        all_timesteps_ok = True
        evaluated_views = 0
        for cam in cams:
            cluster_mask = _render_cluster_binary_mask(
                gaussians, cid_mask, cam, pipe, args, dataset
            )
            if cluster_mask.sum() == 0:
                # Not visible in this view, ignore this timestep
                continue
            evaluated_views += 1
            masks = _load_watershed_instance_masks(cam, mask_root)
            if not masks:
                # Mask is required per timestep; if missing, fail this cluster
                all_timesteps_ok = False
                break
            # Threshold must be met by a single mask (no summation), per timestep
            timestep_ok = any(
                compute_mask_coverage(cluster_mask, m) >= iou_threshold for m in masks
            )
            if not timestep_ok:
                all_timesteps_ok = False
                break
        keep = all_timesteps_ok and evaluated_views > 0
        keep_cluster[cid] = keep

    # Remap cluster ids: kept -> 0..K-1, discarded -> -1
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
        if c in new_id_map:
            new_clusters[i] = new_id_map[c]
        else:
            new_clusters[i] = -1
    return new_clusters


def main():
    # determistic seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # mock render.py config
    args, model_params, pipeline, hyperparam = init_params()

    # construct all objects
    gaussians, scene, dataset = load_all_models(
        args, model_params, pipeline, hyperparam
    )

    # gaussian filtering
    # TODO we should do opacity times feature norm
    # opacity_mask = (gaussians.get_opacity >= 0.03).squeeze()
    opacity_mask = (gaussians.get_opacity >= 0.5).squeeze()
    # opacity_mask = (gaussians._opacity > -3.0).squeeze()
    # opacity_mask = (gaussians.get_opacity > 0.05).squeeze()
    inst_feat_norm_mask = (
        gaussians.get_language_feature[:, 3:].norm(dim=-1) > 0.05
    ).squeeze()
    filter_gaussians(gaussians, opacity_mask & inst_feat_norm_mask)
    # filter_gaussians(gaussians, opacity_mask)\\
    print(
        f"Preliminary gaussian filtering left {gaussians._xyz.shape[0]} for clustering"
    )

    # normalize language features
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

    # cluster, optional mask-based cluster filtering, filter gaussians not in clusters
    clusters = cluster_gaussians(gaussians, scene=scene)
    if args.enable_mask_filter:
        clusters = filter_clusters_by_mask_render(
            gaussians=gaussians,
            clusters=clusters,
            scene=scene,
            args=args,
            dataset=dataset,
            pipe=pipeline,
            mask_dir=args.mask_dir,
            iou_threshold=args.coverage_threshold,
            num_views=args.num_views,
        )
    filter_clusters_by_mask_render(
        gaussians, clusters, scene, args, dataset, pipeline, mask_dir=args.mask_dir
    )
    cluster_mask = clusters >= 0
    filter_gaussians(gaussians, cluster_mask)
    clusters = clusters[cluster_mask]
    # palette = set_cluster_colors(gaussians, clusters)
    # clusters[clusters == -1] = clusters.max()+1
    cluster_colors, palette = clusters_to_rgb(clusters)

    print(
        f"Cluster filtering left {len(np.unique(clusters))} clusters containing {gaussians._xyz.shape[0]} gaussians"
    )

    # cluster features
    timesteps = np.linspace(0, 1, N_TIMESTEPS)
    qwen_per_cluster = cluster_qwen_features(gaussians, clusters, args)
    pos_through_time = np.stack(
        [positions_at_timestep(gaussians, t, scene) for t in timesteps]
    )
    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(pos_through_time, clusters)

    # graph
    graphs = np.stack(
        [timestep_graph(pos_through_time[i], clusters) for i in range(len(timesteps))]
    )

    # query correspondences
    # gaussian_lfs = decode_lfs(gaussians.get_language_feature, args)
    # # queries = ["hand", "egg"]
    # # canonical_corpus = ["object", "things", "stuff", "texture"]
    # queries = ["gallbladder", "liver"]
    # canonical_corpus = ["object", "things", "stuff", "texture", "surgery", "body", "anatomy", "medical"]
    # gaussian_scores = lerf_relevancies(gaussian_lfs, queries, canonical_corpus)
    # cluster_scores = lerf_relevancies(clip_features, queries, canonical_corpus)

    # print("hist:", torch.histogram((gaussians.get_scaling.min(dim=1).values / gaussians.get_scaling.max(dim=1).values).detach().cpu(), bins=10))
    # exit(0)

    # verify_deformed_properties(gaussians); exit(0)

    # render and save everything
    out = Path(args.model_path) / "graph"

    out.mkdir(parents=True, exist_ok=True)
    if args.store_verbose:
        render_and_save_all(
            gaussians, pipeline, scene, args, dataset, out
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
        )  # clip features (n_filtered_gaussians, 3)
        np.save(
            out / "instance_latents.npy",
            gaussians.get_language_feature[:, 3:].detach().cpu().numpy(),
        )  # qwen features (n_filtered_gaussians, 3)
    cluster_feats_out = out / "c_qwen_feats"
    cluster_feats_out.mkdir(parents=True, exist_ok=True)
    for k, v in qwen_per_cluster.items():
        np.save(
            cluster_feats_out / f"{k}.npy", v
        )  # top qwen features of each cluster (min(100, n_cluster_gaussians), 3584)
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
    np.save(out / "opacities.npy", gaussians._opacity.detach().cpu().numpy())
    np.save(out / "positions.npy", gaussians.get_xyz.detach().cpu().numpy())
    np.save(out / "colors.npy", gaussians.get_features.detach().cpu().numpy())
    # Exit if only running script to save debugging files
    # print("Exiting after saving files"); exit()

    # visualize to rerun
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
    # log_correspondences_static(
    #     positions=gaussians.get_xyz.detach().cpu().numpy(),
    #     clusters=clusters,
    #     text_queries=queries,
    #     correspondences=gaussian_scores,
    #     corr_min=0.0,
    #     corr_max=1.0,
    # )


if __name__ == "__main__":
    main()
