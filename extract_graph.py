from typing import List
from sklearn.cluster import HDBSCAN, KMeans
import torch
import argparse
import numpy as np
import torchvision
from pathlib import Path
import copy
import logging
import mmcv
import rerun as rr
import random
import cv2

from eval.openclip_encoder import OpenCLIPNetwork
from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
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
N_TIMESTEPS = 5


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
    parser.add_argument("--qwen_autoencoder_ckpt_path", type=str, default=None)
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
        config = mmcv.Config.fromfile(args.configs)
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

    clustering_feats = np.concatenate([pos_through_time, 2 * instance_features], axis=1)
    clusters = HDBSCAN(min_cluster_size=50, metric="euclidean").fit_predict(
        clustering_feats
    )

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
            if cluster_mask.sum() < 50:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because cluster contained <50 gaussians")
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
    random_idx = random.sample(range(len(test_cams)), N_TIMESTEPS)
    cams = [test_cams[i] for i in random_idx]

    # evenly spaced timesteps
    timesteps = np.linspace(0, 1, N_TIMESTEPS, dtype=np.float32)

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
    opacity_mask = (gaussians._opacity > -3.0).squeeze()
    # opacity_mask = (gaussians.get_opacity > 0.05).squeeze()
    inst_feat_norm_mask = (
        gaussians.get_language_feature[:, :3].norm(dim=-1) > 0.05
    ).squeeze()
    filter_gaussians(gaussians, opacity_mask & inst_feat_norm_mask)

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

    # cluster, filter clusters, filter gaussians that are not in a cluster
    clusters = cluster_gaussians(gaussians, scene=scene)
    filter_clusters(clusters, gaussians, scene)
    cluster_mask = clusters >= 0
    filter_gaussians(gaussians, cluster_mask)
    clusters = clusters[cluster_mask]
    # palette = set_cluster_colors(gaussians, clusters)
    cluster_colors, palette = clusters_to_rgb(clusters)

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
