from sklearn.cluster import HDBSCAN
import torch
import argparse
import os
import numpy as np
import torchvision
from pathlib import Path
import logging
import mmcv
import rerun as rr

from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from utils.sh_utils import RGB2SH
import random
# from utils.render_utils import get_state_at_time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Select which frame index to cluster and render
N_TIMESTEPS = 5  # change this to choose the frame index
TIMES = np.linspace(0 + 1e-6, 1 - 1e-6, N_TIMESTEPS)

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

    # load config file if specified
    args = parser.parse_args()
    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    return args, model_params, pipeline, hyperparam

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
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        lang = gaussians.get_language_feature
        # Ensure time has the same dtype/device as model tensors
        time = torch.full(
            (means3D.shape[0], 1), float(timestep), device=means3D.device, dtype=means3D.dtype
        )
        means3D_final, _, _, _, _, _, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )
    return means3D_final.detach().cpu().numpy()


def cluster_gaussians(gaussians: GaussianModel, timestep: float, scene: Scene):
    pos = normalize_indep_dim(positions_at_timestep(gaussians, timestep, scene))
    lf = gaussians.get_language_feature.detach().cpu().numpy()
    lf = normalize_dep_dim(lf)

    # graph = build_graph(pos, lf, k=10)
    # clusters = ng_jordan_weiss_spectral_clustering(graph, min_cluster_size=100, d_spectral=10)
    clusters = HDBSCAN(min_cluster_size=100, metric="euclidean").fit_predict(
        np.concatenate([pos, lf], axis=1)
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
            if std_lang < 0.1:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because std_lang < 0.1")
                continue
            if opacity < 0.4:
                clusters[cluster_mask] = -1
                logger.info("\tFiltered because opacity < 0.4")
                continue

            # restore contiguousness of cluster ids
            clusters[cluster_mask] = i
            i += 1

def set_cluster_colors(gaussians: GaussianModel, clusters: np.ndarray):
    colors = torch.zeros_like(gaussians._features_dc)  # outliers black
    all_clusters_mask = clusters >= 0
    cluster_colors, palette = clusters_to_rgb(clusters[all_clusters_mask])
    sh_dc = RGB2SH(cluster_colors)  # (N,3)
    colors[all_clusters_mask, 0, :] = torch.tensor(sh_dc, device=colors.device, dtype=colors.dtype)
    gaussians._features_dc.data = colors  # constant part becomes cluster color
    gaussians._features_rest.data = torch.zeros_like(
        gaussians._features_rest
    )  # higher order coefficients (handle view dependence) become 0

    return palette

def render(cam: Camera, timestep: float, gaussians: GaussianModel, pipe: PipelineParams, scene: Scene, args: argparse.Namespace, dataset: ModelParams):
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

def render_and_save_all(gaussians: GaussianModel, pipe: PipelineParams, scene: Scene, args: argparse.Namespace, dataset: ModelParams, out: Path):
    save_dir = out / "cluster_renders"
    save_dir.mkdir(parents=True, exist_ok=True)

    # pick random views
    test_cams = scene.getVideoCameras() # test + train
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

def visualize_cluster_pointcloud(gaussians: GaussianModel, clusters: np.ndarray, scene: Scene):
    rr.init("clusters", spawn=True)
    cluster_idx = clusters != -1
    cols = gaussians._features_dc.detach().cpu().numpy()[cluster_idx] * 255
    labels = [str(i) for i in clusters[cluster_idx]]
    for t in np.linspace(0, 1, 20):
        pos = positions_at_timestep(gaussians, t, scene)[cluster_idx]
        pc = rr.Points3D(
            positions=pos,
            colors=cols,
            labels=labels,
        )
        rr.log("clusters", pc)

def main():
    # determistic seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # mock render.py config
    args, model_params, pipeline, hyperparam = init_params()

    # construct all objects
    dataset = model_params.extract(args)
    pipe = pipeline.extract(args)
    hyper = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)  # type:ignore
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        load_stage=args.load_stage,
    )

    # filter gaussians, cluster, filter clusters, set cluster colors
    mask = (gaussians.get_opacity > 0.1).squeeze()
    filter_gaussians(gaussians, mask)
    clusters = cluster_gaussians(gaussians, timestep=0.0, scene=scene)
    filter_clusters(clusters, gaussians, scene)
    palette = set_cluster_colors(gaussians, clusters)

    # log clusters over time
    visualize_cluster_pointcloud(gaussians, clusters, scene)

    # render and save everything
    out = Path(args.model_path) / "graph"
    out.mkdir(parents=True, exist_ok=True)
    render_and_save_all(gaussians, pipe, scene, args, dataset, out)
    store_palette(palette, out / "cluster_palette.png")
    gaussians.save_ply(out / "clustered_gaussians.ply")

if __name__ == "__main__":
    main()