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
import colorsys

from scene.cameras import Camera
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from cluster_utils import store_palette, clusters_to_rgb
from scene import GaussianModel, Scene
from gaussian_renderer import render as gs_render
from utils.sh_utils import RGB2SH
from autoencoder.model import Autoencoder


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
    parser.add_argument("--autoencoder_ckpt_path", type=str, default=None)

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

def cluster_clip_features(gaussians: GaussianModel, clusters: np.ndarray, args: argparse.Namespace):
    # init ae
    ae = Autoencoder(
        encoder_hidden_dims=[256, 128, 64, 32, 3],
        decoder_hidden_dims=[16, 32, 64, 128, 256, 512],
        feature_dim=512,
    ).to("cuda")
    ae.load_state_dict(torch.load(args.autoencoder_ckpt_path, map_location='cuda'))
    ae.eval() 

    # get average language feature weighted by opacity
    weighted_cluster_lfs = []
    n_nodes = len(np.unique(clusters)) - 1
    opacities = gaussians.get_opacity
    lfs = gaussians.get_language_feature
    for cluster_id in range(n_nodes):
        cluster_mask = torch.tensor(clusters == cluster_id).to("cuda")
        cluster_opacities = opacities[cluster_mask]
        cluster_lfs = lfs[cluster_mask]
        cluster_lf = (cluster_lfs * cluster_opacities).sum(0) / cluster_opacities.sum()
        weighted_cluster_lfs.append(cluster_lf)
    
    lfs_weighted_centroids = torch.stack(weighted_cluster_lfs)

    # decode lfs
    with torch.no_grad():
        decoded_lfs = ae.decode(lfs_weighted_centroids)

    return decoded_lfs.detach().cpu().numpy()

def timestep_cluster_means(positions_through_time, clusters):
    """Returns np(timesteps, clusters, 3) cluster mean positions"""
    cluster_ids = np.unique(clusters)
    cluster_ids = cluster_ids[cluster_ids != -1]

    means = np.empty((len(positions_through_time), len(cluster_ids), 3))
    for t in range(len(positions_through_time)):
        for i in range(len(cluster_ids)):
            means[t, i] = positions_through_time[t][clusters == cluster_ids[i]].mean(0)

    return means
    
def timestep_graph(positions, clusters):
    n_nodes = len(np.unique(clusters)) - 1
    means = np.stack([positions[clusters == i].mean(0) for i in range(n_nodes)])
    covs = np.stack([np.cov(positions[clusters == i].T) for i in range(n_nodes)])

    distances = np.empty((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            distances[i, j] = bhattacharyya_coefficient(means[i], covs[i], means[j], covs[j])

    A = np.where(distances >= 0.05, distances, 0)
    return A

def visualize_rerun(gaussians: GaussianModel, clusters: np.ndarray, timesteps: np.ndarray, pos_through_time: np.ndarray, cluster_pos_through_time: np.ndarray, graphs_through_time: np.ndarray, rerun_file: Path):
    rr.init("clusters")
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    rr.save(rerun_file)

    cluster_ids = np.unique(clusters)
    cluster_ids = cluster_ids[cluster_ids != -1]

    cols = gaussians._features_dc.detach().cpu().numpy() * 255

    for t in range(len(timesteps)):
        # Set timeline for this timestep
        rr.set_time_seconds("timestep", timesteps[t])
        
        pos = pos_through_time[t]
        cluster_means = cluster_pos_through_time[t]
        A = graphs_through_time[t]

        # Visualize individual cluster points
        for c in cluster_ids:
            pc = rr.Points3D(
                positions=pos[clusters == c],
                colors=cols[clusters == c],
                radii=0.02,  # Regular point size
            )
            rr.log(f"clusters/points/cluster_{c}", pc)

        # Visualize cluster means with distinctive appearance
        if cluster_means.shape[0] > 0:
            # Use bright colors for means - cycle through distinct colors
            mean_colors = []
            for i in range(len(cluster_means)):
                # Create distinctive colors for means (bright, saturated)
                hue = (i * 360 / len(cluster_means)) % 360
                # Convert HSV to RGB (bright and saturated)
                rgb = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
                mean_colors.append([int(c * 255) for c in rgb])
            
            means_viz = rr.Points3D(
                positions=cluster_means,
                colors=mean_colors,
                radii=0.2,  # Even larger mean points
            )
            rr.log("clusters/means", means_viz)

        # Visualize graph edges with weights
        if A.shape[0] > 0:
            # Find all non-zero edges in the adjacency matrix
            edge_indices = np.where(A > 0)
            if len(edge_indices[0]) > 0:
                edge_weights = A[edge_indices]
                
                # Normalize weights for visualization
                if len(edge_weights) > 1:
                    min_weight = edge_weights.min()
                    max_weight = edge_weights.max()
                    if max_weight > min_weight:
                        normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
                    else:
                        normalized_weights = np.ones_like(edge_weights)
                else:
                    normalized_weights = np.ones_like(edge_weights)
                
                # Create edge lines
                edge_starts = []
                edge_ends = []
                edge_colors = []
                edge_radii = []
                
                for idx, (i, j) in enumerate(zip(edge_indices[0], edge_indices[1])):
                    if i < j:  # Avoid duplicate edges (since adjacency matrix is symmetric)
                        start_pos = cluster_means[i]
                        end_pos = cluster_means[j]
                        weight = normalized_weights[idx]
                        
                        edge_starts.append(start_pos)
                        edge_ends.append(end_pos)
                        
                        # Color based on weight: red for high weights, blue for low weights
                        color_intensity = int(weight * 255)
                        edge_colors.append([color_intensity, 0, 255 - color_intensity])
                        
                        # Thickness based on weight: thicker lines for higher weights
                        thickness = 0.04 + weight * 0.16  # Range from 0.01 to 0.05 (doubled thickness)
                        edge_radii.append(thickness)
                
                if edge_starts:
                    # Log edges as line strips
                    for idx, (start, end) in enumerate(zip(edge_starts, edge_ends)):
                        edge_line = rr.LineStrips3D(
                            strips=[[start, end]],
                            colors=[edge_colors[idx]],
                            radii=[edge_radii[idx]]
                        )
                        rr.log(f"clusters/edges/edge_{idx}", edge_line)

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

    # gaussian filtering
    mask = (gaussians.get_opacity > 0.1).squeeze()
    filter_gaussians(gaussians, mask)

    # cluster and filter
    clusters = cluster_gaussians(gaussians, timestep=0.0, scene=scene)
    filter_clusters(clusters, gaussians, scene)
    palette = set_cluster_colors(gaussians, clusters)

    # cluster features
    timesteps = np.linspace(0, 1, 20) 
    clip_features = cluster_clip_features(gaussians, clusters, args)
    pos_through_time = np.stack([positions_at_timestep(gaussians, t, scene) for t in timesteps])
    cluster_pos_through_time = timestep_cluster_means(pos_through_time, clusters)

    # graph
    graphs = np.stack([timestep_graph(pos_through_time[i], clusters) for i in range(len(timesteps))])

    # render and save everything
    out = Path(args.model_path) / "graph"
    out.mkdir(parents=True, exist_ok=True)
    visualize_rerun(
        gaussians=gaussians,
        clusters=clusters,
        timesteps=timesteps,
        pos_through_time=pos_through_time,
        cluster_pos_through_time=cluster_pos_through_time,
        graphs_through_time=graphs,
        rerun_file=out / "graph_visualization.rrd"
    )
    render_and_save_all(gaussians, pipe, scene, args, dataset, out)
    store_palette(palette, out / "cluster_palette.png")
    gaussians.save_ply(out / "clustered_gaussians.ply")
    np.save(out / "cluster_clip_features.npy", clip_features)
    np.save(out / "cluster_ids.npy", clusters)
    np.save(out / "cluster_centroids_per_timestep.npy", cluster_pos_through_time)
    np.save(out / "adjacency_matrices_per_timestep.npy", graphs)

if __name__ == "__main__":
    main()