import hdbscan
from pynndescent import NNDescent
import scipy.sparse as sp
from scipy.linalg import eig, eigh
from scipy.spatial import cKDTree
import torch
import argparse
import numpy as np
from pathlib import Path
import logging
import mmcv
import rerun as rr
import random
import hydra
import os
from omegaconf import DictConfig
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import GaussianModel, Scene
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import (
    log_points_through_time,
    log_graph_structure_through_time,
)
from utils.sh_utils import SH2RGB
from utils.gaussian_loading_utils import get_best_model_iteration, get_latest_model_iteration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_autoencoder_path(clip: DictConfig, cfg: DictConfig) -> Path:
    if cfg.graph_extraction.use_global_autoencoder:
        ae_path = (
            Path(cfg.preprocessed_root)
            / cfg.graph_extraction.global_autoencoder_checkpoint_dir
            / "best_ckpt.pth"
        )
    else:
        clip_dir = Path(cfg.preprocessed_root) / clip.name
        ae_path = clip_dir / cfg.graph_extraction.checkpoint_subdir / "best_ckpt.pth"
    return ae_path


def get_autoencoder(clip: DictConfig, cfg: DictConfig) -> QwenAutoencoder:
    """Get autoencoder (global or per-clip based on config)."""
    ae_path = get_autoencoder_path(clip, cfg)
    ae = QwenAutoencoder(
        input_dim=cfg.graph_extraction.full_dim,
        latent_dim=cfg.graph_extraction.latent_dim,
    ).to("cuda")
    ae.load_state_dict(torch.load(ae_path, map_location="cuda"))
    ae.eval()
    return ae


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
    os.environ["num_lang_features"] = str(cfg.graph_extraction.num_lang_features)
    os.environ["lang_feature_dim"] = str(cfg.graph_extraction.lang_feature_dim)
    # centers_num is used by discrete_coff_generator in deformation network
    if hasattr(cfg.splat, "centers_num"):
        os.environ["centers_num"] = str(cfg.splat.centers_num)

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
    parser.add_argument("--use_best_splat_checkpoint", action="store_true")

    # Build command line args
    qwen_ae_path = get_autoencoder_path(clip, cfg)

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
    if cfg.graph_extraction.use_best_splat_checkpoint:
        cmd_args.append("--use_best_splat_checkpoint")

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

    if args.iteration == -1:
        if args.use_best_splat_checkpoint:
            args.iteration = get_best_model_iteration(cfg, clip.name)
        else:
            args.iteration = get_latest_model_iteration(cfg)

    # Set centers_num from config if available (needed for discrete_coff_generator)
    # This must be set before creating GaussianModel
    if hasattr(cfg.splat, "centers_num"):
        os.environ["centers_num"] = str(cfg.splat.centers_num)

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
    n_gaussians = len(mask)

    for prop in dir(gaussians):
        # Skip @property decorated attributes
        if isinstance(getattr(type(gaussians), prop, None), property):
            continue

        attribute = getattr(gaussians, prop)
        a_type = type(attribute)
        if a_type == torch.Tensor or a_type == torch.nn.Parameter:
            if attribute.shape[0] == n_gaussians:
                setattr(gaussians, prop, attribute[mask])
            # Handle _control_point_positions_precomputed which has shape (T, N, 3)
            elif attribute.ndim == 3 and attribute.shape[1] == n_gaussians:
                setattr(gaussians, prop, attribute[:, mask, :])

    n_gaussians_new = mask.sum()
    print(
        f"[filter_gaussians] from {n_gaussians} to {n_gaussians_new} gaussians ({n_gaussians - n_gaussians_new} filtered - {n_gaussians_new / n_gaussians * 100:.2f}% left)"
    )


def deform_at_timestep(gaussians: GaussianModel, timestep: float):
    """Extract deformed positions and language features at a specific timestep.

    Args:
        gaussians (GaussianModel): The gaussian model
        timestep (float): The timestep to extract features at

    Returns:
        tuple: (positions, language_features_list) as numpy arrays
            - positions: (N, 3)
            - language_features_list: list of (N, lang_feature_dim) arrays, one per feature type
              For backward compat with 2 features, returns (positions, lang_patch, lang_instance)
    """
    num_lang_features = int(os.environ.get("num_lang_features", 2))
    lang_feature_dim = int(os.environ.get("lang_feature_dim", 3))

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

        # Normalize each language feature independently before deformation (matching renderer)
        normalized_features = []
        for i in range(num_lang_features):
            start_idx = i * lang_feature_dim
            end_idx = start_idx + lang_feature_dim
            feat = lang[:, start_idx:end_idx]
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-9)
            normalized_features.append(feat)
        lang_normalized = torch.cat(normalized_features, dim=-1)

        # Apply deformation to all Gaussians (same as renderer)
        means3D_final, _, _, _, _, lang_final, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang_normalized, time
        )

        # Replace positions for control-point-driven Gaussians with precomputed positions
        # (same logic as in gaussian_renderer/__init__.py)
        if (
            gaussians._is_control_point_driven is not None
            and gaussians._control_point_positions_precomputed is not None
        ):
            # Convert normalized time (0-1) to frame index
            time_value = time[0, 0].item()
            frame_idx = int(time_value * (gaussians._num_frames - 1))
            frame_idx = max(0, min(frame_idx, gaussians._num_frames - 1))

            # Get precomputed positions for this frame
            control_point_positions_full = (
                gaussians._control_point_positions_precomputed[frame_idx]
            )  # (N_gaussians, 3)

            # Ensure shapes match
            assert control_point_positions_full.shape[0] == means3D_final.shape[0], (
                f"Mismatch: control_point_positions_full has {control_point_positions_full.shape[0]} positions, "
                f"but means3D_final has {means3D_final.shape[0]} Gaussians"
            )

            # Replace means3D_final for control-point-driven Gaussians with precomputed positions
            means3D_final = (
                means3D_final.clone()
            )  # Clone to avoid in-place modification
            means3D_final[gaussians._is_control_point_driven] = (
                control_point_positions_full[
                    gaussians._is_control_point_driven
                ].detach()
            )

        positions = means3D_final.detach().cpu().numpy()

        # Extract each language feature and normalize independently
        # (deformation network already normalizes, but this ensures unit norm for decoder)
        lang_features = []
        for i in range(num_lang_features):
            start_idx = i * lang_feature_dim
            end_idx = start_idx + lang_feature_dim
            feat = lang_final[:, start_idx:end_idx].detach().cpu().numpy()
            # Normalize with epsilon to prevent NaN from zero-norm vectors
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-9)
            lang_features.append(feat)

    # For backward compatibility with code expecting (positions, lang_patch, lang_instance)
    if num_lang_features == 2:
        return positions, lang_features[0], lang_features[1]
    else:
        return positions, *lang_features


def laplacian_sym(A):
    if sp.isspmatrix(A):
        D_diag = np.asarray(A.sum(axis=0)).ravel()
        D_pow_neg_half_diag = np.zeros_like(D_diag)
        nonzero = D_diag != 0
        D_pow_neg_half_diag[nonzero] = D_diag[nonzero] ** -0.5
        D_pow_neg_half = sp.diags(D_pow_neg_half_diag, format="csr")
        normalized_A = D_pow_neg_half @ A @ D_pow_neg_half
        normalized_L = sp.eye(A.shape[0], format="csr") - normalized_A
    else:
        D_diag = np.sum(A, axis=0)
        D_pow_neg_half_diag = np.zeros_like(D_diag)
        nonzero = D_diag != 0
        D_pow_neg_half_diag[nonzero] = D_diag[nonzero] ** -0.5
        D_pow_neg_half = np.diag(D_pow_neg_half_diag)
        normalized_A = D_pow_neg_half @ A @ D_pow_neg_half
        normalized_L = np.eye(A.shape[0]) - normalized_A
    return normalized_L


def spectral_embeddings(L, d, normalize_rows: bool, use_symmetric_eigensolver: bool):
    """Spectral embeddings.

    Args:
        L: Laplacian matrix (n x n)
        d: dimension of embeddings (number of eigenvectors to keep), if set to None, cutoff at largest eigengap
        normalize_rows: Normalize rows of embedding
        use_symmetric_eigensolver: Use symmetric eigensolver

    Returns:
        _type_: _description_
    """
    n = L.shape[0]

    if sp.isspmatrix(L):
        # ncv = number of Lanczos vectors, must be > k and <= n
        # larger ncv = more robust but slower
        ncv = min(n, max(2 * d + 1, 40))
        if use_symmetric_eigensolver:
            eigenvalues, U = sp.linalg.eigsh(L, k=d, which="SA", ncv=ncv)
        else:
            eigenvalues, U = sp.linalg.eigs(L, k=d, which="SR", ncv=ncv)
    else:
        if use_symmetric_eigensolver:
            eigenvalues, U = eigh(L, subset_by_index=[0, d - 1])
        else:
            eigenvalues, eigenvectors = eig(L)
            U = eigenvectors[:, :d]
        assert np.allclose(eigenvalues[0], 0), (
            f"smallest eigenvalue is not 0 but {np.array(eigenvalues[0]).round(6)} with eigenvector {np.array(eigenvectors[0]).round(6)}"
        )
        assert len(eigenvalues) < 2 or not np.allclose(eigenvalues[1], 0), (
            "second smallest eigenvalue is 0 -> graph is not connected"
        )

    # normalize rows if required
    if normalize_rows:
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        T = U / norms
    else:
        T = U

    return T


def assign_noise_to_nearest(samples: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Assign noise points (-1) to their nearest non-noise neighbor's cluster."""
    noise_mask = clusters == -1
    if not noise_mask.any():
        return clusters
    valid_mask = ~noise_mask
    if not valid_mask.any():
        return clusters  # all noise, nothing to assign to
    # for each noise point, find nearest valid point
    tree = cKDTree(samples[valid_mask])
    _, nearest_idx = tree.query(samples[noise_mask], k=1)
    # map back to original indices
    valid_indices = np.where(valid_mask)[0]
    clusters[noise_mask] = clusters[valid_indices[nearest_idx]]
    return clusters


def ng_jordan_weiss_spectral_clustering(
    A, spectral_embedding_dim: int, hdbscan_args: dict, soft_clustering: bool = False
):
    """Spectral clustering with symmetric, normalized Laplacian.
    Heuristically approximates normalized cut (not principled like Shi-Malik),
    but is faster because it uses a symmetric eigensolver.

    Noise points are left as -1 for external assignment (e.g., via k-NN).
    Handles disconnected graphs by clustering each connected component separately.

    (Naming here follows: https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf)

    Args:
        A: Adjacency matrix (n x n)
        spectral_embedding_dim: Number of eigenvectors for embedding
        hdbscan_args: Arguments for HDBSCAN clustering
        soft_clustering: If True, use soft clustering to assign all points (no noise).
            If False, use regular HDBSCAN with noise points marked as -1.

    Returns:
        clusters: Cluster assignments (n,), -1 for noise (only if soft_clustering=False)
    """
    n_samples = A.shape[0]
    n_components, component_labels = sp.csgraph.connected_components(A, directed=False)

    # print component stats
    comp_sizes = np.bincount(component_labels)
    print(
        f"[spectral] {n_samples} samples, {n_components} components, min_cluster_size={hdbscan_args.min_cluster_size}"
    )
    print(f"[spectral] Largest 10 components: {sorted(comp_sizes, reverse=True)[:10]}")
    print(
        f"[spectral] Components >= min_cluster_size: {(comp_sizes >= hdbscan_args.min_cluster_size).sum()}"
    )

    # if graph connected, cluster and return
    if n_components == 1:
        L = laplacian_sym(A)
        T = spectral_embeddings(
            L,
            d=spectral_embedding_dim,
            normalize_rows=True,
            use_symmetric_eigensolver=True,
        )
        clusters = hdbscan.HDBSCAN(**hdbscan_args).fit_predict(T)
        hdbscan_noise = (clusters == -1).sum()
        if soft_clustering:
            clusters = assign_noise_to_nearest(T, clusters)
            print(
                f"[spectral] Assigned {hdbscan_noise} noise points to nearest clusters"
            )
        else:
            print(
                f"[spectral] Noise: {hdbscan_noise} total, 0 disconnected, {hdbscan_noise} HDBSCAN"
            )
        return clusters

    # if disconnected graph, cluster each component separately
    clusters = np.full(n_samples, -1, dtype=np.int32)
    cluster_offset = 0
    disconnected_noise = 0
    hdbscan_noise = 0
    for comp_id in range(n_components):
        comp_mask = component_labels == comp_id
        comp_size = comp_mask.sum()

        # leave component as noise if too small
        if (
            comp_size < hdbscan_args.min_cluster_size
            or comp_size <= spectral_embedding_dim + 1
        ):
            disconnected_noise += comp_size
            continue

        # subgraph
        comp_indices = np.where(comp_mask)[0]
        A_sub = A[comp_indices][:, comp_indices]

        # cluster
        L_sub = laplacian_sym(A_sub)
        d = min(spectral_embedding_dim, comp_size - 1)
        T_sub = spectral_embeddings(
            L_sub, d=d, normalize_rows=True, use_symmetric_eigensolver=True
        )
        comp_clusters = hdbscan.HDBSCAN(**hdbscan_args).fit_predict(T_sub)
        comp_noise = (comp_clusters == -1).sum()
        hdbscan_noise += comp_noise

        if soft_clustering:
            comp_clusters = assign_noise_to_nearest(T_sub, comp_clusters)

        # assign offsetted cluster ids to original indices
        valid_mask = comp_clusters >= 0
        clusters[comp_indices[valid_mask]] = comp_clusters[valid_mask] + cluster_offset
        cluster_offset += comp_clusters.max() + 1

    total_noise = (clusters == -1).sum()
    if soft_clustering:
        print(
            f"[spectral] Assigned {hdbscan_noise} HDBSCAN noise points to nearest clusters, {total_noise} unassigned (from {disconnected_noise} small components)"
        )
    else:
        print(
            f"[spectral] Noise: {total_noise} total, {disconnected_noise} disconnected, {hdbscan_noise} HDBSCAN"
        )
    return clusters


def dump_histogram(
    data: np.ndarray, output_path: Path, title: str, xlabel: str, sigma: float
):
    """Save a histogram of the data distribution with RBF kernel overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # histogram on primary y-axis
    counts, bins, _ = ax1.hist(
        data.ravel(),
        bins=100,
        edgecolor="black",
        alpha=0.7,
        label="Distance distribution",
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # RBF kernel on secondary y-axis
    ax2 = ax1.twinx()
    x = np.linspace(0, data.max(), 200)
    rbf = np.exp(-(x**2) / (2 * sigma**2))
    ax2.plot(x, rbf, color="tab:red", linewidth=2, label=f"RBF kernel (σ={sigma:.4f})")
    ax2.set_ylabel("RBF Similarity", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.1)

    # vertical lines for mean/std
    ax1.axvline(
        data.mean(),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={data.mean():.4f}",
    )
    ax1.axvline(
        data.mean() + data.std(),
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"std={data.std():.4f}",
    )
    ax1.axvline(data.mean() - data.std(), color="orange", linestyle=":", linewidth=1.5)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[build_graph] Saved histogram to {output_path}")


def build_similarity_graph(
    positions: np.ndarray,
    normalized_lf: np.ndarray,
    k: int,
    weight_pos: float,
    weight_lf: float,
    sigma_pos_factor: float,
    sigma_lf_factor: float,
    histogram_output_dir: Path | None = None,
):
    """Build a weighted, symmetric knn graph

    Returns:
        graph: Sparse adjacency matrix (n x n)
        nnd: NNDescent index for additional queries
    """
    n_samples = positions.shape[0]
    k = min(k, n_samples - 1)

    # mutual knn graph with gaussian rbf similarity
    nnd = NNDescent(
        positions.astype(np.float32),
        metric="euclidean",
        n_neighbors=k + 1,  # +1 because it includes self
        n_jobs=-1,
        low_memory=True,
    )
    knn_indices, knn_dists = nnd.neighbor_graph
    rows = np.repeat(np.arange(n_samples), k + 1)
    cols = knn_indices.ravel()
    print(
        f"[build_graph] knn_dists min {knn_dists.min()}, max {knn_dists.max()}, mean {knn_dists.mean()}, std {knn_dists.std()}"
    )
    sigma_pos = sigma_pos_factor * knn_dists.mean()
    if histogram_output_dir is not None:
        dump_histogram(
            knn_dists,
            histogram_output_dir / "knn_position_dists.png",
            "KNN Position Distances",
            "Euclidean Distance",
            sigma_pos,
        )
    knn_dists_rbf = np.exp(-(knn_dists**2) / (2 * sigma_pos**2))
    knn_position = sp.csr_matrix(
        (knn_dists_rbf.ravel(), (rows, cols)), shape=(n_samples, n_samples)
    )
    knn_position = knn_position.maximum(knn_position.T)  # symmetrize to mutual nn

    # cosine similarity graph
    new_rows, new_cols = knn_position.nonzero()  # recompute after symmetrization
    lf_cos_sim = (normalized_lf[new_rows] * normalized_lf[new_cols]).sum(axis=-1)
    lf_cos_sim = np.clip(
        (lf_cos_sim + 1) / 2, 0, 1
    )  # we need range [0, 1] so degree matrix does not get negative entries
    lf_cos_dist = 1 - lf_cos_sim  # we want distances in [0, 1] for rbf kernel
    print(
        f"[build_graph] lf_cos_dist min {lf_cos_dist.min()}, max {lf_cos_dist.max()}, mean {lf_cos_dist.mean()}, std {lf_cos_dist.std()}"
    )
    sigma_lf = sigma_lf_factor * lf_cos_dist.mean()
    if histogram_output_dir is not None:
        dump_histogram(
            lf_cos_dist,
            histogram_output_dir / "knn_lf_cos_dist.png",
            "KNN Language Feature Cosine Distance",
            "Cosine Distance (1 - sim)",
            sigma_lf,
        )
    lf_cos_dist_rbf = np.exp(-(lf_cos_dist**2) / (2 * sigma_lf**2))
    knn_language = sp.csr_matrix(
        (lf_cos_dist_rbf, (new_rows, new_cols)), shape=(n_samples, n_samples)
    )

    print(
        f"[build_graph] pos min {knn_position.data.min()}, max {knn_position.data.max()}, mean {knn_position.data.mean()}, std {knn_position.data.std()}"
    )
    print(
        f"[build_graph] lfs min {knn_language.data.min()}, max {knn_language.data.max()}, mean {knn_language.data.mean()}, std {knn_language.data.std()}"
    )

    graph = weight_pos * knn_position + weight_lf * knn_language
    return graph


def hdbscan_on_precomputed_graph(
    A_dist: sp.csr_matrix,
    samples: np.ndarray,
    hdbscan_args: dict,
    soft_clustering: bool = False,
):
    """HDBSCAN clustering on a precomputed distance graph.

    Handles disconnected graphs by clustering each connected component separately.

    Args:
        A_dist: Distance adjacency matrix (n x n)
        samples: Sample features for noise assignment (n x d)
        hdbscan_args: Arguments for HDBSCAN clustering
        soft_clustering: If True, assign noise points to nearest cluster.
            If False, leave noise points marked as -1.

    Returns:
        clusters: Cluster assignments (n,), -1 for noise (only if soft_clustering=False)
    """
    n_samples = A_dist.shape[0]
    n_components, component_labels = sp.csgraph.connected_components(
        A_dist, directed=False
    )

    # print component stats
    comp_sizes = np.bincount(component_labels)
    print(
        f"[hdbscan] {n_samples} samples, {n_components} components, min_cluster_size={hdbscan_args['min_cluster_size']}"
    )
    print(f"[hdbscan] Largest 10 components: {sorted(comp_sizes, reverse=True)[:10]}")
    print(
        f"[hdbscan] Components >= min_cluster_size: {(comp_sizes >= hdbscan_args['min_cluster_size']).sum()}"
    )

    # if graph connected, cluster and return
    if n_components == 1:
        max_dist = A_dist.data.max()
        clusters = hdbscan.HDBSCAN(
            metric="precomputed", max_dist=max_dist, **hdbscan_args
        ).fit_predict(A_dist)
        hdbscan_noise = (clusters == -1).sum()
        if soft_clustering:
            clusters = assign_noise_to_nearest(samples, clusters)
            print(
                f"[hdbscan] Assigned {hdbscan_noise} noise points to nearest clusters"
            )
        else:
            print(
                f"[hdbscan] Noise: {hdbscan_noise} total, 0 disconnected, {hdbscan_noise} HDBSCAN"
            )
        return clusters

    # if disconnected graph, cluster each component separately
    clusters = np.full(n_samples, -1, dtype=np.int32)
    cluster_offset = 0
    disconnected_noise = 0
    hdbscan_noise = 0

    for comp_id in range(n_components):
        comp_mask = component_labels == comp_id
        comp_size = comp_mask.sum()

        # leave component as noise if too small
        if comp_size < hdbscan_args["min_cluster_size"]:
            disconnected_noise += comp_size
            continue

        # subgraph
        comp_indices = np.where(comp_mask)[0]
        A_sub = A_dist[comp_indices][:, comp_indices]

        # cluster
        max_dist = A_sub.data.max() if A_sub.nnz > 0 else 1.0
        comp_clusters = hdbscan.HDBSCAN(
            metric="precomputed", max_dist=max_dist, **hdbscan_args
        ).fit_predict(A_sub)
        comp_noise = (comp_clusters == -1).sum()
        hdbscan_noise += comp_noise

        if soft_clustering:
            comp_clusters = assign_noise_to_nearest(
                samples[comp_indices], comp_clusters
            )

        # assign offsetted cluster ids to original indices
        valid_mask = comp_clusters >= 0
        clusters[comp_indices[valid_mask]] = comp_clusters[valid_mask] + cluster_offset
        cluster_offset += comp_clusters.max() + 1 if valid_mask.any() else 0

    total_noise = (clusters == -1).sum()
    if soft_clustering:
        print(
            f"[hdbscan] Assigned {hdbscan_noise} HDBSCAN noise points to nearest clusters, {total_noise} unassigned (from {disconnected_noise} small components)"
        )
    else:
        print(
            f"[hdbscan] Noise: {total_noise} total, {disconnected_noise} disconnected, {hdbscan_noise} HDBSCAN"
        )
    return clusters


def hdbscan_cluster_gaussians(
    gaussians: GaussianModel, cfg: DictConfig, histogram_output_dir: Path | None = None
):
    deformed_data = [
        deform_at_timestep(gaussians, t)
        for t in np.linspace(
            0,
            1,
            cfg.graph_extraction.hdbscan_clustering.pos_timesteps,
            dtype=np.float32,
        )
    ]
    pos_through_time = np.concatenate([i[0] for i in deformed_data], axis=-1)
    lf = deformed_data[0][2]
    sparse_A = build_similarity_graph(
        positions=pos_through_time,
        normalized_lf=lf,
        k=cfg.graph_extraction.hdbscan_clustering.knn_graph_k,
        weight_pos=cfg.graph_extraction.hdbscan_clustering.weight_pos,
        weight_lf=cfg.graph_extraction.hdbscan_clustering.weight_lf,
        sigma_pos_factor=cfg.graph_extraction.hdbscan_clustering.rbf_sigma_pos_factor,
        sigma_lf_factor=cfg.graph_extraction.hdbscan_clustering.rbf_sigma_lf_factor,
        histogram_output_dir=histogram_output_dir,
    )

    # hdbscan needs distances - clamp to avoid inf
    sparse_A.data = np.clip(sparse_A.data, 1e-6, None)
    sparse_A.data = 1.0 / sparse_A.data

    clusters = hdbscan_on_precomputed_graph(
        A_dist=sparse_A,
        samples=pos_through_time,
        hdbscan_args=dict(cfg.graph_extraction.hdbscan_clustering.hdbscan_args),
        soft_clustering=cfg.graph_extraction.hdbscan_clustering.soft_clustering,
    )

    print(f"[hdbscan] total clusters: {len(np.unique(clusters[clusters >= 0]))}")
    return clusters


def full_hdbscan_cluster_gaussians(gaussians: GaussianModel, cfg: DictConfig):
    deformed_data = [
        deform_at_timestep(gaussians, t)
        for t in np.linspace(
            0,
            1,
            cfg.graph_extraction.full_hdbscan_clustering.timesteps,
            dtype=np.float32,
        )
    ]
    pos_through_time = np.concatenate([i[0] for i in deformed_data], axis=-1)
    lf_through_times = np.concatenate([i[2] for i in deformed_data], axis=-1)
    weight_pos = cfg.graph_extraction.full_hdbscan_clustering.weight_pos
    weight_lf = cfg.graph_extraction.full_hdbscan_clustering.weight_lf
    samples = np.concatenate(
        [pos_through_time * weight_pos, lf_through_times * weight_lf], axis=-1
    )
    clusters = hdbscan.HDBSCAN(
        **cfg.graph_extraction.full_hdbscan_clustering.hdbscan_args
    ).fit_predict(samples)
    hdbscan_noise = (clusters == -1).sum()

    if cfg.graph_extraction.full_hdbscan_clustering.soft_clustering:
        clusters = assign_noise_to_nearest(samples, clusters)
        print(
            f"[full_hdbscan] Assigned {hdbscan_noise} noise points to nearest clusters"
        )
    else:
        print(
            f"[full_hdbscan] Noise: {hdbscan_noise} total, 0 disconnected, {hdbscan_noise} HDBSCAN"
        )

    print(f"[full_hdbscan] total clusters: {len(np.unique(clusters[clusters >= 0]))}")
    return clusters


def spectral_cluster_gaussians(gaussians: GaussianModel, cfg: DictConfig):
    deformed_data = [
        deform_at_timestep(gaussians, t)
        for t in np.linspace(
            0,
            1,
            cfg.graph_extraction.spectral_clustering.pos_timesteps,
            dtype=np.float32,
        )
    ]
    pos_through_time = np.concatenate([i[0] for i in deformed_data], axis=-1)
    lf = deformed_data[0][2]
    sparse_A = build_similarity_graph(
        positions=pos_through_time,
        normalized_lf=lf,
        k=cfg.graph_extraction.spectral_clustering.knn_graph_k,
        weight_pos=cfg.graph_extraction.spectral_clustering.weight_pos,
        weight_lf=cfg.graph_extraction.spectral_clustering.weight_lf,
        sigma_pos_factor=cfg.graph_extraction.spectral_clustering.rbf_sigma_pos_factor,
        sigma_lf_factor=cfg.graph_extraction.spectral_clustering.rbf_sigma_lf_factor,
    )
    clusters = ng_jordan_weiss_spectral_clustering(
        sparse_A,
        spectral_embedding_dim=cfg.graph_extraction.spectral_clustering.spectral_embedding_dim,
        hdbscan_args=cfg.graph_extraction.spectral_clustering.hdbscan_args,
        soft_clustering=cfg.graph_extraction.spectral_clustering.soft_clustering,
    )
    hdbscan_noise = (clusters == -1).sum()

    if cfg.graph_extraction.spectral_clustering.soft_clustering:
        clusters = assign_noise_to_nearest(pos_through_time, clusters)
        print(f"[spectral] Assigned {hdbscan_noise} noise points to nearest clusters")
    else:
        print(
            f"[spectral] Noise: {hdbscan_noise} total, 0 disconnected, {hdbscan_noise} HDBSCAN"
        )

    print(f"[spectral] total clusters: {len(np.unique(clusters[clusters >= 0]))}")
    return clusters


def clusters_to_rgb(clusters: np.ndarray) -> np.ndarray:
    """compute colors for clusters (n_gaussians, 3) in range(0,1)"""
    unique = np.unique(clusters)
    assert np.all(unique == np.arange(len(unique))), "Cluster ids must be contiguous"
    pal = np.random.rand(len(unique), 3)
    return pal[clusters]


def load_precomputed_instance_clusters(clip: DictConfig, cfg: DictConfig) -> np.ndarray:
    """Load precomputed merged instance assignments from CoTracker preprocessing.

    Args:
        clip: Clip configuration
        cfg: Full hydra configuration

    Returns:
        clusters: (N_gaussians,) array of instance IDs, -1 for background/unassigned
    """
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    merged_ids_path = clip_dir / "cotracker" / "merged_instance_ids.npy"

    clusters = np.load(merged_ids_path)

    n_noise = (clusters == -1).sum()
    print(f"[precomputed] unassigned noise: {n_noise}")
    print(
        f"[precomputed] total clusters: {len(np.unique(clusters[clusters >= 0]))} (excluding noise)"
    )
    return clusters


def bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2):
    mu1, mu2 = np.asarray(mu1), np.asarray(mu2)
    Sigma1, Sigma2 = np.asarray(Sigma1), np.asarray(Sigma2)

    # Regularize covariance matrices to ensure positive definiteness
    # This handles cases where clusters have few points or collinear points
    eps = 1e-6
    dim = Sigma1.shape[0]
    Sigma1_reg = Sigma1 + eps * np.eye(dim)
    Sigma2_reg = Sigma2 + eps * np.eye(dim)

    # Average covariance
    Sigma = 0.5 * (Sigma1_reg + Sigma2_reg)

    # Cholesky factorization for stability
    L = np.linalg.cholesky(Sigma)
    # Solve for (mu2 - mu1) without explicit inverse
    diff = mu2 - mu1
    sol = np.linalg.solve(L, diff)
    sol = np.linalg.solve(L.T, sol)
    term1 = 0.125 * np.dot(diff, sol)  # (1/8) Δμᵀ Σ⁻¹ Δμ

    # log-determinants via Cholesky
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
    logdet_Sigma1 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma1_reg))))
    logdet_Sigma2 = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma2_reg))))
    term2 = 0.5 * (logdet_Sigma - 0.5 * (logdet_Sigma1 + logdet_Sigma2))

    DB = term1 + term2
    return np.exp(-DB)  # Bhattacharyya coefficient


def decode_qwen(ae: QwenAutoencoder, lfs: torch.Tensor, cfg: DictConfig) -> np.ndarray:
    BATCH_SIZE = cfg.graph_extraction.decode_batch_size

    decoded_lfs = []
    with torch.no_grad():
        for i in range(0, lfs.shape[0], BATCH_SIZE):
            batch = lfs[i : min(i + BATCH_SIZE, len(lfs))].to("cuda")
            decoded_lfs.append(ae.decode(batch))
    decoded_lfs = [i.detach().cpu().numpy() for i in decoded_lfs]
    decoded_lfs = np.concatenate(decoded_lfs, axis=0)
    return decoded_lfs


def clusterwise_qwen_feats(
    ae: QwenAutoencoder,
    clusters: np.ndarray,
    cfg: DictConfig,
    patch_lf_through_time: np.ndarray,
    instance_lf_through_time: np.ndarray = None,
    feature_selection: str = "random",
    opacities: np.ndarray = None,
) -> dict[int, np.ndarray]:
    """Extract qwen features from each cluster by sub-clustering its Qwen features.

    Args:
        ae: QwenAutoencoder
        clusters: np.ndarray, cluster ids
        cfg: DictConfig
        patch_lf_through_time: np.ndarray, patch language features through time (T, n_gaussians, lang_dim)
        instance_lf_through_time: np.ndarray, instance language features through time (T, n_gaussians, lang_dim)
        feature_selection: str, one of: [opacity, random]
        opacities: np.ndarray, opacities (n_gaussians,) (only needed if feature_selection is "opacity")

    Returns:
        Dict[int, np.ndarray]: dict str(cluster id) -> tensor(T, n_feats, full_dim)
        or tuple of those dicts for patch and instance features
    """
    assert feature_selection in ["opacity", "random"], "Invalid feature selection"

    cluster_to_patch_through_time = {}
    cluster_to_instance_through_time = {}

    for cid in np.unique(clusters):
        # get feats for that cluster
        cmask = clusters == cid
        cluster_patch_lf = patch_lf_through_time[:, cmask]
        if instance_lf_through_time is not None:
            cluster_instance_lf = instance_lf_through_time[:, cmask]

        # pick random gaussians for that cluster
        n_feats = min(
            cluster_patch_lf.shape[1], cfg.graph_extraction.features_per_cluster
        )
        if feature_selection == "random":
            indices = np.random.choice(
                np.arange(cluster_patch_lf.shape[1]), n_feats, replace=False
            )
        else:
            indices = np.argsort(opacities[cmask])[-n_feats:]

        cluster_patch_lf = cluster_patch_lf[:, indices]
        if instance_lf_through_time is not None:
            cluster_instance_lf = cluster_instance_lf[:, indices]

        # decode them
        flat_cluster_patch_lf = cluster_patch_lf.reshape(-1, cluster_patch_lf.shape[-1])
        decoded_flat_cluster_patch_lf = decode_qwen(
            ae,
            torch.tensor(flat_cluster_patch_lf, device="cuda", dtype=torch.float32),
            cfg,
        )
        decoded_cluster_patch_lf = decoded_flat_cluster_patch_lf.reshape(
            cluster_patch_lf.shape[0], -1, decoded_flat_cluster_patch_lf.shape[-1]
        )
        cluster_to_patch_through_time[str(cid)] = decoded_cluster_patch_lf
        if instance_lf_through_time is not None:
            flat_cluster_instance_lf = cluster_instance_lf.reshape(
                -1, cluster_instance_lf.shape[-1]
            )
            decoded_flat_cluster_instance_lf = decode_qwen(
                ae,
                torch.tensor(
                    flat_cluster_instance_lf, device="cuda", dtype=torch.float32
                ),
                cfg,
            )
            decoded_cluster_instance_lf = decoded_flat_cluster_instance_lf.reshape(
                cluster_instance_lf.shape[0],
                -1,
                decoded_flat_cluster_instance_lf.shape[-1],
            )
            cluster_to_instance_through_time[str(cid)] = decoded_cluster_instance_lf

    return (
        cluster_to_patch_through_time
        if instance_lf_through_time is None
        else (cluster_to_patch_through_time, cluster_to_instance_through_time)
    )


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

    bhattacharyya_coeffs = np.empty((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            bhattacharyya_coeffs[i, j] = bhattacharyya_coefficient(
                means[i], covs[i], means[j], covs[j]
            )

    A = np.where(
        bhattacharyya_coeffs >= cfg.graph_extraction.graph_edge_threshold,
        bhattacharyya_coeffs,
        0,
    )
    return A, bhattacharyya_coeffs


def visualize_cluster_latent_distributions(
    clusters: np.ndarray,
    lf_patch_through_time: np.ndarray,
    output_dir: Path,
    timestep_idx: int = 0,
):
    """Create 3D PCA visualizations of latent features for each cluster.

    Args:
        clusters: (n_gaussians,) cluster assignments
        lf_patch_through_time: (T, n_gaussians, lang_dim) latent features
        output_dir: Directory to save visualizations
        timestep_idx: Which timestep to visualize (default: 0)
    """
    vis_dir = output_dir / "cluster_latent_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get latent features at specified timestep
    lf = lf_patch_through_time[timestep_idx]  # (n_gaussians, lang_dim)

    unique_clusters = np.unique(clusters[clusters >= 0])
    logger.info(
        f"Visualizing latent distributions for {len(unique_clusters)} clusters..."
    )

    # Fit global PCA on all features for consistent projection
    pca_global = PCA(n_components=3)
    pca_global.fit(lf)

    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_lf = lf[mask]  # (n_cluster_gaussians, lang_dim)

        if len(cluster_lf) < 4:
            continue

        # Project to 3D using global PCA
        lf_3d = pca_global.transform(cluster_lf)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            lf_3d[:, 0],
            lf_3d[:, 1],
            lf_3d[:, 2],
            c=np.arange(len(lf_3d)),
            cmap="viridis",
            alpha=0.6,
            s=20,
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"Cluster {cluster_id} Latent Distribution (n={len(cluster_lf)})")

        plt.colorbar(scatter, label="Gaussian Index", shrink=0.6)
        plt.tight_layout()
        plt.savefig(vis_dir / f"cluster_{cluster_id:03d}_latents.png", dpi=150)
        plt.close(fig)

    # Also create combined visualization with all clusters
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_lf = lf[mask]
        if len(cluster_lf) < 4:
            continue
        lf_3d = pca_global.transform(cluster_lf)
        ax.scatter(
            lf_3d[:, 0],
            lf_3d[:, 1],
            lf_3d[:, 2],
            c=[colors[i % len(colors)]],
            alpha=0.4,
            s=10,
            label=f"C{cluster_id}" if i < 20 else None,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"All Clusters Latent Distribution (PCA, t={timestep_idx})")
    if len(unique_clusters) <= 20:
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(vis_dir / "all_clusters_latents.png", dpi=150)
    plt.close(fig)

    logger.info(f"Saved latent visualizations to {vis_dir}")


def extract_graph(clip: DictConfig, cfg: DictConfig):
    """Extract scene graph from trained Gaussian Splatting model.

    Args:
        clip (DictConfig): Clip configuration
        cfg (DictConfig): Full hydra configuration
    """
    # make deterministic
    random.seed(cfg.graph_extraction.random_seed)
    np.random.seed(cfg.graph_extraction.random_seed)
    torch.manual_seed(cfg.graph_extraction.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.graph_extraction.random_seed)

    # load gaussians and autoencoder
    logger.info(f"Loading gaussians and autoencoder...")
    gaussians, scene, dataset, args, pipeline = load_gaussian_model(clip, cfg)
    ae = get_autoencoder(clip, cfg)

    # pre-clustering filtering
    logger.info(f"Pre-clustering filtering...")
    opacities = gaussians.get_opacity.squeeze()
    mask = opacities >= cfg.graph_extraction.precluster_opacity_threshold
    filter_gaussians(gaussians, mask)

    # clustering
    logger.info(f"Clustering with {cfg.graph_extraction.cluster_method} method...")
    model_path = Path(cfg.output_root) / clip.name
    graph_output_dir = Path(model_path) / cfg.graph_extraction.graph_output_subdir
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.graph_extraction.cluster_method in [
        "spectral",
        "precomputed",
        "hdbscan",
        "full_hdbscan",
    ]
    if cfg.graph_extraction.cluster_method == "spectral":
        clusters = spectral_cluster_gaussians(gaussians, cfg=cfg)
    if cfg.graph_extraction.cluster_method == "precomputed":
        clusters = load_precomputed_instance_clusters(clip, cfg)
    if cfg.graph_extraction.cluster_method == "full_hdbscan":
        clusters = full_hdbscan_cluster_gaussians(gaussians, cfg=cfg)
    if cfg.graph_extraction.cluster_method == "hdbscan":
        clusters = hdbscan_cluster_gaussians(
            gaussians, cfg=cfg, histogram_output_dir=graph_output_dir
        )

    # post-clustering filtering
    logger.info(f"Post-clustering filtering...")
    cluster_mask = clusters >= 0
    filter_gaussians(gaussians, cluster_mask)
    clusters = clusters[cluster_mask]

    # cluster properties
    logger.info(f"Computing cluster features...")
    timesteps = np.linspace(0, 1, cfg.graph_extraction.n_timesteps)
    deformed_data = [deform_at_timestep(gaussians, t) for t in timesteps]
    pos_through_time = np.stack([i[0] for i in deformed_data])
    lf_patch_through_time = np.stack([i[1] for i in deformed_data])
    lf_instance_through_time = np.stack([i[2] for i in deformed_data])
    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(pos_through_time, clusters)

    # cluster qwen features
    logger.info(f"Decoding Qwen features...")
    cluster_feats_dict_patch, cluster_feats_dict_instance = clusterwise_qwen_feats(
        ae,
        clusters,
        cfg,
        patch_lf_through_time=lf_patch_through_time,
        instance_lf_through_time=lf_instance_through_time,
        feature_selection=cfg.graph_extraction.feature_selection,
        opacities=gaussians.get_opacity.detach().squeeze().cpu().numpy(),
    )

    # graph structure
    logger.info(f"Building graphs...")
    graph_results = [
        timestep_graph(pos_through_time[i], clusters, cfg)
        for i in range(len(timesteps))
    ]
    graphs = np.stack([g[0] for g in graph_results])
    bhattacharyya_coeffs = np.stack([g[1] for g in graph_results])

    # save outputs
    logger.info(f"Saving outputs...")
    out = graph_output_dir
    # Save per-gaussian RGB colors derived from SH DC coefficients (same as rerun RGB logging).
    dc_sh = gaussians._features_dc.detach().cpu().numpy()  # (N, 1, 3)
    colors_rgb = SH2RGB(dc_sh[:, 0, :])  # (N, 3) in [0,1]
    colors_rgb_uint8 = (np.clip(colors_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.save(out / "colors_rgb.npy", colors_rgb_uint8)
    np.savez(
        out / "c_qwen_feats.npz", **cluster_feats_dict_patch
    )  # qwen features per cluster through time (cluster_id -> (timesteps, n_feats, 3584))
    np.savez(
        out / "c_qwen_feats_instance.npz", **cluster_feats_dict_instance
    )  # qwen instance features per cluster through time (cluster_id -> (timesteps, n_feats, 3584))
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
    np.save(
        out / "bhattacharyya_coeffs.npy", bhattacharyya_coeffs
    )  # dense bhattacharyya coefficients through time (timesteps, n_clusters, n_clusters)
    np.save(out / "positions.npy", pos_through_time)  # (T, n_filtered_gaussians, 3)
    np.save(out / "clusters.npy", clusters)  # (n_filtered_gaussians,)
    np.save(
        out / "patch_latents_through_time.npy", lf_patch_through_time
    )  # patch latents through time (timesteps, n_filtered_gaussians, lang_dim)

    # matplotlib 3D latent feature visualizations
    logger.info(f"Creating cluster latent visualizations...")
    visualize_cluster_latent_distributions(
        clusters=clusters,
        lf_patch_through_time=lf_patch_through_time,
        output_dir=out,
        timestep_idx=0,
    )

    # rerun visualization
    logger.info(f"Visualizing to rerun...")
    rr.init("clusters")
    rr.save(out / "visualization.rrd")
    cluster_colors = clusters_to_rgb(clusters)
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

    # memory cleanup
    logger.info(f"Cleaning up memory...")
    del gaussians
    del scene
    del dataset
    del args
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    for clip in tqdm(cfg.clips, desc="Extracting graphs", unit="clip"):
        extract_graph(clip, cfg)


if __name__ == "__main__":
    main()
