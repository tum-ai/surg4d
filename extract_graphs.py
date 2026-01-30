from sklearn.cluster import HDBSCAN
from pynndescent import NNDescent
import scipy.sparse as sp
from scipy.linalg import eig, eigh
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

from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import GaussianModel, Scene
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import (
    log_points_through_time,
    log_graph_structure_through_time,
)
from utils.gaussian_loading_utils import get_latest_model_iteration


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
        args.iteration = get_latest_model_iteration(cfg)

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


def ng_jordan_weiss_spectral_clustering(
    A, spectral_embedding_dim: int, hdbscan_args: dict
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

    Returns:
        clusters: Cluster assignments (n,), -1 for noise
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
        clusters = HDBSCAN(**hdbscan_args).fit_predict(T)
        hdbscan_noise = (clusters == -1).sum()
        print(
            f"[spectral] Noise: {hdbscan_noise} total, {0} disconnected, {hdbscan_noise} HDBSCAN"
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
        comp_clusters = HDBSCAN(**hdbscan_args).fit_predict(T_sub)

        # count hdbscan noise
        hdbscan_noise += (comp_clusters == -1).sum()

        # assign offsetted cluster ids to original indices
        valid_mask = comp_clusters >= 0
        clusters[comp_indices[valid_mask]] = comp_clusters[valid_mask] + cluster_offset
        cluster_offset += comp_clusters.max() + 1 if valid_mask.any() else 0

    total_noise = (clusters == -1).sum()
    print(
        f"[spectral] Noise: {total_noise} total, {disconnected_noise} disconnected, {hdbscan_noise} HDBSCAN"
    )
    return clusters


def build_graph(
    positions: np.ndarray,
    normalized_lf: np.ndarray,
    k: int,
    weight_pos: float,
    weight_lf: float,
    sigma_pos_factor: float,
    sigma_lf_factor: float,
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
    print(f"[build_graph] knn_dists min {knn_dists.min()}, max {knn_dists.max()}, mean {knn_dists.mean()}, std {knn_dists.std()}")
    sigma_pos = sigma_pos_factor * knn_dists.mean()
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
    lf_cos_dist = 1 - lf_cos_sim # we want distances in [0, 1] for rbf kernel
    print(f"[build_graph] lf_cos_dist min {lf_cos_dist.min()}, max {lf_cos_dist.max()}, mean {lf_cos_dist.mean()}, std {lf_cos_dist.std()}")
    sigma_lf = sigma_lf_factor * lf_cos_dist.mean()
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
    return graph, nnd


def spectral_cluster_gaussians(gaussians: GaussianModel, cfg: DictConfig):
    deformed_data = [
        deform_at_timestep(gaussians, t)
        for t in np.linspace(0, 1, cfg.graph_extraction.spectral_clustering.pos_timesteps, dtype=np.float32)
    ]
    pos_through_time = np.concatenate([i[0] for i in deformed_data], axis=-1)
    lf = deformed_data[0][2]
    graph, nnd = build_graph(
        positions=pos_through_time,
        normalized_lf=lf,
        k=cfg.graph_extraction.spectral_clustering.knn_graph_k,
        weight_pos=cfg.graph_extraction.spectral_clustering.weight_pos,
        weight_lf=cfg.graph_extraction.spectral_clustering.weight_lf,
        sigma_pos_factor=cfg.graph_extraction.spectral_clustering.rbf_sigma_pos_factor,
        sigma_lf_factor=cfg.graph_extraction.spectral_clustering.rbf_sigma_lf_factor,
    )
    clusters = ng_jordan_weiss_spectral_clustering(
        graph,
        spectral_embedding_dim=cfg.graph_extraction.spectral_clustering.spectral_embedding_dim,
        hdbscan_args=cfg.graph_extraction.spectral_clustering.hdbscan_args,
    )

    # # Assign noise points to nearest clustered neighbor via extended k-NN query
    # noise_mask = clusters == -1
    # n_noise = noise_mask.sum()
    # if n_noise > 0:
    #     print(f"[spectral] Assigning {n_noise} noise points to nearest clusters...")
        
    #     noise_positions = pos_through_time[noise_mask]
    #     query_k = min(cfg.graph_extraction.spectral_clustering.noise_assignment_k, pos_through_time.shape[0] - 1)
    #     neighbor_indices, _ = nnd.query(noise_positions.astype(np.float32), k=query_k)
        
    #     noise_indices = np.where(noise_mask)[0]
    #     for i, noise_idx in enumerate(noise_indices):
    #         for j in range(query_k):
    #             neighbor = neighbor_indices[i, j]
    #             if clusters[neighbor] >= 0:
    #                 clusters[noise_idx] = clusters[neighbor]
    #                 break
        
    #     remaining_noise = (clusters == -1).sum()
    #     print(f"[spectral] {remaining_noise} points still unassigned")

    # print(
    #     f"[spectral] total clusters: {len(np.unique(clusters[clusters >= 0]))} (excluding noise)"
    # )
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
    patch_lf: np.ndarray = None,
) -> dict[int, np.ndarray]:
    """Extract qwen features from each cluster by sub-clustering its Qwen features.

    Args:
        gaussians (GaussianModel): Gaussian model.
        clusters (np.ndarray): Cluster ids.
        cfg (DictConfig): Hydra configuration.
        patch_lf (np.ndarray): Language features to use. If None, use static features from gaussians.

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
            ae,
            torch.tensor(cluster_lfs[indices], device="cuda", dtype=torch.float32),
            cfg,
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
    assert cfg.graph_extraction.cluster_method in ["spectral", "precomputed"]
    if cfg.graph_extraction.cluster_method == "spectral":
        clusters = spectral_cluster_gaussians(gaussians, cfg=cfg)
    if cfg.graph_extraction.cluster_method == "precomputed":
        clusters = load_precomputed_instance_clusters(clip, cfg)

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
    selected_decoded_lf_through_time = [
        clusterwise_qwen_feats(
            ae,
            clusters,
            cfg,
            patch_lf=lf_patch_through_time[t_idx],
        )
        for t_idx in range(len(timesteps))
    ]

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
    model_path = Path(cfg.output_root) / clip.name
    out = Path(model_path) / cfg.graph_extraction.graph_output_subdir
    out.mkdir(parents=True, exist_ok=True)
    cluster_feats_dict = {}
    for cluster_id in selected_decoded_lf_through_time[0].keys():
        cluster_feats_dict[str(cluster_id)] = np.stack(
            [lf_dict[cluster_id] for lf_dict in selected_decoded_lf_through_time]
        )
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
    np.save(
        out / "bhattacharyya_coeffs.npy", bhattacharyya_coeffs
    )  # dense bhattacharyya coefficients through time (timesteps, n_clusters, n_clusters)
    np.save(out / "positions.npy", pos_through_time)  # (T, n_filtered_gaussians, 3)
    np.save(out / "clusters.npy", clusters)  # (n_filtered_gaussians,)

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
