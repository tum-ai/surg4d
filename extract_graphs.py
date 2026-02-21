import torch
import numpy as np
from pathlib import Path
import logging
import rerun as rr
import random
import hydra
import json
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from utils.rerun_utils import (
    log_points_through_time,
    log_graph_structure_through_time,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def clusters_to_rgb(clusters: np.ndarray) -> np.ndarray:
    """compute colors for clusters (n_gaussians, 3) in range(0,1)"""
    unique = np.unique(clusters)
    assert np.all(unique == np.arange(len(unique))), "Cluster ids must be contiguous"
    pal = np.random.rand(len(unique), 3)
    return pal[clusters]


def filter_and_reindex_clusters(
    clusters: np.ndarray,
    min_cluster_size: int,
    semantic_labels: dict[int, str] = None,
) -> tuple[np.ndarray, dict[int, str] | None]:
    """Filter small clusters and remap valid cluster ids to contiguous ids starting from 0.

    Args:
        clusters: (N,) cluster ids, with -1 indicating noise.
        min_cluster_size: Minimum size for valid clusters. If -1, keep all non-noise clusters.
        semantic_labels: Optional mapping from original cluster id -> semantic label.

    Returns:
        remapped_clusters: (N,) clusters with small clusters set to -1 and valid ids remapped to [0..C-1].
        remapped_semantic_labels: Optional mapping aligned to remapped cluster ids.
    """
    clusters = clusters.copy()

    if min_cluster_size != -1:
        valid_clusters = clusters[clusters >= 0]
        cluster_ids, cluster_counts = np.unique(valid_clusters, return_counts=True)
        small_cluster_ids = cluster_ids[cluster_counts < min_cluster_size]
        if len(small_cluster_ids) > 0:
            clusters[np.isin(clusters, small_cluster_ids)] = -1
        logger.info(
            f"Filtered {len(small_cluster_ids)} clusters below min size {min_cluster_size}"
        )

    valid_cluster_ids = np.unique(clusters[clusters >= 0])
    remapped_clusters = np.full_like(clusters, fill_value=-1)
    if len(valid_cluster_ids) > 0:
        remapped_clusters[clusters >= 0] = np.searchsorted(
            valid_cluster_ids, clusters[clusters >= 0]
        ).astype(remapped_clusters.dtype)

    remapped_semantic_labels = None
    if semantic_labels is not None:
        remapped_semantic_labels = {
            int(new_id): semantic_labels[int(old_id)]
            for new_id, old_id in enumerate(valid_cluster_ids)
            if int(old_id) in semantic_labels
        }
        logger.info(
            f"Remapped semantic labels for {len(remapped_semantic_labels)} clusters"
        )

    return remapped_clusters, remapped_semantic_labels


def temporal_lof_outlier_mask(
    positions_through_time: np.ndarray,
    clusters: np.ndarray,
    cfg: DictConfig,
    histogram_output_dir: Path | None = None,
) -> np.ndarray:
    """Flag gaussians that are strong LOF outliers in any cluster at any timestep.

    Args:
        positions_through_time: (T, N, 3) gaussian positions through time.
        clusters: (N,) contiguous cluster ids.
        cfg: Full hydra config.

    Returns:
        outlier_mask: (N,) True where gaussian is a strong outlier at >=1 timestep.
    """
    lof_cfg = cfg.graph_extraction.temporal_lof_outlier_filter
    unique_clusters = np.unique(clusters)
    outlier_mask = np.zeros(clusters.shape[0], dtype=bool)
    all_negative_outlier_factors = []

    for timestep_idx in range(positions_through_time.shape[0]):
        positions = positions_through_time[timestep_idx]
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            if cluster_indices.shape[0] < lof_cfg.min_cluster_points:
                continue

            n_neighbors = min(lof_cfg.n_neighbors, cluster_indices.shape[0] - 1)
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=lof_cfg.contamination,
            )
            cluster_positions = positions[cluster_indices]
            lof_labels = lof.fit_predict(cluster_positions)
            all_negative_outlier_factors.append(lof.negative_outlier_factor_)
            strong_outlier_mask = (lof_labels == -1) & (
                lof.negative_outlier_factor_
                < lof_cfg.strong_negative_outlier_factor_threshold
            )
            outlier_mask[cluster_indices[strong_outlier_mask]] = True

    if histogram_output_dir is not None and len(all_negative_outlier_factors) > 0:
        all_negative_outlier_factors = np.concatenate(all_negative_outlier_factors, axis=0)
        plot_negative_outlier_factors = all_negative_outlier_factors[
            (all_negative_outlier_factors >= -5.0) & (all_negative_outlier_factors <= 0.0)
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            plot_negative_outlier_factors,
            bins=100,
            range=(-5.0, 0.0),
            edgecolor="black",
            alpha=0.75,
        )
        if -5.0 <= lof_cfg.strong_negative_outlier_factor_threshold <= 0.0:
            ax.axvline(
                lof_cfg.strong_negative_outlier_factor_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"strong threshold={lof_cfg.strong_negative_outlier_factor_threshold}",
            )
            ax.legend(loc="upper left")
        ax.set_xlim(-5.0, 0.0)
        ax.set_title("LOF Negative Outlier Factor Distribution (Global)")
        ax.set_xlabel("negative_outlier_factor")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(
            histogram_output_dir / "lof_negative_outlier_factor_hist_global.png", dpi=150
        )
        plt.close(fig)

    return outlier_mask


def load_precomputed_instance_clusters(clip: DictConfig, cfg: DictConfig) -> np.ndarray:
    """Load precomputed merged instance assignments from CoTracker preprocessing.

    Args:
        clip: Clip configuration
        cfg: Full hydra configuration
    Returns:
        clusters: (N_gaussians,) array of instance IDs, -1 for background/unassigned
    """
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    merged_ids_path = clip_dir / cfg.graph_extraction.cotracker_subdir / "merged_instance_ids.npy"

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

    # load points
    points_file = Path(cfg.preprocessed_root) / clip.name / cfg.graph_extraction.cotracker_subdir / "point_positions_precomputed.npy"
    points_through_time = np.load(points_file)  # (T, N, 3)

    # sumbsample temporally
    points_through_time = points_through_time[::cfg.graph_extraction.timestep_stride]

    # load point colors from CoTracker preprocessing
    point_colors_file = Path(cfg.preprocessed_root) / clip.name / cfg.graph_extraction.cotracker_subdir / "point_colors.npy"
    ply_colors = np.load(point_colors_file).astype(np.float32) / 255.0
    logger.info(f"Loaded {ply_colors.shape[0]} point colors from {point_colors_file}")
    if points_through_time.shape[1] != ply_colors.shape[0]:
        logger.warning(
            f"number of colors ({ply_colors.shape[0]}) does not match number of gaussians ({points_through_time.shape[1]})"
        )


    # load clusters and semantic labels
    graph_output_dir = Path(cfg.output_root) / clip.name / cfg.graph_extraction.graph_output_subdir
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    clusters = load_precomputed_instance_clusters(clip, cfg)
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    semantic_labels_path = clip_dir / cfg.graph_extraction.cotracker_subdir / "merged_instance_semantic_labels.json"
    with open(semantic_labels_path, "r") as f:
        semantic_labels_raw = json.load(f)
        semantic_labels = {int(k): v for k, v in semantic_labels_raw.items()}

    # filter small clusters and reindex to contiguous ids
    print('BEFORE')
    print(np.unique(clusters, return_counts=True))
    print(semantic_labels)
    clusters, semantic_labels = filter_and_reindex_clusters(
        clusters=clusters,
        min_cluster_size=cfg.graph_extraction.min_cluster_size,
        semantic_labels=semantic_labels,
    )
    print('AFTER')
    print(np.unique(clusters, return_counts=True))
    print(semantic_labels)
    logger.info(f"Post-clustering filtering...")
    cluster_mask = clusters >= 0
    points_through_time = points_through_time[:, cluster_mask, :]
    clusters = clusters[cluster_mask]
    ply_colors = ply_colors[cluster_mask]

    # filter outliers within clusters
    logger.info(f"Temporal LOF outlier filtering...")
    temporal_outlier_mask = temporal_lof_outlier_mask(
        points_through_time,
        clusters,
        cfg,
        histogram_output_dir=graph_output_dir,
    )
    keep_mask = ~temporal_outlier_mask
    logger.info(
        f"Temporal LOF removed {temporal_outlier_mask.sum()} / {len(temporal_outlier_mask)} gaussians"
    )
    points_through_time = points_through_time[:, keep_mask, :]
    clusters = clusters[keep_mask]
    ply_colors = ply_colors[keep_mask]

    # cluster properties
    logger.info(f"Computing cluster features...")
    (
        cluster_pos_through_time,
        cluster_center_through_time,
        cluster_extent_through_time,
    ) = properties_through_time(points_through_time, clusters)

    # graph structure
    logger.info(f"Building graphs...")
    graph_results = [
        timestep_graph(points_through_time[i], clusters, cfg)
        for i in range(len(points_through_time))
    ]
    graphs = np.stack([g[0] for g in graph_results])
    bhattacharyya_coeffs = np.stack([g[1] for g in graph_results])

    # save outputs
    logger.info(f"Saving outputs...")
    out = graph_output_dir
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
    np.save(out / "positions.npy", points_through_time)  # (T, n_filtered_gaussians, 3)
    np.save(out / "clusters.npy", clusters)  # (n_filtered_gaussians,)
    if semantic_labels is not None:
        with open(out / "cluster_semantics.json", "w") as f:
            json.dump({str(k): v for k, v in semantic_labels.items()}, f, indent=2)
        logger.info(
            f"Saved adapted semantic labels to {out / 'cluster_semantics.json'}"
        )

    # rerun visualization
    logger.info(f"Visualizing to rerun...")
    rr.init("clusters")
    rr.save(out / "visualization.rrd")
    cluster_colors = clusters_to_rgb(clusters)
    log_points_through_time(
        clusters=clusters,
        cluster_colors=cluster_colors,
        pos_through_time=points_through_time,
        point_colors=ply_colors,
        cluster_pos_through_time=cluster_pos_through_time,
        semantic_labels=semantic_labels,
    )
    log_graph_structure_through_time(
        cluster_pos_through_time=cluster_pos_through_time,
        graphs_through_time=graphs,
    )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    for clip in tqdm(cfg.clips, desc="Extracting graphs", unit="clip"):
        extract_graph(clip, cfg)


if __name__ == "__main__":
    main()
