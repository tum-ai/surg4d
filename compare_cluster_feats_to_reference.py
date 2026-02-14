"""Compare clusterwise Qwen features to reference patch features.

For each gaussian feature in the clusterwise features (across all timesteps),
compute the minimum L1 distance to all reference patch features.
Report the mean minimum distance per timestep and overall mean.
"""

import numpy as np
from pathlib import Path


def compute_mean_min_l1_distance_for_timestep(
    cluster_data: dict,
    reference_feats: np.ndarray,
    timestep: int,
) -> float:
    """Compute mean minimum L1 distance from cluster features to reference features for a specific timestep.
    
    Args:
        cluster_data: Loaded npz data with cluster features
        reference_feats: Reference patch features (n_patches, feature_dim)
        timestep: Timestep index to extract
        
    Returns:
        Mean minimum L1 distance for this timestep
    """
    # Extract timestep from all clusters
    all_gaussian_feats = []
    for cluster_id in cluster_data.keys():
        cluster_feats = cluster_data[cluster_id]  # (timesteps, n_feats, feature_dim)
        timestep_feats = cluster_feats[timestep]  # (n_feats, feature_dim)
        all_gaussian_feats.append(timestep_feats)
    
    # Concatenate all gaussian features
    gaussian_feats = np.concatenate(all_gaussian_feats, axis=0)  # (n_total_gaussians, feature_dim)
    
    # Check dimension match
    if gaussian_feats.shape[1] != reference_feats.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: gaussian_feats={gaussian_feats.shape[1]}, "
            f"reference_feats={reference_feats.shape[1]}"
        )
    
    # Compute L1 distances: for each gaussian, compute distance to all references
    # Use broadcasting: (n_gaussians, 1, dim) - (1, n_patches, dim) -> (n_gaussians, n_patches)
    distances = np.abs(
        gaussian_feats[:, np.newaxis, :] - reference_feats[np.newaxis, :, :]
    ).mean(axis=2)  # (n_gaussians, n_patches) - mean absolute difference per dimension
    
    # Get minimum distance for each gaussian
    min_distances = distances.min(axis=1)  # (n_gaussians,)
    
    # Compute mean
    mean_min_distance = min_distances.mean()
    
    return mean_min_distance


def compute_all_timesteps(
    cluster_feats_path: Path,
    reference_feats_dir: Path,
    n_timesteps: int = 20,
) -> tuple[list[float], float]:
    """Compute mean minimum L1 distance for all timesteps.
    
    Args:
        cluster_feats_path: Path to c_qwen_feats.npz file
        reference_feats_dir: Directory containing reference patch features
        n_timesteps: Number of timesteps to process
        
    Returns:
        Tuple of (list of per-timestep means, overall mean)
    """
    # Load clusterwise features
    cluster_data = np.load(cluster_feats_path)
    
    # Get number of timesteps from first cluster
    first_cluster_id = list(cluster_data.keys())[0]
    n_timesteps_actual = cluster_data[first_cluster_id].shape[0]
    n_timesteps = min(n_timesteps, n_timesteps_actual)
    
    print(f"Processing {n_timesteps} timesteps (cluster data has {n_timesteps_actual} timesteps)")
    
    per_timestep_means = []
    
    for timestep in range(n_timesteps):
        # Frame number increases by 4 per timestep
        frame_number = timestep * 4
        frame_str = f"{frame_number:06d}"
        reference_feats_path = reference_feats_dir / f"{frame_str}_f.npy"
        
        if not reference_feats_path.exists():
            print(f"  Warning: Reference frame {frame_str} not found, skipping timestep {timestep}")
            continue
        
        # Load reference features for this frame
        reference_feats = np.load(reference_feats_path)  # (n_patches, feature_dim)
        
        # Compute metric for this timestep
        mean_dist = compute_mean_min_l1_distance_for_timestep(
            cluster_data, reference_feats, timestep
        )
        per_timestep_means.append(mean_dist)
        
        print(f"  Timestep {timestep:2d} (frame {frame_str}): {mean_dist:.4f}")
    
    # Compute overall mean
    overall_mean = np.mean(per_timestep_means) if per_timestep_means else 0.0
    
    return per_timestep_means, overall_mean


def main():
    # Paths
    base_dir = Path("/mnt/home/nicolasstellwag/surgery-scene-graphs")
    
    cluster_feats_path_1 = (
        base_dir / "output/aelocal_states15_iter10k/video01_00240/graph/c_qwen_feats.npz"
    )
    cluster_feats_path_2 = (
        base_dir / "output/aenoisy_states15_iter10k/video01_00240/graph/c_qwen_feats.npz"
    )
    reference_feats_dir = (
        base_dir
        / "data/preprocessed/qwen3_da3_subsampled/video01_00240/qwen3_patch_features"
    )
    
    print("=" * 80)
    print("Comparing clusterwise features to reference patch features")
    print("Across 20 timesteps (frame numbers increase by 4 per timestep)")
    print("=" * 80)
    print()
    
    # Compute for first splat file
    print("File 1: aelocal_states15_iter10k")
    print("-" * 80)
    timestep_means_1, overall_mean_1 = compute_all_timesteps(
        cluster_feats_path_1, reference_feats_dir, n_timesteps=20
    )
    print()
    
    # Compute for second splat file
    print("File 2: aenoisy_states15_iter10k")
    print("-" * 80)
    timestep_means_2, overall_mean_2 = compute_all_timesteps(
        cluster_feats_path_2, reference_feats_dir, n_timesteps=20
    )
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nPer-timestep results:")
    print(f"{'Timestep':<10} {'Frame':<10} {'aelocal':<15} {'aenoisy':<15} {'Diff':<10}")
    print("-" * 65)
    for timestep in range(len(timestep_means_1)):
        frame_num = timestep * 4
        frame_str = f"{frame_num:06d}"
        diff = abs(timestep_means_1[timestep] - timestep_means_2[timestep])
        print(
            f"{timestep:<10} {frame_str:<10} {timestep_means_1[timestep]:<15.4f} "
            f"{timestep_means_2[timestep]:<15.4f} {diff:<10.4f}"
        )
    
    print("\nOverall means:")
    print(f"aelocal_states15_iter10k:  {overall_mean_1:.4f}")
    print(f"aenoisy_states15_iter10k:  {overall_mean_2:.4f}")
    print(f"Difference: {abs(overall_mean_1 - overall_mean_2):.4f}")


if __name__ == "__main__":
    main()
