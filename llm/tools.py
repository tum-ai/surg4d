import json
import torch
import numpy as np
from functools import partial
from typing import Dict, Any, List, Callable, Tuple, Optional
from scipy.spatial import KDTree
import rerun as rr

from benchmark.graph_utils import get_coord_transformations
from autoencoder.model_qwen import QwenAutoencoder
from rerun_utils import _compute_scene_extent


IMAGE_PLACEHOLDER = "<image/>"


# helpers


def decode_latents(
    latents: np.ndarray,
    autoencoder: QwenAutoencoder,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Decode latent features to full Qwen features using the autoencoder.

    Args:
        latents: Latent features (N, latent_dim)
        autoencoder: QwenAutoencoder instance
        batch_size: Batch size for decoding

    Returns:
        Decoded features as torch.Tensor (N, full_dim * 4) in concatenated format
        [main | d0 | d1 | d2] ready for use with Qwen3
    """
    device = next(autoencoder.parameters()).device
    decoded_feats = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch_size):
            batch = torch.tensor(
                latents[i : min(i + batch_size, latents.shape[0])],
                device=device,
                dtype=torch.float32,
            )
            decoded = autoencoder.decode(batch)
            decoded_feats.append(decoded.cpu())
    return torch.cat(decoded_feats, dim=0)


# tools


spec_node_distances_through_time = {
    "type": "function",
    "function": {
        "name": "node_distances_through_time",
        "description": "Returns the distances between two nodes for all timesteps. The distances are computed as min distance(p_i, q_j), where p_i is a point in the first node and q_j is a point in the second node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id_1": {
                    "type": "integer",
                    "description": "The first node's id",
                },
                "node_id_2": {
                    "type": "integer",
                    "description": "The second node's id",
                },
            },
            "required": ["node_id_1", "node_id_2"],
        },
    },
}


def node_distances_through_time(
    positions: np.ndarray,
    clusters: np.ndarray,
    node_id_1: int,
    node_id_2: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Compute minimum distances between two nodes through time.
    
    Uses KDTree for efficient pairwise distance computation and takes the mean
    of the lowest 5 percentile distances for robustness (similar to overlap_position).
    
    Args:
        positions: Gaussian positions (T, n_gaussians, 3)
        clusters: Cluster assignment per gaussian (n_gaussians,)
        node_id_1: First node id
        node_id_2: Second node id
        toolkit: Optional GraphTools instance for rerun logging
    """
    n_timesteps = positions.shape[0]
    n_nodes = int(clusters.max()) + 1

    # Validate node ids
    if not (0 <= node_id_1 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_1={node_id_1} out of range [0, {n_nodes})"}
            )
        }
    if not (0 <= node_id_2 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_2={node_id_2} out of range [0, {n_nodes})"}
            )
        }

    # Get gaussian indices for each cluster
    mask1 = clusters == node_id_1
    mask2 = clusters == node_id_2

    distances = []
    # Store boundary points for rerun logging
    boundary_points_1 = []
    boundary_points_2 = []

    for t in range(n_timesteps):
        # Get positions at this timestep
        pos1 = positions[t, mask1]  # (n_gaussians_1, 3)
        pos2 = positions[t, mask2]  # (n_gaussians_2, 3)

        # Build KDTrees for each cluster
        tree1 = KDTree(pos1)
        tree2 = KDTree(pos2)

        # Find distances to closest point in other cluster
        dists1, _ = tree2.query(pos1)  # For each point in cluster 1, dist to closest in cluster 2
        dists2, _ = tree1.query(pos2)  # For each point in cluster 2, dist to closest in cluster 1

        # Take points in bottom percentile of distances (closest to contact)
        threshold1 = np.percentile(dists1, 5)
        threshold2 = np.percentile(dists2, 5)

        boundary_mask1 = dists1 <= threshold1
        boundary_mask2 = dists2 <= threshold2

        boundary1 = pos1[boundary_mask1]
        boundary2 = pos2[boundary_mask2]

        # Compute mean distance from the selected boundary gaussians
        # This is the mean of the lowest-percentile distances
        selected_dists = np.concatenate([dists1[boundary_mask1], dists2[boundary_mask2]])
        mean_min_distance = float(np.mean(selected_dists))

        dist_entry = {
            "timestep": t,
            "distance": round(mean_min_distance, 4),
        }
        distances.append(dist_entry)

        # Store boundary points for visualization
        boundary_points_1.append(boundary1)
        boundary_points_2.append(boundary2)

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_node_distances"
        
        # Get masks for visualization (using original coordinates)
        viz_mask1 = toolkit.clusters == node_id_1
        viz_mask2 = toolkit.clusters == node_id_2
        
        scene_extent = _compute_scene_extent(toolkit.positions.reshape(-1, 3))
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        for t in range(n_timesteps):
            rr.set_time("timestep", sequence=t)
            
            # Log all points for both clusters with highlight colors
            rr.log(
                f"{prefix}/node_{node_id_1}",
                rr.Points3D(
                    positions=toolkit.positions[t][viz_mask1],
                    colors=[[255, 0, 0]],
                    radii=point_radius
                )
            )
            rr.log(
                f"{prefix}/node_{node_id_2}",
                rr.Points3D(
                    positions=toolkit.positions[t][viz_mask2],
                    colors=[[0, 0, 255]],
                    radii=point_radius
                )
            )
            
            # Compute mean positions of boundary gaussians in original coordinates
            boundary1_orig = toolkit.point_n2o(boundary_points_1[t])
            boundary2_orig = toolkit.point_n2o(boundary_points_2[t])
            
            mean_boundary1 = np.mean(boundary1_orig, axis=0)
            mean_boundary2 = np.mean(boundary2_orig, axis=0)
            midpoint = (mean_boundary1 + mean_boundary2) / 2
            dist = distances[t]["distance"]
            
            # Log distance marker at midpoint
            rr.log(
                f"{prefix}/distance_marker",
                rr.Points3D(
                    positions=[midpoint],
                    colors=[[0, 255, 0]],
                    radii=point_radius * 1.5,
                    labels=[f"d={dist:.3f}"],
                    show_labels=True,
                )
            )
            
            # Log connection line from mean of boundary gaussians in cluster 1 to cluster 2
            rr.log(
                f"{prefix}/connection",
                rr.LineStrips3D(
                    strips=[[mean_boundary1, mean_boundary2]],
                    colors=[[128, 128, 128]],
                    radii=[point_radius * 0.5],
                )
            )

    return {
        "text": json.dumps(
            {
                "node_id_1": int(node_id_1),
                "node_id_2": int(node_id_2),
                "distances": distances,
            }
        ),
    }


spec_node_overlap_scores_through_time = {
    "type": "function",
    "function": {
        "name": "node_overlap_scores_through_time",
        "description": "Returns the spatial overlap scores (Bhattacharyya coefficients) between two graph nodes at all timesteps. Higher values (closer to 1) indicate greater spatial overlap between the nodes' gaussian distributions; lower values (closer to 0) indicate the nodes are spatially separated.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id_1": {
                    "type": "integer",
                    "description": "The first node's id",
                },
                "node_id_2": {
                    "type": "integer",
                    "description": "The second node's id",
                },
            },
            "required": ["node_id_1", "node_id_2"],
        },
    },
}


def node_overlap_scores_through_time(
    bhattacharyya_coeffs: np.ndarray,
    node_id_1: int,
    node_id_2: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return the Bhattacharyya coefficients (overlap scores) between two nodes through time.

    Args:
        bhattacharyya_coeffs: Dense Bhattacharyya coefficients (T, n_clusters, n_clusters)
        node_id_1: First node id
        node_id_2: Second node id
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with overlap scores at all timesteps.
    """
    n_timesteps = bhattacharyya_coeffs.shape[0]
    n_nodes = bhattacharyya_coeffs.shape[1]

    # Validate node ids
    if not (0 <= node_id_1 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_1={node_id_1} out of range [0, {n_nodes})"}
            )
        }
    if not (0 <= node_id_2 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_2={node_id_2} out of range [0, {n_nodes})"}
            )
        }

    overlap_scores = []
    for t in range(n_timesteps):
        score_entry = {
            "timestep": t,
            "overlap_score": round(
                float(bhattacharyya_coeffs[t, node_id_1, node_id_2]), 4
            ),
        }
        overlap_scores.append(score_entry)

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_node_overlap_scores"
        
        # Get masks for the two nodes
        mask1 = toolkit.clusters == node_id_1
        mask2 = toolkit.clusters == node_id_2
        
        scene_extent = _compute_scene_extent(toolkit.positions.reshape(-1, 3))
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        for t in range(n_timesteps):
            rr.set_time("timestep", sequence=t)
            
            score = overlap_scores[t]["overlap_score"]
            
            # Color-code nodes based on overlap score (red = low, green = high)
            color_val = int(score * 255)
            node_color = [255 - color_val, color_val, 0]
            
            # Log all points for both clusters with color-coded overlap
            rr.log(
                f"{prefix}/node_{node_id_1}",
                rr.Points3D(
                    positions=toolkit.positions[t][mask1],
                    colors=[node_color],
                    radii=point_radius,
                )
            )
            rr.log(
                f"{prefix}/node_{node_id_2}",
                rr.Points3D(
                    positions=toolkit.positions[t][mask2],
                    colors=[node_color],
                    radii=point_radius,
                )
            )
            
            # Log overlap score as text at midpoint
            pos1 = toolkit.centroids[t, node_id_1]
            pos2 = toolkit.centroids[t, node_id_2]
            midpoint = (pos1 + pos2) / 2
            rr.log(
                f"{prefix}/score_marker",
                rr.Points3D(
                    positions=[midpoint],
                    colors=[[255, 255, 0]],
                    radii=point_radius * 1.5,
                    labels=[f"overlap={score:.3f}"],
                    show_labels=True,
                )
            )

    return {
        "text": json.dumps(
            {
                "node_id_1": int(node_id_1),
                "node_id_2": int(node_id_2),
                "overlap_scores": overlap_scores,
            }
        ),
    }


spec_node_overlap_position_at_time = {
    "type": "function",
    "function": {
        "name": "node_overlap_position_at_time",
        "description": "Return the point in 3D at which two graph nodes overlap at a given timestep.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id_1": {
                    "type": "integer",
                    "description": "The first node's id",
                },
                "node_id_2": {
                    "type": "integer",
                    "description": "The second node's id",
                },
                "timestep": {
                    "type": "integer",
                    "description": "The timestep at which to find the overlap point",
                },
            },
            "required": ["node_id_1", "node_id_2", "timestep"],
        },
    },
}


def node_overlap_position_at_time(
    positions: np.ndarray,
    clusters: np.ndarray,
    centroids: np.ndarray,
    node_id_1: int,
    node_id_2: int,
    timestep: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

    # Validate inputs
    if not (0 <= node_id_1 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_1={node_id_1} out of range [0, {n_nodes})"}
            )
        }
    if not (0 <= node_id_2 < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id_2={node_id_2} out of range [0, {n_nodes})"}
            )
        }
    if not (0 <= timestep < n_timesteps):
        return {
            "text": json.dumps(
                {"error": f"timestep={timestep} out of range [0, {n_timesteps})"}
            )
        }

    # Get gaussian indices for each cluster
    mask1 = clusters == node_id_1
    mask2 = clusters == node_id_2

    # Get positions at this timestep
    pos1 = positions[timestep, mask1]  # (n_gaussians_1, 3)
    pos2 = positions[timestep, mask2]  # (n_gaussians_2, 3)

    # Build KDTrees for each cluster
    tree1 = KDTree(pos1)
    tree2 = KDTree(pos2)

    # Find boundary points (closest to other cluster)
    dists1, _ = tree2.query(pos1)
    dists2, _ = tree1.query(pos2)

    # Take points in bottom percentile of distances (closest to contact)
    threshold1 = np.percentile(dists1, 5)
    threshold2 = np.percentile(dists2, 5)

    boundary1 = pos1[dists1 <= threshold1]
    boundary2 = pos2[dists2 <= threshold2]

    # Centroid of contact region
    contact_point = np.mean(np.vstack([boundary1, boundary2]), axis=0)

    result = {
        "node_id_1": int(node_id_1),
        "node_id_2": int(node_id_2),
        "timestep": int(timestep),
        "point": [round(float(p), 4) for p in contact_point],
    }

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_t{timestep:02d}_overlap_position"
        
        rr.set_time("timestep", sequence=timestep)
        
        # Use original coordinates for visualization
        mask1 = toolkit.clusters == node_id_1
        mask2 = toolkit.clusters == node_id_2
        pos1_orig = toolkit.positions[timestep, mask1]
        pos2_orig = toolkit.positions[timestep, mask2]
        
        scene_extent = _compute_scene_extent(toolkit.positions[timestep])
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        # Log both nodes
        rr.log(
            f"{prefix}/node_{node_id_1}",
            rr.Points3D(positions=pos1_orig, colors=[[255, 0, 0]], radii=point_radius)
        )
        rr.log(
            f"{prefix}/node_{node_id_2}",
            rr.Points3D(positions=pos2_orig, colors=[[0, 0, 255]], radii=point_radius)
        )
        
        # Convert contact point back to original coordinates
        contact_point_orig = toolkit.point_n2o(contact_point.reshape(1, 3))[0]
        
        # Log overlap position marker (larger, bright color)
        rr.log(
            f"{prefix}/overlap_position",
            rr.Points3D(
                positions=[contact_point_orig],
                colors=[[0, 255, 0]],
                radii=point_radius * 3,
                labels=["overlap"],
                show_labels=True,
            )
        )

    return {"text": json.dumps(result)}


spec_inspect_highres_node_at_time = {
    "type": "function",
    "function": {
        "name": "inspect_highres_node_at_time",
        "description": "Returns a detailed pseudo-image containing visual features of ALL gaussians belonging to a node at a given timestep. Use this to get a more complete visual understanding of a node than the summary descriptor provides. DO NOT CALL THIS TOOL MORE THAN 5 TIMES! The results are long and can cause issues with your context window.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "The node's id",
                },
                "timestep": {
                    "type": "integer",
                    "description": "The timestep at which to inspect the node",
                },
            },
            "required": ["node_id", "timestep"],
        },
    },
}


def inspect_highres_node_at_time(
    patch_latents_through_time: np.ndarray,
    clusters: np.ndarray,
    autoencoder: QwenAutoencoder,
    node_id: int,
    timestep: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Decode and return all gaussian features for a node at a timestep.

    Args:
        patch_latents_through_time: Latent patch features (T, n_filtered_gaussians, latent_dim)
        clusters: Cluster assignment per gaussian (n_filtered_gaussians,)
        autoencoder: QwenAutoencoder to decode latents to full features
        node_id: The node/cluster id to inspect
        timestep: The timestep to inspect
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with "text" and "vision_features" for the node.
        vision_features contains a single tensor of shape (n_gaussians, full_dim * 4)
        in concatenated format [main | d0 | d1 | d2] as output by the autoencoder.
    """
    MAX_GAUSSIANS_PER_CLUSTER = 2000

    n_timesteps = patch_latents_through_time.shape[0]
    n_nodes = int(clusters.max()) + 1

    # Validate inputs
    if not (0 <= node_id < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id={node_id} out of range [0, {n_nodes})"}
            )
        }
    if not (0 <= timestep < n_timesteps):
        return {
            "text": json.dumps(
                {"error": f"timestep={timestep} out of range [0, {n_timesteps})"}
            )
        }

    # Get latent features for this cluster at this timestep
    cluster_mask = clusters == node_id
    latents = patch_latents_through_time[
        timestep, cluster_mask
    ]  # (n_cluster_gaussians, latent_dim)

    if latents.shape[0] == 0:
        return {"text": json.dumps({"error": f"node_id={node_id} has no gaussians"})}

    # Limit to at most 2k gaussians per cluster
    if latents.shape[0] > MAX_GAUSSIANS_PER_CLUSTER:
        latents = latents[:MAX_GAUSSIANS_PER_CLUSTER]

    # Decode latents to full Qwen features (already in concatenated format)
    vision_features = decode_latents(latents, autoencoder)

    result = {
        "node_id": int(node_id),
        "timestep": int(timestep),
        "n_gaussians": min(MAX_GAUSSIANS_PER_CLUSTER, latents.shape[0]),
        "detailed_view": IMAGE_PLACEHOLDER,
    }

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_t{timestep:02d}_highres_node"
        
        rr.set_time("timestep", sequence=timestep)
        
        # Just highlight the cluster - same approach as other tools
        node_mask = toolkit.clusters == node_id
        node_positions = toolkit.positions[timestep, node_mask]
        
        scene_extent = _compute_scene_extent(toolkit.positions[timestep])
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        # Log the node points
        rr.log(
            f"{prefix}/node_{node_id}",
            rr.Points3D(
                positions=node_positions,
                colors=[[255, 200, 0]],
                radii=point_radius,
            )
        )

    return {
        "text": json.dumps(result),
        "vision_features": [vision_features],
    }


spec_inspect_node_through_time = {
    "type": "function",
    "function": {
        "name": "inspect_node_through_time",
        "description": "Returns the lowres visual descriptors and geometric properties (centroid, bbox-center, bbox-extents) of a single node through ALL timesteps. Use this to track how an object's appearance and position evolve over time.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "The node's id",
                },
            },
            "required": ["node_id"],
        },
    },
}


def inspect_node_through_time(
    qwen_feats,
    centroids: np.ndarray,
    centers: np.ndarray,
    extents: np.ndarray,
    node_id: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return lowres visual descriptors and properties for a node through all timesteps.

    Args:
        qwen_feats: NpzFile or dict mapping cluster_id to features (T, n_feats, 3584)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        node_id: The node/cluster id to inspect
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with "text" (JSON) and "vision_features" for the node through all timesteps.
    """
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

    # Validate node_id
    if not (0 <= node_id < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id={node_id} out of range [0, {n_nodes})"}
            )
        }

    # Get features for this node - handle both NpzFile and dict
    if str(node_id) not in qwen_feats:
        return {
            "text": json.dumps({"error": f"node_id={node_id} not found in qwen_feats"})
        }
    node_features = qwen_feats[str(node_id)]  # (T, n_feats, dim)

    # Build JSON representation
    timesteps_data = []
    vision_features = []

    for t in range(n_timesteps):
        c = centroids[t, node_id]
        ctr = centers[t, node_id]
        ext = extents[t, node_id]

        entry = {
            "timestep": int(t),
            "rough_image": IMAGE_PLACEHOLDER,
            "centroid": {
                "x": round(float(c[0]), 2),
                "y": round(float(c[1]), 2),
                "z": round(float(c[2]), 2),
            },
            "bbox_center": {
                "x": round(float(ctr[0]), 2),
                "y": round(float(ctr[1]), 2),
                "z": round(float(ctr[2]), 2),
            },
            "bbox_extent": {
                "x": round(float(ext[0]), 2),
                "y": round(float(ext[1]), 2),
                "z": round(float(ext[2]), 2),
            },
        }
        timesteps_data.append(entry)

        # Add vision features for this timestep
        vision_features.append(torch.Tensor(node_features[t]))

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_node_through_time"
        
        # Get positions for this node through time
        node_mask = toolkit.clusters == node_id
        scene_extent = _compute_scene_extent(toolkit.positions.reshape(-1, 3))
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        for t in range(n_timesteps):
            rr.set_time("timestep", sequence=t)
            
            node_positions = toolkit.positions[t, node_mask]
            
            rr.log(
                f"{prefix}/node_{node_id}",
                rr.Points3D(
                    positions=node_positions,
                    colors=[[100, 200, 255]],
                    radii=point_radius,
                )
            )

    return {
        "text": json.dumps({
            "node_id": int(node_id),
            "timesteps": timesteps_data,
        }),
        "vision_features": vision_features,
    }


spec_inspect_scene_at_time = {
    "type": "function",
    "function": {
        "name": "inspect_scene_at_time",
        "description": "Returns the complete scene state at a given timestep: all nodes with their lowres visual descriptors and geometric properties (centroid, bbox-center, bbox-extents). Use this to get a full snapshot of the scene at any point in time. DO NOT CALL THIS TOOL MORE THAN 3 TIMES! The results are long and can cause issues with your context window.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestep": {
                    "type": "integer",
                    "description": "The timestep at which to inspect the scene",
                },
            },
            "required": ["timestep"],
        },
    },
}


def inspect_scene_at_time(
    qwen_feats,
    centroids: np.ndarray,
    centers: np.ndarray,
    extents: np.ndarray,
    timestep: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return the complete scene state at a given timestep.

    Args:
        qwen_feats: NpzFile or dict mapping cluster_id to features (T, n_feats, 3584)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        timestep: The timestep to inspect
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with "text" (JSON) and "vision_features" for all nodes at the timestep.
    """
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

    # Validate timestep
    if not (0 <= timestep < n_timesteps):
        return {
            "text": json.dumps(
                {"error": f"timestep={timestep} out of range [0, {n_timesteps})"}
            )
        }

    # Get node features at this timestep, sorted by node id
    node_feat_indices = sorted(list(qwen_feats.keys()), key=lambda x: int(x))
    node_feats_at_t = [qwen_feats[idx][timestep] for idx in node_feat_indices]

    # Build JSON representation
    nodes_data = []
    vision_features = []

    for n in range(n_nodes):
        c = centroids[timestep, n]
        ctr = centers[timestep, n]
        ext = extents[timestep, n]

        nodes_data.append({
            "node_id": int(n),
            "rough_image": IMAGE_PLACEHOLDER,
            "centroid": {
                "x": round(float(c[0]), 2),
                "y": round(float(c[1]), 2),
                "z": round(float(c[2]), 2),
            },
            "bbox_center": {
                "x": round(float(ctr[0]), 2),
                "y": round(float(ctr[1]), 2),
                "z": round(float(ctr[2]), 2),
            },
            "bbox_extent": {
                "x": round(float(ext[0]), 2),
                "y": round(float(ext[1]), 2),
                "z": round(float(ext[2]), 2),
            },
        })

        # Add vision features for this node
        vision_features.append(torch.Tensor(node_feats_at_t[n]))

    result = {
        "timestep": int(timestep),
        "nodes": nodes_data,
    }

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_t{timestep:02d}_scene_at_time"
        
        rr.set_time("timestep", sequence=timestep)
        
        # Log graph edges at this timestep (use original coordinates)
        A = toolkit.adjacency[timestep]
        edge_indices = np.where(A > 0)
        
        if len(edge_indices[0]) > 0:
            edge_weights = A[edge_indices]
            scene_extent = _compute_scene_extent(toolkit.centroids[timestep])
            
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
            
            # Log edges
            for idx, (u, v) in enumerate(zip(edge_indices[0], edge_indices[1])):
                if u < v:  # Avoid duplicate edges for symmetric adjacency
                    start_pos = toolkit.centroids[timestep, u]
                    end_pos = toolkit.centroids[timestep, v]
                    weight = normalized_weights[idx]
                    
                    color = [0, 0, 0]
                    thickness = max(scene_extent * (0.002 + 0.008 * weight), 1e-5)
                    
                    rr.log(
                        f"{prefix}/edges/edge_{idx}",
                        rr.LineStrips3D(
                            strips=[[start_pos, end_pos]],
                            colors=[color],
                            radii=[thickness],
                        )
                    )
        
        # Log all nodes at this timestep
        scene_extent = _compute_scene_extent(toolkit.positions[timestep])
        point_radius = max(scene_extent * 0.008, 1e-5)
        
        for node_id in range(n_nodes):
            node_mask = toolkit.clusters == node_id
            node_positions = toolkit.positions[timestep, node_mask]
            
            rr.log(
                f"{prefix}/nodes/{node_id}",
                rr.Points3D(
                    positions=node_positions,
                    radii=point_radius,
                )
            )

    return {
        "text": json.dumps(result),
        "vision_features": vision_features,
    }


spec_node_movement = {
    "type": "function",
    "function": {
        "name": "node_movement",
        "description": "Returns the centroid and bounding box of a node at each timestep. Use this to track how a node moves through time.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "The node's id",
                },
            },
            "required": ["node_id"],
        },
    },
}


def node_movement(
    centroids: np.ndarray,
    centers: np.ndarray,
    extents: np.ndarray,
    node_id: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return centroid and bounding box for a node at each timestep.

    Args:
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        node_id: The node/cluster id to track
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with centroid and bbox at each timestep.
    """
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

    # Validate node_id
    if not (0 <= node_id < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id={node_id} out of range [0, {n_nodes})"}
            )
        }

    # Build movement data
    movement_data = []
    for t in range(n_timesteps):
        c = centroids[t, node_id]
        ctr = centers[t, node_id]
        ext = extents[t, node_id]

        entry = {
            "timestep": int(t),
            "centroid": {
                "x": round(float(c[0]), 4),
                "y": round(float(c[1]), 4),
                "z": round(float(c[2]), 4),
            },
            "bbox_center": {
                "x": round(float(ctr[0]), 4),
                "y": round(float(ctr[1]), 4),
                "z": round(float(ctr[2]), 4),
            },
            "bbox_extent": {
                "x": round(float(ext[0]), 4),
                "y": round(float(ext[1]), 4),
                "z": round(float(ext[2]), 4),
            },
        }
        movement_data.append(entry)

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_node_movement"

        # Get mask for this node
        node_mask = toolkit.clusters == node_id
        scene_extent = _compute_scene_extent(toolkit.positions.reshape(-1, 3))
        point_radius = max(scene_extent * 0.008, 1e-5)

        for t in range(n_timesteps):
            rr.set_time("timestep", sequence=t)

            # Log node points at this timestep
            node_positions = toolkit.positions[t, node_mask]
            rr.log(
                f"{prefix}/node_{node_id}",
                rr.Points3D(
                    positions=node_positions,
                    colors=[[255, 128, 0]],  # Orange
                    radii=point_radius,
                )
            )

    return {
        "text": json.dumps({
            "node_id": int(node_id),
            "movement": movement_data,
        }),
    }


spec_voxelize_scene = {
    "type": "function",
    "function": {
        "name": "voxelize_scene",
        "description": "Voxelizes the scene at a given timestep and returns visual descriptors for each voxel containing scene content. The specified bounding box (or full scene if no bbox is provided) is divided into 5x5x5 voxels. Use this to inspect the scene detached from its graph representation.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestep": {
                    "type": "integer",
                    "description": "The timestep at which to voxelize the scene",
                },
                "bbox": {
                    "type": "array",
                    "description": "Optional bounding box [x_center, y_center, z_center, x_size, y_size, z_size] defining the region to voxelize. If null/omitted, voxelizes the entire scene.",
                    "items": {"type": "number"},
                },
            },
            "required": ["timestep"],
        },
    },
}


def voxelize_scene(
    positions: np.ndarray,
    patch_latents_through_time: np.ndarray,
    autoencoder: QwenAutoencoder,
    timestep: int,
    bbox: List[float] = None,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Sample the scene at regular 3D spatial locations with visual descriptors per location.

    Each spatial sample (voxel) contains visual descriptors aggregated from all scene content
    within that location, allowing the agent to spatially query "where is X" by examining
    descriptors at different locations.

    The scene is divided into grid_size x grid_size x grid_size voxels. Only non-empty
    voxels (those containing scene content) are returned.

    Args:
        positions: Gaussian positions through time (T, n_filtered_gaussians, 3)
        patch_latents_through_time: Latent patch features (T, n_filtered_gaussians, latent_dim)
        autoencoder: QwenAutoencoder to decode latents to full features
        timestep: The timestep at which to sample
        bbox: Optional [x_center, y_center, z_center, x_size, y_size, z_size].
              If None, samples the entire scene.
              NOTE: This format corresponds to Omni3D bbox format without rotation,
              which Qwen3 was trained on - important for paper justification.
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with "text" (JSON description) and "vision_features" (one tensor per non-empty sample).
        Text contains sample metadata (voxel_index, bbox, content_density, grid index).
        voxel_index is a 3D tuple (i, j, k) indicating the voxel's position in the grid.
        Vision features are visual descriptors for scene content at each sampled location.
        Only non-empty voxels are included in the output.
    """
    MAX_GAUSSIANS_PER_VOXEL = 64
    GRID_SIZE = 5

    n_timesteps = positions.shape[0]

    # Validate timestep
    if not (0 <= timestep < n_timesteps):
        return {
            "text": json.dumps(
                {"error": f"timestep={timestep} out of range [0, {n_timesteps})"}
            )
        }

    # Get positions at this timestep
    pos_t = positions[timestep]  # (n_gaussians, 3)

    # Determine bounding box for sampling
    if bbox is None:
        # Compute bbox from all scene content
        min_coords = pos_t.min(axis=0)
        max_coords = pos_t.max(axis=0)
        bbox_center = (min_coords + max_coords) / 2
        bbox_size = max_coords - min_coords
    else:
        # Parse provided bbox [x_center, y_center, z_center, x_size, y_size, z_size]
        if len(bbox) != 6:
            return {
                "text": json.dumps(
                    {"error": f"bbox must have 6 elements, got {len(bbox)}"}
                )
            }
        bbox_center = np.array(bbox[:3])
        bbox_size = np.array(bbox[3:])

    # Compute voxel size from grid size
    voxel_size = bbox_size / GRID_SIZE

    # Compute grid origin (bottom-left-back corner)
    grid_min = bbox_center - bbox_size / 2

    # Assign each gaussian to a voxel
    # For each gaussian position, compute which voxel (i, j, k) it belongs to
    gaussian_voxel_indices = np.floor((pos_t - grid_min) / voxel_size).astype(int)

    # Build output - iterate over all voxels, skip empty ones
    voxels_data = []
    vision_features = []
    voxel_centers_for_logging = []
    voxel_indices_for_logging = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                voxel_idx = (i, j, k)
                # Find gaussians in this voxel
                gaussian_indices = np.where(np.all(gaussian_voxel_indices == voxel_idx, axis=1))[0]
                if len(gaussian_indices) == 0:
                    continue

                # Compute this voxel's 3D bounding box
                voxel_min = grid_min + np.array(voxel_idx) * voxel_size
                voxel_max = voxel_min + voxel_size
                voxel_center = (voxel_min + voxel_max) / 2

                # Get latents for all gaussians in this voxel
                n_gaussians = min(len(gaussian_indices), MAX_GAUSSIANS_PER_VOXEL)
                voxel_latents = patch_latents_through_time[timestep][gaussian_indices[:n_gaussians]]

                # Decode latents to visual descriptors
                voxel_visual_descriptor = decode_latents(voxel_latents, autoencoder)

                # Store sample metadata
                voxels_data.append(
                    {
                        "voxel_index": [int(i), int(j), int(k)],
                        "bbox": [round(float(x), 2) for x in voxel_center]
                        + [round(float(x), 2) for x in voxel_size],
                        "visual_descriptor": IMAGE_PLACEHOLDER,
                    }
                )
                vision_features.append(voxel_visual_descriptor)
                voxel_centers_for_logging.append(voxel_center)
                voxel_indices_for_logging.append(voxel_idx)

    if len(voxels_data) == 0:
        return {
            "text": json.dumps(
                {"error": f"No voxels containing scene content found for timestep {timestep} and bbox {bbox} - are you sure the bbox is sensible?"}
            )
        }

    # Build response
    response_data = {
        "timestep": int(timestep),
        "query_bbox": [float(x) for x in bbox_center] + [float(x) for x in bbox_size]
        if bbox is not None
        else None,
        "voxels": voxels_data,
    }

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_t{timestep:02d}_voxelize_scene"
        
        rr.set_time("timestep", sequence=timestep)
        
        # Recompute grid in ORIGINAL coordinates for visualization
        pos_t_orig = toolkit.positions[timestep]
        
        # Determine bounding box for visualization (original coordinates)
        if bbox is None:
            min_coords_orig = pos_t_orig.min(axis=0)
            max_coords_orig = pos_t_orig.max(axis=0)
            bbox_center_orig = (min_coords_orig + max_coords_orig) / 2
            bbox_size_orig = max_coords_orig - min_coords_orig
        else:
            # Convert bbox from normalized to original coordinates
            # Transform the min and max corners instead of center/size separately
            bbox_center_norm = np.array(bbox[:3])
            bbox_size_norm = np.array(bbox[3:])
            bbox_min_norm = bbox_center_norm - bbox_size_norm / 2
            bbox_max_norm = bbox_center_norm + bbox_size_norm / 2
            bbox_min_orig = toolkit.point_n2o(bbox_min_norm.reshape(1, 3))[0]
            bbox_max_orig = toolkit.point_n2o(bbox_max_norm.reshape(1, 3))[0]
            bbox_center_orig = (bbox_min_orig + bbox_max_orig) / 2
            bbox_size_orig = bbox_max_orig - bbox_min_orig
        
        voxel_size_orig = bbox_size_orig / GRID_SIZE
        grid_min_orig = bbox_center_orig - bbox_size_orig / 2
        
        scene_extent = _compute_scene_extent(pos_t_orig)
        voxel_marker_radius = max(scene_extent * 0.005, 1e-5)  # Smaller markers
        thin_line_radius = max(scene_extent * 0.001, 1e-6)  # Very thin lines
        
        # Log voxel centers and wireframes for occupied voxels
        for voxel_idx in voxel_indices_for_logging:
            i, j, k = voxel_idx
            
            # Compute voxel position in original coordinates
            voxel_center_orig = grid_min_orig + (np.array([i, j, k]) + 0.5) * voxel_size_orig
            
            # Log center point with label (show only on hover)
            rr.log(
                f"{prefix}/voxels/{i}_{j}_{k}/center",
                rr.Points3D(
                    positions=[voxel_center_orig],
                    colors=[[200, 100, 200]],
                    radii=voxel_marker_radius,
                    labels=[f"({i},{j},{k})"],
                    show_labels=False,  # Only show on hover
                )
            )
            
            # Compute voxel corners in original coordinates
            voxel_min_orig = grid_min_orig + np.array([i, j, k]) * voxel_size_orig
            voxel_max_orig = voxel_min_orig + voxel_size_orig
            
            # Create 12 edges for this voxel's wireframe
            voxel_edges = [
                # Bottom face (z=min)
                [voxel_min_orig, [voxel_max_orig[0], voxel_min_orig[1], voxel_min_orig[2]]],
                [[voxel_max_orig[0], voxel_min_orig[1], voxel_min_orig[2]], [voxel_max_orig[0], voxel_max_orig[1], voxel_min_orig[2]]],
                [[voxel_max_orig[0], voxel_max_orig[1], voxel_min_orig[2]], [voxel_min_orig[0], voxel_max_orig[1], voxel_min_orig[2]]],
                [[voxel_min_orig[0], voxel_max_orig[1], voxel_min_orig[2]], voxel_min_orig],
                # Top face (z=max)
                [[voxel_min_orig[0], voxel_min_orig[1], voxel_max_orig[2]], [voxel_max_orig[0], voxel_min_orig[1], voxel_max_orig[2]]],
                [[voxel_max_orig[0], voxel_min_orig[1], voxel_max_orig[2]], voxel_max_orig],
                [voxel_max_orig, [voxel_min_orig[0], voxel_max_orig[1], voxel_max_orig[2]]],
                [[voxel_min_orig[0], voxel_max_orig[1], voxel_max_orig[2]], [voxel_min_orig[0], voxel_min_orig[1], voxel_max_orig[2]]],
                # Vertical edges
                [voxel_min_orig, [voxel_min_orig[0], voxel_min_orig[1], voxel_max_orig[2]]],
                [[voxel_max_orig[0], voxel_min_orig[1], voxel_min_orig[2]], [voxel_max_orig[0], voxel_min_orig[1], voxel_max_orig[2]]],
                [[voxel_max_orig[0], voxel_max_orig[1], voxel_min_orig[2]], voxel_max_orig],
                [[voxel_min_orig[0], voxel_max_orig[1], voxel_min_orig[2]], [voxel_min_orig[0], voxel_max_orig[1], voxel_max_orig[2]]],
            ]
            
            # Log all edges for this voxel
            for edge_idx, edge in enumerate(voxel_edges):
                rr.log(
                    f"{prefix}/voxels/{i}_{j}_{k}/edges/edge_{edge_idx}",
                    rr.LineStrips3D(
                        strips=[edge],
                        colors=[[150, 150, 200]],  # Light blue/purple
                        radii=[thin_line_radius],
                    )
                )
        
        # Log grid bounding box in original coordinates
        bbox_min_orig = grid_min_orig
        bbox_max_orig = grid_min_orig + np.array([GRID_SIZE, GRID_SIZE, GRID_SIZE]) * voxel_size_orig
        
        # Create 12 edges of the bounding box
        edges = [
            # Bottom face
            [bbox_min_orig, [bbox_max_orig[0], bbox_min_orig[1], bbox_min_orig[2]]],
            [[bbox_max_orig[0], bbox_min_orig[1], bbox_min_orig[2]], [bbox_max_orig[0], bbox_max_orig[1], bbox_min_orig[2]]],
            [[bbox_max_orig[0], bbox_max_orig[1], bbox_min_orig[2]], [bbox_min_orig[0], bbox_max_orig[1], bbox_min_orig[2]]],
            [[bbox_min_orig[0], bbox_max_orig[1], bbox_min_orig[2]], bbox_min_orig],
            # Top face
            [[bbox_min_orig[0], bbox_min_orig[1], bbox_max_orig[2]], [bbox_max_orig[0], bbox_min_orig[1], bbox_max_orig[2]]],
            [[bbox_max_orig[0], bbox_min_orig[1], bbox_max_orig[2]], bbox_max_orig],
            [bbox_max_orig, [bbox_min_orig[0], bbox_max_orig[1], bbox_max_orig[2]]],
            [[bbox_min_orig[0], bbox_max_orig[1], bbox_max_orig[2]], [bbox_min_orig[0], bbox_min_orig[1], bbox_max_orig[2]]],
            # Vertical edges
            [bbox_min_orig, [bbox_min_orig[0], bbox_min_orig[1], bbox_max_orig[2]]],
            [[bbox_max_orig[0], bbox_min_orig[1], bbox_min_orig[2]], [bbox_max_orig[0], bbox_min_orig[1], bbox_max_orig[2]]],
            [[bbox_max_orig[0], bbox_max_orig[1], bbox_min_orig[2]], bbox_max_orig],
            [[bbox_min_orig[0], bbox_max_orig[1], bbox_min_orig[2]], [bbox_min_orig[0], bbox_max_orig[1], bbox_max_orig[2]]],
        ]
        
        for idx, edge in enumerate(edges):
            rr.log(
                f"{prefix}/grid_bbox/edge_{idx}",
                rr.LineStrips3D(
                    strips=[edge],
                    colors=[[100, 100, 100]],
                    radii=[voxel_marker_radius * 0.3],
                )
            )

    result = {"text": json.dumps(response_data, indent=2)}
    if vision_features:
        result["vision_features"] = vision_features
    return result


class GraphTools:
    """Tool registry for scene graph operations.

    Provides tools for querying and inspecting scene graph data extracted by
    extract_graphs.py. The autoencoder is required for tools that decode latent
    features (like inspect_highres_node_at_time).

    Args:
        positions: Gaussian positions through time (T, n_filtered_gaussians, 3)
        clusters: Cluster assignment per gaussian (n_filtered_gaussians,)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        adjacency: Adjacency matrices through time (T, n_clusters, n_clusters)
        bhattacharyya_coeffs: Dense Bhattacharyya coefficients through time (T, n_clusters, n_clusters)
        qwen_feats: Qwen features per cluster. Can be NpzFile or dict
            {cluster_id: (T, n_feats, 3584)}
        patch_latents_through_time: Latent patch features (T, n_filtered_gaussians, latent_dim).
            Required for inspect_highres_node_at_time. Only available when store_verbose=True.
        autoencoder: QwenAutoencoder instance for decoding latent features.
            Required for inspect_highres_node_at_time.
    """

    def __init__(
        self,
        positions: np.ndarray,
        clusters: np.ndarray,
        centroids: np.ndarray,
        centers: np.ndarray,
        extents: np.ndarray,
        adjacency: np.ndarray,
        bhattacharyya_coeffs: np.ndarray,
        qwen_feats,
        patch_latents_through_time: np.ndarray,
        autoencoder: QwenAutoencoder,
    ):
        self.positions = positions
        self.clusters = clusters
        self.centroids = centroids
        self.centers = centers
        self.extents = extents
        self.adjacency = adjacency
        self.bhattacharyya_coeffs = bhattacharyya_coeffs
        self.qwen_feats = qwen_feats
        self.patch_latents_through_time = patch_latents_through_time
        self.autoencoder = autoencoder

        self.point_o2n, self.point_n2o, self.distance_o2n, self.distance_n2o = (
            get_coord_transformations(positions)
        )
        
        # Tool call logging state
        self.call_counter = 0
        self.recording_active = False
        self.rr = rr  # Store rerun instance for external logging access

    def start_recording(self, rrd_file: str):
        """Initialize rerun recording to the specified file and log initial graph state.
        
        Args:
            rrd_file: Path to the .rrd file to save visualizations
        """
        self.call_counter = 0
        self.recording_active = True
        
        # Initialize rerun and set output file
        rr.init("tool_calls")
        rr.save(rrd_file)
        
        # Log initial graph structure through all timesteps (use original coordinates)
        n_timesteps = self.positions.shape[0]
        scene_extent = _compute_scene_extent(self.positions.reshape(-1, 3))
        point_radius = max(scene_extent * 0.005, 1e-5)
        
        for t in range(n_timesteps):
            rr.set_time("timestep", sequence=t)
            
            for cluster_id in np.unique(self.clusters):
                mask = self.clusters == cluster_id
                rr.log(
                    f"00_initial_graph/nodes/{cluster_id}",
                    rr.Points3D(
                        positions=self.positions[t][mask],
                        radii=point_radius,
                    ),
                )
    
    def stop_recording(self):
        """Stop recording and reset state."""
        self.recording_active = False
        self.call_counter = 0
    
    def increase_logging_tool_counter(self) -> int:
        """Increment and return the tool call counter."""
        self.call_counter += 1
        return self.call_counter
    
    def log_final_prediction(
        self,
        position: np.ndarray,
        timestep_idx: int,
        label: str,
        entity_name: str = "zz_final_prediction",
    ):
        """Log a final prediction point to the rerun trace.
        
        Args:
            position: 3D position in original coordinates (1, 3) or (3,)
            timestep_idx: Timestep index to log at
            label: Label text to display with the point
            entity_name: Entity name for the rerun log (default: "zz_final_prediction")
        """
        from rerun_utils import _compute_scene_extent
        
        # Ensure position is (1, 3) shape
        pos_arr = np.array(position, dtype=np.float32).reshape(1, 3)
        
        # Set the timestep
        self.rr.set_time("timestep", sequence=int(timestep_idx))
        
        # Compute appropriate point size based on scene extent
        scene_extent = _compute_scene_extent(self.positions[timestep_idx])
        point_radius = max(scene_extent * 0.025, 1e-4)  # Larger than tool points (0.008)
        
        # Log the final prediction as a big red point
        self.rr.log(
            entity_name,
            self.rr.Points3D(
                positions=pos_arr,
                colors=[[255, 0, 0]],  # Bright red
                radii=point_radius,
                labels=[f"prediction: {label}"],
                show_labels=True,
            )
        )

    def get_all_tools(self) -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
        """Returns tools dict with graph data already bound.

        The returned dict maps tool names to (callable, spec) tuples,
        where the callable only requires semantic arguments (e.g. node IDs).
        """
        return {
            "node_distances_through_time": (
                partial(
                    node_distances_through_time,
                    positions=self.point_o2n(self.positions),
                    clusters=self.clusters,
                    toolkit=self,
                ),
                spec_node_distances_through_time,
            ),
            "node_overlap_scores_through_time": (
                partial(
                    node_overlap_scores_through_time,
                    bhattacharyya_coeffs=self.bhattacharyya_coeffs,
                    toolkit=self,
                ),
                spec_node_overlap_scores_through_time,
            ),
            "node_overlap_position_at_time": (
                partial(
                    node_overlap_position_at_time,
                    positions=self.point_o2n(self.positions),
                    clusters=self.clusters,
                    centroids=self.point_o2n(self.centroids),
                    toolkit=self,
                ),
                spec_node_overlap_position_at_time,
            ),
            "inspect_highres_node_at_time": (
                partial(
                    inspect_highres_node_at_time,
                    patch_latents_through_time=self.patch_latents_through_time,
                    clusters=self.clusters,
                    autoencoder=self.autoencoder,
                    toolkit=self,
                ),
                spec_inspect_highres_node_at_time,
            ),
            "inspect_node_through_time": (
                partial(
                    inspect_node_through_time,
                    qwen_feats=self.qwen_feats,
                    centroids=self.point_o2n(self.centroids),
                    centers=self.point_o2n(self.centers),
                    extents=self.distance_o2n(self.extents),
                    toolkit=self,
                ),
                spec_inspect_node_through_time,
            ),
            "inspect_scene_at_time": (
                partial(
                    inspect_scene_at_time,
                    qwen_feats=self.qwen_feats,
                    centroids=self.point_o2n(self.centroids),
                    centers=self.point_o2n(self.centers),
                    extents=self.distance_o2n(self.extents),
                    toolkit=self,
                ),
                spec_inspect_scene_at_time,
            ),
            "voxelize_scene": (
                partial(
                    voxelize_scene,
                    positions=self.point_o2n(self.positions),
                    patch_latents_through_time=self.patch_latents_through_time,
                    autoencoder=self.autoencoder,
                    toolkit=self,
                ),
                spec_voxelize_scene,
            ),
            "node_movement": (
                partial(
                    node_movement,
                    centroids=self.point_o2n(self.centroids),
                    centers=self.point_o2n(self.centers),
                    extents=self.distance_o2n(self.extents),
                    toolkit=self,
                ),
                spec_node_movement,
            ),
        }

    def get_tools_by_name(
        self, tool_names: List[str]
    ) -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
        """Get a subset of tools by name."""
        tools = self.get_all_tools()
        assert all(name in tools for name in tool_names), (
            f"Invalid tool names: {set(tool_names) - set(tools.keys())}"
        )
        return {name: tools[name] for name in tool_names}
