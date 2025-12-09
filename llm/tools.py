import json
import torch
import numpy as np
from functools import partial
from typing import Dict, Any, List, Callable, Tuple
from scipy.spatial import KDTree

from benchmark.graph_utils import get_coord_transformations
from autoencoder.model_qwen import QwenAutoencoder


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
        "description": "Returns the Euclidean distances between the centroids of two graph nodes at all timesteps.",
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
    centroids: np.ndarray,
    node_id_1: int,
    node_id_2: int,
) -> Dict[str, Any]:
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

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

    distances = [
        {
            "timestep": t,
            "distance": round(
                float(
                    np.linalg.norm(centroids[t, node_id_1] - centroids[t, node_id_2])
                ),
                4,
            ),
        }
        for t in range(n_timesteps)
    ]

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
) -> Dict[str, Any]:
    """Return the Bhattacharyya coefficients (overlap scores) between two nodes through time.

    Args:
        bhattacharyya_coeffs: Dense Bhattacharyya coefficients (T, n_clusters, n_clusters)
        node_id_1: First node id
        node_id_2: Second node id

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

    overlap_scores = [
        {
            "timestep": t,
            "overlap_score": round(
                float(bhattacharyya_coeffs[t, node_id_1, node_id_2]), 4
            ),
        }
        for t in range(n_timesteps)
    ]

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

    return {
        "text": json.dumps(
            {
                "node_id_1": int(node_id_1),
                "node_id_2": int(node_id_2),
                "timestep": int(timestep),
                "point": [round(float(p), 4) for p in contact_point],
            }
        ),
    }


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
) -> Dict[str, Any]:
    """Decode and return all gaussian features for a node at a timestep.

    Args:
        patch_latents_through_time: Latent patch features (T, n_filtered_gaussians, latent_dim)
        clusters: Cluster assignment per gaussian (n_filtered_gaussians,)
        autoencoder: QwenAutoencoder to decode latents to full features
        node_id: The node/cluster id to inspect
        timestep: The timestep to inspect

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

    return {
        "text": json.dumps(
            {
                "node_id": int(node_id),
                "timestep": int(timestep),
                "n_gaussians": min(MAX_GAUSSIANS_PER_CLUSTER, latents.shape[0]),
                "detailed_view": IMAGE_PLACEHOLDER,
            }
        ),
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
) -> Dict[str, Any]:
    """Return lowres visual descriptors and properties for a node through all timesteps.

    Args:
        qwen_feats: NpzFile or dict mapping cluster_id to features (T, n_feats, 3584)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        node_id: The node/cluster id to inspect

    Returns:
        Dict with "text" and "vision_features" for the node through all timesteps.
        Text format matches the initial graph representation in prompt_graph_agent,
        with timestep wrappers around each timestep's data.
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

    # Build XML-like text representation similar to prompt_graph_agent
    text_parts = [f'<node-through-time node-id="{node_id}">\n']
    vision_features = []

    for t in range(n_timesteps):
        c = centroids[t, node_id]
        ctr = centers[t, node_id]
        ext = extents[t, node_id]

        text_parts.extend(
            [
                f'<timestep t="{t}">\n',
                "<lowres-visual-descriptor>",
                IMAGE_PLACEHOLDER,
                "</lowres-visual-descriptor>\n",
                f'<centroid x="{c[0]:.2f}" y="{c[1]:.2f}" z="{c[2]:.2f}"/>\n',
                f'<bbox-center x="{ctr[0]:.2f}" y="{ctr[1]:.2f}" z="{ctr[2]:.2f}"/>\n',
                f'<bbox-extent x="{ext[0]:.2f}" y="{ext[1]:.2f}" z="{ext[2]:.2f}"/>\n',
                "</timestep>\n",
            ]
        )

        # Add vision features for this timestep
        vision_features.append(torch.Tensor(node_features[t]))

    text_parts.append("</node-through-time>")

    return {
        "text": "".join(text_parts),
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
) -> Dict[str, Any]:
    """Return the complete scene state at a given timestep.

    Args:
        qwen_feats: NpzFile or dict mapping cluster_id to features (T, n_feats, 3584)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        centers: Cluster centers through time (T, n_clusters, 3)
        extents: Cluster extents through time (T, n_clusters, 3)
        timestep: The timestep to inspect

    Returns:
        Dict with "text" and "vision_features" for all nodes at the timestep.
        Text format matches the initial graph representation in prompt_graph_agent.
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

    # Build XML-like text representation matching prompt_graph_agent format
    text_parts = [f'<graph-nodes t="{timestep}">\n']
    vision_features = []

    for n in range(n_nodes):
        c = centroids[timestep, n]
        ctr = centers[timestep, n]
        ext = extents[timestep, n]

        text_parts.extend(
            [
                f'<node id="{n}">\n',
                "<lowres-visual-descriptor>",
                IMAGE_PLACEHOLDER,
                "</lowres-visual-descriptor>\n",
                f'<centroid x="{c[0]:.2f}" y="{c[1]:.2f}" z="{c[2]:.2f}"/>\n',
                f'<bbox-center x="{ctr[0]:.2f}" y="{ctr[1]:.2f}" z="{ctr[2]:.2f}"/>\n',
                f'<bbox-extent x="{ext[0]:.2f}" y="{ext[1]:.2f}" z="{ext[2]:.2f}"/>\n',
                "</node>\n",
            ]
        )

        # Add vision features for this node
        vision_features.append(torch.Tensor(node_feats_at_t[n]))

    text_parts.append("</graph-nodes>")

    return {
        "text": "".join(text_parts),
        "vision_features": vision_features,
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
        grid_size: Number of voxels per dimension (e.g., 5 creates 5x5x5=125 voxels).
                   Maximum 10 (1000 voxels total).
        bbox: Optional [x_center, y_center, z_center, x_size, y_size, z_size].
              If None, samples the entire scene.
              NOTE: This format corresponds to Omni3D bbox format without rotation,
              which Qwen3 was trained on - important for paper justification.

    Returns:
        Dict with "text" (JSON description) and "vision_features" (one tensor per non-empty sample).
        Text contains sample metadata (bbox, content_density, grid index).
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
                        "bbox": [round(float(x), 2) for x in voxel_center]
                        + [round(float(x), 2) for x in voxel_size],
                        "visual_descriptor": IMAGE_PLACEHOLDER,
                    }
                )
                vision_features.append(voxel_visual_descriptor)

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

    def get_all_tools(self) -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
        """Returns tools dict with graph data already bound.

        The returned dict maps tool names to (callable, spec) tuples,
        where the callable only requires semantic arguments (e.g. node IDs).
        """
        return {
            "node_distances_through_time": (
                partial(
                    node_distances_through_time,
                    centroids=self.point_o2n(self.centroids),
                ),
                spec_node_distances_through_time,
            ),
            "node_overlap_scores_through_time": (
                partial(
                    node_overlap_scores_through_time,
                    bhattacharyya_coeffs=self.bhattacharyya_coeffs,
                ),
                spec_node_overlap_scores_through_time,
            ),
            "node_overlap_position_at_time": (
                partial(
                    node_overlap_position_at_time,
                    positions=self.point_o2n(self.positions),
                    clusters=self.clusters,
                    centroids=self.point_o2n(self.centroids),
                ),
                spec_node_overlap_position_at_time,
            ),
            "inspect_highres_node_at_time": (
                partial(
                    inspect_highres_node_at_time,
                    patch_latents_through_time=self.patch_latents_through_time,
                    clusters=self.clusters,
                    autoencoder=self.autoencoder,
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
                ),
                spec_inspect_scene_at_time,
            ),
            "voxelize_scene": (
                partial(
                    voxelize_scene,
                    positions=self.point_o2n(self.positions),
                    patch_latents_through_time=self.patch_latents_through_time,
                    autoencoder=self.autoencoder,
                ),
                spec_voxelize_scene,
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
