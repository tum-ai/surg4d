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
        {"timestep": t, "distance": round(float(np.linalg.norm(centroids[t, node_id_1] - centroids[t, node_id_2])), 4)}
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
        {"timestep": t, "overlap_score": round(float(bhattacharyya_coeffs[t, node_id_1, node_id_2]), 4)}
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
        "description": "Returns a detailed pseudo-image containing visual features of ALL gaussians belonging to a node at a given timestep. Use this to get a more complete visual understanding of a node than the summary descriptor provides.",
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

    # Decode latents to full Qwen features (already in concatenated format)
    vision_features = decode_latents(latents, autoencoder)

    return {
        "text": json.dumps(
            {
                "node_id": int(node_id),
                "timestep": int(timestep),
                "n_gaussians": int(vision_features.shape[0]),
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
        "description": "Returns the complete scene state at a given timestep: all nodes with their lowres visual descriptors and geometric properties (centroid, bbox-center, bbox-extents). Use this to get a full snapshot of the scene at any point in time.",
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
