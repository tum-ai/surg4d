import json
import numpy as np
from pathlib import Path
from functools import partial
from typing import Dict, Any, List, Callable, Tuple, Optional
from scipy.spatial import KDTree
import rerun as rr
from PIL import Image

from benchmark.graph_utils import get_coord_transformations
from utils.rerun_utils import _compute_scene_extent


IMAGE_PLACEHOLDER = "<image/>"

# Percentile used to select boundary gaussians for KDTree contact/overlap calculations.
# Lower values (e.g. 2) make the "boundary" smaller and reduce influence from outliers.
BOUNDARY_PERCENTILE = 2.0



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
    of the lowest BOUNDARY_PERCENTILE percentile distances for robustness (see `BOUNDARY_PERCENTILE`).
    
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
        threshold1 = np.percentile(dists1, BOUNDARY_PERCENTILE)
        threshold2 = np.percentile(dists2, BOUNDARY_PERCENTILE)

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
        "description": "Returns the spatial overlap scores (Bhattacharyya coefficients) between two graph nodes at all timesteps. Higher values (closer to 1) indicate greater spatial overlap between the nodes' gaussian distributions; lower values (closer to 0) indicate the nodes are spatially separated. Note that Bhattacharyya coefficients are computed for a single Gaussian approximation of each node, so the scores are only a rough estimate of true spatial overlap.",
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
        "description": "Return the point in 3D at which two graph nodes overlap at a given timestep. If there is no overlap between the nodes at the given timestep, returns a message indicating no overlap instead of a point.",
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
    bhattacharyya_coeffs: np.ndarray,
    node_id_1: int,
    node_id_2: int,
    timestep: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return the point in 3D at which two graph nodes overlap at a given timestep.

    If there is no spatial overlap between the nodes (Bhattacharyya coefficient of 0.0),
    returns a message indicating no overlap instead of computing a contact point.

    Args:
        positions: Gaussian positions (T, n_gaussians, 3)
        clusters: Cluster assignment per gaussian (n_gaussians,)
        centroids: Cluster centroids through time (T, n_clusters, 3)
        bhattacharyya_coeffs: Dense Bhattacharyya coefficients (T, n_clusters, n_clusters)
        node_id_1: First node id
        node_id_2: Second node id
        timestep: The timestep at which to find the overlap point
        toolkit: Optional GraphTools instance for rerun logging

    Returns:
        Dict with either the overlap point or a message indicating no overlap.
    """
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

    # Check if there is spatial overlap between the nodes
    overlap_coeff = bhattacharyya_coeffs[timestep, node_id_1, node_id_2]
    if np.isclose(overlap_coeff, 0.0):
        return {
            "text": json.dumps({
                "node_id_1": int(node_id_1),
                "node_id_2": int(node_id_2),
                "timestep": int(timestep),
                "message": f"No spatial overlap between node {node_id_1} and node {node_id_2} at timestep {timestep}.",
            })
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
    threshold1 = np.percentile(dists1, BOUNDARY_PERCENTILE)
    threshold2 = np.percentile(dists2, BOUNDARY_PERCENTILE)

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


spec_show_scene_at_timestep = {
    "type": "function",
    "function": {
        "name": "show_scene_at_timestep",
        "description": "Returns the original video frame for a given graph timestep index.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestep_idx": {
                    "type": "integer",
                    "description": "Graph timestep index",
                },
            },
            "required": ["timestep_idx"],
        },
    },
}


def show_scene_at_timestep(
    video_frames: List[Path],
    annotation_stride: int,
    timestep_idx: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return the RGB video frame corresponding to a graph timestep.

    Resolves frame_number = timestep_idx * annotation_stride and returns the
    actual image payload so the agent loop can inject it as visual context.
    """
    if timestep_idx < 0:
        return {
            "text": json.dumps(
                {"error": f"timestep_idx={timestep_idx} must be >= 0"}
            )
        }

    frame_number = int(timestep_idx) * int(annotation_stride)
    if frame_number >= len(video_frames):
        return {
            "text": json.dumps(
                {
                    "error": (
                        f"resolved frame_number={frame_number} out of range "
                        f"for {len(video_frames)} available frames"
                    )
                }
            )
        }

    frame_path = Path(video_frames[frame_number])
    with Image.open(frame_path) as frame_img:
        rgb_frame = frame_img.convert("RGB").copy()

    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_t{int(timestep_idx):02d}_show_scene"

        rr.set_time("timestep", sequence=int(timestep_idx))
        rr.log(
            f"{prefix}/timestep_{timestep_idx}_frame_{frame_number}",
            rr.Image(np.array(rgb_frame)),
        )

    payload = {
        "timestep_idx": int(timestep_idx),
        "frame": IMAGE_PLACEHOLDER,
    }
    return {
        "text": json.dumps(payload),
        "images": [rgb_frame],
        "image_paths": [str(frame_path)],
    }


spec_node_movement_through_time = {
    "type": "function",
    "function": {
        "name": "node_movement_through_time",
        "description": "Returns the node centroid and its movement (centroid delta to previous timestep) at each timestep.",
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


def node_movement_through_time(
    centroids: np.ndarray,
    node_id: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Return the node centroid and its movement at each timestep.

    Args:
        centroids: Cluster centroids through time (T, n_clusters, 3)
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
        # centroid
        c = centroids[t, node_id]
        entry = {
            "timestep": int(t),
            "centroid": {
                "x": round(float(c[0]), 4),
                "y": round(float(c[1]), 4),
                "z": round(float(c[2]), 4),
            },
        }

        # delta (movement)
        if t > 0:
            prev_c = centroids[t - 1, node_id]
            movement = {
                "x": round(float(c[0]) - float(prev_c[0]), 4),
                "y": round(float(c[1]) - float(prev_c[1]), 4),
                "z": round(float(c[2]) - float(prev_c[2]), 4),
            }
            entry["delta_to_previous"] = movement

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


spec_relative_node_movement_through_time = {
    "type": "function",
    "function": {
        "name": "relative_node_movement_through_time",
        "description": "Returns the centroid difference vector between two nodes for all timesteps. This can be used to see if two nodes move together.",
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


spec_aggregated_node_movement = {
    "type": "function",
    "function": {
        "name": "aggregated_node_movement",
        "description": "Returns centroid movement between two timesteps for a single node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "The node's id",
                },
                "start_timestep": {
                    "type": "integer",
                    "description": "The start timestep index",
                },
                "end_timestep": {
                    "type": "integer",
                    "description": "The end timestep index",
                },
            },
            "required": ["node_id", "start_timestep", "end_timestep"],
        },
    },
}


def aggregated_node_movement(
    centroids: np.ndarray,
    node_id: int,
    start_timestep: int,
    end_timestep: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Compute centroid_tend - centroid_tstart for a single node.

    Args:
        centroids: Cluster centroids through time (T, n_clusters, 3)
        node_id: Node/cluster id
        start_timestep: Start timestep index
        end_timestep: End timestep index
        toolkit: Optional GraphTools instance for rerun logging
    """
    n_timesteps = centroids.shape[0]
    n_nodes = centroids.shape[1]

    if not (0 <= node_id < n_nodes):
        return {
            "text": json.dumps(
                {"error": f"node_id={node_id} out of range [0, {n_nodes})"}
            )
        }

    if not (0 <= start_timestep < n_timesteps):
        return {
            "text": json.dumps(
                {
                    "error": f"start_timestep={start_timestep} out of range [0, {n_timesteps})"
                }
            )
        }
    if not (0 <= end_timestep < n_timesteps):
        return {
            "text": json.dumps(
                {"error": f"end_timestep={end_timestep} out of range [0, {n_timesteps})"}
            )
        }

    centroid_start = centroids[start_timestep, node_id]
    centroid_end = centroids[end_timestep, node_id]
    delta = centroid_end - centroid_start

    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_aggregated_node_movement"

        rr.set_time("timestep", sequence=start_timestep)
        rr.log(
            f"{prefix}/start_centroid",
            rr.Points3D(
                positions=[toolkit.point_n2o(centroid_start)],
                colors=[[0, 255, 0]],
                labels=["start"],
                show_labels=True,
            ),
        )

        rr.set_time("timestep", sequence=end_timestep)
        rr.log(
            f"{prefix}/end_centroid",
            rr.Points3D(
                positions=[toolkit.point_n2o(centroid_end)],
                colors=[[255, 0, 0]],
                labels=["end"],
                show_labels=True,
            ),
        )

    return {
        "text": json.dumps(
            {
                "node_id": int(node_id),
                "start_timestep": int(start_timestep),
                "end_timestep": int(end_timestep),
                "movement": {
                    "x": round(float(delta[0]), 4),
                    "y": round(float(delta[1]), 4),
                    "z": round(float(delta[2]), 4),
                },
            }
        )
    }


def relative_node_movement_through_time(
    centroids: np.ndarray,
    node_id_1: int,
    node_id_2: int,
    toolkit: Optional['GraphTools'] = None,
) -> Dict[str, Any]:
    """Compute the centroid difference vector between two nodes through time.

    Args:
        centroids: Cluster centroids through time (T, n_clusters, 3)
        node_id_1: First node id
        node_id_2: Second node id
        toolkit: Optional GraphTools instance for rerun logging
    """
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

    relative_movements = []
    for t in range(n_timesteps):
        c1 = centroids[t, node_id_1]
        c2 = centroids[t, node_id_2]
        # Vector from node 1 to node 2
        diff = c2 - c1
        
        entry = {
            "timestep": t,
            "centroid_difference": {
                "x": round(float(diff[0]), 4),
                "y": round(float(diff[1]), 4),
                "z": round(float(diff[2]), 4),
            },
            "centroid_distance": round(float(np.linalg.norm(diff)), 4),
        }
        relative_movements.append(entry)

    # Rerun logging
    if toolkit is not None and toolkit.recording_active:
        counter = toolkit.increase_logging_tool_counter()
        prefix = f"tool_calls/{counter:02d}_relative_movement"
        
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
                    colors=[[255, 0, 0]], # Red
                    radii=point_radius
                )
            )
            rr.log(
                f"{prefix}/node_{node_id_2}",
                rr.Points3D(
                    positions=toolkit.positions[t][viz_mask2],
                    colors=[[0, 0, 255]], # Blue
                    radii=point_radius
                )
            )
            
            # Log connection line between centroids
            c1_orig = toolkit.centroids[t, node_id_1]
            c2_orig = toolkit.centroids[t, node_id_2]
            
            rr.log(
                f"{prefix}/connection",
                rr.LineStrips3D(
                    strips=[[c1_orig, c2_orig]],
                    colors=[[255, 255, 255]],
                    radii=[point_radius * 0.5],
                )
            )

    return {
        "text": json.dumps(
            {
                "node_id_1": int(node_id_1),
                "node_id_2": int(node_id_2),
                "relative_movements": relative_movements,
            }
        ),
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
        video_frames: List[Path],
        annotation_stride: int,
    ):
        self.positions = positions
        self.clusters = clusters
        self.centroids = centroids
        self.centers = centers
        self.extents = extents
        self.adjacency = adjacency
        self.bhattacharyya_coeffs = bhattacharyya_coeffs
        self.video_frames = [Path(frame) for frame in video_frames]
        self.annotation_stride = int(annotation_stride)

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
        from utils.rerun_utils import _compute_scene_extent
        
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
                    bhattacharyya_coeffs=self.bhattacharyya_coeffs,
                ),
                spec_node_overlap_position_at_time,
            ),
            "node_movement_through_time": (
                partial(
                    node_movement_through_time,
                    centroids=self.point_o2n(self.centroids),
                    toolkit=self,
                ),
                spec_node_movement_through_time,
            ),
            "relative_node_movement_through_time": (
                partial(
                    relative_node_movement_through_time,
                    centroids=self.point_o2n(self.centroids),
                    toolkit=self,
                ),
                spec_relative_node_movement_through_time,
            ),
            "aggregated_node_movement": (
                partial(
                    aggregated_node_movement,
                    centroids=self.point_o2n(self.centroids),
                    toolkit=self,
                ),
                spec_aggregated_node_movement,
            ),
            "show_scene_at_timestep": (
                partial(
                    show_scene_at_timestep,
                    video_frames=self.video_frames,
                    annotation_stride=self.annotation_stride,
                    toolkit=self,
                ),
                spec_show_scene_at_timestep,
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
