import numpy as np
from typing import List, Dict, Any


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
    positions: np.ndarray,
    node_centroids: np.ndarray,
    node_id_1: int,
    node_id_2: int,
) -> List[Dict[str, Any]]:
    # todo normalize scene coords properly so they are const over tools and graph
    min_pos = positions.min(axis=-1)
    max_pos = positions.max(axis=-1)
    scene_extent = float((max_pos - min_pos).max())

    n_timesteps = node_centroids.shape[0]
    n_nodes = node_centroids.shape[1]

    # Validate node ids
    if not (0 <= node_id_1 < n_nodes):
        return {"error": f"node_id_1={node_id_1} out of range [0, {n_nodes})"}
    if not (0 <= node_id_2 < n_nodes):
        return {"error": f"node_id_2={node_id_2} out of range [0, {n_nodes})"}

    results = []
    for t in range(n_timesteps):
        centroid_dist = float(
            np.linalg.norm(node_centroids[t, node_id_1] - node_centroids[t, node_id_2])
        )
        dist = float(centroid_dist / scene_extent)
        results.append({"timestep": t, "distance": round(dist, 4)})

    return {
        "tool_name": "node_distances_through_time",
        "arguments": {
            "node_id_1": int(node_id_1),
            "node_id_2": int(node_id_2),
        },
        "result": results,
    }


ALL_TOOLS = {
    "node_distances_through_time": (node_distances_through_time, spec_node_distances_through_time),
}