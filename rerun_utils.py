import numpy as np
import rerun as rr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Optional
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB


def _compute_scene_extent(positions: np.ndarray) -> float:
    """Compute a robust scene extent from 3D positions.

    Uses the diagonal length of the axis-aligned bounding box. Falls back to 1.0
    for empty inputs or degenerate extents to keep sizes reasonable.
    """
    if positions is None or positions.size == 0:
        return 1.0
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = np.linalg.norm(maxs - mins)
    if extent <= 0.0 or not np.isfinite(extent):
        return 1.0
    return float(extent)

def log_points_through_time(
    gaussians: GaussianModel,
    clusters: np.ndarray,
    cluster_colors: np.ndarray,
    timesteps: np.ndarray,
    pos_through_time: np.ndarray,
    cluster_pos_through_time: np.ndarray,
    text_queries: list[str],
    cluster_correspondences: np.ndarray,
    patch_lf_through_time: np.ndarray,
    instance_lf_through_time: np.ndarray,
):
    """Log cluster pointclouds (points + cluster means) over time to Rerun.

    Args:
        gaussians: Gaussian model containing color features used for point coloring.
        clusters: Array of cluster ids per gaussian (length = num_gaussians).
        timesteps: Array of timesteps corresponding to positions.
        pos_through_time: Positions per timestep; shape (T, N, 3).
        cluster_pos_through_time: Cluster mean positions per timestep; shape (T, C, 3).
        text_queries: List of text queries used for cluster correspondences.
        cluster_correspondences: Cluster correspondences of shape (C, n_queries).
        patch_lf_through_time: Patch language features per timestep; shape (T, N, D).
        instance_lf_through_time: Instance language features per timestep; shape (T, N, D).
    """
    cluster_ids = np.unique(clusters)

    # Convert SH DC coefficients to RGB in [0,255]
    dc_sh = gaussians._features_dc.detach().cpu().numpy()  # (N, 1, 3)
    cols_rgb = SH2RGB(dc_sh[:, 0, :])  # (N, 3) in [0,1] ideally
    cols_rgb = (np.clip(cols_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Scale uniformity
    uniformity = gaussians.get_scaling.min(dim=1).values / gaussians.get_scaling.max(dim=1).values
    uniformity = (uniformity.detach().cpu().numpy() * 255.0).astype(np.uint8)
    uniformity = np.repeat(uniformity[:, None], 3, axis=-1)
    # print(np.histogram(uniformity, bins=10))

    for i in range(len(timesteps)):
        rr.set_time("timestep", sequence=i)
        pos = pos_through_time[i]
        cluster_means = cluster_pos_through_time[i]

        scene_extent = _compute_scene_extent(pos)
        point_radius = max(scene_extent * 0.005, 1e-5)
        mean_radius = max(scene_extent * 0.015, point_radius * 3.0)

        # Use timestep-specific language features if available
        cols_patch_t = patch_lf_through_time[i]
        cols_patch_t = (((cols_patch_t + 1.0) / 2.0).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        cols_instance_t = instance_lf_through_time[i]
        cols_instance_t = (((cols_instance_t + 1.0) / 2.0).clip(0.0, 1.0) * 255.0).astype(np.uint8)

        rr.log(
            "rgb",
            rr.Points3D(
                positions=pos,
                colors=cols_rgb,
                radii=point_radius,
            ),
        )
        rr.log(
            "qwen_patch",
            rr.Points3D(
                positions=pos,
                colors=cols_patch_t,
                radii=point_radius,
            ),
        )
        rr.log(
            "qwen_instance",
            rr.Points3D(
                positions=pos,
                colors=cols_instance_t,
                radii=point_radius,
            ),
        )

        rr.log(
            "uniformity",
            rr.Points3D(
                positions=pos,
                colors=uniformity,
                radii=point_radius
            )
        )

        # Log individual cluster points
        for c in cluster_ids:
            rr.log(
                f"clusters/points/{c}",
                rr.Points3D(
                    positions=pos[clusters == c],
                    colors=cluster_colors[clusters == c],
                    radii=point_radius,
                ),
            )

        # Log cluster means
        mean_colors = np.stack([cluster_colors[clusters == c][0] for c in cluster_ids])
        mean_labels = None
        if text_queries is not None and cluster_correspondences is not None:
            mean_labels = []
            for c in cluster_ids:
                mean_labels.append(
                    "\n".join(
                        [
                            f"{text_queries[i]}\t\t{cluster_correspondences[i, c]:.2f}"
                            for i in range(len(text_queries))
                        ]
                    )
                )
        means_viz = rr.Points3D(
            positions=cluster_means,
            colors=mean_colors,
            radii=mean_radius,
            labels=mean_labels,
            show_labels=False,
        )
        rr.log("clusters/means", means_viz)

# def log_points_through_time(
#     gaussians_rgb,
#     gaussians_qwen_patch,
#     gaussians_qwen_instance,
#     clusters: np.ndarray,
#     cluster_colors: np.ndarray,
#     timesteps: np.ndarray,
#     pos_through_time: np.ndarray,
#     cluster_pos_through_time: np.ndarray,
#     text_queries: list[str],
#     cluster_correspondences: np.ndarray,
# ):
#     """Log cluster pointclouds (points + cluster means) over time to Rerun.

#     Args:
#         gaussians: Gaussian model containing color features used for point coloring.
#         clusters: Array of cluster ids per gaussian (length = num_gaussians).
#         timesteps: Array of timesteps corresponding to positions.
#         pos_through_time: Positions per timestep; shape (T, N, 3).
#         cluster_pos_through_time: Cluster mean positions per timestep; shape (T, C, 3).
#         text_queries: List of text queries used for cluster correspondences.
#         cluster_correspondences: Cluster correspondences of shape (C, n_queries).
#     """
#     cluster_ids = np.unique(clusters)

#     # Convert SH DC coefficients to RGB in [0,255]
#     dc_sh = gaussians_rgb._features_dc.detach().cpu().numpy()  # (N, 1, 3)
#     cols_rgb = SH2RGB(dc_sh[:, 0, :])  # (N, 3) in [0,1] ideally
#     cols_rgb = (np.clip(cols_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

#     # Language features assumed to be in roughly [-1,1] → map to [0,255]
#     cols_patch = gaussians_qwen_patch.get_language_feature.detach().cpu().numpy()
#     cols_patch = (((cols_patch + 1.0) / 2.0).clip(0.0, 1.0) * 255.0).astype(np.uint8)

#     cols_instance = gaussians_qwen_instance.get_language_feature.detach().cpu().numpy()
#     cols_instance = (((cols_instance + 1.0) / 2.0).clip(0.0, 1.0) * 255.0).astype(np.uint8)

#     for i in range(len(timesteps)):
#         rr.set_time("timestep", sequence=i)
#         pos = pos_through_time[i]
#         cluster_means = cluster_pos_through_time[i]

#         scene_extent = _compute_scene_extent(pos)
#         point_radius = max(scene_extent * 0.005, 1e-5)
#         mean_radius = max(scene_extent * 0.015, point_radius * 3.0)

#         rr.log(
#             "rgb",
#             rr.Points3D(
#                 positions=pos,
#                 colors=cols_rgb,
#                 radii=point_radius,
#             ),
#         )
#         rr.log(
#             "qwen_patch",
#             rr.Points3D(
#                 positions=pos,
#                 colors=cols_patch,
#                 radii=point_radius,
#             ),
#         )
#         rr.log(
#             "qwen_instance",
#             rr.Points3D(
#                 positions=pos,
#                 colors=cols_instance,
#                 radii=point_radius,
#             ),
#         )

#         # Log individual cluster points
#         for c in cluster_ids:
#             rr.log(
#                 f"clusters/points/{c}",
#                 rr.Points3D(
#                     positions=pos[clusters == c],
#                     colors=cluster_colors[clusters == c],
#                     radii=point_radius,
#                 ),
#             )

#         # Log cluster means
#         mean_colors = np.stack([cluster_colors[clusters == c][0] for c in cluster_ids])
#         mean_labels = None
#         if text_queries is not None and cluster_correspondences is not None:
#             mean_labels = []
#             for c in cluster_ids:
#                 mean_labels.append(
#                     "\n".join(
#                         [
#                             f"{text_queries[i]}\t\t{cluster_correspondences[i, c]:.2f}"
#                             for i in range(len(text_queries))
#                         ]
#                     )
#                 )
#         means_viz = rr.Points3D(
#             positions=cluster_means,
#             colors=mean_colors,
#             radii=mean_radius,
#             labels=mean_labels,
#             show_labels=False,
#         )
#         rr.log("clusters/means", means_viz)


def log_graph_structure_through_time(
    cluster_pos_through_time: np.ndarray,
    graphs_through_time: np.ndarray,
):
    """Log graph edges to rerun.
    Edge thickness is proportional to the weight.

    Args:
        cluster_pos_through_time: Cluster mean positions per timestep; shape (T, C, 3).
        graphs_through_time: Adjacency matrices per timestep; shape (T, C, C).
    """
    num_timesteps = len(graphs_through_time)
    for i in range(num_timesteps):
        rr.set_time("timestep", sequence=i)
        cluster_means = cluster_pos_through_time[i]
        A = graphs_through_time[i]

        if A.shape[0] == 0:
            continue

        # Clear previously logged edges for this timestep so stale edges don't persist
        rr.log("clusters/edges", rr.Clear(recursive=True))

        edge_indices = np.where(A > 0)
        if len(edge_indices[0]) == 0:
            continue

        edge_weights = A[edge_indices]

        scene_extent = _compute_scene_extent(cluster_means)

        # Normalize weights for visualization
        if len(edge_weights) > 1:
            min_weight = edge_weights.min()
            max_weight = edge_weights.max()
            if max_weight > min_weight:
                normalized_weights = (edge_weights - min_weight) / (
                    max_weight - min_weight
                )
            else:
                normalized_weights = np.ones_like(edge_weights)
        else:
            normalized_weights = np.ones_like(edge_weights)

        # Create and log edges as line strips
        for idx, (u, v) in enumerate(zip(edge_indices[0], edge_indices[1])):
            if u < v:  # Avoid duplicate edges for symmetric adjacency
                start_pos = cluster_means[u]
                end_pos = cluster_means[v]
                weight = normalized_weights[idx]

                color = [0, 0, 0]
                thickness = max(scene_extent * (0.002 + 0.008 * weight), 1e-5)

                edge_line = rr.LineStrips3D(
                    strips=[[start_pos, end_pos]],
                    colors=[color],
                    radii=[thickness],
                )
                rr.log(f"clusters/edges/edge_{idx}", edge_line)


def log_correspondences_static(
    positions,
    clusters,
    text_queries,
    correspondences,
    corr_min,
    corr_max,
):
    """Log static correspondence heatmaps.

    Args:
        positions: Positions of shape (N, 3).
        text_queries: List of text queries used for cluster correspondences.
        clusters: Array of cluster ids per gaussian (length = num_gaussians).
        correspondences: Correspondences of shape (n_texts, N).
        corr_min: Minimum value for the color map.
        corr_max: Maximum value for the color map.
    """
    mask = clusters >= 0
    positions = positions[mask]
    correspondences = correspondences[:, mask]

    norm = mcolors.Normalize(vmin=corr_min, vmax=corr_max, clip=True)
    cmap = cm.get_cmap("seismic")

    scene_extent = _compute_scene_extent(positions)
    corr_point_radius = max(scene_extent * 0.004, 1e-5)

    for i, query in enumerate(text_queries):
        corr = correspondences[i]
        rgba = cmap(norm(corr))
        rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
        points = rr.Points3D(
            positions=positions,
            colors=rgb,
            radii=corr_point_radius,
            labels=[str(i) for i in corr],
            show_labels=False,
        )
        rr.log(f"correspondences/{query}", points)


# =============================
# Spatial grounding visualizers
# =============================

def _colorize_values(values: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map scalar values to RGB colors in [0,255] using a colormap with per-vector min-max normalization."""
    vmin = float(values.min())
    vmax = float(values.max())
    denom = (vmax - vmin) if vmax > vmin else 1.0
    normed = (values - vmin) / denom
    rgba = cm.get_cmap(cmap_name)(normed)
    rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
    return rgb


def log_scalar_values_over_points(
    entity_path: str,
    positions: np.ndarray,
    values: np.ndarray,
    labels: Optional[List[str]] = None,
    cmap_name: str = "jet",
    point_radius_scale: float = 0.004,
    timestep: Optional[int] = None,
):
    """Log a colored point cloud where color encodes scalar values per point.

    Args:
        entity_path: Rerun path to log into.
        positions: (N, 3)
        values: (N,)
        labels: Optional list[str] length N used as per-point labels (hidden by default)
        cmap_name: Matplotlib colormap name
        point_radius_scale: Relative radius based on scene extent
        timestep: Optional timestep index for time axis alignment
    """
    if timestep is not None:
        rr.set_time("timestep", sequence=int(timestep))

    scene_extent = _compute_scene_extent(positions)
    point_radius = max(scene_extent * point_radius_scale, 1e-5)
    rgb = _colorize_values(values.astype(np.float64), cmap_name)

    points = rr.Points3D(
        positions=positions,
        colors=rgb,
        radii=point_radius,
        labels=labels,
        show_labels=False,
    )
    rr.log(entity_path, points)


def log_basic_points(
    entity_path: str,
    positions: np.ndarray,
    color: List[int] = [200, 200, 200],
    point_radius_scale: float = 0.004,
    timestep: Optional[int] = None,
):
    """Log a basic point cloud with uniform color, used as an overlay base."""
    if timestep is not None:
        rr.set_time("timestep", sequence=int(timestep))
    scene_extent = _compute_scene_extent(positions)
    point_radius = max(scene_extent * point_radius_scale, 1e-5)
    cols = np.tile(np.array(color, dtype=np.uint8)[None, :], (positions.shape[0], 1))
    points = rr.Points3D(positions=positions, colors=cols, radii=point_radius)
    rr.log(entity_path, points)


def init_and_save_rerun(output_rrd_path):
    """Initialize and set file sink. Call once per run before logging."""
    rr.init("clusters")
    rr.save(output_rrd_path)


def log_spatial_grounding_heatmaps(
    base_path: str,
    positions: np.ndarray,
    layers: List[int],
    tokens: List[str],
    query_token_indices: List[int],
    scores: np.ndarray,
    cmap_name: str,
    timestep: int,
):
    """Log per-token and aggregated heatmaps for spatial grounding.

    Args:
        base_path: Root entity path under which to log heatmaps
        positions: (N, 3) positions for the sampled points
        layers: list of layer indices as configured
        tokens: decoded token strings for the sequence
        query_token_indices: indices into tokens corresponding to the substring span
        scores: (L, Q, N) attention scores (layers x query_tokens x points)
        cmap_name: matplotlib colormap name
        timestep: time index to align with the base recording
    """
    # Log tokens metadata once
    rr.set_time("timestep", sequence=int(timestep))
    rr.log(
        f"{base_path}/tokens",
        rr.TextLog(
            text=f"query_token_indices={query_token_indices}\n" +
                 f"tokens: {tokens}"
        ),
    )

    for lpos, layer_idx in enumerate(layers):
        layer_path = f"{base_path}/layer_{layer_idx}"
        # Per-token
        for qpos, tok_idx in enumerate(query_token_indices):
            tok = tokens[tok_idx]
            vals = scores[lpos, qpos]
            labels = [f"{tok}:{float(v):.4f}" for v in vals]
            log_scalar_values_over_points(
                entity_path=f"{layer_path}/token_{qpos}_{tok}",
                positions=positions,
                values=vals,
                labels=labels,
                cmap_name=cmap_name,
                timestep=timestep,
            )
        # Aggregate (mean)
        agg = scores[lpos].mean(axis=0)
        labels = [f"mean:{float(v):.4f}" for v in agg]
        log_scalar_values_over_points(
            entity_path=f"{layer_path}/aggregate_mean",
            positions=positions,
            values=agg,
            labels=labels,
            cmap_name=cmap_name,
            timestep=timestep,
        )


def log_spatial_predictions(
    base_path: str,
    clip_name: str,
    positions_through_time: np.ndarray,
    results: dict,
    cmap_name: str = "jet",
):
    """Visualize spatial grounding predictions.

    Expects grouped results per timestep: {"objects": [...], "actions": [...]}.
    Logs, per timestep, a base point cloud for context and per-query, per-layer
    top-k predictions colored by score under .../objects/... and .../actions/....

    Args:
        base_path: Root entity path under which to log heatmaps.
        clip_name: Name of the clip; used to namespace logs.
        positions_through_time: Array of shape (T, N, 3) or (N, 3) with point positions.
        results: Grouped prediction dict as produced by splat_feat_queries.
        cmap_name: Matplotlib colormap name.
    """
    for timestep_key, queries in results.items():
        try:
            timestep_int = int(timestep_key)
        except Exception:
            timestep_int = timestep_key  # type: ignore[assignment]

        # Choose positions for this timestep; support (T, N, 3) and (N, 3)
        try:
            point_positions = positions_through_time[timestep_int]
        except Exception:
            point_positions = positions_through_time

        # Base point cloud for context
        log_basic_points(
            entity_path=f"{base_path}/{clip_name}/t{int(timestep_int):06d}/base_points",
            positions=point_positions,
            color=[180, 180, 180],
            timestep=int(timestep_int),
        )

        # Support only grouped (dict with objects/actions) format
        assert isinstance(queries, dict), "log_spatial_predictions expects grouped results (dict with 'objects' and 'actions')."
        log_spatial_query_group(
            base_path=base_path,
            clip_name=clip_name,
            timestep_int=int(timestep_int),
            group_name="objects",
            query_list=queries.get("objects", []),
            cmap_name=cmap_name,
        )
        log_spatial_query_group(
            base_path=base_path,
            clip_name=clip_name,
            timestep_int=int(timestep_int),
            group_name="actions",
            query_list=queries.get("actions", []),
            cmap_name=cmap_name,
        )


def log_spatial_query_group(
    *,
    base_path: str,
    clip_name: str,
    timestep_int: int,
    group_name: str | None,
    query_list: list,
    cmap_name: str,
):
    prefix = f"{base_path}/{clip_name}/t{int(timestep_int):06d}/"
    if group_name:
        prefix = f"{prefix}{group_name}/"
    for qidx, qitem in enumerate(query_list):
        qname = qitem.get("query", f"query_{qidx}")
        preds = qitem.get("predictions", {})
        for layer_idx, pred in preds.items():
            layer_str = str(layer_idx)
            pos_arr = np.array(pred.get("positions", []), dtype=float)
            score_arr = np.array(pred.get("scores", []), dtype=float)
            if pos_arr.size == 0 or score_arr.size == 0:
                continue
            score_arr = score_arr.reshape(-1)
            log_scalar_values_over_points(
                entity_path=f"{prefix}{qname}/layer_{layer_str}",
                positions=pos_arr,
                values=score_arr,
                labels=[f"{qname}:{float(s):.4f}" for s in score_arr],
                cmap_name=cmap_name,
                timestep=int(timestep_int),
            )
