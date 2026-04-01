"""Build project-page Rerun .rrd assets without touching preprocessed data or ``output/`` graphs.

1) **Graph clips** (interactive triptych): recomputes the same point-cloud pipeline as
   ``extract_graphs.py`` from preprocessed numpy (points, colors, instance ids), then writes **two**
   recordings under ``<repo>/project_page/assets/rrd/<clip>/<graph_subdir>/``:
   ``visualization_rgb.rrd`` (RGB point cloud) and ``visualization_semantic.rrd`` (per-cluster
   semantic colors). No graph edges, cluster means, or ``.npy`` writes.

   Config paths ``preprocessed_root`` / ``output_root`` are resolved against the **dataset checkout**
   (see ``_dataset_checkout_root()``): set env ``SURG4D_DATA_CHECKOUT`` to that repo, or place a
   ``surgery-scene-graphs`` sibling next to this repo; otherwise paths are relative to the surg4d
   repo root.

2) **Tool-viz recordings** (spatial / temporal / directional): replays ``tool_calls`` from the
   per-clip prediction JSONs under ``output_root/predictions_final/{spatial|temporal|directional}/``
   only (not ``eval.*.output_dir`` or other variants). If replay fails, falls back to copying an
   existing ``output_root`` ``.rrd`` at the same relative path as the asset (e.g.
   ``predictions_final/tool_viz/...``).

Uses Hydra ``conf/config.yaml`` at the surg4d repo root (``config_path`` is relative to this file).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

# This file lives under project_page/; repo-root packages (benchmark, extract_graphs, llm, …)
# are imported from the surg4d parent directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import hydra
import numpy as np
import rerun as rr
import torch
from benchmark.serialization_utils import parse_json
from extract_graphs import (
    filter_and_reindex_clusters,
    load_precomputed_instance_clusters,
    temporal_lof_outlier_mask,
)
from llm.tools import GraphTools
from omegaconf import DictConfig, OmegaConf
from rerun.blueprint import Blueprint, Spatial3DView, TimePanel
from rerun.blueprint.archetypes import Background, LineGrid3D

logger = logging.getLogger(__name__)


def _dataset_checkout_root() -> Path:
    """Root where ``data/`` and ``output/`` from config.yaml live (often full surgery-scene-graphs checkout)."""
    env = os.environ.get("SURG4D_DATA_CHECKOUT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (
        _REPO_ROOT.parent / "surgery-scene-graphs",
        _REPO_ROOT.parent.parent / "surgery-scene-graphs",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return _REPO_ROOT.resolve()


def resolve_dataset_path(cfg: DictConfig, key: str) -> Path:
    """Resolve ``cfg.preprocessed_root`` / ``cfg.output_root`` against the dataset checkout."""
    p = Path(cfg[key])
    if p.is_absolute():
        return p
    return _dataset_checkout_root() / p


# Clips used in index.html for graph_final visualization .rrd files
GRAPH_CLIP_NAMES: frozenset[str] = frozenset(
    {"video01_28580", "video12_19500", "video17_01563"}
)

# Tool-viz paths under project_page/assets/rrd/ (and fallback copy from output_root).
# Replay uses ``query_id`` to locate the row in ``graph_agent_semantics`` inside the clip JSON.
@dataclass(frozen=True)
class ToolVizReplaySpec:
    rrd_relpath: str
    kind: Literal["spatial", "temporal", "directional"]
    clip_name: str
    query_id: str


TOOL_VIZ_REPLAY_SPECS: tuple[ToolVizReplaySpec, ...] = (
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video24_09676/graph_agent_semantics_t015_video24_09676_spatial_0_Where_is_the_grasper_in_the_back_gripping_the_gall.rrd",
    #     "spatial",
    #     "video24_09676",
    #     "video24_09676_spatial_0",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video12_19980/graph_agent_semantics_t018_video12_19980_spatial_0_Where_is_the_l-hook_coagulating_the_gallbladder.rrd",
    #     "spatial",
    #     "video12_19980",
    #     "video12_19980_spatial_0",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video12_19500/graph_agent_semantics_t007_video12_19500_spatial_1_Where_is_the_grasper_behind_the_l-hook_holding_the.rrd",
    #     "spatial",
    #     "video12_19500",
    #     "video12_19500_spatial_1",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video17_01803/graph_agent_semantics_t013_video17_01803_spatial_0_Where_does_the_grasper_grip_the_gallbladder.rrd",
    #     "spatial",
    #     "video17_01803",
    #     "video17_01803_spatial_0",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video55_00588/graph_agent_semantics_t016_video55_00588_spatial_2_Where_is_the_l-hook_hooked_into_the_connective_tis.rrd",
    #     "spatial",
    #     "video55_00588",
    #     "video55_00588_spatial_2",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video52_02826/graph_agent_semantics_t004_video52_02826_spatial_0_Where_is_the_front_grasper_gripping_the_gallbladde.rrd",
    #     "spatial",
    #     "video52_02826",
    #     "video52_02826_spatial_0",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/spatial/video52_02826/graph_agent_semantics_t004_video52_02826_spatial_1_Where_is_the_back_grasper_gripping_the_gallbladder.rrd",
    #     "spatial",
    #     "video52_02826",
    #     "video52_02826_spatial_1",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/temporal/video01_28580/graph_agent_semantics_video01_28580_temporal_rng_1_When_is_the_l-hook_touching_the_gallbladder.rrd",
    #     "temporal",
    #     "video01_28580",
    #     "video01_28580_temporal_rng_1",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/directional/video25_00402/graph_agent_semantics_video25_00402_directional_1_In_which_direction_is_the_liver_moving.rrd",
    #     "directional",
    #     "video25_00402",
    #     "video25_00402_directional_1",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/directional/video52_00160/graph_agent_semantics_video52_00160_directional_1_In_which_direction_is_the_gallbladder_moving.rrd",
    #     "directional",
    #     "video52_00160",
    #     "video52_00160_directional_1",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/directional/video52_00160/graph_agent_semantics_video52_00160_directional_2_In_which_direction_is_the_liver_moving.rrd",
    #     "directional",
    #     "video52_00160",
    #     "video52_00160_directional_2",
    # ),
    # ToolVizReplaySpec(
    #     "predictions_final/tool_viz/directional/video43_00787/graph_agent_semantics_video43_00787_directional_0_In_which_direction_is_the_grasper_pulling_the_gall.rrd",
    #     "directional",
    #     "video43_00787",
    #     "video43_00787_directional_0",
    # ),
)


def clusters_to_rgb(clusters: np.ndarray) -> np.ndarray:
    """Same as ``extract_graphs.clusters_to_rgb`` (not re-exported from that module)."""
    unique = np.unique(clusters)
    assert np.all(unique == np.arange(len(unique))), "Cluster ids must be contiguous"
    pal = np.random.rand(len(unique), 3)
    return pal[clusters]


def _compute_scene_extent(positions: np.ndarray) -> float:
    if positions is None or positions.size == 0:
        return 1.0
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = float(np.linalg.norm(maxs - mins))
    if extent <= 0.0 or not np.isfinite(extent):
        return 1.0
    return extent


def _project_page_blueprint() -> Blueprint:
    return Blueprint(
        Spatial3DView(
            origin="/",
            contents="/**",
            name="Scene",
            background=Background(color=(0, 0, 0, 255)),
            line_grid=LineGrid3D(visible=False),
        ),
        TimePanel(
            fps=6,
            # loop_mode=LoopMode(3)
            ),
        auto_layout=True,
    )


def log_rgb_points_through_time(
    pos_through_time: np.ndarray,
    point_colors: np.ndarray,
) -> None:
    """Log only the RGB-colored point cloud at ``rgb``."""
    for i in range(len(pos_through_time)):
        rr.set_time("timestep", sequence=i)
        pos = pos_through_time[i]
        scene_extent = _compute_scene_extent(pos)
        point_radius = max(scene_extent * 0.005, 1e-5)
        rr.log(
            "rgb",
            rr.Points3D(
                positions=pos,
                colors=point_colors,
                radii=point_radius,
            ),
        )


def log_semantic_points_through_time(
    clusters: np.ndarray,
    cluster_colors: np.ndarray,
    pos_through_time: np.ndarray,
) -> None:
    """Log per-cluster semantic-colored points under ``semantic/points/{id}`` (no means/edges)."""
    cluster_ids = np.unique(clusters)
    for i in range(len(pos_through_time)):
        rr.set_time("timestep", sequence=i)
        pos = pos_through_time[i]
        scene_extent = _compute_scene_extent(pos)
        point_radius = max(scene_extent * 0.005, 1e-5)
        for c in cluster_ids:
            rr.log(
                f"semantic/points/{c}",
                rr.Points3D(
                    positions=pos[clusters == c],
                    colors=cluster_colors[clusters == c],
                    radii=point_radius,
                ),
            )


def _disconnect_rerun_recording() -> None:
    try:
        rr.disconnect()
    except Exception:
        pass


def _write_single_graph_rrd(rrd_path: Path, log_fn: Callable[[], None]) -> None:
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    if rrd_path.exists():
        rrd_path.unlink()
    rr.init("graph_export")
    rr.send_blueprint(_project_page_blueprint())
    rr.save(str(rrd_path))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_fn()
    _disconnect_rerun_recording()


def recompute_graph_and_log_rrds(
    clip: DictConfig,
    cfg: DictConfig,
    rrd_rgb_out: Path,
    rrd_semantic_out: Path,
) -> None:
    """Recompute point cloud from preprocessed numpy; write RGB and semantic .rrd files (no graph npy)."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    pre_root = resolve_dataset_path(cfg, "preprocessed_root")
    points_file = (
        pre_root
        / clip.name
        / cfg.graph_extraction.cotracker_subdir
        / "point_positions_precomputed.npy"
    )
    points_through_time = np.load(points_file)
    points_through_time = points_through_time[:: cfg.graph_extraction.timestep_stride]

    point_colors_file = (
        pre_root
        / clip.name
        / cfg.graph_extraction.cotracker_subdir
        / "point_colors.npy"
    )
    ply_colors = np.load(point_colors_file).astype(np.float32) / 255.0

    clusters = load_precomputed_instance_clusters(clip, cfg)
    clip_dir = pre_root / clip.name
    semantic_labels_path = (
        clip_dir / cfg.graph_extraction.cotracker_subdir / "merged_instance_semantic_labels.json"
    )
    with open(semantic_labels_path, "r") as f:
        semantic_labels_raw = json.load(f)
        semantic_labels = {int(k): v for k, v in semantic_labels_raw.items()}

    clusters, _semantic_labels = filter_and_reindex_clusters(
        clusters=clusters,
        min_cluster_size=cfg.graph_extraction.min_cluster_size,
        semantic_labels=semantic_labels,
    )
    cluster_mask = clusters >= 0
    points_through_time = points_through_time[:, cluster_mask, :]
    clusters = clusters[cluster_mask]
    ply_colors = ply_colors[cluster_mask]

    if cfg.graph_extraction.temporal_lof_outlier_filter.enabled:
        temporal_outlier_mask = temporal_lof_outlier_mask(
            points_through_time,
            clusters,
            cfg,
            histogram_output_dir=None,
        )
        keep_mask = ~temporal_outlier_mask
        points_through_time = points_through_time[:, keep_mask, :]
        clusters = clusters[keep_mask]
        ply_colors = ply_colors[keep_mask]

    cluster_colors = clusters_to_rgb(clusters)

    _write_single_graph_rrd(
        rrd_rgb_out,
        lambda: log_rgb_points_through_time(points_through_time, ply_colors),
    )
    _write_single_graph_rrd(
        rrd_semantic_out,
        lambda: log_semantic_points_through_time(
            clusters, cluster_colors, points_through_time
        ),
    )

    logger.info("Wrote %s", rrd_rgb_out)
    logger.info("Wrote %s", rrd_semantic_out)


_INT_TOOL_KEYS = frozenset(
    {"node_id", "node_id_1", "node_id_2", "timestep", "start_timestep", "end_timestep"}
)


def _coerce_tool_arguments(args: dict[str, Any] | None) -> dict[str, Any]:
    if not args:
        return {}
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k in _INT_TOOL_KEYS and v is not None:
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _reset_rerun_for_next_recording() -> None:
    """Allow a fresh ``rr.init`` for the next ``GraphTools.start_recording`` (new .rrd file)."""
    import llm.tools as lt

    lt.RERUN_INITIALIZED = False
    try:
        rr.disconnect()
    except Exception:
        pass


def _load_graph_agent_semantics_list(pred_file: Path) -> list[dict[str, Any]] | None:
    if not pred_file.is_file():
        return None
    with open(pred_file, "r") as f:
        data = json.load(f)
    if isinstance(data.get("methods"), dict) and "graph_agent_semantics" in data["methods"]:
        return data["methods"]["graph_agent_semantics"]
    if "graph_agent_semantics" in data:
        return data["graph_agent_semantics"]
    return None


def _predictions_json_path(cfg: DictConfig, kind: str, clip_name: str) -> Path:
    """Single canonical prediction JSON: ``output_root/predictions_final/<kind>/<clip>.json``."""
    fb_root = resolve_dataset_path(cfg, "output_root") / "predictions_final"
    sub = {"spatial": "spatial", "temporal": "temporal", "directional": "directional"}[kind]
    return fb_root / sub / f"{clip_name}.json"


def _find_result_entry(entries: list[dict[str, Any]], query_id: str) -> dict[str, Any] | None:
    for e in entries:
        if e.get("id") == query_id:
            return e
    return None


def _tool_names_for_kind(cfg: DictConfig, kind: str) -> list[str]:
    if kind == "spatial":
        tool_config = cfg.eval.spatial.graph_agent_semantics_tools
    elif kind == "temporal":
        tool_config = cfg.eval.temporal.graph_agent_semantics_tools
    elif kind == "directional":
        tool_config = cfg.eval.directional.graph_agent_semantics_tools
    else:
        raise ValueError(kind)
    return [t.name for t in tool_config]


def build_graph_tools_for_clip(
    clip_name: str,
    cfg: DictConfig,
    *,
    recording_rerun_minimal: bool = False,
) -> GraphTools:
    graph_dir = resolve_dataset_path(cfg, "output_root") / clip_name / cfg.eval.paths.graph_subdir
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"
    positions_path = graph_dir / "positions.npy"
    clusters_path = graph_dir / "clusters.npy"
    adjacency_path = graph_dir / "graph.npy"
    bhattacharyya_path = graph_dir / "bhattacharyya_coeffs.npy"

    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)
    positions = np.load(positions_path)
    clusters = np.load(clusters_path)
    adjacency = np.load(adjacency_path)
    bhattacharyya_coeffs = np.load(bhattacharyya_path)

    images_dir = resolve_dataset_path(cfg, "preprocessed_root") / clip_name / cfg.eval.paths.images_subdir
    video_frames = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    return GraphTools(
        positions=positions,
        clusters=clusters,
        centroids=node_centroids,
        centers=node_centers,
        extents=node_extents,
        adjacency=adjacency,
        bhattacharyya_coeffs=bhattacharyya_coeffs,
        video_frames=video_frames,
        annotation_stride=cfg.eval.annotation_stride,
        recording_rerun_minimal=recording_rerun_minimal,
    )


def _replay_tool_calls(
    graph_tools: GraphTools,
    tools: dict[str, tuple[Any, Any]],
    tool_calls: list[dict[str, Any]],
) -> None:
    for tc in tool_calls:
        name = tc.get("tool_name")
        if not name or name not in tools:
            logger.warning("Skipping unknown or missing tool: %s", name)
            continue
        fn, _ = tools[name]
        args = _coerce_tool_arguments(tc.get("arguments"))
        fn(**args)


def _spatial_final_prediction(
    entry: dict[str, Any],
    graph_tools: GraphTools,
) -> tuple[np.ndarray, int] | None:
    ts = int(entry["timestep"])
    p3d = entry.get("predicted_3d_original")
    if (
        isinstance(p3d, list)
        and len(p3d) > 0
        and isinstance(p3d[0], list)
        and len(p3d[0]) >= 3
        and p3d[0][0] is not None
    ):
        arr = np.array(p3d, dtype=np.float32).reshape(1, 3)
        return arr, ts
    parsed = parse_json(entry.get("raw_response") or "")
    if parsed and all(k in parsed for k in ("x", "y", "z")):
        pos_arr = np.array([[parsed["x"], parsed["y"], parsed["z"]]], dtype=np.float32)
        return graph_tools.point_n2o(pos_arr), ts
    return None


def replay_tool_viz_to_rrd(
    cfg: DictConfig,
    spec: ToolVizReplaySpec,
    dst_rrd: Path,
) -> None:
    """Replay one tool-viz job from prediction JSON into ``dst_rrd``."""
    pred_path: Path | None = None
    entries: list[dict[str, Any]] | None = None
    pred_path = _predictions_json_path(cfg, spec.kind, spec.clip_name)
    entries = _load_graph_agent_semantics_list(pred_path)
    if not entries:
        raise FileNotFoundError(
            f"No prediction JSON with graph_agent_semantics for {spec.kind}/{spec.clip_name} "
            f"(expected {pred_path.resolve()})"
        )

    row = _find_result_entry(entries, spec.query_id)
    if row is None:
        raise KeyError(f"query id {spec.query_id!r} not in {pred_path}")

    tool_calls = row.get("tool_calls") or []
    if not tool_calls:
        raise ValueError("empty tool_calls")

    print(
        "tool_viz prediction json:",
        pred_path.resolve(),
        "query_id",
        spec.query_id,
        "tool_names",
        [tc.get("tool_name") for tc in tool_calls],
        flush=True,
    )

    graph_tools = build_graph_tools_for_clip(
        spec.clip_name,
        cfg,
        recording_rerun_minimal=spec.kind in ("spatial", "temporal"),
    )
    tool_names = _tool_names_for_kind(cfg, spec.kind)
    tools = graph_tools.get_tools_by_name(tool_names)
    print("tool_viz GraphTools registry keys:", sorted(tools.keys()), flush=True)

    dst_rrd.parent.mkdir(parents=True, exist_ok=True)
    if dst_rrd.exists():
        dst_rrd.unlink()

    _reset_rerun_for_next_recording()
    graph_tools.start_recording(str(dst_rrd))
    # TODO: not sure about this
    rr.send_blueprint(_project_page_blueprint())
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    try:
        _replay_tool_calls(graph_tools, tools, tool_calls)
        if spec.kind == "spatial":
            fin = _spatial_final_prediction(row, graph_tools)
            if fin is not None:
                pos_arr, timestep_idx = fin
                graph_tools.log_final_prediction(
                    position=pos_arr,
                    timestep_idx=timestep_idx,
                    label=str(row.get("query") or ""),
                )
    finally:
        graph_tools.stop_recording()


def export_tool_viz_rrds(cfg: DictConfig) -> None:
    out_root = resolve_dataset_path(cfg, "output_root")
    asset_root = _REPO_ROOT / "project_page" / "assets" / "rrd"
    for spec in TOOL_VIZ_REPLAY_SPECS:
        dst = asset_root / spec.rrd_relpath
        try:
            replay_tool_viz_to_rrd(cfg, spec, dst)
            logger.info("Replayed tool viz -> %s", dst)
            print("tool_viz replay ok:", dst.resolve(), flush=True)
        except Exception as e:
            print("tool_viz replay failed:", spec.rrd_relpath, e, flush=True)
            logger.warning("Replay failed (%s), falling back to copy: %s", spec.rrd_relpath, e)
            src = out_root / spec.rrd_relpath
            if not src.is_file():
                logger.warning("Missing tool-viz fallback source (skip): %s", src)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print("tool_viz fallback copy:", src.resolve(), "->", dst.resolve(), flush=True)
            logger.info("Copied fallback %s -> %s", src, dst)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    # Make preprocessed/output absolute so extract_graphs helpers (and eval paths) hit the dataset checkout.
    OmegaConf.update(
        cfg,
        "preprocessed_root",
        str(resolve_dataset_path(cfg, "preprocessed_root")),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "output_root",
        str(resolve_dataset_path(cfg, "output_root")),
        merge=False,
    )
    logger.info("Dataset checkout root: %s", _dataset_checkout_root())
    logger.info("cfg.preprocessed_root -> %s", cfg.preprocessed_root)
    logger.info("cfg.output_root -> %s", cfg.output_root)
    logger.info("Project assets root: %s", _REPO_ROOT / "project_page")

    for clip in cfg.clips:
        if clip.name not in GRAPH_CLIP_NAMES:
            continue
        rrd_dir = (
            _REPO_ROOT
            / "project_page"
            / "assets"
            / "rrd"
            / clip.name
            / cfg.graph_extraction.graph_output_subdir
        )
        rrd_rgb = rrd_dir / "visualization_rgb.rrd"
        rrd_sem = rrd_dir / "visualization_semantic.rrd"
        logger.info("Graph RRDs for clip %s -> %s , %s", clip.name, rrd_rgb, rrd_sem)
        recompute_graph_and_log_rrds(clip, cfg, rrd_rgb, rrd_sem)

    export_tool_viz_rrds(cfg)
    logger.info("Done.")


if __name__ == "__main__":
    main()
