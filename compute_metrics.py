import hydra
from omegaconf import DictConfig
import random
import numpy as np
import torch
from pathlib import Path
import json

def compute_spatial_metrics(cfg: DictConfig):
    if cfg.compute_metrics.spatial is None:
        return
    
    cm_cfg = cfg.compute_metrics.spatial
    gt_filename: str = cm_cfg.gt_filename
    pred_root = Path(cm_cfg.pred_root)
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset-wide accumulators per method (query-wise for micro average)
    method_all_distances: dict[str, list[float]] = {}
    method_all_parse_failures: dict[str, list[str]] = {}
    
    # Per-clip results
    for clip in cfg.clips:
        clip_name = str(clip.name)
        gt_path = Path(cfg.preprocessed_root) / clip_name / gt_filename
        pred_path = pred_root / f"{clip_name}.json"
        
        if not gt_path.exists() or not pred_path.exists():
            continue
        
        with gt_path.open("r") as f:
            gt_data = json.load(f)
        with pred_path.open("r") as f:
            preds_data = json.load(f)
        
        annotations = gt_data.get("annotations", [])
        methods_preds = preds_data
        
        clip_results: dict[str, dict] = {}
        
        # Process each method
        for method_name, method_preds in methods_preds.items():
            # Initialize method accumulators if not exists
            if method_name not in method_all_distances:
                method_all_distances[method_name] = []
                method_all_parse_failures[method_name] = []
            
            method_distances: list[float] = []
            query_results: list[dict] = []
            
            # Create a mapping from query_id to predictions
            pred_by_query_id: dict[str, dict] = {}
            for pred_item in method_preds:
                query_id = pred_item.get("id")
                if query_id:
                    pred_by_query_id[query_id] = pred_item
            
            # Process each annotation
            for annotation in annotations:
                query_id = str(annotation.get("id"))
                question = str(annotation.get("query"))
                timestep = int(annotation.get("timestep"))
                
                # Ground-truth pixel point (x, y)
                pil_coords = annotation.get("pil_coords", [])
                if len(pil_coords) != 2:
                    continue
                gx = float(pil_coords[0])
                gy = float(pil_coords[1])
                gt_xy = np.array([gx, gy], dtype=np.float64)
                
                pred_item = pred_by_query_id.get(query_id)
                parse_success = False
                
                if pred_item and "predicted" in pred_item and pred_item["predicted"] is not None:
                    pred_coords = pred_item["predicted"]
                    if (
                        isinstance(pred_coords, list)
                        and len(pred_coords) == 2
                        and pred_coords[0] is not None
                        and pred_coords[1] is not None
                    ):
                        try:
                            px = float(pred_coords[0])
                            py = float(pred_coords[1])
                            pred_xy = np.array([px, py], dtype=np.float64)

                            # Compute L2 distance
                            diff = pred_xy - gt_xy
                            l2_distance = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2))
                            parse_success = True
                        except (TypeError, ValueError):
                            l2_distance = float(cm_cfg.l2_error_no_prediction)
                            px, py = None, None
                    else:
                        # Invalid prediction format
                        l2_distance = float(cm_cfg.l2_error_no_prediction)
                        px, py = None, None
                else:
                    # No prediction or parsing failed
                    l2_distance = float(cm_cfg.l2_error_no_prediction)
                    px, py = None, None

                if not parse_success:
                    method_all_parse_failures[method_name].append(f"{clip_name}:{query_id}")
                
                method_distances.append(l2_distance)
                method_all_distances[method_name].append(l2_distance)
                
                query_results.append({
                    "id": query_id,
                    "timestep": timestep,
                    "query": question,
                    "ground_truth_pixel": [gx, gy],
                    "predicted_pixel": [px, py] if px is not None and py is not None else None,
                    "l2_distance": round(l2_distance, 2) if l2_distance != float("inf") else None,
                })
            
            # Compute per-method averages for this clip
            clip_results[method_name] = {
                "queries": query_results,
                "num_queries": len(query_results),
                "mean_l2_distance": round(float(np.mean(method_distances)), 2) if method_distances else None,
                "std_l2_distance": round(float(np.std(method_distances)), 2) if method_distances else None,
            }
        
        # Save per-clip results
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump({
                "clip": clip_name,
                "methods": clip_results,
            }, f, indent=2)
    
    # Compute aggregated metrics per method (micro average over all queries)
    aggregated: dict[str, dict] = {}
    for method_name in method_all_distances.keys():
        distances_list = method_all_distances[method_name]
        
        aggregated[method_name] = {
            "num_queries": len(distances_list),
            "mean_l2_distance": round(float(np.mean(distances_list)), 2) if distances_list else None,
            "std_l2_distance": round(float(np.std(distances_list)), 2) if distances_list else None,
            "num_parsing_failures": len(method_all_parse_failures[method_name]),
            "parsing_failure_query_ids": method_all_parse_failures[method_name],
        }
    
    with aggregated_file.open("w") as f:
        json.dump({"methods": aggregated}, f, indent=2)

def compute_temporal_metrics(cfg: DictConfig):
    if cfg.compute_metrics.temporal is None:
        return
    
    cm_cfg = cfg.compute_metrics.temporal
    pred_root = Path(cm_cfg.pred_root)

    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset-wide accumulators per method (query-wise for micro average)
    method_all_errors: dict[str, list[float]] = {}
    method_all_ious: dict[str, list[float]] = {}
    method_all_parse_failures: dict[str, list[str]] = {}
    
    # Per-clip results
    for clip in cfg.clips:
        clip_name = str(clip.name)
        labels_path = Path(cfg.compute_metrics.annotations_root) / "temporal" / f"{clip_name}.json"
        pred_path = pred_root / f"{clip_name}.json"
        
        if not labels_path.exists() or not pred_path.exists():
            continue
        
        with labels_path.open("r") as f:
            labels_data = json.load(f)
        with pred_path.open("r") as f:
            preds_data = json.load(f)

        annotations = labels_data.get("annotations", [])
        methods_preds = preds_data.get("methods", {})
        
        clip_results: dict[str, dict] = {}
        
        # Process each method
        for method_name, method_preds in methods_preds.items():
            # Initialize method accumulators if not exists
            if method_name not in method_all_errors:
                method_all_errors[method_name] = []
                method_all_ious[method_name] = []
                method_all_parse_failures[method_name] = []
            
            method_errors: list[float] = []
            method_ious: list[float] = []
            query_results: list[dict] = []
            
            # Create a mapping from query_id to predictions
            pred_by_query_id: dict[str, dict] = {}
            for pred_item in method_preds:
                query_id = pred_item.get("id")
                if query_id:
                    pred_by_query_id[query_id] = pred_item
            
            # Process each annotation
            for annotation in annotations:
                query_id = str(annotation.get("id"))
                query_type = str(annotation.get("type"))
                question = str(annotation.get("query"))
                
                pred_item = pred_by_query_id.get(query_id)
                
                if query_type == "pit":
                    gt_timestep = int(annotation["timestep"])
                    
                    if pred_item and "predicted" in pred_item and pred_item["predicted"] is not None:
                        try:
                            pred_timestep = int(pred_item["predicted"])
                            error = float(abs(pred_timestep - gt_timestep))
                        except (TypeError, ValueError):
                            pred_timestep = None
                            error = float(cm_cfg.pit_noprediction_error)
                            method_all_parse_failures[method_name].append(f"{clip_name}:pit:{query_id}")
                    else:
                        # No prediction or parsing failed
                        pred_timestep = None
                        error = float(cm_cfg.pit_noprediction_error)
                        method_all_parse_failures[method_name].append(f"{clip_name}:pit:{query_id}")
                    
                    method_errors.append(error)
                    method_all_errors[method_name].append(error)
                    
                    query_results.append({
                        "id": query_id,
                        "type": query_type,
                        "query": question,
                        "ground_truth_timestep": gt_timestep,
                        "predicted_timestep": pred_timestep,
                        "absolute_error": error,
                    })
                    
                elif query_type == "range":
                    # Ground truth ranges (inclusive)
                    gt_ranges = annotation["ranges"]
                    
                    if pred_item and pred_item.get("predicted"):
                        pred_ranges = pred_item["predicted"]
                        try:
                            iou = compute_temporal_iou(gt_ranges, pred_ranges, cfg.compute_metrics.n_timesteps)
                        except (TypeError, ValueError):
                            pred_ranges = None
                            iou = 0.0
                            method_all_parse_failures[method_name].append(f"{clip_name}:range:{query_id}")
                    else:
                        # No prediction or parsing failed
                        pred_ranges = None
                        iou = 0.0
                        method_all_parse_failures[method_name].append(f"{clip_name}:range:{query_id}")
                    
                    method_ious.append(iou)
                    method_all_ious[method_name].append(iou)
                    
                    query_results.append({
                        "id": query_id,
                        "type": query_type,
                        "query": question,
                        "ground_truth_ranges": gt_ranges,
                        "predicted_ranges": pred_ranges,
                        "iou": iou,
                    })

                else:
                    raise ValueError(f"Unsupported query type for {clip_name} {query_id}: {query_type}")
            
            # Compute per-method averages for this clip
            clip_results[method_name] = {
                "queries": query_results,
                "num_queries": len(query_results),
                "mean_absolute_error": round(float(np.mean(method_errors)), 2) if method_errors else None,
                "mean_iou": round(float(np.mean(method_ious)), 2) if method_ious else None,
            }
        
        # Save per-clip results
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump({
                "clip": clip_name,
                "methods": clip_results,
            }, f, indent=2)
    
    # Compute aggregated metrics per method (micro average over all queries)
    aggregated: dict[str, dict] = {}
    for method_name in method_all_errors.keys() | method_all_ious.keys():
        errors = method_all_errors.get(method_name, [])
        ious = method_all_ious.get(method_name, [])
        
        aggregated[method_name] = {
            "mean_absolute_error": round(float(np.mean(errors)), 2) if errors else None,
            "std_absolute_error": round(float(np.std(errors)), 2) if errors else None,
            "mean_iou": round(float(np.mean(ious)), 3) if ious else None,
            "std_iou": round(float(np.std(ious)), 3) if ious else None,
            "num_pit_queries": len(errors),
            "num_range_queries": len(ious),
            "num_queries": len(errors) + len(ious),
            "num_parsing_failures": len(method_all_parse_failures.get(method_name, [])),
            "parsing_failure_query_ids": method_all_parse_failures.get(method_name, []),
        }
    
    with aggregated_file.open("w") as f:
        json.dump({"methods": aggregated}, f, indent=2)


def compute_directional_metrics(cfg: DictConfig):
    if cfg.compute_metrics.directional is None:
        return

    cm_cfg = cfg.compute_metrics.directional
    pred_root = Path(cm_cfg.pred_root)

    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)

    method_all_errors: dict[str, list[float]] = {}
    method_all_axis_errors: dict[str, dict[str, list[float]]] = {}
    method_all_axis_errors_by_gt_value: dict[str, dict[str, dict[int, list[float]]]] = {}
    method_all_parse_failures: dict[str, list[str]] = {}

    for clip in cfg.clips:
        clip_name = str(clip.name)
        labels_path = Path(cfg.compute_metrics.annotations_root) / "directional" / f"{clip_name}.json"
        pred_path = pred_root / f"{clip_name}.json"

        if not labels_path.exists() or not pred_path.exists():
            continue

        with labels_path.open("r") as f:
            labels_data = json.load(f)
        with pred_path.open("r") as f:
            preds_data = json.load(f)

        annotations = labels_data.get("annotations", [])
        methods_preds = preds_data.get("methods", {})

        clip_results: dict[str, dict] = {}

        for method_name, method_preds in methods_preds.items():
            if method_name not in method_all_errors:
                method_all_errors[method_name] = []
                method_all_axis_errors[method_name] = {"x": [], "y": [], "z": []}
                method_all_axis_errors_by_gt_value[method_name] = {
                    "x": {-1: [], 0: [], 1: []},
                    "y": {-1: [], 0: [], 1: []},
                    "z": {-1: [], 0: [], 1: []},
                }
                method_all_parse_failures[method_name] = []

            method_errors: list[float] = []
            method_axis_errors = {"x": [], "y": [], "z": []}
            method_axis_errors_by_gt_value = {
                "x": {-1: [], 0: [], 1: []},
                "y": {-1: [], 0: [], 1: []},
                "z": {-1: [], 0: [], 1: []},
            }
            query_results: list[dict] = []

            pred_by_query_id: dict[str, dict] = {}
            for pred_item in method_preds:
                query_id = pred_item.get("id")
                if query_id:
                    pred_by_query_id[query_id] = pred_item

            for annotation in annotations:
                query_id = str(annotation.get("id"))
                question = str(annotation.get("query"))
                gt_direction = annotation.get("direction")
                gt_range = annotation.get("range")

                gx = float(gt_direction["x"])
                gy = float(gt_direction["y"])
                gz = float(gt_direction["z"])

                pred_item = pred_by_query_id.get(query_id)
                parse_success = False
                if pred_item and pred_item.get("predicted") is not None:
                    pred_direction = pred_item["predicted"]
                    if (
                        isinstance(pred_direction, dict)
                        and pred_direction.get("x") is not None
                        and pred_direction.get("y") is not None
                        and pred_direction.get("z") is not None
                    ):
                        try:
                            px = float(pred_direction["x"])
                            py = float(pred_direction["y"])
                            pz = float(pred_direction["z"])
                            x_abs_error = float(abs(px - gx))
                            y_abs_error = float(abs(py - gy))
                            z_abs_error = float(abs(pz - gz))
                            l1_per_axis_mean = float((x_abs_error + y_abs_error + z_abs_error) / 3.0)
                            parse_success = True
                        except (TypeError, ValueError):
                            px, py, pz = None, None, None
                            x_abs_error = float(cm_cfg.noprediction_error)
                            y_abs_error = float(cm_cfg.noprediction_error)
                            z_abs_error = float(cm_cfg.noprediction_error)
                            l1_per_axis_mean = float(cm_cfg.noprediction_error)
                    else:
                        px, py, pz = None, None, None
                        x_abs_error = float(cm_cfg.noprediction_error)
                        y_abs_error = float(cm_cfg.noprediction_error)
                        z_abs_error = float(cm_cfg.noprediction_error)
                        l1_per_axis_mean = float(cm_cfg.noprediction_error)
                else:
                    px, py, pz = None, None, None
                    x_abs_error = float(cm_cfg.noprediction_error)
                    y_abs_error = float(cm_cfg.noprediction_error)
                    z_abs_error = float(cm_cfg.noprediction_error)
                    l1_per_axis_mean = float(cm_cfg.noprediction_error)

                if not parse_success:
                    method_all_parse_failures[method_name].append(f"{clip_name}:{query_id}")

                gx_int = int(gx)
                gy_int = int(gy)
                gz_int = int(gz)

                method_errors.append(l1_per_axis_mean)
                method_all_errors[method_name].append(l1_per_axis_mean)
                method_axis_errors["x"].append(x_abs_error)
                method_axis_errors["y"].append(y_abs_error)
                method_axis_errors["z"].append(z_abs_error)
                method_all_axis_errors[method_name]["x"].append(x_abs_error)
                method_all_axis_errors[method_name]["y"].append(y_abs_error)
                method_all_axis_errors[method_name]["z"].append(z_abs_error)

                method_axis_errors_by_gt_value["x"][gx_int].append(x_abs_error)
                method_axis_errors_by_gt_value["y"][gy_int].append(y_abs_error)
                method_axis_errors_by_gt_value["z"][gz_int].append(z_abs_error)
                method_all_axis_errors_by_gt_value[method_name]["x"][gx_int].append(x_abs_error)
                method_all_axis_errors_by_gt_value[method_name]["y"][gy_int].append(y_abs_error)
                method_all_axis_errors_by_gt_value[method_name]["z"][gz_int].append(z_abs_error)

                query_results.append(
                    {
                        "id": query_id,
                        "query": question,
                        "range": gt_range,
                        "ground_truth_direction": {"x": gx, "y": gy, "z": gz},
                        "predicted_direction": {"x": px, "y": py, "z": pz} if px is not None else None,
                        "axis_l1_distances": {
                            "x": round(x_abs_error, 3),
                            "y": round(y_abs_error, 3),
                            "z": round(z_abs_error, 3),
                        },
                        "mean_axis_l1_distance": round(l1_per_axis_mean, 3),
                    }
                )

            clip_results[method_name] = {
                "queries": query_results,
                "num_queries": len(query_results),
                "mean_axis_l1_distance": round(float(np.mean(method_errors)), 3) if method_errors else None,
                "std_axis_l1_distance": round(float(np.std(method_errors)), 3) if method_errors else None,
                "mean_axis_l1_distance_per_axis": {
                    axis_name: (
                        round(float(np.mean(axis_errors)), 3) if axis_errors else None
                    )
                    for axis_name, axis_errors in method_axis_errors.items()
                },
                "mean_axis_l1_distance_per_axis_by_gt_value": {
                    axis_name: {
                        str(gt_value): (
                            round(float(np.mean(gt_value_errors)), 3) if gt_value_errors else None
                        )
                        for gt_value, gt_value_errors in by_gt.items()
                    }
                    for axis_name, by_gt in method_axis_errors_by_gt_value.items()
                },
            }

        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump(
                {
                    "clip": clip_name,
                    "methods": clip_results,
                },
                f,
                indent=2,
            )

    aggregated: dict[str, dict] = {}
    for method_name, errors in method_all_errors.items():
        aggregated[method_name] = {
            "mean_axis_l1_distance": round(float(np.mean(errors)), 3) if errors else None,
            "std_axis_l1_distance": round(float(np.std(errors)), 3) if errors else None,
            "mean_axis_l1_distance_per_axis": {
                axis_name: (
                    round(float(np.mean(axis_errors)), 3) if axis_errors else None
                )
                for axis_name, axis_errors in method_all_axis_errors[method_name].items()
            },
            "mean_axis_l1_distance_per_axis_by_gt_value": {
                axis_name: {
                    str(gt_value): (
                        round(float(np.mean(gt_value_errors)), 3) if gt_value_errors else None
                    )
                    for gt_value, gt_value_errors in by_gt.items()
                }
                for axis_name, by_gt in method_all_axis_errors_by_gt_value[method_name].items()
            },
            "num_queries": len(errors),
            "num_parsing_failures": len(method_all_parse_failures[method_name]),
            "parsing_failure_query_ids": method_all_parse_failures[method_name],
        }

    with aggregated_file.open("w") as f:
        json.dump({"methods": aggregated}, f, indent=2)


def compute_temporal_iou(gt_ranges: list[list[int]], pred_ranges: list[list[int]], max_timestep: int) -> float:
    """Compute IoU between ground truth and predicted temporal ranges.
    
    Args:
        gt_ranges: List of [start, end] ranges (inclusive) from ground truth
        pred_ranges: List of [start, end] ranges (inclusive) from predictions
        max_timestep: Maximum timestep value (for clipping)
        
    Returns:
        IoU score between 0 and 1
    """
    # Convert ranges to sets of timesteps
    gt_timesteps: set[int] = set()
    for start, end in gt_ranges:
        # Ranges are inclusive
        for t in range(int(start), int(end) + 1):
            if 0 <= t < max_timestep:
                gt_timesteps.add(t)
    
    pred_timesteps: set[int] = set()
    for start, end in pred_ranges:
        # Ranges are inclusive
        for t in range(int(start), int(end) + 1):
            if 0 <= t < max_timestep:
                pred_timesteps.add(t)
    
    if len(gt_timesteps) == 0 and len(pred_timesteps) == 0:
        return 1.0
    
    if len(gt_timesteps) == 0 or len(pred_timesteps) == 0:
        return 0.0
    
    intersection = len(gt_timesteps & pred_timesteps)
    union = len(gt_timesteps | pred_timesteps)
    
    return float(intersection) / float(union)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup (harmless for CPU-only metrics)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    compute_spatial_metrics(cfg)
    compute_temporal_metrics(cfg)
    compute_directional_metrics(cfg)

if __name__ == "__main__":
    main()