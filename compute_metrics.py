import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import numpy as np

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

    ks: list[int] = list(cm_cfg.l2_top_ks)
    layers_filter = {str(layer_idx) for layer_idx in cm_cfg.layers}

    methods = ["splat", "static_graph", "frame_attn"]

    # Dataset-wide accumulators (layerwise):
    # methods -> class -> layer_key -> {sum: np.array[K], count: int}
    dataset_stats: dict[str, dict[str, dict[str, dict[str, np.ndarray | int]]]] = {
        m: {"objects": {}, "actions": {}, "all": {}} for m in methods
    }

    def _min_l2_at_k(pred_coords: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
        # pred_coords: [N, 2] (x, y); gt_xy: [2]
        if pred_coords.size == 0:
            return np.full(len(ks), np.inf, dtype=np.float64)
        diffs = pred_coords.astype(np.float64) - gt_xy[None, :]
        dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))  # [N]
        out = np.empty(len(ks), dtype=np.float64)
        for i, k in enumerate(ks):
            kk = min(k, dists.shape[0])
            if kk <= 0:
                out[i] = np.inf
            else:
                out[i] = float(np.min(dists[:kk]))
        return out

    for clip in cfg.clips:
        clip_name = str(clip.name)
        gt_path = Path(cfg.preprocessed_root) / clip_name / gt_filename
        pred_path = pred_root / f"{clip_name}.json"

        if not gt_path.exists() or not pred_path.exists():
            continue

        with gt_path.open("r") as f:
            gt_data = json.load(f)
        with pred_path.open("r") as f:
            preds_all = json.load(f)

        per_clip_results = {}

        for method in methods:
            if method not in preds_all:
                continue

            # Per-class, per-layer accumulators for this clip
            sums: dict[str, dict[str, np.ndarray]] = {
                "objects": {},
                "actions": {},
                "all": {},
            }
            counts: dict[str, dict[str, int]] = {"objects": {}, "actions": {}, "all": {}}
            clip_method_items: list[dict] = []

            method_preds = preds_all[method]

            # Iterate timesteps present in both GT and predictions
            for t_key, gt_entry in gt_data.items():
                if t_key not in method_preds:
                    continue
                pred_entry = method_preds[t_key]

                for group in ("objects", "actions"):
                    gt_list = gt_entry.get(group, [])
                    pred_list = pred_entry.get(group, [])
                    n = min(len(gt_list), len(pred_list))
                    if n <= 0:
                        continue

                    for i in range(n):
                        gt_item = gt_list[i]
                        pred_item = pred_list[i]

                        # Ground-truth pixel point (x, y)
                        gx = float(gt_item["pixel_x"])  # assume set in config pipeline
                        gy = float(gt_item["pixel_y"])  # assume set in config pipeline
                        gt_xy = np.array([gx, gy], dtype=np.float64)

                        # Collect per-layer min_l2@k for this query
                        preds_by_layer = pred_item.get("predictions", {})
                        per_layer_out: dict[str, dict[str, float]] = {}
                        for layer_key, layer_pred in preds_by_layer.items():
                            lkey = str(layer_key)
                            if lkey not in layers_filter:
                                continue
                            coords = np.array(layer_pred.get("pixel_coords", []), dtype=np.float64)
                            vals = _min_l2_at_k(coords, gt_xy)
                            per_layer_out[lkey] = {f"min_l2@{k}": float(v) for k, v in zip(ks, vals.tolist())}
                            # accumulate per group/layer
                            if lkey not in sums[group]:
                                sums[group][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts[group][lkey] = 0
                            sums[group][lkey] += vals
                            counts[group][lkey] += 1
                            # overall
                            if lkey not in sums["all"]:
                                sums["all"][lkey] = np.zeros(len(ks), dtype=np.float64)
                                counts["all"][lkey] = 0
                            sums["all"][lkey] += vals
                            counts["all"][lkey] += 1

                        # record per-query item if we computed any layer metrics
                        if per_layer_out:
                            query_text = pred_item.get("query") or gt_item.get("query")
                            clip_method_items.append(
                                {
                                    "timestep": t_key,
                                    "frame_number": int(gt_entry.get("frame_number", -1)),
                                    "group": group,
                                    "query": query_text,
                                    "per_layer": per_layer_out,
                                }
                            )

            # Compute averages for this clip and method
            method_out = {"per_class": {}, "counts": {}, "items": clip_method_items}
            for group in ("objects", "actions", "all"):
                per_layer_avgs = {}
                per_layer_counts = {}
                for lkey, svec in sums[group].items():
                    c = counts[group].get(lkey, 0)
                    if c > 0:
                        avg = (svec / c).tolist()
                    else:
                        avg = [float("nan")] * len(ks)
                    per_layer_avgs[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg)}
                    per_layer_counts[lkey] = c

                    # Update dataset-wide accumulators
                    if lkey not in dataset_stats[method][group]:
                        dataset_stats[method][group][lkey] = {
                            "sum": np.zeros(len(ks), dtype=np.float64),
                            "count": 0,
                        }
                    dataset_stats[method][group][lkey]["sum"] += svec
                    dataset_stats[method][group][lkey]["count"] += c

                method_out["per_class"][group] = per_layer_avgs
                method_out["counts"][group] = per_layer_counts

            per_clip_results[method] = method_out

        # Save per-clip file with query-wise metrics
        clip_out = {
            "clip": clip_name,
            "config": {"l2_top_ks": ks, "layers": sorted(list(layers_filter), key=lambda x: int(x))},
            "methods": per_clip_results,
        }
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump(clip_out, f, indent=2)

    # Save dataset-wide summary
    summary = {"config": {"l2_top_ks": ks, "layers": sorted(list(layers_filter), key=lambda x: int(x))}, "methods": {}}
    for method in methods:
        if method not in dataset_stats:
            continue
        mstats = dataset_stats[method]
        out_m = {"per_class": {}, "counts": {}}
        for group in ("objects", "actions", "all"):
            per_layer = {}
            per_layer_counts = {}
            for lkey, stat in mstats[group].items():
                total_count = int(stat["count"])  # type: ignore[index]
                sums_arr = stat["sum"]  # type: ignore[index]
                if total_count > 0:
                    avg = (sums_arr / total_count).tolist()  # type: ignore[operator]
                else:
                    avg = [float("nan")] * len(ks)
                per_layer[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg)}
                per_layer_counts[lkey] = total_count
            out_m["per_class"][group] = per_layer
            out_m["counts"][group] = per_layer_counts
        summary["methods"][method] = out_m

    with aggregated_file.open("w") as f:
        json.dump({"summary": summary, "config": {"l2_top_ks": ks, "layers": sorted(list(layers_filter), key=lambda x: int(x))}}, f, indent=2)

def compute_temporal_metrics(cfg: DictConfig):
    if cfg.compute_metrics.temporal is None:
        return

def compute_triplets_metrics(cfg: DictConfig):
    if cfg.compute_metrics.triplets is None:
        return

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    compute_spatial_metrics(cfg)
    compute_temporal_metrics(cfg)
    compute_triplets_metrics(cfg)

if __name__ == "__main__":
    main()