import hydra
from omegaconf import DictConfig
import os
import random
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List
from benchmark.cholect50_utils import CholecT50Loader
from benchmark.benchmark_config import normalize_for_matching
from sklearn.metrics import average_precision_score

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

    methods = ["splat", "static_graph", "frame_attn", "splat_graph", "frame_attn_refine"]

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
                            vals = np.round(vals, 2)
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
                        avg_arr = svec / c
                        avg_list = np.round(avg_arr, 2).tolist()
                    else:
                        avg_list = [float("nan")] * len(ks)
                    per_layer_avgs[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
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
            "methods": per_clip_results,
        }
        with (out_dir / f"{clip_name}.json").open("w") as f:
            json.dump(clip_out, f, indent=2)

    # Save dataset-wide summary
    summary = {"methods": {}}
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
                    avg_arr = sums_arr / total_count  # type: ignore[operator]
                    avg_list = np.round(avg_arr, 2).tolist()
                else:
                    avg_list = [float("nan")] * len(ks)
                per_layer[lkey] = {f"min_l2@{k}": v for k, v in zip(ks, avg_list)}
                per_layer_counts[lkey] = total_count
            out_m["per_class"][group] = per_layer
            out_m["counts"][group] = per_layer_counts
        summary["methods"][method] = out_m

    with aggregated_file.open("w") as f:
        json.dump({"summary": summary}, f, indent=2)

def compute_temporal_metrics(cfg: DictConfig):
    if cfg.compute_metrics.temporal is None:
        return
    cm_cfg = cfg.compute_metrics.temporal

    pred_root = Path(cm_cfg.pred_root)
    labels_root = Path(cm_cfg.labels_root)
    labels_tmpl: str = cm_cfg.labels_filename_template
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)

    # Metric params: prefer compute_metrics config, fallback to eval config if present
    tcfg = None
    try:
        if hasattr(cm_cfg, 'metrics') and cm_cfg.metrics is not None:
            tcfg = cm_cfg.metrics
    except Exception:
        tcfg = None
    if tcfg is None:
        try:
            if hasattr(cfg, 'eval') and cfg.eval is not None and hasattr(cfg.eval, 'temporal') and cfg.eval.temporal is not None and hasattr(cfg.eval.temporal, 'metrics'):
                tcfg = cfg.eval.temporal.metrics
        except Exception:
            tcfg = None
    if tcfg is None:
        tcfg = {}

    # Dataset accumulators per ablation and query type
    dataset: dict[str, dict[str, list[dict]]] = {}

    # Helpers mirroring evaluator logic
    def _eval_frame_error(predicted: dict | None, gt: dict, tol: int) -> dict:
        if predicted is None or 'frame' not in predicted:
            return {
                'frame_error': float('inf'),
                'within_tolerance': False,
                'tolerance_used': tol,
                'success': False,
            }
        err = abs(int(predicted['frame']) - int(gt['frame']))
        return {
            'frame_error': err,
            'within_tolerance': err <= tol,
            'tolerance_used': tol,
            'success': err <= tol,
        }

    def _eval_iou(predicted: dict | None, gt: dict, thr: float) -> dict:
        if predicted is None or 'ranges' not in predicted:
            return {
                'iou': round(0.0, 2),
                'precision': round(0.0, 2),
                'recall': round(0.0, 2),
                'success': False,
                'threshold': thr,
            }
        def ranges_to_set(ranges):
            s = set()
            for a, b in ranges:
                s.update(range(int(a), int(b) + 1))
            return s
        gt_frames = ranges_to_set(gt['ranges'])
        pred_frames = ranges_to_set(predicted['ranges'])
        inter = len(gt_frames & pred_frames)
        union = len(gt_frames | pred_frames)
        iou = inter / union if union > 0 else 0.0
        prec = inter / len(pred_frames) if len(pred_frames) > 0 else 0.0
        rec = inter / len(gt_frames) if len(gt_frames) > 0 else 0.0
        return {
            'iou': round(float(iou), 2),
            'precision': round(float(prec), 2),
            'recall': round(float(rec), 2),
            'success': iou >= thr,
            'threshold': thr,
        }

    def _eval_ordering(predicted: dict | None, gt: dict, order_weight: float, iou_weight: float) -> dict:
        if predicted is None or 'events' not in predicted:
            return {
                'order_correct': False,
                'per_event_iou': [],
                'mean_iou': round(0.0, 2),
                'composite_score': round(0.0, 2),
                'success': False,
            }
        pred_events = predicted['events']
        gt_events = gt['events']
        order_correct = len(pred_events) == len(gt_events)
        if order_correct:
            for i in range(len(pred_events)):
                if int(pred_events[i].get('order', -1)) != int(gt_events[i]['order']):
                    order_correct = False
                    break
        per_event_iou: list[float] = []
        for i in range(min(len(pred_events), len(gt_events))):
            pr = pred_events[i]['frame_range']
            gr = gt_events[i]['frame_range']
            pf = set(range(int(pr[0]), int(pr[1]) + 1))
            gf = set(range(int(gr[0]), int(gr[1]) + 1))
            inter = len(pf & gf)
            union = len(pf | gf)
            per_event_iou.append(inter / union if union > 0 else 0.0)
        mean_iou = float(np.mean(per_event_iou)) if per_event_iou else 0.0
        composite = order_weight * (1.0 if order_correct else 0.0) + iou_weight * mean_iou
        return {
            'order_correct': order_correct,
            'per_event_iou': [round(float(x), 2) for x in per_event_iou],
            'mean_iou': round(float(mean_iou), 2),
            'composite_score': round(float(composite), 2),
            'success': composite >= 0.5,
        }

    def _eval_count(predicted: dict | None, gt: dict, count_weight: float, iou_weight: float) -> dict:
        if predicted is None or 'count' not in predicted:
            return {
                'count_correct': False,
                'count_error': int(gt['count']),
                'per_occurrence_iou': [],
                'mean_iou': round(0.0, 2),
                'composite_score': round(0.0, 2),
                'success': False,
            }
        pred_count = int(predicted['count'])
        gt_count = int(gt['count'])
        pred_occs = predicted.get('occurrences', []) or []
        gt_occs = gt.get('occurrences', []) or []
        count_correct = (pred_count == gt_count)
        count_error = abs(pred_count - gt_count)
        per_iou: list[float] = []
        matched = set()
        for po in pred_occs[:len(gt_occs)]:
            pf = set(range(int(po[0]), int(po[1]) + 1))
            best_iou = 0.0
            best_idx = -1
            for j, go in enumerate(gt_occs):
                if j in matched:
                    continue
                gf = set(range(int(go[0]), int(go[1]) + 1))
                inter = len(pf & gf)
                union = len(pf | gf)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0:
                matched.add(best_idx)
                per_iou.append(best_iou)
        mean_iou = float(np.mean(per_iou)) if per_iou else 0.0
        composite = count_weight * (1.0 if count_correct else 0.0) + iou_weight * mean_iou
        return {
            'count_correct': count_correct,
            'count_error': count_error,
            'pred_count': pred_count,
            'gt_count': gt_count,
            'per_occurrence_iou': [round(float(x), 2) for x in per_iou],
            'mean_iou': round(float(mean_iou), 2),
            'composite_score': round(float(composite), 2),
            'success': composite >= 0.5,
        }

    # Per-clip processing
    for clip in cfg.clips:
        clip_name = str(clip.name)
        pred_path = pred_root / f"{clip_name}.json"
        gt_path = labels_root / labels_tmpl.format(clip_name=clip_name)
        if not pred_path.exists() or not gt_path.exists():
            continue

        with pred_path.open('r') as f:
            preds = json.load(f)
        with gt_path.open('r') as f:
            gt_full = json.load(f)
        gt_by_id = {a['query_id']: a for a in gt_full.get('annotations', [])}

        per_clip_out: dict[str, dict] = {}

        for ablation, items in preds.get('ablations', {}).items():
            results = []
            for item in items:
                qid = item.get('query_id')
                qtype = item.get('query_type')
                pred = item.get('predicted')
                gt_entry = gt_by_id.get(qid, {})
                gt = gt_entry.get('ground_truth')
                metrics = {}
                if gt is not None:
                    if qtype in ('action_onset', 'action_offset'):
                        tol = int(tcfg[qtype]['tolerance'])
                        metrics = _eval_frame_error(pred, gt, tol)
                    elif qtype == 'action_duration':
                        thr = float(tcfg[qtype]['threshold'])
                        metrics = _eval_iou(pred, gt, thr)
                    elif qtype == 'multiple_event_ordering':
                        ow = float(tcfg[qtype]['order_weight'])
                        iw = float(tcfg[qtype]['iou_weight'])
                        metrics = _eval_ordering(pred, gt, ow, iw)
                    elif qtype == 'count_frequency':
                        cw = float(tcfg[qtype]['count_weight'])
                        iw = float(tcfg[qtype]['iou_weight'])
                        metrics = _eval_count(pred, gt, cw, iw)
                results.append({
                    'query_id': qid,
                    'query_type': qtype,
                    'question': item.get('question'),
                    'predicted': pred,
                    'ground_truth': gt,
                    'metrics': metrics,
                    'raw_response': item.get('raw_response'),
                })

            # Aggregate per ablation for this clip
            by_type: dict[str, list[dict]] = {}
            for r in results:
                by_type.setdefault(r['query_type'], []).append(r)

            aggregated: dict[str, dict] = {}
            # action_onset/offset
            for qtype in ('action_onset', 'action_offset'):
                qres = by_type.get(qtype, [])
                if qres:
                    errors = [m['metrics']['frame_error'] for m in qres if m['metrics'].get('frame_error') != float('inf')]
                    success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                    aggregated[qtype] = {
                        'mean_frame_error': (round(float(np.mean(errors)), 2) if errors else float('inf')),
                        'success_rate': round(float(success_rate), 2),
                        'count': len(qres),
                    }
            # action_duration
            qres = by_type.get('action_duration', [])
            if qres:
                ious = [m['metrics'].get('iou', 0.0) for m in qres]
                success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                aggregated['action_duration'] = {
                    'mean_iou': round(float(np.mean(ious)), 2) if ious else round(0.0, 2),
                    'success_rate': round(float(success_rate), 2),
                    'count': len(qres),
                }
            # multiple_event_ordering
            qres = by_type.get('multiple_event_ordering', [])
            if qres:
                scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
                order_correct_rate = sum(1 for m in qres if m['metrics'].get('order_correct')) / len(qres)
                mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
                aggregated['multiple_event_ordering'] = {
                    'mean_composite_score': round(float(np.mean(scores)), 2) if scores else round(0.0, 2),
                    'order_correct_rate': round(float(order_correct_rate), 2),
                    'mean_iou': round(float(np.mean(mean_ious)), 2) if mean_ious else round(0.0, 2),
                    'count': len(qres),
                }
            # count_frequency
            qres = by_type.get('count_frequency', [])
            if qres:
                scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
                count_correct_rate = sum(1 for m in qres if m['metrics'].get('count_correct')) / len(qres)
                mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
                aggregated['count_frequency'] = {
                    'mean_composite_score': round(float(np.mean(scores)), 2) if scores else round(0.0, 2),
                    'count_correct_rate': round(float(count_correct_rate), 2),
                    'mean_iou': round(float(np.mean(mean_ious)), 2) if mean_ious else round(0.0, 2),
                    'count': len(qres),
                }

            per_clip_out[ablation] = {
                'metrics': aggregated,
                'results': results,
            }

            # Add to dataset accumulators
            dataset.setdefault(ablation, {}).setdefault('items', []).extend(results)

        # Save per-clip results file
        with (out_dir / f"{clip_name}.json").open('w') as f:
            json.dump({'clip': clip_name, 'ablations': per_clip_out}, f, indent=2)

    # Aggregate dataset-wide per ablation
    summary: dict[str, dict] = {}
    for ablation, data in dataset.items():
        items = data.get('items', [])
        by_type: dict[str, list[dict]] = {}
        for r in items:
            by_type.setdefault(r['query_type'], []).append(r)
        agg: dict[str, dict] = {}
        for qtype in ('action_onset', 'action_offset'):
            qres = by_type.get(qtype, [])
            if qres:
                errors = [m['metrics']['frame_error'] for m in qres if m['metrics'].get('frame_error') != float('inf')]
                success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
                agg[qtype] = {
                    'mean_frame_error': (round(float(np.mean(errors)), 2) if errors else float('inf')),
                    'success_rate': round(float(success_rate), 2),
                    'count': len(qres),
                }
        qres = by_type.get('action_duration', [])
        if qres:
            ious = [m['metrics'].get('iou', 0.0) for m in qres]
            success_rate = sum(1 for m in qres if m['metrics'].get('success')) / len(qres)
            agg['action_duration'] = {
                'mean_iou': round(float(np.mean(ious)), 2) if ious else round(0.0, 2),
                'success_rate': round(float(success_rate), 2),
                'count': len(qres),
            }
        qres = by_type.get('multiple_event_ordering', [])
        if qres:
            scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
            order_correct_rate = sum(1 for m in qres if m['metrics'].get('order_correct')) / len(qres)
            mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
            agg['multiple_event_ordering'] = {
                'mean_composite_score': round(float(np.mean(scores)), 2) if scores else round(0.0, 2),
                'order_correct_rate': round(float(order_correct_rate), 2),
                'mean_iou': round(float(np.mean(mean_ious)), 2) if mean_ious else round(0.0, 2),
                'count': len(qres),
            }
        qres = by_type.get('count_frequency', [])
        if qres:
            scores = [m['metrics'].get('composite_score', 0.0) for m in qres]
            count_correct_rate = sum(1 for m in qres if m['metrics'].get('count_correct')) / len(qres)
            mean_ious = [m['metrics'].get('mean_iou', 0.0) for m in qres]
            agg['count_frequency'] = {
                'mean_composite_score': round(float(np.mean(scores)), 2) if scores else round(0.0, 2),
                'count_correct_rate': round(float(count_correct_rate), 2),
                'mean_iou': round(float(np.mean(mean_ious)), 2) if mean_ious else round(0.0, 2),
                'count': len(qres),
            }
        summary[ablation] = {'metrics': agg}

    with aggregated_file.open('w') as f:
        json.dump({'ablations': summary}, f, indent=2)

def compute_triplets_metrics(cfg: DictConfig):
    if cfg.compute_metrics.triplets is None:
        return

    cm_cfg = cfg.compute_metrics.triplets

    pred_root = Path(cm_cfg.pred_root)
    out_dir = Path(cm_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = Path(cm_cfg.aggregated_output_filename)
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)

    # Loader for ground-truth labels (CholecT50)
    loader = CholecT50Loader(str(cfg.cholect50_root))
    video_cache: Dict[int, Dict] = {}

    # Dataset accumulators per ablation
    dataset: Dict[str, List[Dict]] = {}

    def _eval_triplets(pred: List[Dict], gt: List[Dict]) -> Dict:
        best = {'instrument': False, 'verb': False, 'target': False, 'triplet': False}
        for g in gt:
            for p in pred:
                inst = False
                verb = False
                targ = False
                if p.get('instrument') is not None:
                    inst = normalize_for_matching(p['instrument']) == normalize_for_matching(g.get('instrument', ''))
                if p.get('verb') is not None:
                    verb = normalize_for_matching(p['verb']) == normalize_for_matching(g.get('verb', ''))
                if p.get('target') is not None:
                    targ = normalize_for_matching(p['target']) == normalize_for_matching(g.get('target', ''))
                if inst:
                    best['instrument'] = True
                if verb:
                    best['verb'] = True
                if targ:
                    best['target'] = True
                if inst and verb and targ:
                    best['triplet'] = True
        return best

    def _sets_from_triplets(trips: List[Dict]) -> Dict[str, set[str]]:
        inst_set: set[str] = set()
        verb_set: set[str] = set()
        targ_set: set[str] = set()
        iv_set: set[str] = set()
        it_set: set[str] = set()
        ivt_set: set[str] = set()
        for t in trips or []:
            i = normalize_for_matching(t.get('instrument')) if t.get('instrument') is not None else None
            v = normalize_for_matching(t.get('verb')) if t.get('verb') is not None else None
            tg = normalize_for_matching(t.get('target')) if t.get('target') is not None else None
            if i:
                inst_set.add(i)
            if v:
                verb_set.add(v)
            if tg:
                targ_set.add(tg)
            if i and v:
                iv_set.add(f"{i}|{v}")
            if i and tg:
                it_set.add(f"{i}|{tg}")
            if i and v and tg:
                ivt_set.add(f"{i}|{v}|{tg}")
        return {
            'i': inst_set,
            'v': verb_set,
            't': targ_set,
            'iv': iv_set,
            'it': it_set,
            'ivt': ivt_set,
        }

    # Per-clip processing
    for clip in cfg.clips:
        clip_name = str(clip.name)
        pred_path = pred_root / f"{clip_name}.json"
        if not pred_path.exists():
            continue

        with pred_path.open('r') as f:
            preds = json.load(f)

        per_clip_out: Dict[str, Dict] = {}

        for ablation, items in preds.get('ablations', {}).items():
            results = []
            for item in items:
                video_id = int(item.get('video_id')) if item.get('video_id') is not None else None
                second_idx = int(item.get('second_idx')) if item.get('second_idx') is not None else None
                predicted = item.get('predicted') or []

                gt_trips: List[Dict] = []
                if video_id is not None and second_idx is not None:
                    if video_id not in video_cache:
                        try:
                            video_cache[video_id] = loader.load_video_annotations(video_id)
                        except Exception:
                            video_cache[video_id] = {}
                    vdata = video_cache.get(video_id) or {}
                    if vdata:
                        try:
                            gt_trips = loader.get_frame_triplets(vdata, second_idx)
                        except Exception:
                            gt_trips = []

                metrics = _eval_triplets(predicted, gt_trips)

                results.append({
                    'sample_id': item.get('sample_id'),
                    'video_id': video_id,
                    'second_idx': second_idx,
                    'predicted': predicted,
                    'ground_truth': gt_trips,
                    'metrics': metrics,
                    'raw_response': item.get('raw_response'),
                })

            # Aggregate per ablation for this clip
            n = max(1, len(results))
            instrument_acc = sum(1 for r in results if r['metrics'].get('instrument')) / n
            verb_acc = sum(1 for r in results if r['metrics'].get('verb')) / n
            target_acc = sum(1 for r in results if r['metrics'].get('target')) / n
            triplet_acc = sum(1 for r in results if r['metrics'].get('triplet')) / n

            per_clip_out[ablation] = {
                'metrics': {
                    'instrument_acc': round(float(instrument_acc), 2),
                    'verb_acc': round(float(verb_acc), 2),
                    'target_acc': round(float(target_acc), 2),
                    'triplet_acc': round(float(triplet_acc), 2),
                    'count': len(results),
                },
                'results': results,
            }

            # Add to dataset accumulators
            dataset.setdefault(ablation, []).extend(results)

        # Save per-clip results file
        with (out_dir / f"{clip_name}.json").open('w') as f:
            json.dump({'clip': clip_name, 'ablations': per_clip_out}, f, indent=2)

    # Aggregate dataset-wide per ablation
    summary: Dict[str, Dict] = {}
    for ablation, items in dataset.items():
        n = max(1, len(items))
        instrument_acc = sum(1 for r in items if r['metrics'].get('instrument')) / n
        verb_acc = sum(1 for r in items if r['metrics'].get('verb')) / n
        target_acc = sum(1 for r in items if r['metrics'].get('target')) / n
        triplet_acc = sum(1 for r in items if r['metrics'].get('triplet')) / n
        # Compute mAPs for i, v, t, iv, it, ivt using sklearn average_precision_score
        # Build per-sample label presence sets for GT and predictions
        gt_sets = []
        pred_sets = []
        for r in items:
            gt_sets.append(_sets_from_triplets(r.get('ground_truth') or []))
            pred_sets.append(_sets_from_triplets(r.get('predicted') or []))

        def _compute_map(key: str) -> float:
            # Build class list: classes that appear at least once in GT across samples
            classes: set[str] = set()
            for s in gt_sets:
                classes.update(s[key])
            if not classes:
                return 0.0
            ap_values: list[float] = []
            for cls in sorted(classes):
                y_true = [1 if (cls in s[key]) else 0 for s in gt_sets]
                # Use binary presence as score (degenerate but consistent when no confidences)
                y_score = [1.0 if (cls in s[key]) else 0.0 for s in pred_sets]
                if sum(y_true) == 0:
                    continue
                try:
                    ap = float(average_precision_score(y_true, y_score))
                except Exception:
                    ap = 0.0
                ap_values.append(ap)
            return float(np.mean(ap_values)) if ap_values else 0.0

        map_i = _compute_map('i')
        map_v = _compute_map('v')
        map_t = _compute_map('t')
        map_iv = _compute_map('iv')
        map_it = _compute_map('it')
        map_ivt = _compute_map('ivt')

        summary[ablation] = {
            'metrics': {
                'instrument_acc': round(float(instrument_acc), 2),
                'verb_acc': round(float(verb_acc), 2),
                'target_acc': round(float(target_acc), 2),
                'triplet_acc': round(float(triplet_acc), 2),
                'mAP_i': round(float(map_i), 3),
                'mAP_v': round(float(map_v), 3),
                'mAP_t': round(float(map_t), 3),
                'mAP_iv': round(float(map_iv), 3),
                'mAP_it': round(float(map_it), 3),
                'mAP_ivt': round(float(map_ivt), 3),
                'count': len(items),
            }
        }

    with aggregated_file.open('w') as f:
        json.dump({'ablations': summary}, f, indent=2)

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup (harmless for CPU-only metrics)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    compute_spatial_metrics(cfg)
    compute_temporal_metrics(cfg)
    compute_triplets_metrics(cfg)

if __name__ == "__main__":
    main()