#!/usr/bin/env python3
"""
Debug script to understand why mAP scores don't reflect triplet_acc improvements.

This script analyzes the predictions and shows:
1. How many predictions are made per sample
2. How set-based mAP differs from exact triplet matching
3. Cases where individual components match but full triplets don't
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import numpy as np

def normalize_for_matching(s):
    """Normalize labels for matching (from benchmark_config.py)"""
    if s is None:
        return None
    return str(s).lower().strip().replace(' ', '_').replace('-', '_')

def sets_from_triplets(trips: List[Dict]) -> Dict[str, set]:
    """Convert triplets to sets (same logic as compute_metrics.py line 593)"""
    result = {'i': set(), 'v': set(), 't': set(), 'iv': set(), 'it': set(), 'ivt': set()}
    for t in trips or []:
        i = normalize_for_matching(t.get('instrument')) if t.get('instrument') else None
        v = normalize_for_matching(t.get('verb')) if t.get('verb') else None
        tg = normalize_for_matching(t.get('target')) if t.get('target') else None
        if i:
            result['i'].add(i)
        if v:
            result['v'].add(v)
        if tg:
            result['t'].add(tg)
        if i and v:
            result['iv'].add(f"{i}|{v}")
        if i and tg:
            result['it'].add(f"{i}|{tg}")
        if i and v and tg:
            result['ivt'].add(f"{i}|{v}|{tg}")
    return result

def eval_triplets(pred: List[Dict], gt: List[Dict]) -> Dict:
    """Evaluate triplets (same logic as compute_metrics.py line 570)"""
    best = {'instrument': False, 'verb': False, 'target': False, 'triplet': False}
    for g in gt:
        for p in pred:
            inst = normalize_for_matching(p.get('instrument')) == normalize_for_matching(g.get('instrument'))
            verb = normalize_for_matching(p.get('verb')) == normalize_for_matching(g.get('verb'))
            targ = normalize_for_matching(p.get('target')) == normalize_for_matching(g.get('target'))
            if inst:
                best['instrument'] = True
            if verb:
                best['verb'] = True
            if targ:
                best['target'] = True
            if inst and verb and targ:
                best['triplet'] = True
    return best

def analyze_predictions(metrics_file: Path, ablation: str):
    """Analyze predictions for a specific ablation"""
    with metrics_file.open('r') as f:
        data = json.load(f)
    
    if ablation not in data.get('ablations', {}):
        return None
    
    results = data['ablations'][ablation]['results']
    
    # Statistics
    stats = {
        'total_samples': len(results),
        'num_predictions_dist': Counter(),  # Distribution of prediction counts
        'num_gt_dist': Counter(),  # Distribution of GT counts
        'partial_matches': 0,  # Samples with some but not all components correct
        'full_matches': 0,  # Samples with complete triplet match
        'no_matches': 0,  # Samples with no component matches
        'examples': []
    }
    
    for r in results:
        pred = r['predicted'] or []
        gt = r['ground_truth'] or []
        metrics = r['metrics']
        
        stats['num_predictions_dist'][len(pred)] += 1
        stats['num_gt_dist'][len(gt)] += 1
        
        if metrics['triplet']:
            stats['full_matches'] += 1
        elif metrics['instrument'] or metrics['verb'] or metrics['target']:
            stats['partial_matches'] += 1
            # Collect examples of partial matches
            if len(stats['examples']) < 5:
                stats['examples'].append({
                    'sample_id': r['sample_id'],
                    'predicted': pred,
                    'ground_truth': gt,
                    'metrics': metrics
                })
        else:
            stats['no_matches'] += 1
    
    return stats

def compute_set_based_accuracy(metrics_file: Path, ablation: str):
    """Compute set-based accuracy (similar to how mAP works)"""
    with metrics_file.open('r') as f:
        data = json.load(f)
    
    if ablation not in data.get('ablations', {}):
        return None
    
    results = data['ablations'][ablation]['results']
    
    # For each component type, compute precision/recall
    component_stats = {
        'i': {'tp': 0, 'fp': 0, 'fn': 0},
        'v': {'tp': 0, 'fp': 0, 'fn': 0},
        't': {'tp': 0, 'fp': 0, 'fn': 0},
        'ivt': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    for r in results:
        pred_sets = sets_from_triplets(r['predicted'] or [])
        gt_sets = sets_from_triplets(r['ground_truth'] or [])
        
        for key in ['i', 'v', 't', 'ivt']:
            tp = len(pred_sets[key] & gt_sets[key])
            fp = len(pred_sets[key] - gt_sets[key])
            fn = len(gt_sets[key] - pred_sets[key])
            
            component_stats[key]['tp'] += tp
            component_stats[key]['fp'] += fp
            component_stats[key]['fn'] += fn
    
    # Compute precision/recall/F1
    metrics = {}
    for key in ['i', 'v', 't', 'ivt']:
        tp = component_stats[key]['tp']
        fp = component_stats[key]['fp']
        fn = component_stats[key]['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics[key] = {
            'precision': round(prec, 3),
            'recall': round(rec, 3),
            'f1': round(f1, 3),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics

def main():
    metrics_dir = Path("output/final_scenes_qwen3_False/metrics/triplets")
    
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found: {metrics_dir}")
        return
    
    print("=" * 80)
    print("DEBUGGING mAP vs triplet_acc DISCREPANCY")
    print("=" * 80)
    print()
    
    # Analyze all clips for specific ablations
    ablations_to_compare = ['single_frame', 'dynamic_graph']
    
    aggregated_stats = {abl: defaultdict(int) for abl in ablations_to_compare}
    
    print("Analyzing per-clip predictions...\n")
    
    for clip_file in sorted(metrics_dir.glob("*.json")):
        if clip_file.name == "aggregated.json":
            continue
        
        print(f"\n{'='*80}")
        print(f"CLIP: {clip_file.stem}")
        print(f"{'='*80}")
        
        for ablation in ablations_to_compare:
            print(f"\n{ablation}:")
            print("-" * 40)
            
            stats = analyze_predictions(clip_file, ablation)
            if stats is None:
                print(f"  No data for {ablation}")
                continue
            
            set_metrics = compute_set_based_accuracy(clip_file, ablation)
            
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Full matches (triplet_acc): {stats['full_matches']} ({stats['full_matches']/stats['total_samples']*100:.1f}%)")
            print(f"  Partial matches: {stats['partial_matches']} ({stats['partial_matches']/stats['total_samples']*100:.1f}%)")
            print(f"  No matches: {stats['no_matches']} ({stats['no_matches']/stats['total_samples']*100:.1f}%)")
            print(f"\n  Predictions per sample: {dict(stats['num_predictions_dist'])}")
            print(f"  Ground truth per sample: {dict(stats['num_gt_dist'])}")
            
            print(f"\n  Set-based metrics (similar to mAP logic):")
            for key, name in [('i', 'Instrument'), ('v', 'Verb'), ('t', 'Target'), ('ivt', 'Full Triplet')]:
                m = set_metrics[key]
                print(f"    {name:12s}: Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
            
            # Show examples of partial matches
            if stats['examples']:
                print(f"\n  Example partial matches:")
                for ex in stats['examples'][:2]:
                    print(f"    Sample: {ex['sample_id']}")
                    print(f"      Predicted: {ex['predicted']}")
                    print(f"      GT:        {ex['ground_truth']}")
                    print(f"      Matches:   I={ex['metrics']['instrument']}, V={ex['metrics']['verb']}, T={ex['metrics']['target']}, Full={ex['metrics']['triplet']}")
            
            # Aggregate for summary
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    aggregated_stats[ablation][k] += v
    
    print("\n\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    # Load aggregated metrics
    agg_file = metrics_dir / "aggregated.json"
    if agg_file.exists():
        with agg_file.open('r') as f:
            agg_data = json.load(f)
        
        print("\nAggregated metrics from compute_metrics.py:")
        print("-" * 40)
        for ablation in ablations_to_compare:
            metrics = agg_data['ablations'][ablation]['metrics']
            print(f"\n{ablation}:")
            print(f"  triplet_acc: {metrics['triplet_acc']:.2f}")
            print(f"  mAP_i:       {metrics['mAP_i']:.3f}")
            print(f"  mAP_v:       {metrics['mAP_v']:.3f}")
            print(f"  mAP_t:       {metrics['mAP_t']:.3f}")
            print(f"  mAP_ivt:     {metrics['mAP_ivt']:.3f}")
    
    print("\n\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. triplet_acc requires EXACT matching of all three components (I, V, T) in the SAME triplet.
   
2. mAP_* uses SET-BASED evaluation:
   - Collects all unique instruments/verbs/targets from ALL predicted triplets
   - Computes per-class Average Precision
   - Averages across all classes in ground truth
   
3. This means:
   - If you predict [{grasper, cut, liver}, {scissors, grasp, cystic_duct}]
   - Your sets are: i={grasper, scissors}, v={cut, grasp}, t={liver, cystic_duct}
   - You get credit for predicting these classes even if in WRONG COMBINATIONS
   
4. Additionally, the mAP implementation uses BINARY scores (1.0 or 0.0) instead of 
   confidence scores, which makes it behave more like set-based recall than true mAP.

5. PARTIAL MATCHES are the key:
   - Samples where instrument or verb or target match but not all three
   - These contribute to mAP but NOT to triplet_acc
   - If dynamic_graph has fewer but more accurate predictions, it may actually
     have LOWER mAP than single_frame which predicts many triplets with wrong combinations
""")

if __name__ == "__main__":
    main()

