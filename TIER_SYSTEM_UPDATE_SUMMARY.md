# Tier-Based Confidence System - Update Summary

## ✅ What Was Done

I've implemented a comprehensive **tier-based confidence system** to solve the mAP computation issue where `dynamic_graph` had 4x better triplet accuracy (0.58 vs 0.15) but nearly identical mAP scores.

### Root Cause Identified
The problem was that the model was outputting confidence scores with **insufficient variance** - clustering around 0.95, 0.85, 0.8. When all predictions have similar scores, mAP cannot effectively distinguish between better and worse predictions.

### Solution Implemented
A structured **5-tier confidence system** embedded directly in all 8 evaluation prompts with:
- **Clear tier definitions** with specific confidence ranges (0.90-0.95, 0.70-0.80, 0.50-0.65, 0.30-0.45, 0.10-0.25)
- **Context-specific criteria** for each ablation type
- **Explicit variance requirements** that force the model to use different tiers
- **Multiple ground truth support** (already working in mAP computation)

---

## 📝 Files Modified

### 1. `/workspace/paddy/surgery-scene-graphs/conf/eval/triplets/default.yaml`
**Updated all 8 prompts with tier-based confidence system:**
- ✅ `single_frame`
- ✅ `single_frame_mask_overlay`
- ✅ `multiframe`
- ✅ `multiframe_mask_overlay`
- ✅ `static_descriptors`
- ✅ `static_graph`
- ✅ `dynamic_descriptors`
- ✅ `dynamic_graph`

Each prompt now includes:
```yaml
CONFIDENCE TIER SYSTEM (use specific values, DO NOT always use the same tier):
Tier 1 (0.90-0.95): [context-specific high confidence criteria]
Tier 2 (0.70-0.80): [context-specific medium-high confidence criteria]
Tier 3 (0.50-0.65): [context-specific medium confidence criteria]
Tier 4 (0.30-0.45): [context-specific low confidence criteria]
Tier 5 (0.10-0.25): [context-specific very low confidence criteria]

VARIANCE REQUIREMENTS:
- NEVER assign the same confidence to all triplets in a single response
- If predicting multiple triplets, they MUST span at least 2 different tiers unless only 1 triplet is predicted
- Use the FULL range within each tier (e.g., use 0.92, 0.73, 0.54, not just 0.95, 0.75, 0.55)
```

### 2. `/workspace/paddy/surgery-scene-graphs/CONFIDENCE_SCORING_IMPLEMENTATION.md`
**Updated documentation to reflect tier-based approach:**
- Added explanation of the tier system
- Documented variance requirements
- Explained multiple ground truth handling (already working correctly)
- Added notes on tier system benefits

### 3. `/workspace/paddy/surgery-scene-graphs/TIER_BASED_CONFIDENCE_GUIDE.md` *(NEW)*
**Created comprehensive quick reference guide:**
- Detailed criteria for each tier
- Examples of correct and incorrect usage
- Common mistakes to avoid
- Troubleshooting guidance

### 4. `/workspace/paddy/surgery-scene-graphs/TIER_SYSTEM_UPDATE_SUMMARY.md` *(THIS FILE)*
Summary of changes and next steps

---

## 🎯 Key Features of Tier System

### Context-Specific Criteria

Each ablation type has tier criteria tailored to its available information:

#### Single Frame / Mask Overlay
- **Tier 1**: ALL components clearly visible AND action definitively happening
- **Tier 2**: Components visible BUT action partially obscured OR minor ambiguity about verb
- **Tier 3**: Instrument visible BUT target partially obscured OR action inferred from position
- **Tier 4**: Multiple interpretations OR significant occlusion but context suggests combination
- **Tier 5**: Speculative OR very limited evidence BUT anatomically plausible

#### Multiframe / Mask Overlay
- **Tier 1**: Triplet clearly active at t AND temporal context strongly confirms continuity
- **Tier 2**: Active at t BUT temporal context shows action starting/ending OR minor discontinuity
- **Tier 3**: Likely active at t BUT temporal context ambiguous OR action inferred from nearby frames
- **Tier 4**: Uncertain at exact timestep t BUT temporal pattern suggests plausible
- **Tier 5**: Speculative based on temporal context OR action visible before/after but unclear at t

#### Static Graph
- **Tier 1**: Strong edge connection (overlap>0.7) AND descriptors unambiguous
- **Tier 2**: Moderate edge (overlap 0.4-0.7) OR descriptors clear but action inferred from spatial proximity
- **Tier 3**: Weak edge (overlap<0.4) OR descriptors partially ambiguous but graph structure supports interpretation
- **Tier 4**: No direct edge BUT close centroid distance OR descriptors allow multiple interpretations
- **Tier 5**: Minimal graph evidence OR highly speculative based on surgical context alone

#### Dynamic Graph (Most Important - highest triplet_acc)
- **Tier 1**: Strong edge at t (overlap>0.7) AND descriptors unambiguous AND temporal trajectory confirms sustained action
- **Tier 2**: Moderate edge at t (overlap 0.4-0.7) OR temporal context shows action starting/ending at t
- **Tier 3**: Weak edge at t (overlap<0.4) BUT temporal pattern strongly suggests interaction OR descriptors partially ambiguous
- **Tier 4**: No direct edge at t BUT close centroids AND temporal evolution suggests transient interaction
- **Tier 5**: Minimal graph evidence at t OR speculative based on temporal trends without strong structural support

#### Static/Dynamic Descriptors
- **Tier 1**: Descriptors unambiguously describe all components AND action explicitly indicated
- **Tier 2**: Descriptors clearly identify components BUT action must be inferred from spatial relationships
- **Tier 3**: Descriptors partially match components OR action inferred from typical surgical workflow
- **Tier 4**: Descriptors suggest multiple possible triplets OR incomplete information but plausible
- **Tier 5**: Descriptors provide minimal evidence OR highly speculative interpretation

### Variance Enforcement

Three mechanisms ensure variance:
1. **Explicit instruction**: "NEVER assign the same confidence to all triplets"
2. **Multi-tier requirement**: "MUST span at least 2 different tiers"
3. **Full range usage**: "Use the FULL range within each tier (e.g., 0.92, 0.73, 0.54)"

### Multiple Ground Truth Handling

The system already handles multiple ground truth triplets correctly:
```python
# From compute_metrics.py lines 770-774
gt_has = any(
    f"{normalize_for_matching(t.get('instrument'))}|{...}" == cls
    for t in gt_trips  # Iterates over ALL ground truth triplets
    if t.get('instrument') and t.get('verb') and t.get('target')
)
```

This means:
- If a frame has multiple valid triplets, each is evaluated independently
- Model can predict all of them with different confidences
- mAP computation treats each triplet class separately

---

## 🚀 Next Steps

### 1. Re-run Evaluation (Required)
You need to re-run the evaluation to generate predictions with the new tier-based confidence system:

```bash
# Using your existing evaluation script
bash RUN_NEW_EVALUATION.sh
```

Or directly:
```bash
python evaluate_benchmark.py
```

**What this does:**
- Sends updated prompts (with tier system) to the model
- Model outputs triplets with varied confidence scores following tier guidelines
- Predictions saved to `output/*/predictions/triplets/*.json`

**Expected output format:**
```json
{
  "predicted": [
    {
      "instrument": "grasper",
      "verb": "grasp",
      "target": "gallbladder",
      "confidence": 0.92
    },
    {
      "instrument": "bipolar",
      "verb": "coagulate",
      "target": "blood_vessel",
      "confidence": 0.71
    }
  ]
}
```

### 2. Compute New Metrics (Required)
After evaluation completes, compute metrics with the confidence-based mAP:

```bash
python compute_metrics.py
```

**What this does:**
- Loads predictions with confidence scores
- Computes mAP using confidence for ranking (not binary)
- Evaluates full I|V|T combinations as classes
- Outputs to `output/*/metrics/triplets/aggregated.json`

### 3. Verify Results (Recommended)

Check if mAP now reflects triplet accuracy differences:

```bash
# View the aggregated metrics
cat output/final_scenes_qwen3_False/metrics/triplets/aggregated.json | jq
```

**Expected improvement:**
```json
{
  "ablations": {
    "single_frame": {
      "metrics": {
        "triplet_acc": 0.15,
        "mAP_ivt": 0.20-0.30    // Should be lower than before
      }
    },
    "dynamic_graph": {
      "metrics": {
        "triplet_acc": 0.58,
        "mAP_ivt": 0.45-0.55    // Should be MUCH higher than single_frame!
      }
    }
  }
}
```

The key is that **mAP_ivt for dynamic_graph should now be significantly higher** than for single_frame, reflecting the 4x better triplet accuracy.

### 4. Analyze Confidence Distributions (Optional)

Check if the model is actually using different tiers:

```bash
# Extract all confidence scores from predictions
grep -roh '"confidence": [0-9.]*' output/*/predictions/triplets/*.json | \
  cut -d' ' -f2 | \
  sort -n | \
  uniq -c

# Or more detailed analysis
python -c "
import json
import glob
from collections import Counter

confs = []
for f in glob.glob('output/*/predictions/triplets/*.json'):
    with open(f) as fp:
        data = json.load(fp)
        for abl, items in data.get('ablations', {}).items():
            for item in items:
                for pred in item.get('predicted', []):
                    if 'confidence' in pred:
                        confs.append(pred['confidence'])

# Count by tier
tiers = {
    'Tier 1 (0.90-0.95)': sum(1 for c in confs if 0.90 <= c <= 0.95),
    'Tier 2 (0.70-0.80)': sum(1 for c in confs if 0.70 <= c <= 0.80),
    'Tier 3 (0.50-0.65)': sum(1 for c in confs if 0.50 <= c <= 0.65),
    'Tier 4 (0.30-0.45)': sum(1 for c in confs if 0.30 <= c <= 0.45),
    'Tier 5 (0.10-0.25)': sum(1 for c in confs if 0.10 <= c <= 0.25),
}

for tier, count in tiers.items():
    print(f'{tier}: {count} ({100*count/len(confs):.1f}%)')
"
```

**Good distribution** should show usage across multiple tiers, not just Tier 1.

---

## 📊 Troubleshooting

### Issue: Model still outputs similar confidence scores

**Symptoms:**
- All predictions still around 0.95, 0.85, 0.8
- mAP scores still don't differentiate ablations

**Solutions:**
1. Check if prompts are being loaded correctly:
   ```bash
   # Verify the tier system is in the config
   grep -A 5 "CONFIDENCE TIER SYSTEM" conf/eval/triplets/default.yaml
   ```

2. Check model's raw responses:
   ```bash
   # Look at a few raw responses to see if model acknowledges tier system
   jq '.ablations.dynamic_graph[0].raw_response' output/*/predictions/triplets/*.json | head -5
   ```

3. If model ignores tier requirements, you may need to:
   - Make tier requirements even more explicit in prompts
   - Add few-shot examples showing correct tier usage
   - Consider post-processing to enforce variance (last resort)

### Issue: mAP computation fails

**Symptoms:**
- Error running `compute_metrics.py`
- Missing confidence scores in output

**Solutions:**
1. Ensure parser handles confidence correctly:
   ```python
   # Already implemented in benchmark/frame_evaluators.py lines 487-494
   confidence = item.get('confidence', 1.0)
   try:
       confidence = float(confidence)
       confidence = max(0.0, min(1.0, confidence))
   except (TypeError, ValueError):
       confidence = 1.0
   ```

2. Check if old predictions are being used (without confidence):
   - Delete old prediction files: `rm -rf output/*/predictions/triplets/*.json`
   - Re-run evaluation: `python evaluate_benchmark.py`

### Issue: Multiple ground truth triplets not handled

**Symptoms:**
- mAP seems too low when multiple valid actions occur simultaneously
- Predictions of valid alternative triplets penalized

**Solutions:**
- Already handled correctly in `compute_metrics.py` lines 770-774
- Each ground truth triplet is checked independently using `any()`
- If still seeing issues, verify ground truth data has multiple triplets when expected

---

## 📖 Documentation

Three documents are now available:

1. **TIER_SYSTEM_UPDATE_SUMMARY.md** *(this file)*
   - Overview of changes
   - Next steps and troubleshooting

2. **TIER_BASED_CONFIDENCE_GUIDE.md**
   - Quick reference for tier criteria
   - Examples and common mistakes
   - Impact on mAP explanation

3. **CONFIDENCE_SCORING_IMPLEMENTATION.md** *(updated)*
   - Technical implementation details
   - Code changes explained
   - Backward compatibility notes

---

## 🎓 Understanding the Impact

### Why Tier System Solves the Problem

**Before (Model-generated confidence):**
```
Prediction 1: grasper|grasp|gallbladder     confidence=0.95
Prediction 2: grasper|retract|liver         confidence=0.95  ❌ WRONG
Prediction 3: bipolar|coagulate|blood_vessel confidence=0.85
Prediction 4: hook|dissect|cystic_duct      confidence=0.85  ❌ WRONG
```

mAP can't distinguish because confidences don't reflect correctness!

**After (Tier-based system):**
```
Prediction 1: grasper|grasp|gallbladder     confidence=0.92  ✓ (Tier 1 - clear evidence)
Prediction 2: grasper|retract|liver         confidence=0.43  ❌ (Tier 4 - uncertain, turned out wrong)
Prediction 3: bipolar|coagulate|blood_vessel confidence=0.76  ✓ (Tier 2 - clear but minor ambiguity)
Prediction 4: hook|dissect|cystic_duct      confidence=0.28  ❌ (Tier 4 - speculative, turned out wrong)
```

Now mAP rewards the model for being **more confident in correct predictions and less confident in incorrect ones**!

### Why This Helps Dynamic Graph

The `dynamic_graph` ablation has the most information:
- Spatial graph structure at each timestep
- Temporal evolution of nodes and edges
- Object descriptors at each timestep

This allows it to:
- Assign **Tier 1 confidence** (0.90-0.95) when strong edge + clear descriptors + sustained temporal trajectory
- Assign **lower tiers** (0.70-0.80 or 0.50-0.65) when evidence is partial or action is transient
- Correctly **differentiate** between highly certain predictions and uncertain ones

In contrast, `single_frame` only sees one image:
- Less able to distinguish true actions from momentary positions
- More likely to assign similar confidences to different quality predictions
- Results in lower mAP because ranking is less accurate

---

## ✅ Summary

You now have:
- ✅ **All 8 prompts updated** with tier-based confidence system
- ✅ **Variance requirements** enforced through explicit instructions
- ✅ **Context-specific criteria** for each ablation type
- ✅ **Multiple ground truth support** verified in mAP computation
- ✅ **Comprehensive documentation** for reference

**Next action:** Re-run evaluation and compute metrics to see the improved mAP scores!

```bash
# Step 1: Re-run evaluation (this will take time)
python evaluate_benchmark.py

# Step 2: Compute metrics with new confidence scores
python compute_metrics.py

# Step 3: Check results
cat output/final_scenes_qwen3_False/metrics/triplets/aggregated.json | jq '.ablations | to_entries[] | {name: .key, triplet_acc: .value.metrics.triplet_acc, mAP_ivt: .value.metrics.mAP_ivt}'
```

Expected outcome: **dynamic_graph should now have significantly higher mAP_ivt than single_frame**, reflecting its superior triplet accuracy!

