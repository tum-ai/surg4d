# Confidence-Based mAP Implementation for Triplet Evaluation

## 📋 Summary of Changes

This document describes the implementation of a **TIER-BASED CONFIDENCE SYSTEM** for triplet predictions to fix the discrepancy where mAP scores didn't reflect improvements in triplet accuracy.

### Problem
- **Issue 1**: mAP used binary (0/1) scores and set-based decomposition, giving credit for predicting correct components even in wrong combinations
- **Issue 2**: Model-generated confidence scores had insufficient variance (clustering around 0.95, 0.85, 0.8)
- **Result**: `dynamic_graph` had 4x better `triplet_acc` (0.58 vs 0.15) but nearly identical mAP scores to `single_frame`

### Solution
- **Tier-based confidence system** embedded in prompts with 5 distinct tiers (0.90-0.95, 0.70-0.80, 0.50-0.65, 0.30-0.45, 0.10-0.25)
- **Variance requirements** force model to use different tiers for multiple predictions
- **Context-specific criteria** for each ablation type (visual clarity, temporal continuity, graph structure)
- **Multiple ground truth handling** properly supported in mAP computation
- mAP computation now treats full I|V|T combinations as classes with confidence scores

---

## 🔧 Changes Made

### 1. **Updated Prompts with Tier-Based System** (`conf/eval/triplets/default.yaml`)

All 8 prompts now include a structured tier-based confidence system:

**Tier System Structure:**
```
Tier 1 (0.90-0.95): Highest confidence - all evidence clear and unambiguous
Tier 2 (0.70-0.80): High confidence - clear evidence with minor ambiguity
Tier 3 (0.50-0.65): Medium confidence - partial evidence or inferred action
Tier 4 (0.30-0.45): Low confidence - multiple interpretations or weak evidence
Tier 5 (0.10-0.25): Very low confidence - speculative or minimal evidence
```

**Variance Requirements Added:**
- NEVER assign same confidence to all triplets in a response
- Multiple triplets MUST span at least 2 different tiers (unless only 1 predicted)
- Use FULL range within each tier (e.g., 0.92, 0.73, 0.54, not just 0.95, 0.75, 0.55)

**Context-Specific Criteria for Each Ablation:**

1. **single_frame / single_frame_mask_overlay**: Based on visual clarity of components and action
2. **multiframe / multiframe_mask_overlay**: Based on temporal continuity and action clarity at timestep t
3. **static_descriptors**: Based on descriptor specificity and action inference from text
4. **static_graph**: Based on edge strength (overlap score) and descriptor clarity
5. **dynamic_descriptors**: Based on descriptor match at t and temporal trajectory
6. **dynamic_graph**: Based on edge strength at t, descriptor clarity, AND temporal evolution

**Updated ablations:**
- `single_frame` ✓
- `single_frame_mask_overlay` ✓
- `multiframe` ✓
- `multiframe_mask_overlay` ✓
- `static_descriptors` ✓
- `static_graph` ✓
- `dynamic_descriptors` ✓
- `dynamic_graph` ✓

### 2. **Updated Parser** (`benchmark/frame_evaluators.py`)

Modified `_normalize_triplets()` function to extract and validate confidence scores:

```python
# Extract confidence score, default to 1.0 if not present or invalid
confidence = item.get('confidence', 1.0)
try:
    confidence = float(confidence)
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
except (TypeError, ValueError):
    confidence = 1.0

normalized.append({
    'instrument': inst, 
    'verb': verb, 
    'target': targ,
    'confidence': confidence
})
```

**Behavior:**
- Extracts `confidence` field from model output
- Defaults to 1.0 if missing (backward compatible)
- Clamps values to [0.0, 1.0] range
- Handles invalid values gracefully

### 3. **Proper mAP Computation** (`compute_metrics.py`)

Completely rewrote mAP computation with multiple ground truth support:

**Old approach (broken):**
```python
# Decomposed triplets into sets
gt_sets = [{'i': {instruments}, 'v': {verbs}, 't': {targets}}]
y_score = [1.0 if (cls in pred_set) else 0.0]  # Binary!
```

**New approach (correct):**
```python
# Treats full I|V|T combinations as classes
def _compute_map_proper(key: str, use_full_triplet: bool = False):
    # For each unique triplet combination in GT
    for cls in gt_classes:
        y_true = [1 if cls in gt else 0 for each sample]
        y_score = [confidence of cls in predictions for each sample]
        ap = average_precision_score(y_true, y_score)
```

**Multiple Ground Truth Handling:**
The mAP computation correctly handles cases where a frame has multiple valid ground truth triplets:
```python
# Lines 770-774 in compute_metrics.py
gt_has = any(
    f"{normalize_for_matching(t.get('instrument'))}|{...}" == cls
    for t in gt_trips  # Iterates over ALL ground truth triplets
    if t.get('instrument') and t.get('verb') and t.get('target')
)
```

This means if ground truth has:
- `grasper|grasp|gallbladder` 
- `bipolar|coagulate|blood_vessel`

The model can predict both with different confidences, and mAP will properly evaluate each triplet class independently.

**What changed:**
- **`mAP_i`, `mAP_v`, `mAP_t`**: Now use confidence scores (was binary)
- **`mAP_ivt`**: Now evaluates full triplet combinations with confidence (was set-based)
- **`mAP_iv`, `mAP_it`**: Kept legacy implementation for now
- **Multiple GT support**: Already properly implemented using `any()` checks

---

## 🚀 How to Run

### Step 1: Re-run Evaluation

The evaluation will now prompt the model to output confidence scores:

```bash
pixi run python evaluate_benchmark.py
```

**What happens:**
- Model receives updated prompts requesting confidence
- Parser extracts confidence scores (or defaults to 1.0)
- Predictions saved with confidence scores in JSON

### Step 2: Compute New Metrics

After evaluation completes, compute metrics with the new mAP implementation:

```bash
pixi run python compute_metrics.py
```

**Expected improvements:**
- `mAP_ivt` for `dynamic_graph` should now be significantly higher than `single_frame`
- Should align with the 3x improvement in `triplet_acc`
- Component-level mAPs (`mAP_i`, `mAP_v`, `mAP_t`) will also better reflect confidence

---

## 📊 Expected Results

### Before (Old mAP):
```json
{
  "single_frame": {
    "triplet_acc": 0.15,
    "mAP_ivt": 0.217
  },
  "dynamic_graph": {
    "triplet_acc": 0.46,  // 3x better!
    "mAP_ivt": 0.212      // Barely different ❌
  }
}
```

### After (New mAP with Confidence):
```json
{
  "single_frame": {
    "triplet_acc": 0.15,
    "mAP_ivt": 0.2-0.3    // Should be similar or lower
  },
  "dynamic_graph": {
    "triplet_acc": 0.46,
    "mAP_ivt": 0.4-0.5    // Should be much higher! ✅
  }
}
```

**Why this makes sense:**
- `dynamic_graph` predicts fewer but more accurate triplets with higher confidence
- `single_frame` predicts many triplets with wrong combinations
- New mAP rewards correct combinations with high confidence

---

## 🔍 Technical Details

### How Confidence Affects mAP

mAP (mean Average Precision) measures how well a model **ranks** predictions:

1. **For each triplet class** (e.g., `grasper|grasp|gallbladder`):
   - Sort all predictions by confidence (highest first)
   - Compute precision at each recall level
   - Average these precisions = AP for this class

2. **Mean across all classes** = mAP

**Example:**
```
Ground truth: {grasper|grasp|gallbladder} in samples [1, 3, 5]

Predictions sorted by confidence:
1. Sample 1: grasper|grasp|gallbladder (conf=0.95) ✅ Precision=1/1=1.00
2. Sample 2: grasper|retract|liver (conf=0.80) ❌     Precision=1/2=0.50
3. Sample 3: grasper|grasp|gallbladder (conf=0.75) ✅ Precision=2/3=0.67
4. Sample 5: grasper|grasp|gallbladder (conf=0.60) ✅ Precision=3/4=0.75

AP = (1.00 + 0.67 + 0.75) / 3 = 0.81
```

If the model had wrong confidences (high conf for wrong predictions), AP would be lower!

### Backward Compatibility

The implementation is **fully backward compatible**:
- If model doesn't output confidence, parser defaults to 1.0
- Old predictions without confidence still work (treated as equally confident)
- No breaking changes to data formats

---

## 🐛 Debugging

If you see issues after running evaluation:

### Check predictions have confidence:
```bash
# Look at a prediction file
cat output/*/predictions/triplets/video01_00240.json | head -50
```

Should see:
```json
{
  "predicted": [
    {
      "instrument": "grasper",
      "verb": "grasp",
      "target": "gallbladder",
      "confidence": 0.95
    }
  ]
}
```

### Verify model is following new prompt:
Check `raw_response` field in predictions to see if model includes confidence scores.

### Test mAP computation:
```bash
# Run debug script to compare old vs new mAP
pixi run python debug_map_vs_triplet_acc.py
```

---

## 📝 Important Notes

### Tier-Based System Benefits

1. **Forced variance**: The tier system with explicit variance requirements prevents the model from clustering all predictions around similar values (e.g., all 0.95)

2. **Contextual guidance**: Each ablation has specific criteria tied to the available information (visual clarity, temporal context, graph structure, etc.)

3. **Interpretability**: Clear tier definitions make it easier to understand why a prediction received a particular confidence score

4. **Robustness to model behavior**: Even if the model doesn't perfectly calibrate probabilities, the tier system ensures sufficient variance for mAP to distinguish between better and worse predictions

### Multiple Ground Truth Scenarios

The system properly handles cases where:
- Multiple instruments are active simultaneously (e.g., left and right hand instruments)
- Same instrument performing multiple actions (e.g., grasper retracting gallbladder AND grasper grasping cystic_duct)
- Ambiguous cases where annotators might have labeled different but equally valid interpretations

### Model Calibration

While the tier system ensures variance, the model should still aim for calibration:
- Tier 1 predictions should be correct >90% of the time
- Tier 2 predictions correct ~75% of the time
- And so on...

### Future Improvements

- Could also implement proper mAP for `mAP_iv` and `mAP_it` 
- Could add calibration metrics (ECE - Expected Calibration Error)
- Could visualize confidence distributions per tier
- Could analyze which ablations use which tiers most frequently

---

## 📚 References

- Original debug analysis: `debug_map_vs_triplet_acc.py`
- sklearn documentation: [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- COCO evaluation metrics (similar approach): [cocodataset.org](https://cocodataset.org/#detection-eval)

