# Tier-Based Confidence System - Quick Reference

## 🎯 Overview

This guide provides a quick reference for the tier-based confidence system implemented in all 8 triplet evaluation ablations. The system ensures proper variance in confidence scores for accurate mAP computation.

## 📊 Confidence Tiers

### Tier 1: 0.90-0.95 (Highest Confidence)
**Use when:** All evidence is clear, unambiguous, and definitively supports the triplet

**Criteria by ablation:**
- **Single frame**: ALL components clearly visible AND action definitively happening AND no ambiguity
- **Multiframe**: Triplet clearly active at t AND temporal context strongly confirms continuity
- **Static graph**: Strong edge connection (overlap>0.7) AND descriptors unambiguous
- **Dynamic graph**: Strong edge at t (overlap>0.7) AND descriptors clear AND temporal trajectory confirms sustained action
- **Descriptors**: Descriptors unambiguously describe all components AND action explicitly indicated

**Example:** Grasper tip clearly grasping gallbladder tissue, visible in high quality with no occlusion

---

### Tier 2: 0.70-0.80 (High Confidence)
**Use when:** Clear evidence with minor ambiguity or partial information

**Criteria by ablation:**
- **Single frame**: Instrument and target visible BUT action partially obscured OR minor ambiguity about exact verb
- **Multiframe**: Triplet active at t BUT temporal context shows action just starting/ending OR minor discontinuity
- **Static graph**: Moderate edge connection (overlap 0.4-0.7) OR descriptors clear but action inferred from spatial proximity
- **Dynamic graph**: Moderate edge at t (overlap 0.4-0.7) OR temporal context shows action starting/ending at t
- **Descriptors**: Descriptors clearly identify components BUT action must be inferred from spatial relationships

**Example:** Instrument clearly visible and positioned near target, but exact action moment is slightly unclear

---

### Tier 3: 0.50-0.65 (Medium Confidence)
**Use when:** Partial evidence or action must be inferred from context

**Criteria by ablation:**
- **Single frame**: Instrument visible BUT target partially obscured OR action inferred from instrument position/orientation
- **Multiframe**: Triplet likely active at t BUT temporal context ambiguous OR action inferred from nearby frames
- **Static graph**: Weak edge connection (overlap<0.4) OR descriptors partially ambiguous but graph structure supports interpretation
- **Dynamic graph**: Weak edge at t (overlap<0.4) BUT temporal pattern strongly suggests interaction OR descriptors partially ambiguous
- **Descriptors**: Descriptors partially match components OR action inferred from typical surgical workflow

**Example:** Instrument present and target region visible, but occlusion or angle makes exact action uncertain

---

### Tier 4: 0.30-0.45 (Low Confidence)
**Use when:** Multiple valid interpretations possible or significant ambiguity

**Criteria by ablation:**
- **Single frame**: Multiple valid interpretations possible OR significant portion obscured but context strongly suggests combination
- **Multiframe**: Uncertain if active at exact timestep t BUT temporal pattern suggests this is plausible
- **Static graph**: No direct edge BUT close centroid distance suggests interaction OR descriptors allow multiple interpretations
- **Dynamic graph**: No direct edge at t BUT close centroids AND temporal evolution suggests transient interaction
- **Descriptors**: Descriptors suggest multiple possible triplets OR incomplete information but plausible combination

**Example:** Scene suggests a specific action based on surgical context, but visual evidence is ambiguous

---

### Tier 5: 0.10-0.25 (Very Low Confidence)
**Use when:** Speculative interpretation with minimal supporting evidence

**Criteria by ablation:**
- **Single frame**: Speculative interpretation OR very limited visible evidence BUT anatomically plausible
- **Multiframe**: Speculative based on temporal context OR action visible before/after but unclear at t
- **Static graph**: Minimal graph evidence OR highly speculative based on surgical context alone
- **Dynamic graph**: Minimal graph evidence at t OR speculative based on temporal trends without strong structural support
- **Descriptors**: Descriptors provide minimal evidence OR highly speculative interpretation

**Example:** Prediction based primarily on typical surgical workflow rather than direct visual evidence

---

## ⚠️ Variance Requirements

### Critical Rules:
1. **NEVER** assign the same confidence to all triplets in a single response
2. If predicting **multiple triplets**, they **MUST span at least 2 different tiers** (unless only 1 triplet is predicted)
3. Use the **FULL range** within each tier
   - ✅ Good: 0.92, 0.91, 0.73, 0.54
   - ❌ Bad: 0.95, 0.95, 0.75, 0.75, 0.55 (no variance)
   - ❌ Bad: 0.95, 0.75, 0.55 (only using tier boundaries)

---

## 🔍 Examples by Scenario

### Scenario 1: Single Clear Action
```json
{
  "triplets": [
    {
      "instrument": "grasper",
      "verb": "grasp",
      "target": "gallbladder",
      "confidence": 0.93
    }
  ]
}
```
✅ Only one triplet, Tier 1 confidence is appropriate

---

### Scenario 2: Primary + Secondary Action
```json
{
  "triplets": [
    {
      "instrument": "grasper",
      "verb": "retract",
      "target": "gallbladder",
      "confidence": 0.92
    },
    {
      "instrument": "hook",
      "verb": "dissect",
      "target": "cystic_plate",
      "confidence": 0.74
    }
  ]
}
```
✅ Two triplets spanning Tier 1 (0.92) and Tier 2 (0.74)

---

### Scenario 3: Clear + Ambiguous Actions
```json
{
  "triplets": [
    {
      "instrument": "bipolar",
      "verb": "coagulate",
      "target": "blood_vessel",
      "confidence": 0.91
    },
    {
      "instrument": "grasper",
      "verb": "grasp",
      "target": "cystic_duct",
      "confidence": 0.71
    },
    {
      "instrument": "irrigator",
      "verb": "irrigate",
      "target": "abdominal_wall_cavity",
      "confidence": 0.43
    }
  ]
}
```
✅ Three triplets spanning Tier 1 (0.91), Tier 2 (0.71), and Tier 4 (0.43)

---

## 🚫 Common Mistakes

### Mistake 1: All High Confidence
```json
{
  "triplets": [
    {"instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 0.95},
    {"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel", "confidence": 0.95},
    {"instrument": "hook", "verb": "dissect", "target": "cystic_plate", "confidence": 0.95}
  ]
}
```
❌ All same confidence, breaks variance requirement

---

### Mistake 2: Only Tier Boundaries
```json
{
  "triplets": [
    {"instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 0.95},
    {"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel", "confidence": 0.80},
    {"instrument": "hook", "verb": "dissect", "target": "cystic_plate", "confidence": 0.65}
  ]
}
```
❌ Using only tier boundaries (0.95, 0.80, 0.65), not full range

---

### Mistake 3: Single Tier Only
```json
{
  "triplets": [
    {"instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 0.92},
    {"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel", "confidence": 0.91},
    {"instrument": "hook", "verb": "dissect", "target": "cystic_plate", "confidence": 0.93}
  ]
}
```
❌ All in Tier 1, doesn't span at least 2 different tiers

---

## 📈 Impact on mAP

The tier-based system ensures proper mAP computation by:

1. **Providing ranking information**: Confidence scores rank predictions from most to least confident
2. **Enabling discrimination**: Different ablations with different accuracy should have different mAP scores
3. **Rewarding precision**: Models that correctly identify which predictions are more/less certain get higher mAP

### Before Tier System:
```
dynamic_graph:  triplet_acc=0.58, mAP_ivt=0.215  ❌
single_frame:   triplet_acc=0.15, mAP_ivt=0.224  ❌
```
Nearly identical mAP despite 4x difference in accuracy!

### After Tier System (Expected):
```
dynamic_graph:  triplet_acc=0.58, mAP_ivt=0.45-0.55  ✅
single_frame:   triplet_acc=0.15, mAP_ivt=0.20-0.30  ✅
```
mAP now reflects the accuracy difference!

---

## 🔧 Troubleshooting

### If mAP scores still don't differentiate ablations:

1. **Check prediction files**: Verify confidence scores have variance
   ```bash
   grep -o '"confidence": [0-9.]*' output/.../predictions/triplets/*.json | sort | uniq -c
   ```

2. **Verify model is following instructions**: Check `raw_response` fields to see model reasoning

3. **Analyze confidence distributions**: See which tiers are actually being used

4. **Consider prompt adjustments**: May need to emphasize variance requirements more

---

## 📚 Related Documents

- `CONFIDENCE_SCORING_IMPLEMENTATION.md` - Detailed implementation guide
- `conf/eval/triplets/default.yaml` - Actual prompts with tier system
- `compute_metrics.py` - mAP computation code (lines 731-807)

