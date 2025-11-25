# Re-run Evaluation Instructions

## What was fixed:
1. ✅ Aggregated.json computation (use `skip_compute_metrics=false` override)
2. ✅ All 8 ablation prompts now have forceful confidence variance requirements

## Steps to verify the fix:

### Step 1: Re-run evaluation with updated prompts
```bash
# This will use the new prompts with strong variance requirements
pixi run python evaluate_benchmark.py
```

This will:
- Use the updated prompts that force the model to output diverse confidence values
- Generate new predictions with varied confidence scores
- Save predictions to `output/final_scenes_qwen3_False/predictions/triplets/`

### Step 2: Compute metrics
```bash
pixi run python compute_metrics.py skip_compute_metrics=false
```

This will:
- Read the new predictions with diverse confidence values
- Compute mAP using the confidence-based approach
- Save aggregated metrics to `output/final_scenes_qwen3_False/metrics/triplets/aggregated.json`

### Step 3: Verify confidence variance improved
```bash
# Check distribution of confidence values
grep -r '"confidence":' output/final_scenes_qwen3_False/predictions/triplets/*.json | \
  awk -F'"confidence": ' '{print $2}' | awk -F',' '{print $1}' | \
  sort | uniq -c | sort -rn | head -20
```

Expected improvement:
- **Before:** 79 × 0.92, 10 × 0.73 (basically 2 values)
- **After:** Should see 10-20+ unique values spread across tiers (0.20-0.95)

### Step 4: Check mAP improvements
```bash
# View the aggregated metrics
cat output/final_scenes_qwen3_False/metrics/triplets/aggregated.json | jq '.ablations.dynamic_graph.metrics'
cat output/final_scenes_qwen3_False/metrics/triplets/aggregated.json | jq '.ablations.single_frame.metrics'
```

Expected:
- `dynamic_graph` should have higher `mAP_ivt` than before (was 0.275, should increase)
- The gap between `dynamic_graph` and `single_frame` mAP should widen (reflecting better predictions)

## Troubleshooting:

### If confidence values are still all the same:
1. Check that you're using the updated config file:
   ```bash
   grep "MANDATORY CONFIDENCE VARIANCE" conf/eval/triplets/default.yaml
   ```
   Should see 8 matches (one per ablation)

2. Check raw model responses:
   ```bash
   cat output/final_scenes_qwen3_False/predictions/triplets/video01_00240.json | jq '.ablations.single_frame[0].raw_response'
   ```
   Look for confidence values in the JSON response

### If aggregated.json is not created:
- Make sure you're passing `skip_compute_metrics=false`
- Check that predictions exist: `ls output/final_scenes_qwen3_False/predictions/triplets/`

## Expected Results

### Before fixes:
- Distribution: 79% = 0.92, 10% = 0.73, rest scattered
- mAP doesn't reflect quality differences between ablations
- Aggregated.json not computed without override

### After fixes:
- Distribution: 10-20+ unique values spread across 0.20-0.95
- mAP properly reflects prediction quality (dynamic_graph >> single_frame)
- Aggregated.json computes successfully with override flag

