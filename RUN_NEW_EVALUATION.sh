#!/bin/bash
# Script to run evaluation with new confidence-based mAP

set -e

echo "=========================================="
echo "Running Triplet Evaluation with Confidence Scores"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Run evaluate_benchmark.py with updated prompts (model outputs confidence)"
echo "2. Run compute_metrics.py with new mAP computation (uses confidence)"
echo "3. Compare results to show mAP now aligns with triplet_acc"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Step 1: Run evaluation
echo ""
echo "Step 1/2: Running evaluation (this may take a while)..."
echo "----------------------------------------"
pixi run python evaluate_benchmark.py

# Step 2: Compute metrics
echo ""
echo "Step 2/2: Computing metrics with new mAP..."
echo "----------------------------------------"
pixi run python compute_metrics.py

# Step 3: Show results
echo ""
echo "=========================================="
echo "✅ Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Predictions: output/*/predictions/triplets/"
echo "  - Metrics: output/*/metrics/triplets/aggregated.json"
echo ""
echo "To view aggregated results:"
echo "  cat output/*/metrics/triplets/aggregated.json | jq '.ablations | to_entries[] | {ablation: .key, triplet_acc: .value.metrics.triplet_acc, mAP_ivt: .value.metrics.mAP_ivt}'"
echo ""
echo "Expected: mAP_ivt for dynamic_graph should now be significantly higher than single_frame!"

