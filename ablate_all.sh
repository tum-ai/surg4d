# ==================================================================
# Test: First scene only
# ==================================================================
# top config + frame tool ablation
pixi run python track_objects.py clips=final_dataset_first
pixi run python extract_graphs.py clips=final_dataset_first
pixi run python evaluate_benchmark.py clips=final_dataset_first
pixi run python compute_metrics.py clips=final_dataset_first

# ablation 1: disable jump filtering
pixi run python track_objects.py clips=final_dataset_first \
    "track_objects.cotracker_filter_depth_jumps=False" \
    "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
    "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
    "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

pixi run python extract_graphs.py clips=final_dataset_first \
    "track_objects.cotracker_filter_depth_jumps=False" \
    "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
    "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
    "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

pixi run python evaluate_benchmark.py clips=final_dataset_first \
    "track_objects.cotracker_filter_depth_jumps=False" \
    "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
    "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
    "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

pixi run python compute_metrics.py clips=final_dataset_first \
    "track_objects.cotracker_filter_depth_jumps=False" \
    "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
    "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
    "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# ablation 2: introduce multiframe tracking
pixi run python track_objects.py clips=final_dataset_first \
    "track_objects.cotracker_init_from_multiple_frames=True" \
    "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
    "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
    "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

pixi run python extract_graphs.py clips=final_dataset_first \
    "track_objects.cotracker_init_from_multiple_frames=True" \
    "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
    "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
    "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

pixi run python evaluate_benchmark.py clips=final_dataset_first \
    "track_objects.cotracker_init_from_multiple_frames=True" \
    "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
    "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
    "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

pixi run python compute_metrics.py clips=final_dataset_first \
    "track_objects.cotracker_init_from_multiple_frames=True" \
    "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
    "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
    "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# ablation 3: disable occlusion filling
pixi run python track_objects.py clips=final_dataset_first \
    "track_objects.cotracker_fill_occlusions=False" \
    "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
    "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
    "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

pixi run python extract_graphs.py clips=final_dataset_first \
    "track_objects.cotracker_fill_occlusions=False" \
    "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
    "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
    "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

pixi run python evaluate_benchmark.py clips=final_dataset_first \
    "track_objects.cotracker_fill_occlusions=False" \
    "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
    "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
    "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

pixi run python compute_metrics.py clips=final_dataset_first \
    "track_objects.cotracker_fill_occlusions=False" \
    "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
    "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
    "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# ==================================================================
# Test: 8B First scene
# ==================================================================
pixi run python evaluate_benchmark.py clips=final_dataset_first \
    "eval.qwen3_size=8B" \
    "eval.qwen3_use_fp8=False" \
    "eval.output_dir=\${output_root}/predictions_final_8b" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_8b"

pixi run python compute_metrics.py clips=final_dataset_first \
    "eval.qwen3_size=8B" \
    "eval.qwen3_use_fp8=False" \
    "eval.output_dir=\${output_root}/predictions_final_8b" \
    "eval.directional.methods=['graph_agent_semantics']" \
    "eval.spatial.methods=['graph_agent_semantics']" \
    "eval.temporal.methods=['graph_agent_semantics']" \
    "compute_metrics.output_dir=\${output_root}/metrics_final_8b"

# # ==================================================================
# # First half of dataset
# # ==================================================================
# # top config + frame tool ablation
# pixi run python track_objects.py clips=first_half
# pixi run python extract_graphs.py clips=first_half
# pixi run python evaluate_benchmark.py clips=first_half
# pixi run python compute_metrics.py clips=first_half

# # ablation 1: disable jump filtering
# pixi run python track_objects.py clips=first_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python extract_graphs.py clips=first_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python evaluate_benchmark.py clips=first_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python compute_metrics.py clips=first_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# # ablation 2: introduce multiframe tracking
# pixi run python track_objects.py clips=first_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python extract_graphs.py clips=first_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python evaluate_benchmark.py clips=first_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python compute_metrics.py clips=first_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# # ablation 3: disable occlusion filling
# pixi run python track_objects.py clips=first_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python extract_graphs.py clips=first_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python evaluate_benchmark.py clips=first_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python compute_metrics.py clips=first_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"


# # ==================================================================
# # Second half of dataset
# # ==================================================================
# # top config + frame tool ablation
# pixi run python track_objects.py clips=second_half
# pixi run python extract_graphs.py clips=second_half
# pixi run python evaluate_benchmark.py clips=second_half
# pixi run python compute_metrics.py clips=second_half

# # ablation 1: disable jump filtering
# pixi run python track_objects.py clips=second_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python extract_graphs.py clips=second_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python evaluate_benchmark.py clips=second_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# pixi run python compute_metrics.py clips=second_half \
#     "track_objects.cotracker_filter_depth_jumps=False" \
#     "track_objects.cotracker_subdir=cotracker_final_nojumpfilter" \
#     "graph_extraction.graph_output_subdir=graph_final_nojumpfilter" \
#     "eval.output_dir=\${output_root}/predictions_final_nojumpfilter" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_nojumpfilter"

# # ablation 2: introduce multiframe tracking
# pixi run python track_objects.py clips=second_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python extract_graphs.py clips=second_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python evaluate_benchmark.py clips=second_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# pixi run python compute_metrics.py clips=second_half \
#     "track_objects.cotracker_init_from_multiple_frames=True" \
#     "track_objects.cotracker_subdir=cotracker_final_multiframeinit" \
#     "graph_extraction.graph_output_subdir=graph_final_multiframeinit" \
#     "eval.output_dir=\${output_root}/predictions_final_multiframeinit" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_multiframeinit"

# # ablation 3: disable occlusion filling
# pixi run python track_objects.py clips=second_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python extract_graphs.py clips=second_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python evaluate_benchmark.py clips=second_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"

# pixi run python compute_metrics.py clips=second_half \
#     "track_objects.cotracker_fill_occlusions=False" \
#     "track_objects.cotracker_subdir=cotracker_final_noocclusionfilling" \
#     "graph_extraction.graph_output_subdir=graph_final_noocclusionfilling" \
#     "eval.output_dir=\${output_root}/predictions_final_noocclusionfilling" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_noocclusionfilling"


# # ==================================================================
# # Test: 8B Full DS
# # ==================================================================
# pixi run python evaluate_benchmark.py clips=final_dataset \
#     "eval.qwen3_size=8B" \
#     "eval.qwen3_use_fp8=False" \
#     "eval.output_dir=\${output_root}/predictions_final_8b" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_8b"

# pixi run python compute_metrics.py clips=final_dataset \
#     "eval.qwen3_size=8B" \
#     "eval.qwen3_use_fp8=False" \
#     "eval.output_dir=\${output_root}/predictions_final_8b" \
#     "eval.directional.methods=['graph_agent_semantics']" \
#     "eval.spatial.methods=['graph_agent_semantics']" \
#     "eval.temporal.methods=['graph_agent_semantics']" \
#     "compute_metrics.output_dir=\${output_root}/metrics_final_8b"