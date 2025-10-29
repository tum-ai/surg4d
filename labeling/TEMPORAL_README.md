# Temporal Annotations

Manual temporal VQA annotations for 25 surgical video clips (55 queries total).

## Structure

- **Annotations**: `annotations_backup/temporal/*.json` (git-tracked backup)
- **Primary location**: `data/temporal_annotations/` (symlinked, used by evaluation)

## Query Types

- **action_onset** (28 queries): When does action X start? → Frame N
- **action_duration** (27 queries): During which frames does X happen? → Ranges [[start, end], ...]

Frame numbers are 0-indexed (0-19 for 20-frame clips).

## Scripts

```bash
# Validate annotations
pixi run python labeling/validate_temporal_annotations.py

# Show statistics
pixi run python labeling/temporal_annotation_stats.py

# Generate config snippets for clip YAML files
pixi run python labeling/generate_temporal_config.py

# Regenerate templates (if needed)
pixi run python labeling/prepare_temporal_labeling.py
```

## Usage

Add to clip configs in `conf/clips/*.yaml`:
```yaml
- name: video17_01563
  temporal_eval_file: data/temporal_annotations/video17_01563_temporal.json
```

Run evaluation:
```bash
pixi run python evaluate_benchmark.py +clips=final_dataset +eval=temporal
```

## Format

```json
{
  "clip_info": {"clip_name": "video17_01563", "num_frames": 20},
  "annotations": [
    {
      "query_id": "video17_01563_q1",
      "query_type": "action_onset",
      "question": "When does the grasper enter?",
      "ground_truth": {"frame": 5}
    }
  ]
}
```

## Notes

- Empty ranges `[[]]` indicate negative cases (action doesn't occur)
- All 25 clips validated and complete
- Clips: 8 seg80_only + 17 seg80_t50_intersection

