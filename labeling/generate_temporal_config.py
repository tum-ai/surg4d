#!/usr/bin/env python3
"""
Generate config snippets for temporal evaluation files.
Outputs YAML that can be added to clip config files.
"""

import json
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent
    clips_file = repo_root / "final_dataset_clips.json"
    
    # Load clips
    with open(clips_file, 'r') as f:
        clips_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Temporal Evaluation Config Snippets")
    print(f"{'='*70}\n")
    print("Add these lines to your clip configs in conf/clips/*.yaml:\n")
    
    for category, clips in clips_data.items():
        print(f"# {category}")
        print(f"# {'='*60}")
        for clip_name in clips:
            print(f"- name: {clip_name}")
            print(f"  temporal_eval_file: data/temporal_annotations/{clip_name}_temporal.json")
            print()
    
    print(f"\n{'='*70}")
    print(f"Example: Complete clip config")
    print(f"{'='*70}\n")
    
    # Show complete example
    example_clip = clips_data["seg80_only"][0]  # video17_01563
    print(f"""- name: {example_clip}
  video_id: 17
  first_frame: 1563
  last_frame: 1643  # 80 frames total
  frame_stride: 1
  temporal_eval_file: data/temporal_annotations/{example_clip}_temporal.json
  spatial_eval_file: data/labels/seg80_only/{example_clip}_spatial.json
""")


if __name__ == "__main__":
    main()

