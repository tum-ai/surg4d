#!/usr/bin/env python3
"""
Script to prepare temporal annotation template files.
Creates empty temporal JSON files for each clip in final_dataset_clips.json.
"""

import json
from pathlib import Path


def create_temporal_template(clip_name: str, num_frames: int = 80) -> dict:
    """Create an empty temporal annotation template for a clip."""
    return {
        "clip_info": {
            "clip_name": clip_name,
            "num_frames": num_frames,
            "annotation_instructions": "Add temporal queries below. Each query should have a unique query_id.",
        },
        "annotations": [
            # Example queries (delete and replace with your annotations)
            {
                "query_id": f"{clip_name}_q1",
                "query_type": "action_onset",
                "question": "When does the grasper first enter the surgical field?",
                "answer_format": "Frame X",
                "ground_truth": {"frame": None},  # Fill in after reviewing video
                "notes": "Optional: Add notes about this annotation",
            },
            {
                "query_id": f"{clip_name}_q2",
                "query_type": "action_duration",
                "question": "During which frames is tissue being grasped?",
                "answer_format": "Frames X-Y (or multiple ranges: X-Y, A-B)",
                "ground_truth": {
                    "ranges": []  # Fill in: [[start1, end1], [start2, end2], ...]
                },
                "notes": "Use multiple ranges if action happens multiple times",
            },
        ],
    }


def main():
    # Paths
    repo_root = Path(__file__).parent.parent
    clips_file = repo_root / "final_dataset_clips.json"
    output_dir = repo_root / "data/temporal_annotations"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clips
    with open(clips_file, "r") as f:
        clips_data = json.load(f)

    # Count total clips
    total_clips = sum(len(clips) for clips in clips_data.values())
    print(f"\n{'='*60}")
    print(f"Creating temporal annotation templates")
    print(f"{'='*60}")
    print(f"Total clips: {total_clips}")

    # Create template for each clip
    created_count = 0
    skipped_count = 0

    for category, clips in clips_data.items():
        print(f"\n{category}:")
        for clip_name in clips:
            output_file = output_dir / f"{clip_name}_temporal.json"

            # Skip if file already exists (don't overwrite)
            if output_file.exists():
                print(f"  ⊙ {clip_name} (already exists, skipping)")
                skipped_count += 1
                continue

            # Create template
            template = create_temporal_template(clip_name)

            # Save to file
            with open(output_file, "w") as f:
                json.dump(template, f, indent=2)

            print(f"  ✓ {clip_name}")
            created_count += 1

    print(f"\n{'='*60}")
    print(f"Template creation complete!")
    print(f"  Created: {created_count} files")
    print(f"  Skipped: {skipped_count} files (already exist)")
    print(f"  Location: {output_dir}")
    print(f"{'='*60}")

    # Print next steps
    print("\n📋 Next Steps:")
    print("1. Watch each clip video to understand the actions")
    print("2. Edit each *_temporal.json file to add your annotations")
    print("3. For each clip, create 2-4 temporal queries of different types")
    print("4. Focus on action_onset and action_duration for simplicity")
    print("5. Commit completed annotations to git")
    print("\n💡 Tips:")
    print(
        "- Use labeling/clips/{category}/{clip_name}/{clip_name}.mp4 to review videos"
    )
    print("- Frame numbers are 0-indexed (0 to 79)")
    print("- For action_duration, ranges are inclusive: [start, end]")
    print("- Add descriptive 'notes' field for complex cases")


if __name__ == "__main__":
    main()
