#!/usr/bin/env python3
"""
Display statistics and progress for temporal annotations.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter


def is_annotation_complete(anno: dict) -> bool:
    """Check if an annotation is complete (not a template)."""
    gt = anno.get("ground_truth", {})
    qtype = anno.get("query_type", "")
    
    if qtype in ["action_onset", "action_offset"]:
        return gt.get("frame") is not None
    elif qtype == "action_duration":
        return len(gt.get("ranges", [])) > 0
    elif qtype == "multiple_event_ordering":
        return len(gt.get("events", [])) >= 2
    elif qtype == "count_frequency":
        return gt.get("count", 0) > 0 and len(gt.get("occurrences", [])) > 0
    
    return False


def main():
    repo_root = Path(__file__).parent.parent
    annotations_dir = repo_root / "data/temporal_annotations"
    clips_file = repo_root / "final_dataset_clips.json"
    
    if not annotations_dir.exists():
        print(f"Annotations directory not found: {annotations_dir}")
        print("Run prepare_temporal_labeling.py first.")
        return
    
    # Load expected clips
    with open(clips_file, 'r') as f:
        clips_data = json.load(f)
    
    all_clips = []
    for category, clips in clips_data.items():
        all_clips.extend(clips)
    
    print(f"\n{'='*70}")
    print(f"Temporal Annotation Statistics")
    print(f"{'='*70}\n")
    
    # Track statistics
    completed_files = []
    incomplete_files = []
    query_type_counter = Counter()
    total_queries = 0
    complete_queries = 0
    
    for clip_name in sorted(all_clips):
        file_path = annotations_dir / f"{clip_name}_temporal.json"
        
        if not file_path.exists():
            incomplete_files.append(clip_name)
            print(f"⊙ {clip_name:30s} - NOT STARTED (no file)")
            continue
        
        # Load file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"✗ {clip_name:30s} - INVALID JSON")
            incomplete_files.append(clip_name)
            continue
        
        annotations = data.get("annotations", [])
        
        if not annotations:
            print(f"⊙ {clip_name:30s} - EMPTY (0 queries)")
            incomplete_files.append(clip_name)
            continue
        
        # Check completion status
        complete_annos = [a for a in annotations if is_annotation_complete(a)]
        num_complete = len(complete_annos)
        num_total = len(annotations)
        
        total_queries += num_total
        complete_queries += num_complete
        
        # Count query types
        for anno in complete_annos:
            qtype = anno.get("query_type", "unknown")
            query_type_counter[qtype] += 1
        
        # Status
        if num_complete == num_total and num_total > 0:
            print(f"✓ {clip_name:30s} - COMPLETE ({num_complete} queries)")
            completed_files.append(clip_name)
        elif num_complete > 0:
            print(f"◐ {clip_name:30s} - IN PROGRESS ({num_complete}/{num_total} queries)")
            incomplete_files.append(clip_name)
        else:
            print(f"⊙ {clip_name:30s} - TEMPLATE ({num_total} queries, none filled)")
            incomplete_files.append(clip_name)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Total clips:              {len(all_clips)}")
    print(f"Completed clips:          {len(completed_files)}")
    print(f"Incomplete/not started:   {len(incomplete_files)}")
    print(f"Completion rate:          {len(completed_files)/len(all_clips)*100:.1f}%")
    print()
    print(f"Total queries:            {total_queries}")
    print(f"Complete queries:         {complete_queries}")
    print(f"Incomplete queries:       {total_queries - complete_queries}")
    
    if complete_queries > 0:
        avg_per_clip = complete_queries / len(completed_files) if completed_files else 0
        print(f"Avg queries per clip:     {avg_per_clip:.1f}")
    
    # Query type breakdown
    if query_type_counter:
        print(f"\n{'='*70}")
        print(f"Query Type Distribution (completed queries only)")
        print(f"{'='*70}")
        for qtype, count in query_type_counter.most_common():
            percentage = count / complete_queries * 100 if complete_queries > 0 else 0
            print(f"  {qtype:30s} {count:4d} ({percentage:5.1f}%)")
    
    # Recommendations
    if incomplete_files:
        print(f"\n{'='*70}")
        print(f"Next Steps")
        print(f"{'='*70}")
        print(f"Continue annotating these {len(incomplete_files)} clips:")
        for clip in incomplete_files[:10]:  # Show first 10
            print(f"  - {clip}")
        if len(incomplete_files) > 10:
            print(f"  ... and {len(incomplete_files) - 10} more")
    
    print()


if __name__ == "__main__":
    main()

