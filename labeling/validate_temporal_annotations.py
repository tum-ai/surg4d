#!/usr/bin/env python3
"""
Validation script for temporal annotations.
Checks format, ranges, and required fields.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class TemporalAnnotationValidator:
    """Validator for temporal annotation JSON files."""
    
    def __init__(self, num_frames: int = 80):
        self.num_frames = num_frames
        self.errors = []
        self.warnings = []
    
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate a single temporal annotation file."""
        self.errors = []
        self.warnings = []
        
        # Check file exists
        if not file_path.exists():
            self.errors.append(f"File not found: {file_path}")
            return False, self.errors, self.warnings
        
        # Load JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False, self.errors, self.warnings
        
        # Validate structure
        self._validate_structure(data)
        
        # Validate annotations
        if "annotations" in data:
            self._validate_annotations(data["annotations"])
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_structure(self, data: Dict):
        """Validate top-level structure."""
        if "annotations" not in data:
            self.errors.append("Missing 'annotations' key")
            return
        
        if not isinstance(data["annotations"], list):
            self.errors.append("'annotations' must be a list")
            return
        
        if len(data["annotations"]) == 0:
            self.warnings.append("No annotations found (empty list)")
    
    def _validate_annotations(self, annotations: List[Dict]):
        """Validate each annotation."""
        query_ids = set()
        
        for i, anno in enumerate(annotations):
            # Check required fields
            required_fields = ["query_id", "query_type", "question", "answer_format", "ground_truth"]
            for field in required_fields:
                if field not in anno:
                    self.errors.append(f"Annotation {i}: Missing required field '{field}'")
            
            # Check query_id uniqueness
            if "query_id" in anno:
                qid = anno["query_id"]
                if qid in query_ids:
                    self.errors.append(f"Annotation {i}: Duplicate query_id '{qid}'")
                query_ids.add(qid)
            
            # Validate query type
            if "query_type" in anno:
                self._validate_query_type(anno, i)
    
    def _validate_query_type(self, anno: Dict, index: int):
        """Validate query type-specific fields."""
        qtype = anno["query_type"]
        gt = anno.get("ground_truth", {})
        
        valid_types = ["action_onset", "action_offset", "action_duration", 
                       "multiple_event_ordering", "count_frequency"]
        
        if qtype not in valid_types:
            self.errors.append(
                f"Annotation {index}: Invalid query_type '{qtype}'. "
                f"Must be one of {valid_types}"
            )
            return
        
        # Validate based on type
        if qtype in ["action_onset", "action_offset"]:
            self._validate_single_frame(gt, index, qtype)
        
        elif qtype == "action_duration":
            self._validate_ranges(gt, index)
        
        elif qtype == "multiple_event_ordering":
            self._validate_events(gt, index)
        
        elif qtype == "count_frequency":
            self._validate_count_frequency(gt, index)
    
    def _validate_single_frame(self, gt: Dict, index: int, qtype: str):
        """Validate single frame ground truth."""
        if "frame" not in gt:
            self.errors.append(f"Annotation {index}: Missing 'frame' in ground_truth for {qtype}")
            return
        
        frame = gt["frame"]
        
        # Allow None for templates
        if frame is None:
            self.warnings.append(f"Annotation {index}: Frame is None (template not filled)")
            return
        
        if not isinstance(frame, int):
            self.errors.append(f"Annotation {index}: Frame must be an integer, got {type(frame)}")
            return
        
        if frame < 0 or frame >= self.num_frames:
            self.errors.append(
                f"Annotation {index}: Frame {frame} out of range [0, {self.num_frames-1}]"
            )
    
    def _validate_ranges(self, gt: Dict, index: int):
        """Validate frame ranges."""
        if "ranges" not in gt:
            self.errors.append(f"Annotation {index}: Missing 'ranges' in ground_truth")
            return
        
        ranges = gt["ranges"]
        
        if not isinstance(ranges, list):
            self.errors.append(f"Annotation {index}: 'ranges' must be a list")
            return
        
        # Allow empty ranges for negative cases (action doesn't occur)
        if len(ranges) == 0:
            self.warnings.append(f"Annotation {index}: Empty ranges (negative case: action doesn't occur)")
            return
        
        for j, r in enumerate(ranges):
            # Allow empty lists for negative test cases (action doesn't occur in that instance)
            if isinstance(r, list) and len(r) == 0:
                # This is a negative case within a multi-range scenario - acceptable
                continue
            
            if not isinstance(r, list) or len(r) != 2:
                self.errors.append(
                    f"Annotation {index}, Range {j}: Must be [start, end] or [], got {r}"
                )
                continue
            
            start, end = r
            
            if not isinstance(start, int) or not isinstance(end, int):
                self.errors.append(
                    f"Annotation {index}, Range {j}: Start and end must be integers"
                )
                continue
            
            if start < 0 or start >= self.num_frames:
                self.errors.append(
                    f"Annotation {index}, Range {j}: Start {start} out of range [0, {self.num_frames-1}]"
                )
            
            if end < 0 or end >= self.num_frames:
                self.errors.append(
                    f"Annotation {index}, Range {j}: End {end} out of range [0, {self.num_frames-1}]"
                )
            
            if start > end:
                self.errors.append(
                    f"Annotation {index}, Range {j}: Start {start} > End {end}"
                )
    
    def _validate_events(self, gt: Dict, index: int):
        """Validate multiple event ordering."""
        if "events" not in gt:
            self.errors.append(f"Annotation {index}: Missing 'events' in ground_truth")
            return
        
        events = gt["events"]
        
        if not isinstance(events, list):
            self.errors.append(f"Annotation {index}: 'events' must be a list")
            return
        
        if len(events) < 2:
            self.errors.append(
                f"Annotation {index}: At least 2 events required for ordering, got {len(events)}"
            )
            return
        
        prev_frame = -1
        for j, event in enumerate(events):
            if not isinstance(event, dict):
                self.errors.append(f"Annotation {index}, Event {j}: Must be a dict")
                continue
            
            if "name" not in event or "frame" not in event:
                self.errors.append(
                    f"Annotation {index}, Event {j}: Must have 'name' and 'frame' fields"
                )
                continue
            
            frame = event["frame"]
            
            if not isinstance(frame, int):
                self.errors.append(
                    f"Annotation {index}, Event {j}: Frame must be integer"
                )
                continue
            
            if frame < 0 or frame >= self.num_frames:
                self.errors.append(
                    f"Annotation {index}, Event {j}: Frame {frame} out of range [0, {self.num_frames-1}]"
                )
            
            # Check chronological order
            if frame <= prev_frame:
                self.warnings.append(
                    f"Annotation {index}, Event {j}: Events not in chronological order "
                    f"(frame {frame} <= previous frame {prev_frame})"
                )
            
            prev_frame = frame
    
    def _validate_count_frequency(self, gt: Dict, index: int):
        """Validate count/frequency."""
        if "count" not in gt or "occurrences" not in gt:
            self.errors.append(
                f"Annotation {index}: Missing 'count' or 'occurrences' in ground_truth"
            )
            return
        
        count = gt["count"]
        occurrences = gt["occurrences"]
        
        if not isinstance(count, int) or count < 0:
            self.errors.append(f"Annotation {index}: 'count' must be non-negative integer")
        
        if not isinstance(occurrences, list):
            self.errors.append(f"Annotation {index}: 'occurrences' must be a list")
            return
        
        if len(occurrences) != count:
            self.warnings.append(
                f"Annotation {index}: Count {count} doesn't match number of occurrences {len(occurrences)}"
            )
        
        # Validate each occurrence as a range
        for j, r in enumerate(occurrences):
            if not isinstance(r, list) or len(r) != 2:
                self.errors.append(
                    f"Annotation {index}, Occurrence {j}: Must be [start, end]"
                )
                continue
            
            start, end = r
            if start < 0 or start >= self.num_frames or end < 0 or end >= self.num_frames:
                self.errors.append(
                    f"Annotation {index}, Occurrence {j}: Frames out of range"
                )
            
            if start > end:
                self.errors.append(
                    f"Annotation {index}, Occurrence {j}: Start > End"
                )


def main():
    repo_root = Path(__file__).parent.parent
    annotations_dir = repo_root / "data/temporal_annotations"
    
    if not annotations_dir.exists():
        print(f"ERROR: Annotations directory not found: {annotations_dir}")
        print("Run prepare_temporal_labeling.py first to create template files.")
        return
    
    # Find all temporal annotation files
    annotation_files = sorted(annotations_dir.glob("*_temporal.json"))
    
    if not annotation_files:
        print(f"No temporal annotation files found in {annotations_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Validating {len(annotation_files)} temporal annotation files")
    print(f"{'='*70}\n")
    
    validator = TemporalAnnotationValidator(num_frames=80)
    
    valid_count = 0
    invalid_count = 0
    warning_count = 0
    
    for file_path in annotation_files:
        is_valid, errors, warnings = validator.validate_file(file_path)
        
        # Print results
        status = "✓" if is_valid else "✗"
        color = "\033[92m" if is_valid else "\033[91m"  # Green or Red
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} {file_path.name}")
        
        if errors:
            for error in errors:
                print(f"    ERROR: {error}")
            invalid_count += 1
        else:
            valid_count += 1
        
        if warnings:
            for warning in warnings:
                print(f"    WARNING: {warning}")
            warning_count += 1
        
        if errors or warnings:
            print()
    
    # Summary
    print(f"{'='*70}")
    print(f"Validation Summary:")
    print(f"  Valid:    {valid_count}")
    print(f"  Invalid:  {invalid_count}")
    print(f"  Warnings: {warning_count}")
    print(f"{'='*70}\n")
    
    if invalid_count > 0:
        print("⚠️  Some files have errors. Please fix them before running evaluation.")
        exit(1)
    elif warning_count > 0:
        print("⚠️  Some files have warnings (likely unfilled templates).")
        exit(0)
    else:
        print("✓ All files are valid!")
        exit(0)


if __name__ == "__main__":
    main()

