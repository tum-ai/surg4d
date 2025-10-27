#!/usr/bin/env python3
"""
Generate a new dataset combining cholecseg8k clips with cholect50 annotations.
Only processes clips whose videos are in both datasets.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
CHOLECSEG8K_DIR = Path("/home/tumai/nico/surgery-scene-graphs/data/cholecseg8k")
CHOLECT50_DIR = Path("/home/tumai/nico/surgery-scene-graphs/data/cholect50")
OUTPUT_DIR = Path("/home/tumai/nico/surgery-scene-graphs/data/cholecseg8k_with_t50_annotations")

# Videos present in both datasets
COMMON_VIDEOS = [1, 12, 18, 25, 26, 27, 35, 43, 48, 52]

# Frame rate info
FULL_FPS = 25  # cholecseg8k is at full framerate
T50_FPS = 1    # cholect50 is at 1fps


def video_num_to_vid_str(video_num: int) -> str:
    """Convert video number to VID format (e.g., 1 -> VID01)."""
    return f"VID{video_num:02d}"


def video_num_to_video_str(video_num: int) -> str:
    """Convert video number to video format (e.g., 1 -> video01)."""
    return f"video{video_num:02d}"


def parse_clip_name(clip_name: str) -> Tuple[int, int]:
    """
    Parse clip name to extract video number and start frame.
    E.g., 'video01_00080' -> (1, 80)
    """
    parts = clip_name.split('_')
    video_num = int(parts[0].replace('video', ''))
    start_frame = int(parts[1])
    return video_num, start_frame


def map_frame_to_t50(frame_num: int) -> int:
    """
    Map a 25fps frame number to the corresponding 1fps cholect50 frame.
    E.g., frame 80 -> t50 frame 3 (80 // 25 = 3)
    """
    return frame_num // FULL_FPS


def load_cholect50_annotations(video_num: int) -> Dict:
    """Load cholect50 annotations for a video."""
    vid_str = video_num_to_vid_str(video_num)
    label_path = CHOLECT50_DIR / "labels" / f"{vid_str}.json"
    
    with open(label_path, 'r') as f:
        return json.load(f)


def get_frame_files(clip_dir: Path) -> List[Path]:
    """Get sorted list of frame files (frame_*_endo.png) in a clip directory."""
    frame_files = sorted([
        f for f in clip_dir.iterdir() 
        if f.name.startswith('frame_') and f.name.endswith('_endo.png')
    ], key=lambda x: int(x.name.split('_')[1]))
    return frame_files


def extract_frame_number(frame_file: Path) -> int:
    """Extract frame number from filename like 'frame_80_endo.png' -> 80."""
    return int(frame_file.name.split('_')[1])


def decode_annotation(annotation_vector: List, categories: Dict) -> Dict:
    """
    Decode an annotation vector into human-readable format.
    
    Annotation vector format: [triplet_id, triplet_sc, triplet_bx, triplet_by, triplet_bw, triplet_bh,
                                instrument_id, instrument_sc, verb_id, verb_sc, target_id, target_sc,
                                phase_id, phase_sc, phase_bx]
    """
    if len(annotation_vector) != 15:
        return None
        
    triplet_id, triplet_sc, triplet_bx, triplet_by, triplet_bw, triplet_bh = annotation_vector[:6]
    instrument_id, instrument_sc, verb_id, verb_sc, target_id, target_sc = annotation_vector[6:12]
    phase_id, phase_sc, phase_bx = annotation_vector[12:15]
    
    # Convert IDs to labels
    decoded = {
        "triplet": {
            "id": triplet_id,
            "label": categories["triplet"].get(str(triplet_id), "unknown") if triplet_id != -1 else None,
            "confidence": triplet_sc if triplet_sc != -1 else None,
            "bbox": {
                "x": triplet_bx,
                "y": triplet_by,
                "width": triplet_bw,
                "height": triplet_bh
            } if triplet_bx != -1 else None
        },
        "instrument": {
            "id": instrument_id,
            "label": categories["instrument"].get(str(instrument_id), "unknown") if instrument_id != -1 else None,
            "confidence": instrument_sc if instrument_sc != -1 else None
        },
        "verb": {
            "id": verb_id,
            "label": categories["verb"].get(str(verb_id), "unknown") if verb_id != -1 else None,
            "confidence": verb_sc if verb_sc != -1 else None
        },
        "target": {
            "id": target_id,
            "label": categories["target"].get(str(target_id), "unknown") if target_id != -1 else None,
            "confidence": target_sc if target_sc != -1 else None
        },
        "phase": {
            "id": phase_id,
            "label": categories["phase"].get(str(phase_id), "unknown") if phase_id != -1 else None,
            "confidence": phase_sc if phase_sc != -1 else None
        }
    }
    
    return decoded


def extract_clip_annotations(clip_dir: Path, video_num: int, t50_data: Dict) -> Dict:
    """
    Extract and decode cholect50 annotations for a clip.
    
    Returns a dictionary mapping frame numbers (in the clip) to annotations.
    """
    frame_files = get_frame_files(clip_dir)
    
    if not frame_files:
        return {}
    
    # Get frame range in the original video
    start_frame = extract_frame_number(frame_files[0])
    end_frame = extract_frame_number(frame_files[-1])
    
    # Map to t50 frame range
    start_t50 = map_frame_to_t50(start_frame)
    end_t50 = map_frame_to_t50(end_frame)
    
    categories = t50_data["categories"]
    annotations_raw = t50_data["annotations"]
    
    clip_annotations = {
        "video_number": video_num,
        "video_id": video_num_to_vid_str(video_num),
        "clip_name": clip_dir.name,
        "original_fps": FULL_FPS,
        "cholect50_fps": T50_FPS,
        "frame_range_25fps": {
            "start": start_frame,
            "end": end_frame,
            "count": len(frame_files)
        },
        "frame_range_1fps": {
            "start": start_t50,
            "end": end_t50
        },
        "frames": {}
    }
    
    # Track all unique triplets in the clip
    unique_triplets = set()
    
    # For each frame in the clip, extract annotations
    for frame_file in frame_files:
        frame_num = extract_frame_number(frame_file)
        t50_frame = map_frame_to_t50(frame_num)
        
        # Get raw annotations from cholect50
        raw_annotations = annotations_raw.get(str(t50_frame), [])
        
        # Decode annotations
        decoded_annotations = []
        for ann_vector in raw_annotations:
            decoded = decode_annotation(ann_vector, categories)
            if decoded:
                decoded_annotations.append(decoded)
                # Track triplet if it's not null
                if decoded["triplet"]["label"] and decoded["triplet"]["label"] != "unknown":
                    unique_triplets.add(decoded["triplet"]["label"])
        
        clip_annotations["frames"][frame_num] = {
            "frame_number_25fps": frame_num,
            "corresponding_t50_frame_1fps": t50_frame,
            "annotations": decoded_annotations
        }
    
    # Add summary of all unique triplets present in the clip
    clip_annotations["unique_triplets_in_clip"] = sorted(list(unique_triplets))
    
    # Add category mappings for reference
    clip_annotations["category_mappings"] = categories
    
    return clip_annotations


def create_video_from_frames(clip_dir: Path, output_path: Path):
    """Create an MP4 video from frame images using ffmpeg."""
    frame_files = get_frame_files(clip_dir)
    
    if not frame_files:
        print(f"  No frames found in {clip_dir}")
        return False
    
    # Create a temporary file list for ffmpeg
    temp_list = clip_dir / "frame_list.txt"
    
    with open(temp_list, 'w') as f:
        for frame_file in frame_files:
            # ffmpeg concat demuxer format
            f.write(f"file '{frame_file.absolute()}'\n")
            f.write(f"duration {1.0/FULL_FPS}\n")
        # Repeat last frame to ensure it's shown
        f.write(f"file '{frame_files[-1].absolute()}'\n")
    
    # Run ffmpeg
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(temp_list),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-r', str(FULL_FPS),
        '-y',  # Overwrite output
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        temp_list.unlink()  # Clean up temp file
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error creating video: {e}")
        print(f"  stderr: {e.stderr.decode()}")
        if temp_list.exists():
            temp_list.unlink()
        return False


def process_clip(clip_dir: Path, video_num: int, t50_data: Dict):
    """Process a single clip: create video and extract annotations."""
    clip_name = clip_dir.name
    print(f"  Processing {clip_name}...")
    
    # Create output directory
    video_str = video_num_to_video_str(video_num)
    output_clip_dir = OUTPUT_DIR / video_str / clip_name
    output_clip_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video
    video_path = output_clip_dir / f"{clip_name}.mp4"
    print(f"    Creating video...")
    success = create_video_from_frames(clip_dir, video_path)
    
    if not success:
        print(f"    Failed to create video")
        return
    
    print(f"    Video created: {video_path}")
    
    # Extract annotations
    print(f"    Extracting annotations...")
    annotations = extract_clip_annotations(clip_dir, video_num, t50_data)
    
    # Save annotations
    annotations_path = output_clip_dir / f"{clip_name}_annotations.json"
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"    Annotations saved: {annotations_path}")


def main():
    """Main processing loop."""
    print("=" * 80)
    print("Generating Cholecseg8k dataset with CholecT50 annotations")
    print("=" * 80)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_clips = 0
    
    # Process each common video
    for video_num in COMMON_VIDEOS:
        video_str = video_num_to_video_str(video_num)
        print(f"Processing {video_str}...")
        
        # Load cholect50 annotations for this video
        try:
            t50_data = load_cholect50_annotations(video_num)
        except FileNotFoundError:
            print(f"  Warning: No cholect50 annotations found for video {video_num}")
            continue
        
        # Get all clips for this video
        video_dir = CHOLECSEG8K_DIR / video_str
        if not video_dir.exists():
            print(f"  Warning: {video_dir} not found")
            continue
        
        clips = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        print(f"  Found {len(clips)} clips")
        
        # Process each clip
        for clip_dir in clips:
            process_clip(clip_dir, video_num, t50_data)
            total_clips += 1
        
        print()
    
    print("=" * 80)
    print(f"Processing complete! Total clips processed: {total_clips}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

