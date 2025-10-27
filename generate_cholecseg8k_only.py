"""
Generate videos for cholecseg8k clips that are NOT in cholect50.
These clips don't have cholect50 annotations.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

# Configuration
CHOLECSEG8K_DIR = Path("/home/tumai/nico/surgery-scene-graphs/data/cholecseg8k")
OUTPUT_DIR = Path("/home/tumai/nico/surgery-scene-graphs/data/cholecseg8k_only")

# Videos in cholecseg8k but NOT in cholect50
CHOLECSEG8K_ONLY_VIDEOS = [9, 17, 20, 24, 28, 37, 55]

# Frame rate info
FULL_FPS = 25


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


def create_clip_metadata(clip_dir: Path, video_num: int) -> dict:
    """Create basic metadata for a clip (no annotations)."""
    frame_files = get_frame_files(clip_dir)
    
    if not frame_files:
        return {}
    
    # Get frame range in the original video
    start_frame = extract_frame_number(frame_files[0])
    end_frame = extract_frame_number(frame_files[-1])
    
    metadata = {
        "video_number": video_num,
        "video_id": video_num_to_video_str(video_num),
        "clip_name": clip_dir.name,
        "fps": FULL_FPS,
        "frame_range": {
            "start": start_frame,
            "end": end_frame,
            "count": len(frame_files)
        },
        "note": "This video is from cholecseg8k only and does not have cholect50 annotations"
    }
    
    return metadata


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


def process_clip(clip_dir: Path, video_num: int):
    """Process a single clip: create video and metadata."""
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
    
    # Create metadata
    print(f"    Creating metadata...")
    metadata = create_clip_metadata(clip_dir, video_num)
    
    # Save metadata
    metadata_path = output_clip_dir / f"{clip_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    Metadata saved: {metadata_path}")


def main():
    """Main processing loop."""
    print("=" * 80)
    print("Generating Cholecseg8k-only dataset (videos not in CholecT50)")
    print("=" * 80)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_clips = 0
    
    # Process each cholecseg8k-only video
    for video_num in CHOLECSEG8K_ONLY_VIDEOS:
        video_str = video_num_to_video_str(video_num)
        print(f"Processing {video_str}...")
        
        # Get all clips for this video
        video_dir = CHOLECSEG8K_DIR / video_str
        if not video_dir.exists():
            print(f"  Warning: {video_dir} not found")
            continue
        
        clips = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        print(f"  Found {len(clips)} clips")
        
        # Process each clip
        for clip_dir in clips:
            process_clip(clip_dir, video_num)
            total_clips += 1
        
        print()
    
    print("=" * 80)
    print(f"Processing complete! Total clips processed: {total_clips}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()




