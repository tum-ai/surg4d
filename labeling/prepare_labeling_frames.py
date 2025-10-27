#!/usr/bin/env python3
"""
Script to prepare frames for labeling from the cholecseg8k dataset.
Copies frames from clips listed in final_dataset_clips.json with a stride of 4.
Also creates full framerate MP4 videos for each clip.
"""

import json
import shutil
from pathlib import Path
import subprocess


def main():
    # Configuration (paths relative to repository root)
    repo_root = Path(__file__).parent.parent
    dataset_root = repo_root / "data/cholecseg8k"
    labeling_root = repo_root / "labeling/clips"
    clips_file = repo_root / "final_dataset_clips.json"
    frame_stride = 4
    frames_per_clip = 80
    
    # Load clips to process
    with open(clips_file, 'r') as f:
        clips_data = json.load(f)
    
    # Process each category
    for category, clips in clips_data.items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        # Create output directory for this category
        output_dir = labeling_root / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each clip
        for clip_name in clips:
            print(f"\nProcessing clip: {clip_name}")
            
            # Parse clip name (e.g., "video09_00912" -> video="video09", start_frame=912)
            video_name, start_frame_str = clip_name.split('_')
            start_frame = int(start_frame_str)
            
            # Source directory
            source_dir = dataset_root / video_name / clip_name
            
            if not source_dir.exists():
                print(f"  WARNING: Source directory not found: {source_dir}")
                continue
            
            # Create clip directory in output
            clip_output_dir = output_dir / clip_name
            clip_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy frames with stride (RGB images only)
            frames_copied = 0
            for i in range(0, frames_per_clip, frame_stride):
                frame_num = start_frame + i
                
                # Copy only RGB image
                source_file = source_dir / f"frame_{frame_num}_endo.png"
                dest_file = clip_output_dir / f"frame_{frame_num}_endo.png"
                
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                    frames_copied += 1
                else:
                    print(f"  WARNING: File not found: {source_file}")
            
            print(f"  Copied {frames_copied} frames (stride={frame_stride}) from {clip_name}")
            
            # Create full framerate video from all 80 frames using ffmpeg
            video_path = clip_output_dir / f"{clip_name}.mp4"
            print(f"  Creating video: {video_path.name}")
            
            # Create a temporary file list for ffmpeg
            file_list_path = clip_output_dir / "frame_list.txt"
            with open(file_list_path, 'w') as f:
                for i in range(frames_per_clip):
                    frame_num = start_frame + i
                    frame_path = source_dir / f"frame_{frame_num}_endo.png"
                    if frame_path.exists():
                        f.write(f"file '{frame_path.absolute()}'\n")
                        f.write(f"duration 0.04\n")  # 25 fps = 0.04 seconds per frame
            
            # Use ffmpeg to create video from frame list
            try:
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output file
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(file_list_path),
                    '-vsync', 'vfr',
                    '-pix_fmt', 'yuv420p',
                    '-c:v', 'libx264',
                    '-crf', '23',
                    str(video_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✓ Video created with {frames_per_clip} frames")
                else:
                    print(f"  ERROR creating video: {result.stderr}")
            except Exception as e:
                print(f"  ERROR: Failed to create video: {e}")
            finally:
                # Clean up temporary file list
                if file_list_path.exists():
                    file_list_path.unlink()
    
    print(f"\n{'='*60}")
    print("Frame preparation complete!")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for category in clips_data.keys():
        category_dir = labeling_root / category
        num_clips = len(list(category_dir.glob("*"))) if category_dir.exists() else 0
        print(f"  {category}: {num_clips} clips")


if __name__ == "__main__":
    main()

