#!/usr/bin/env python3
"""
Sample selector for multi-frame evaluation.

Selects continuous sequences of frames with consistent triplet configurations.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.cholect50_utils import CholecT50Loader

@dataclass
class MultiFrameSample:
    """Sample with multiple frames for temporal evaluation"""
    video_id: int
    start_frame: int
    end_frame: int
    clip_start: int
    image_paths: List[Path]  # Multiple frames
    graph_path: Optional[Path]
    # TODO: For now, loading these also for non-triplet tasks; might change this
    gt_triplets: List[Dict]  # Ground truth for the sequence
    gt_phase: Optional[str]
    
    @property
    def sample_id(self) -> str:
        abs_start = int(self.clip_start + self.start_frame)
        abs_end = int(self.clip_start + self.end_frame)
        return f"v{self.video_id:02d}_c{int(self.clip_start):05d}_f{abs_start:05d}-{abs_end:05d}"
    
    @property
    def num_frames(self) -> int:
        return len(self.image_paths)


class TripletsFrameSelector:
    """Select multi-frame samples for evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        video_id, clip_start = self._find_video_data(config.video_dir)
        self.available_graph = (video_id, clip_start, config.graph_dir)

    def _find_video_data(self, video_dir):
        video_id = int(video_dir.name.split("_")[0].replace("video", ""))
        print(f"video_id: {video_id}")
        clip_name = video_dir.name
        print(f"clip_name: {clip_name}")
        clip_start = int(clip_name.split("_")[1])
        return video_id, clip_start
            
    def select_sequences(self) -> List[MultiFrameSample]:
        """
        Select multi-frame sequences for evaluation.
        
        Args:
            num_sequences: Number of sequences to select
            frames_per_sequence: Number of frames in each sequence
            min_config_length: Minimum length of consistent triplet configuration
        """
        video_id, clip_start, graph_path = self.available_graph

        # Frame indexing and mapping
        FRAMERATE = self.config.triplets_config['FRAMERATE']  # 25 fps
        NUM_FRAMES = self.config.triplets_config['NUM_FRAMES']  # 80
        TEMP_CONTEXT_FR_MULTIPLIER = self.config.triplets_config['TEMP_CONTEXT_FR_MULTIPLIER']
        FRAME_STRIDE = int(self.config.triplets_config.get('frame_stride', 4))
        ALIGN_TO_GRAPH = bool(self.config.triplets_config.get('align_to_graph', True))
        
        # Need the video id here to go with that into cholect50; does not correspond to the same frames, different fps!
        video_data = self.loader.load_video_annotations(video_id)
        print(f"Loaded video data for video {video_id}")
            
        # Find image paths
        clip_dir = self.config.video_dir
            
        # Get image paths for selected frames
        # Use configured images subdirectory
        images_dir = clip_dir / self.config.images_subdir
        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            print(f"Expected structure: {clip_dir}/{self.config.images_subdir}/")
            return []
        
        image_paths = list(sorted(images_dir.glob("*.jpg")))
        if not image_paths:
            print(f"ERROR: No .jpg images found in {images_dir}")
            return []
        
        print(f"Found {len(image_paths)} images in {images_dir}")
        print(f"first 10 image_paths: {image_paths[:10]}")

        samples = []
        # Absolute frame range of this clip in the original video
        clip_abs_start = clip_start
        clip_abs_end = clip_start + NUM_FRAMES - 1

        # CholecT50 annotations exist at every 25th frame. Compute the second indices
        # that fall into this clip, then map each annotation to the nearest stride frame.
        # Convert absolute frame range to second indices (floor/ceil bounds)
        from math import floor, ceil
        s_start = ceil(clip_abs_start / FRAMERATE)
        s_end = floor(clip_abs_end / FRAMERATE)

        for second_idx in range(s_start, s_end + 1):
            annotation_abs_frame = second_idx * FRAMERATE
            if not (clip_abs_start <= annotation_abs_frame <= clip_abs_end):
                continue

            # Fetch GT triplets for this annotated frame index in CholecT50
            triplets = self.loader.get_frame_triplets(video_data, int(second_idx))
            print(f"triplets for abs_frame {annotation_abs_frame} (sec {second_idx}): {triplets}")

            # Map to clip-relative end_frame
            end_rel = annotation_abs_frame - clip_abs_start  # 0..79
            if ALIGN_TO_GRAPH and FRAME_STRIDE > 1:
                # Snap to nearest stride-aligned frame (e.g., every 4th frame)
                end_rel = int(round(end_rel / FRAME_STRIDE) * FRAME_STRIDE)
                end_rel = max(0, min(NUM_FRAMES - 1, end_rel))

            end_frame = end_rel
            start_frame = end_frame - TEMP_CONTEXT_FR_MULTIPLIER * FRAMERATE
            if start_frame < 0:
                start_frame = 0

            samples.append(MultiFrameSample(
                video_id=video_id,
                start_frame=start_frame,
                end_frame=end_frame,
                clip_start=clip_start,
                image_paths=image_paths,
                graph_path=graph_path,
                gt_triplets=triplets,
                gt_phase=triplets[0]['phase'] if triplets else None
            ))
        
        return samples
    
    def print_summary(self, samples: List[MultiFrameSample]):
        """Print summary of selected samples"""
        print("\n" + "="*80)
        print("SELECTED MULTI-FRAME SAMPLES")
        print("="*80)
        print(f"\nTotal samples: {len(samples)}")
        print(f"Frames per sample: {samples[0].num_frames if samples else 0}")
        print()
        
        # for i, sample in enumerate(samples, 1):
        #     print(f"{i}. {sample.sample_id}")
        #     print(f"   Video: {sample.video_id}, Frames: {sample.start_frame}-{sample.end_frame}")
        #     print(f"   Triplets: {[t['triplet_name'] for t in sample.gt_triplets]}")
        #     print(f"   Graph: {'Yes' if sample.graph_path else 'No'}")
        
        print("="*80)

