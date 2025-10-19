"""
Sample selection for benchmark evaluation
"""
import random
from pathlib import Path
from typing import List

from benchmark_config import TestSample, BenchmarkConfig
from cholect50_utils import CholecT50Loader, get_precomputed_graphs


class SampleSelector:
    """Select test samples for evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(str(config.cholect50_root))
        self.available_graphs = get_precomputed_graphs()
        random.seed(config.seed)
        
    def select_samples(self, num_samples: int, strategy: str = "diverse") -> List[TestSample]:
        """Select test samples
        
        Args:
            num_samples: Number of samples to select
            strategy: Selection strategy
                - "diverse": Maximize diversity of triplets
                - "random": Random selection
                - "concentrated": From single video clip
                - "by_triplet_config": Frames with same triplet configuration(s)
                - "continuous_sequence": Continuous frames with same triplet config
                
        Returns:
            List of TestSample objects
        """
        if strategy == "diverse":
            return self._select_diverse(num_samples)
        elif strategy == "random":
            return self._select_random(num_samples)
        elif strategy == "concentrated":
            return self._select_concentrated(num_samples)
        elif strategy == "by_triplet_config":
            return self._select_by_triplet_config(num_samples)
        elif strategy == "continuous_sequence":
            return self._select_continuous_sequence(num_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_diverse(self, num_samples: int) -> List[TestSample]:
        """Select samples with diverse triplets"""
        samples = []
        triplets_seen = set()
        
        # Iterate through available graphs
        for video_id, clip_start, graph_path in self.available_graphs:
            if len(samples) >= num_samples:
                break
            
            # Load annotations for this video
            try:
                video_data = self.loader.load_video_annotations(video_id)
            except FileNotFoundError:
                continue
            
            # Get preprocessed data path
            video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
            clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
            if not clip_dirs:
                continue
            clip_dir = clip_dirs[0]
            image_dir = clip_dir  # Images are directly in clip directory
            
            # Sample frames from this clip (80 frames per clip typically)
            frames_to_check = list(range(clip_start, clip_start + 80, 5))  # Every 5th frame
            random.shuffle(frames_to_check)
            
            for frame_num in frames_to_check:
                if len(samples) >= num_samples:
                    break
                
                # Get triplets for this frame
                triplets = self.loader.get_frame_triplets(video_data, frame_num)
                if not triplets:
                    continue
                
                # Check if we've seen these triplets
                triplet_ids = tuple(sorted([t['triplet_id'] for t in triplets]))
                if triplet_ids in triplets_seen:
                    continue  # Skip, we want diversity
                
                # Check if image exists
                image_path = self._find_image_path(image_dir, frame_num, clip_start)
                if not image_path or not image_path.exists():
                    continue
                
                # Create sample
                sample = TestSample(
                    video_id=video_id,
                    frame_num=frame_num,
                    clip_start=clip_start,
                    image_path=image_path,
                    graph_path=graph_path if self.config.use_scene_graph else None,
                    gt_triplets=triplets,
                    gt_phase=triplets[0]['phase'] if triplets else None,
                )
                
                samples.append(sample)
                triplets_seen.add(triplet_ids)
                
                if self.config.verbose:
                    print(f"  Selected {sample.sample_id}: {[t['triplet_name'] for t in triplets]}")
        
        return samples[:num_samples]
    
    def _select_random(self, num_samples: int) -> List[TestSample]:
        """Randomly select samples"""
        all_candidates = []
        
        for video_id, clip_start, graph_path in self.available_graphs:
            try:
                video_data = self.loader.load_video_annotations(video_id)
            except FileNotFoundError:
                continue
            
            video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
            clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
            if not clip_dirs:
                continue
            clip_dir = clip_dirs[0]
            image_dir = clip_dir / "images"
            
            # All frames in this clip
            for frame_num in range(clip_start, clip_start + 80):
                triplets = self.loader.get_frame_triplets(video_data, frame_num)
                if not triplets:
                    continue
                
                image_path = self._find_image_path(image_dir, frame_num, clip_start)
                if not image_path or not image_path.exists():
                    continue
                
                all_candidates.append((video_id, frame_num, clip_start, image_path, 
                                      graph_path, triplets))
        
        # Random sample
        if len(all_candidates) > num_samples:
            selected = random.sample(all_candidates, num_samples)
        else:
            selected = all_candidates
        
        samples = []
        for video_id, frame_num, clip_start, image_path, graph_path, triplets in selected:
            sample = TestSample(
                video_id=video_id,
                frame_num=frame_num,
                clip_start=clip_start,
                image_path=image_path,
                graph_path=graph_path if self.config.use_scene_graph else None,
                gt_triplets=triplets,
                gt_phase=triplets[0]['phase'] if triplets else None,
            )
            samples.append(sample)
        
        return samples
    
    def _select_concentrated(self, num_samples: int) -> List[TestSample]:
        """Select all samples from a single video clip"""
        # Use the first available graph
        if not self.available_graphs:
            return []
        
        video_id, clip_start, graph_path = self.available_graphs[0]
        
        samples = []
        try:
            video_data = self.loader.load_video_annotations(video_id)
        except FileNotFoundError:
            return []
        
        video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
        clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
        if not clip_dirs:
            return []
        clip_dir = clip_dirs[0]
        image_dir = clip_dir / "images"
        
        # Sample evenly spaced frames
        total_frames = 80
        step = max(1, total_frames // num_samples)
        
        for i in range(0, total_frames, step):
            if len(samples) >= num_samples:
                break
            
            frame_num = clip_start + i
            triplets = self.loader.get_frame_triplets(video_data, frame_num)
            if not triplets:
                continue
            
            image_path = self._find_image_path(image_dir, frame_num, clip_start)
            if not image_path or not image_path.exists():
                continue
            
            sample = TestSample(
                video_id=video_id,
                frame_num=frame_num,
                clip_start=clip_start,
                image_path=image_path,
                graph_path=graph_path if self.config.use_scene_graph else None,
                gt_triplets=triplets,
                gt_phase=triplets[0]['phase'] if triplets else None,
            )
            samples.append(sample)
        
        return samples
    
    def _select_by_triplet_config(self, num_samples: int) -> List[TestSample]:
        """Select samples with the same triplet configuration
        
        This strategy groups frames by their exact triplet configuration and selects
        multiple frames that share the same set of actions.
        """
        samples = []
        
        # Try each available video
        for video_id, clip_start, graph_path in self.available_graphs:
            if len(samples) >= num_samples:
                break
                
            try:
                video_data = self.loader.load_video_annotations(video_id)
            except FileNotFoundError:
                continue
            
            # Group frames by triplet configuration
            triplet_groups = self.loader.group_frames_by_triplet_config(video_id)
            
            # Find the largest group (most common configuration)
            if not triplet_groups:
                continue
            
            # Sort by group size
            sorted_groups = sorted(triplet_groups.items(), 
                                  key=lambda x: len(x[1]), reverse=True)
            
            for triplet_config, frame_list in sorted_groups:
                if len(samples) >= num_samples:
                    break
                
                # Sample evenly from this configuration
                step = max(1, len(frame_list) // num_samples)
                sampled_frames = frame_list[::step][:num_samples - len(samples)]
                
                video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
                clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
                if not clip_dirs:
                    continue
                clip_dir = clip_dirs[0]
                image_dir = clip_dir
                
                for frame_num in sampled_frames:
                    triplets = self.loader.get_frame_triplets(video_data, frame_num)
                    if not triplets:
                        continue
                    
                    image_path = self._find_image_path(image_dir, frame_num, clip_start)
                    if not image_path or not image_path.exists():
                        continue
                    
                    sample = TestSample(
                        video_id=video_id,
                        frame_num=frame_num,
                        clip_start=clip_start,
                        image_path=image_path,
                        graph_path=graph_path if self.config.use_scene_graph else None,
                        gt_triplets=triplets,
                        gt_phase=triplets[0]['phase'] if triplets else None,
                    )
                    samples.append(sample)
                    
                    if self.config.verbose:
                        print(f"  Selected {sample.sample_id}: {[t['triplet_name'] for t in triplets]}")
                    
                    if len(samples) >= num_samples:
                        break
                
                # If we got enough samples from this config, stop
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def _select_continuous_sequence(self, num_samples: int, min_length: int = 10) -> List[TestSample]:
        """Select a continuous sequence of frames with same triplet configuration
        
        This finds the longest continuous sequence where the same actions are performed.
        """
        samples = []
        
        # Try each available video
        for video_id, clip_start, graph_path in self.available_graphs:
            if samples:  # Already found a sequence
                break
                
            try:
                video_data = self.loader.load_video_annotations(video_id)
            except FileNotFoundError:
                continue
            
            # Find continuous sequences
            sequences = self.loader.find_continuous_triplet_sequences(
                video_id, min_sequence_length=min_length
            )
            
            if not sequences:
                continue
            
            # Use the longest sequence
            best_seq = sequences[0]
            
            # Sample frames evenly from this sequence
            seq_frames = list(range(best_seq['start_frame'], best_seq['end_frame'] + 1))
            step = max(1, len(seq_frames) // num_samples)
            sampled_frames = seq_frames[::step][:num_samples]
            
            video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
            clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}"))
            if not clip_dirs:
                continue
            clip_dir = clip_dirs[0]
            image_dir = clip_dir
            
            for frame_num in sampled_frames:
                triplets = self.loader.get_frame_triplets(video_data, frame_num)
                if not triplets:
                    continue
                
                image_path = self._find_image_path(image_dir, frame_num, clip_start)
                if not image_path or not image_path.exists():
                    continue
                
                sample = TestSample(
                    video_id=video_id,
                    frame_num=frame_num,
                    clip_start=clip_start,
                    image_path=image_path,
                    graph_path=graph_path if self.config.use_scene_graph else None,
                    gt_triplets=triplets,
                    gt_phase=triplets[0]['phase'] if triplets else None,
                )
                samples.append(sample)
                
                if self.config.verbose:
                    print(f"  Selected {sample.sample_id}: {[t['triplet_name'] for t in triplets]}")
            
            if self.config.verbose:
                print("\nSelected continuous sequence:")
                print(f"  Video: {video_id}")
                print(f"  Frames: {best_seq['start_frame']}-{best_seq['end_frame']} ({best_seq['length']} total)")
                print(f"  Triplets: {best_seq['triplets']}")
                print(f"  Sampled {len(samples)} frames")
            
            break  # Found a good sequence
        
        return samples[:num_samples]
    
    def _find_image_path(self, image_dir: Path, frame_num: int, clip_start: int) -> Path:
        """Find the image file for a given frame"""
        # Frames are named with absolute frame numbers from the video
        # Format: frame_XXX_endo.png
        
        candidates = [
            image_dir / f"frame_{frame_num}_endo.png",
            image_dir / f"frame_{frame_num:03d}_endo.png",
            image_dir / f"frame_{frame_num:06d}_endo.png",
            image_dir / f"frame_{frame_num}.jpg",
            image_dir / f"frame_{frame_num}.png",
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        return None
    
    def print_sample_summary(self, samples: List[TestSample]):
        """Print summary of selected samples"""
        print(f"\n{'='*70}")
        print(f"Selected {len(samples)} test samples")
        print('='*70)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. {sample.sample_id}")
            print(f"   Image: {sample.image_path.name if sample.image_path else 'N/A'}")
            print(f"   Graph: {'Yes' if sample.graph_path else 'No'}")
            print(f"   Phase: {sample.gt_phase}")
            print("   Triplets:")
            for triplet in sample.gt_triplets:
                print(f"     - {triplet['triplet_name']}")
        
        print('='*70 + '\n')

