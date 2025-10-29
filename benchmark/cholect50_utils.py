"""
Utilities for loading and processing CholecT50 annotations
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CholecT50Loader:
    """Load and process CholecT50 action triplet annotations"""
    
    def __init__(self, cholect50_path: str = "/home/students/lmu_proj/shared_data/data/CholecT50"):
        self.root = Path(cholect50_path)
        self.labels_dir = self.root / "labels"
        self.videos_dir = self.root / "videos"
        
    def load_video_annotations(self, video_id: int) -> Dict:
        """Load annotations for a specific video
        
        Args:
            video_id: Video number (e.g., 1 for VID01)
            
        Returns:
            Dictionary with annotations and metadata
        """
        label_file = self.labels_dir / f"VID{video_id:02d}.json"
        print(f"label_file: {label_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"Labels not found for video {video_id}")
            
        with open(label_file) as f:
            data = json.load(f)
        return data
    
    def get_frame_triplets(self, video_data: Dict, frame_num: int) -> List[Dict]:
        """Extract triplets for a specific frame
        
        Args:
            video_data: Output from load_video_annotations
            frame_num: Frame number
            
        Returns:
            List of triplet dictionaries with keys: instrument, verb, target, phase
        """
        annotations = video_data['annotations']
        categories = video_data['categories']
        
        frame_key = str(frame_num)
        if frame_key not in annotations:
            print(f"⚠️ No triplets found for frame {frame_num}")
            return []
        
        triplets = []
        for triplet_data in annotations[frame_key]:
            print(f"triplet_data: {triplet_data}")
            # Format: [triplet_id, i, v, ...padding..., t, ...padding..., phase]
            # Based on the observed pattern
            triplet_id = triplet_data[0]
            instrument_id = triplet_data[1]
            verb_id = triplet_data[7]
            if verb_id == -1:
                verb_id = 9
            target_id = triplet_data[8]
            if target_id == -1:
                target_id = 14
            phase_id = triplet_data[14]
                            
            triplet = {
                'triplet_id': int(triplet_id),
                # 'triplet_name': categories['triplet'][str(int(triplet_id))],
                'instrument': categories['instrument'][str(int(instrument_id))],
                'verb': categories['verb'][str(int(verb_id))],
                'target': categories['target'][str(int(target_id))],
                'phase': categories['phase'][str(int(phase_id))],
                'instrument_id': int(instrument_id),
                'verb_id': int(verb_id),
                'target_id': int(target_id),
                'phase_id': int(phase_id),
            }
            triplets.append(triplet)
        
        return triplets
    
    def get_video_frame_range(self, video_id: int) -> Tuple[int, int]:
        """Get the frame range for a video
        
        Returns:
            Tuple of (start_frame, end_frame)
        """
        data = self.load_video_annotations(video_id)
        frame_nums = [int(k) for k in data['annotations'].keys()]
        return min(frame_nums), max(frame_nums)
    
    def find_frames_with_triplet(self, video_id: int, 
                                  instrument: Optional[str] = None,
                                  verb: Optional[str] = None,
                                  target: Optional[str] = None) -> List[int]:
        """Find frames containing specific triplet components
        
        Args:
            video_id: Video number
            instrument: Instrument name to filter (e.g., "grasper")
            verb: Verb name to filter (e.g., "grasp")
            target: Target name to filter (e.g., "gallbladder")
            
        Returns:
            List of frame numbers matching the criteria
        """
        data = self.load_video_annotations(video_id)
        matching_frames = []
        
        for frame_num in data['annotations'].keys():
            triplets = self.get_frame_triplets(data, int(frame_num))
            
            for triplet in triplets:
                match = True
                if instrument and triplet['instrument'] != instrument:
                    match = False
                if verb and triplet['verb'] != verb:
                    match = False
                if target and triplet['target'] != target:
                    match = False
                
                if match:
                    matching_frames.append(int(frame_num))
                    break  # Found a match in this frame
        
        return sorted(matching_frames)
    
    def group_frames_by_triplet_config(self, video_id: int) -> Dict[str, List[int]]:
        """Group frames by their triplet configuration
        
        Returns a dictionary where:
        - Key: sorted tuple of triplet names (e.g., "grasper,grasp,gallbladder")
        - Value: list of frame numbers with exactly that set of triplets
        """
        data = self.load_video_annotations(video_id)
        triplet_groups = {}
        
        for frame_num in data['annotations'].keys():
            triplets = self.get_frame_triplets(data, int(frame_num))
            
            if not triplets:
                continue
            
            # Create a sorted tuple of triplet names as the key
            triplet_key = tuple(sorted([t['triplet_name'] for t in triplets]))
            
            if triplet_key not in triplet_groups:
                triplet_groups[triplet_key] = []
            
            triplet_groups[triplet_key].append(int(frame_num))
        
        # Sort frame lists
        for key in triplet_groups:
            triplet_groups[key].sort()
        
        return triplet_groups
    
    def find_continuous_triplet_sequences(self, video_id: int, 
                                         min_sequence_length: int = 5) -> List[Dict]:
        """Find continuous sequences of frames with the same triplet configuration
        
        Args:
            video_id: Video number
            min_sequence_length: Minimum number of consecutive frames required
            
        Returns:
            List of dictionaries containing:
            - triplets: tuple of triplet names
            - start_frame: first frame in sequence
            - end_frame: last frame in sequence  
            - length: number of frames
        """
        triplet_groups = self.group_frames_by_triplet_config(video_id)
        sequences = []
        
        for triplet_key, frames in triplet_groups.items():
            # Find continuous sequences in the frame list
            if not frames:
                continue
                
            seq_start = frames[0]
            prev_frame = frames[0]
            
            for frame in frames[1:] + [None]:  # Add None to trigger final sequence save
                if frame is None or frame != prev_frame + 1:
                    # Sequence ended
                    seq_length = prev_frame - seq_start + 1
                    if seq_length >= min_sequence_length:
                        sequences.append({
                            'video_id': video_id,
                            'triplets': triplet_key,
                            'start_frame': seq_start,
                            'end_frame': prev_frame,
                            'length': seq_length,
                            'phase': None,  # Will be filled later if needed
                        })
                    
                    if frame is not None:
                        seq_start = frame
                
                prev_frame = frame
        
        # Sort by length (longest first)
        sequences.sort(key=lambda x: x['length'], reverse=True)
        return sequences


def video_has_precomputed_graph(video_id: int, clip_start: int) -> bool:
    """Check if a video clip has a pre-computed scene graph
    
    Args:
        video_id: Video number
        clip_start: Start frame of the clip
        
    Returns:
        True if graph exists
    """
    output_dir = Path("/home/students/lmu_proj/shared_data/output/cholecseg8k")
    pattern = f"video{video_id:02d}_{clip_start:05d}_*"
    
    matching = list(output_dir.glob(pattern))
    for path in matching:
        if (path / "graph").exists():
            return True
    return False


def get_precomputed_graphs() -> List[Tuple[int, int, Path]]:
    """Get all pre-computed graphs
    
    Returns:
        List of tuples: (video_id, clip_start, graph_path)
    """
    output_dir = Path("/home/students/lmu_proj/shared_data/output/cholecseg8k")
    graphs = []
    
    for graph_dir in output_dir.glob("*/graph"):
        parent = graph_dir.parent.name
        # Parse video01_00080_qwen_cat format
        parts = parent.split('_')
        if len(parts) >= 2:
            try:
                video_id = int(parts[0].replace('video', ''))
                clip_start = int(parts[1])
                graphs.append((video_id, clip_start, graph_dir))
            except ValueError:
                continue
    
    return sorted(graphs)


if __name__ == "__main__":
    # Test the loader
    loader = CholecT50Loader()
    
    print("Testing CholecT50 loader...")
    
    print("\nPre-computed graphs:")
    for vid, clip, path in get_precomputed_graphs():
        print(f"  Video {vid:02d}, clip starting at frame {clip}: {path}")
    
    print("\nSample annotations from VID01, frame 100:")
    data = loader.load_video_annotations(1)
    triplets = loader.get_frame_triplets(data, 100)
    for t in triplets:
        print(f"  - {t['instrument']} {t['verb']} {t['target']} (phase: {t['phase']})")
    
    print("\nFrames with 'grasper grasp gallbladder' in VID01:")
    frames = loader.find_frames_with_triplet(1, instrument="grasper", verb="grasp", target="gallbladder")
    print(f"  Found {len(frames)} frames: {frames[:10]}...")
    
    print("\nGrouping frames by triplet configuration in VID01:")
    triplet_groups = loader.group_frames_by_triplet_config(1)
    print(f"  Found {len(triplet_groups)} unique triplet configurations")
    for i, (triplets, frames) in enumerate(list(triplet_groups.items())[:5]):
        print(f"  Config {i+1}: {len(frames)} frames with {triplets}")
    
    print("\nFinding continuous sequences in VID01:")
    sequences = loader.find_continuous_triplet_sequences(1, min_sequence_length=10)
    print(f"  Found {len(sequences)} continuous sequences (min length 10)")
    for seq in sequences[:5]:
        print(f"  {seq['length']} frames: {seq['start_frame']}-{seq['end_frame']}, triplets: {seq['triplets']}")

