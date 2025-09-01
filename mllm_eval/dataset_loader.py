"""
Dataset loader that integrates SSG-VQA data structure with scene graph functionality
"""
import os
import glob
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from scene_graph_loader import load_scene_graph_for_frame, SceneGraph, TemporalSceneGraph
from scene_graph_encoder import SceneGraphEncoderFactory, TemporalGraphProcessor
from config_manager import SceneGraphConfig


class SceneGraphVQADataset(Dataset):
    """
    Extended VQA dataset that integrates scene graphs with the SSG-VQA data structure
    """
    
    def __init__(self, 
                 sequences: List[str],
                 data_root: str,
                 scene_graph_config: SceneGraphConfig,
                 question_types: Optional[List[str]] = None,
                 load_images: bool = True):
        
        self.sequences = sequences
        self.data_root = Path(data_root)
        self.scene_graph_config = scene_graph_config
        self.load_images = load_images
        
        # Initialize scene graph components
        self.scene_graph_encoder = SceneGraphEncoderFactory.create_encoder(
            scene_graph_config.encoding_format
        )
        self.temporal_processor = TemporalGraphProcessor(
            window_size=scene_graph_config.temporal_frames
        )
        
        # Load VQA data following SSG-VQA structure
        self.vqa_samples = self._load_vqa_samples(question_types)
        
        # SSG-VQA label mapping
        self.labels = [
            "0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9",
            "False", "True",
            "abdominal_wall_cavity", "adhesion", "anatomy", "aspirate", "bipolar",
            "blood_vessel", "blue", "brown", "clip", "clipper", "coagulate", "cut",
            "cystic_artery", "cystic_duct", "cystic_pedicle", "cystic_plate",
            "dissect", "fluid", "gallbladder", "grasp", "grasper", "gut", "hook",
            "instrument", "irrigate", "irrigator", "liver", "omentum", "pack",
            "peritoneum", "red", "retract", "scissors", "silver", "specimen_bag",
            "specimenbag", "white", "yellow"
        ]
        
        print(f"Loaded {len(self.vqa_samples)} VQA samples from {len(sequences)} sequences")
    
    def _load_vqa_samples(self, question_types: Optional[List[str]]) -> List[Dict]:
        """Load VQA samples following SSG-VQA data structure"""
        samples = []
        
        for sequence in self.sequences:
            # Load QA text files
            qa_pattern = self.data_root / "qa_txt" / sequence / "*.txt"
            qa_files = glob.glob(str(qa_pattern))
            
            for qa_file in qa_files:
                try:
                    with open(qa_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    
                    # Parse QA file (following SSG-VQA format)
                    for idx, line in enumerate(lines):
                        if idx >= 2 and "|" in line:  # Skip header lines
                            parts = line.split("|")
                            if len(parts) >= 2:
                                question = parts[0].strip()
                                answer = parts[1].strip()
                                
                                # Optional: filter by question type
                                if question_types and len(parts) >= 3:
                                    q_type = parts[2].strip() if len(parts) > 2 else ""
                                    if question_types != ["all"] and q_type not in question_types:
                                        continue
                                
                                sample = {
                                    "sequence_id": sequence,
                                    "frame_id": Path(qa_file).stem,
                                    "question": question,
                                    "answer": answer,
                                    "question_type": parts[2].strip() if len(parts) > 2 else "",
                                    "question_category": parts[3].strip() if len(parts) > 3 else "",
                                    "qa_file_path": qa_file
                                }
                                samples.append(sample)
                
                except Exception as e:
                    print(f"Error loading QA file {qa_file}: {e}")
                    continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.vqa_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single VQA sample with integrated scene graph"""
        sample = self.vqa_samples[idx]
        
        # Basic sample info
        result = {
            "sequence_id": sample["sequence_id"],
            "frame_id": sample["frame_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "question_type": sample.get("question_type", ""),
            "question_category": sample.get("question_category", ""),
            "label_idx": self._get_label_index(sample["answer"])
        }
        
        # Load image if requested
        if self.load_images:
            image = self._load_image(sample["sequence_id"], sample["frame_id"])
            result["image"] = image
        
        # Load visual features (maintaining SSG-VQA compatibility)
        visual_features = self._load_visual_features(sample["sequence_id"], sample["frame_id"])
        if visual_features is not None:
            result["visual_features"] = visual_features
        
        # Load and encode scene graph
        scene_graph_data = self._load_scene_graph(sample["sequence_id"], sample["frame_id"])
        if scene_graph_data:
            encoded_graph = self.scene_graph_encoder.encode(scene_graph_data)
            result["scene_graph"] = encoded_graph
            result["scene_graph_raw"] = scene_graph_data
            
            # Add temporal context if available and requested
            if (self.scene_graph_config.include_temporal_info and 
                isinstance(scene_graph_data, TemporalSceneGraph)):
                temporal_context = self.temporal_processor.extract_temporal_context(
                    scene_graph_data, float(sample["frame_id"])
                )
                result["temporal_context"] = temporal_context
        else:
            result["scene_graph"] = None
            result["scene_graph_raw"] = None
        
        return result
    
    def _get_label_index(self, answer: str) -> int:
        """Convert answer to label index for compatibility with SSG-VQA evaluation"""
        try:
            return self.labels.index(str(answer))
        except ValueError:
            print(f"Warning: Answer '{answer}' not found in label set")
            return 0  # Default to first label
    
    def _load_image(self, sequence_id: str, frame_id: str) -> Optional[Image.Image]:
        """Load image for the given sequence and frame"""
        # Try multiple possible image paths
        image_paths = [
            self.data_root / "images" / sequence_id / f"{frame_id}.png",
            self.data_root / "images" / sequence_id / f"{int(frame_id):06d}.png",
            self.data_root / "CholecT50" / sequence_id / f"{int(frame_id):06d}.png",
        ]
        
        for image_path in image_paths:
            if image_path.exists():
                try:
                    return Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
        
        print(f"Warning: Could not find image for {sequence_id}/{frame_id}")
        return None
    
    def _load_visual_features(self, sequence_id: str, frame_id: str) -> Optional[torch.Tensor]:
        """Load pre-computed visual features following SSG-VQA structure"""
        try:
            frame_num = int(frame_id)
            
            # Try pixel-level features
            pix_feature_path = (self.data_root / "visual_feats" / "cropped_images" / 
                              sequence_id / "vqa" / "img_features" / "1x1" / 
                              f"{frame_num:06d}.hdf5")
            
            # Try ROI features
            roi_feature_path = (self.data_root / "visual_feats" / "roi_yolo_coord" / 
                              sequence_id / "labels" / "vqa" / "img_features" / "roi" / 
                              f"{frame_num:06d}.hdf5")
            
            visual_features = None
            
            if roi_feature_path.exists():
                with h5py.File(roi_feature_path, 'r') as f:
                    roi_features = torch.from_numpy(f["visual_features"][:])
                
                # Combine with pixel features if available
                if pix_feature_path.exists():
                    with h5py.File(pix_feature_path, 'r') as f:
                        pix_features = torch.from_numpy(f["visual_features"][:])
                    
                    # Following SSG-VQA combination strategy
                    roi_features[:, 18:] = pix_features
                
                visual_features = roi_features
            
            elif pix_feature_path.exists():
                with h5py.File(pix_feature_path, 'r') as f:
                    visual_features = torch.from_numpy(f["visual_features"][:])
            
            return visual_features
            
        except Exception as e:
            print(f"Error loading visual features for {sequence_id}/{frame_id}: {e}")
            return None
    
    def _load_scene_graph(self, sequence_id: str, frame_id: str):
        """Load scene graph for the given sequence and frame"""
        if not self.scene_graph_config.graph_dir:
            return None
        
        try:
            scene_graph = load_scene_graph_for_frame(
                self.scene_graph_config.graph_dir,
                sequence_id,
                frame_id,
                self.scene_graph_config.source_type
            )
            return scene_graph
            
        except Exception as e:
            print(f"Error loading scene graph for {sequence_id}/{frame_id}: {e}")
            return None
    
    def get_question_type_stats(self) -> Dict[str, int]:
        """Get statistics on question types in the dataset"""
        stats = {}
        for sample in self.vqa_samples:
            q_type = sample.get("question_type", "unknown")
            stats[q_type] = stats.get(q_type, 0) + 1
        return stats
    
    def get_samples_by_question_type(self, question_type: str) -> List[Dict]:
        """Get all samples of a specific question type"""
        return [s for s in self.vqa_samples if s.get("question_type", "") == question_type]


class SceneGraphVQADatasetAnalysis(SceneGraphVQADataset):
    """
    Extended dataset class for analysis of different question complexities
    (following SSG-VQA analysis structure)
    """
    
    def __init__(self, 
                 sequences: List[str],
                 data_root: str, 
                 scene_graph_config: SceneGraphConfig,
                 analysis_types: List[str],
                 load_images: bool = True):
        
        # Initialize with all question types first
        super().__init__(sequences, data_root, scene_graph_config, None, load_images)
        
        # Filter samples based on analysis types
        self.analysis_types = analysis_types
        self.vqa_samples = self._filter_samples_for_analysis()
        
        print(f"Filtered to {len(self.vqa_samples)} samples for analysis types: {analysis_types}")
    
    def _filter_samples_for_analysis(self) -> List[Dict]:
        """Filter samples based on analysis type requirements"""
        if not self.analysis_types:
            return self.vqa_samples
        
        filtered_samples = []
        
        for sample in self.vqa_samples:
            q_type = sample.get("question_type", "")
            q_category = sample.get("question_category", "")
            
            # Check if sample matches any of the analysis types
            for analysis_type in self.analysis_types:
                if self._matches_analysis_type(analysis_type, q_type, q_category):
                    filtered_samples.append(sample)
                    break
        
        return filtered_samples
    
    def _matches_analysis_type(self, analysis_type: str, q_type: str, q_category: str) -> bool:
        """Check if a sample matches the analysis type criteria"""
        analysis_type = analysis_type.lower()
        
        # Direct question type matches
        if analysis_type == q_type.lower():
            return True
        
        # Category-based matches
        if analysis_type in ["query_color", "query_type", "query_location"] and q_category.lower() == analysis_type:
            return True
        
        # Complexity-based matches
        complexity_mapping = {
            "zero_hop": ["exist", "count", "query_component"],
            "one_hop": ["query_color", "query_type", "query_location"],
            "single_and": ["single_and"]
        }
        
        for complexity, types in complexity_mapping.items():
            if analysis_type == complexity and q_type.lower() in types:
                return True
        
        return False


class DatasetFactory:
    """Factory for creating different dataset configurations"""
    
    @staticmethod
    def create_dataset(sequences: List[str],
                      data_root: str,
                      scene_graph_config: SceneGraphConfig,
                      dataset_type: str = "standard",
                      question_types: Optional[List[str]] = None,
                      analysis_types: Optional[List[str]] = None,
                      load_images: bool = True) -> SceneGraphVQADataset:
        """Create a dataset based on the specified type"""
        
        if dataset_type == "analysis":
            if not analysis_types:
                raise ValueError("analysis_types must be provided for analysis dataset")
            return SceneGraphVQADatasetAnalysis(
                sequences, data_root, scene_graph_config, analysis_types, load_images
            )
        else:
            return SceneGraphVQADataset(
                sequences, data_root, scene_graph_config, question_types, load_images
            )


def create_data_loaders(data_config, scene_graph_config, batch_size: int = 16):
    """Create train, validation, and test data loaders"""
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    # Create datasets
    if data_config.train_sequences:
        train_dataset = DatasetFactory.create_dataset(
            data_config.train_sequences,
            data_config.data_root,
            scene_graph_config,
            question_types=data_config.question_types
        )
        loaders["train"] = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_vqa_batch
        )
    
    if data_config.val_sequences:
        val_dataset = DatasetFactory.create_dataset(
            data_config.val_sequences,
            data_config.data_root,
            scene_graph_config,
            question_types=data_config.question_types
        )
        loaders["val"] = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_vqa_batch
        )
    
    if data_config.test_sequences:
        test_dataset = DatasetFactory.create_dataset(
            data_config.test_sequences,
            data_config.data_root,
            scene_graph_config,
            question_types=data_config.question_types
        )
        loaders["test"] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_vqa_batch
        )
    
    return loaders


def collate_vqa_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for VQA batches"""
    
    # Initialize batch containers
    collated = {
        "sequence_ids": [],
        "frame_ids": [],
        "questions": [],
        "answers": [],
        "question_types": [],
        "question_categories": [],
        "label_indices": [],
        "scene_graphs": [],
        "scene_graphs_raw": [],
        "temporal_contexts": []
    }
    
    # Handle optional fields
    images = []
    visual_features = []
    has_images = False
    has_visual_features = False
    
    for item in batch:
        # Required fields
        collated["sequence_ids"].append(item["sequence_id"])
        collated["frame_ids"].append(item["frame_id"])
        collated["questions"].append(item["question"])
        collated["answers"].append(item["answer"])
        collated["question_types"].append(item["question_type"])
        collated["question_categories"].append(item["question_category"])
        collated["label_indices"].append(item["label_idx"])
        collated["scene_graphs"].append(item.get("scene_graph"))
        collated["scene_graphs_raw"].append(item.get("scene_graph_raw"))
        collated["temporal_contexts"].append(item.get("temporal_context"))
        
        # Optional fields
        if "image" in item and item["image"] is not None:
            images.append(item["image"])
            has_images = True
        
        if "visual_features" in item and item["visual_features"] is not None:
            visual_features.append(item["visual_features"])
            has_visual_features = True
    
    # Add optional fields to batch if available
    if has_images:
        collated["images"] = images
    
    if has_visual_features:
        # Stack visual features if they have the same shape
        try:
            collated["visual_features"] = torch.stack(visual_features)
        except:
            # If shapes differ, keep as list
            collated["visual_features"] = visual_features
    
    # Convert label indices to tensor
    collated["label_indices"] = torch.tensor(collated["label_indices"], dtype=torch.long)
    
    return collated


if __name__ == "__main__":
    # Example usage
    from config_manager import SceneGraphConfig
    
    # Test configuration
    scene_graph_config = SceneGraphConfig(
        source_type="2d",
        encoding_format="json",
        temporal_frames=1,
        graph_dir="./data/scene_graphs/",
        include_spatial_info=True,
        include_temporal_info=False
    )
    
    # Test dataset creation
    test_sequences = ["VID01", "VID02"]
    dataset = DatasetFactory.create_dataset(
        sequences=test_sequences,
        data_root="./data/",
        scene_graph_config=scene_graph_config,
        question_types=["all"],
        load_images=False  # Set to False for testing without images
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")

        