#!/usr/bin/env python3
"""
Consistent Feature Processor

This module provides a feature processor that ensures consistent CLIP feature extraction
between preprocessing and evaluation phases. It addresses the issue where the same model
and weights produce different language features when loaded differently.

Key Features:
- Uses ConsistentCLIPExtractor for reliable feature extraction
- Maintains compatibility with existing feature processing pipeline
- Provides enhanced object detection capabilities
- Supports multiple CLIP model configurations
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add the preprocessing path to import the CLIP model
sys.path.append("/home/tumai/team1/Ken/4DLangSplatSurgery/preprocess")

# Import the exact same CLIP model used in preprocessing
from generate_clip_features_cholecseg8k import OpenCLIPNetwork, OpenCLIPNetworkConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConsistentCLIPExtractor:
    """CLIP feature extractor that ensures consistency with preprocessing"""

    def __init__(self, model_type="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        self.model_type = model_type
        self.pretrained = pretrained

        logger.info(
            f"Initializing ConsistentCLIPExtractor with {model_type}/{pretrained}"
        )

        # Initialize exactly as in preprocessing
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_type, pretrained, precision="fp16"
        )
        self.model.eval()
        self.model = self.model.to(device)

        self.tokenizer = open_clip.get_tokenizer(model_type)

        # Preprocessing pipeline
        import torchvision.transforms as transforms

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        logger.info("ConsistentCLIPExtractor initialized successfully")

    def preprocess_image(self, image):
        """Preprocess image consistently"""
        from PIL import Image
        import torchvision.transforms as transforms

        if isinstance(image, Image.Image):
            # Convert to tensor
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image)
        else:
            image_tensor = image

        # Apply preprocessing
        processed = self.preprocess(image_tensor).unsqueeze(0)
        return processed

    def encode_image(self, image):
        """Encode image with consistent method"""
        processed = self.preprocess_image(image)

        with torch.no_grad():
            embedding = self.model.encode_image(processed.half())
            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding

    def encode_text(self, text):
        """Encode text with consistent method"""
        tokens = self.tokenizer(text).to(device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding

    def compute_similarity(self, image, text):
        """Compute similarity between image and text"""
        img_emb = self.encode_image(image)
        text_emb = self.encode_text(text)

        with torch.no_grad():
            similarity = torch.mm(img_emb, text_emb.T).item()

        return similarity

    def compute_similarities_batch(self, image, text_list):
        """Compute similarities between image and multiple texts"""
        img_emb = self.encode_image(image)

        with torch.no_grad():
            text_embeddings = []
            for text in text_list:
                text_emb = self.encode_text(text)
                text_embeddings.append(text_emb)

            text_embeddings = torch.cat(text_embeddings, dim=0)
            similarities = torch.mm(img_emb, text_embeddings.T)
            similarities = similarities.cpu().numpy().flatten()

        return similarities


class ConsistentFeatureProcessor:
    """Enhanced feature processor with consistent CLIP extraction"""

    def __init__(
        self,
        clip_model_type="ViT-B-16",
        clip_pretrained="laion2b_s34b_b88k",
        enable_feature_decoding=True,
        top_k_matches=5,
        confidence_threshold=0.05,
        vocabulary_type="general",
    ):
        """
        Initialize the consistent feature processor

        Args:
            clip_model_type: CLIP model type (e.g., "ViT-B-16", "ViT-B-32")
            clip_pretrained: CLIP pretrained weights (e.g., "laion2b_s34b_b88k", "openai")
            enable_feature_decoding: Whether to decode CLIP features to text descriptions
            top_k_matches: Number of top matches to return for each feature
            confidence_threshold: Minimum confidence threshold for matches
            vocabulary_type: Type of vocabulary to use ("general", "medical", "custom")
        """
        self.clip_model_type = clip_model_type
        self.clip_pretrained = clip_pretrained
        self.enable_feature_decoding = enable_feature_decoding
        self.top_k_matches = top_k_matches
        self.confidence_threshold = confidence_threshold
        self.vocabulary_type = vocabulary_type

        # Initialize consistent CLIP extractor
        self.clip_extractor = ConsistentCLIPExtractor(clip_model_type, clip_pretrained)

        # Initialize vocabulary
        self.vocabulary = self._initialize_vocabulary(vocabulary_type)

        logger.info(
            f"ConsistentFeatureProcessor initialized with {clip_model_type}/{clip_pretrained}"
        )

    def _initialize_vocabulary(self, vocabulary_type):
        """Initialize vocabulary based on type"""
        if vocabulary_type == "general":
            return self._get_general_vocabulary()
        elif vocabulary_type == "medical":
            return self._get_medical_vocabulary()
        elif vocabulary_type == "custom":
            return self._get_custom_vocabulary()
        else:
            logger.warning(f"Unknown vocabulary type: {vocabulary_type}, using general")
            return self._get_general_vocabulary()

    def _get_general_vocabulary(self):
        """Get general vocabulary for object detection"""
        return [
            # Human body parts
            "hand",
            "hands",
            "finger",
            "fingers",
            "arm",
            "arms",
            "body",
            "person",
            # Common objects
            "object",
            "thing",
            "item",
            "tool",
            "device",
            "equipment",
            # Colors
            "red",
            "green",
            "blue",
            "yellow",
            "white",
            "black",
            "purple",
            "orange",
            "brown",
            "gray",
            # Materials
            "plastic",
            "metal",
            "wood",
            "fabric",
            "glass",
            "paper",
            "ceramic",
            # Shapes
            "round",
            "square",
            "rectangular",
            "circular",
            "oval",
            "triangular",
            # Sizes
            "small",
            "large",
            "big",
            "tiny",
            "huge",
            "medium",
            # Textures
            "smooth",
            "rough",
            "shiny",
            "matte",
            "textured",
            "flat",
            # Common objects
            "cup",
            "bowl",
            "plate",
            "spoon",
            "fork",
            "knife",
            "container",
            "box",
            "toy",
            "ball",
            "block",
            "piece",
            "part",
            "component",
            # Surfaces
            "table",
            "surface",
            "board",
            "platform",
            "base",
            "stand",
            # Actions
            "holding",
            "touching",
            "grasping",
            "lifting",
            "moving",
            "placing",
        ]

    def _get_medical_vocabulary(self):
        """Get medical vocabulary for surgical scenes"""
        return [
            # Surgical instruments
            "scalpel",
            "forceps",
            "clamp",
            "scissors",
            "needle",
            "suture",
            "cautery",
            "hook",
            "retractor",
            "speculum",
            "probe",
            "catheter",
            "tube",
            "cannula",
            # Anatomical structures
            "tissue",
            "muscle",
            "bone",
            "skin",
            "fat",
            "vessel",
            "nerve",
            "organ",
            "liver",
            "kidney",
            "heart",
            "lung",
            "stomach",
            "intestine",
            "bladder",
            # Surgical materials
            "mesh",
            "graft",
            "implant",
            "stent",
            "screw",
            "plate",
            "wire",
            "suture",
            "gauze",
            "sponge",
            "drape",
            "tape",
            "glue",
            "cement",
            # Surgical procedures
            "incision",
            "dissection",
            "resection",
            "anastomosis",
            "ligation",
            "cauterization",
            # Medical conditions
            "tumor",
            "lesion",
            "abscess",
            "hernia",
            "adhesion",
            "stricture",
            "stenosis",
            # Colors in medical context
            "pink",
            "red",
            "purple",
            "yellow",
            "white",
            "brown",
            "gray",
            "black",
            # Textures in medical context
            "smooth",
            "rough",
            "granular",
            "fibrous",
            "calcified",
            "necrotic",
            "viable",
        ]

    def _get_custom_vocabulary(self):
        """Get custom vocabulary - can be extended based on specific needs"""
        return self._get_general_vocabulary() + [
            # Add custom terms here
            "chicken",
            "egg",
            "food",
            "kitchen",
            "cooking",
            "preparation",
        ]

    def process_node_features(
        self, node_id: str, attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process features for a single node with consistent CLIP extraction

        Args:
            node_id: Node identifier
            attributes: Node attributes containing features

        Returns:
            Dictionary with processed features
        """
        result = {
            "node_id": node_id,
            "original_attributes": attributes.copy(),
            "decoded_features": {},
            "raw_features": {},
            "enhanced_descriptions": [],
        }

        # Extract CLIP features if present
        if "clip_features" in attributes:
            clip_features = attributes["clip_features"]

            if isinstance(clip_features, list) and len(clip_features) == 512:
                # Convert to tensor and ensure correct dtype
                feature_tensor = torch.tensor(clip_features, dtype=torch.float32).to(
                    device
                )
                feature_tensor = (
                    feature_tensor.half()
                )  # Convert to half precision to match CLIP
                feature_tensor /= feature_tensor.norm(dim=-1, keepdim=True)

                # Store raw features
                result["raw_features"]["clip_features"] = clip_features

                # Decode features if enabled
                if self.enable_feature_decoding:
                    decoded = self._decode_clip_features(feature_tensor)
                    result["decoded_features"]["clip_features"] = decoded

                    # Create enhanced description
                    if decoded.get("primary_description"):
                        confidence = decoded.get("confidence", 0)
                        alternatives = decoded.get("alternative_descriptions", [])

                        desc = f"{decoded['primary_description']} (confidence: {confidence:.3f})"
                        if alternatives:
                            desc += f", alternatives: {', '.join(alternatives[:3])}"

                        result["enhanced_descriptions"].append(desc)

        # Process other feature types if present
        for key, value in attributes.items():
            if key != "clip_features" and isinstance(value, (list, np.ndarray)):
                if len(value) > 10:  # Likely a feature vector
                    result["raw_features"][key] = value

                    # Try to decode if it looks like a CLIP feature
                    if len(value) == 512:
                        try:
                            feature_tensor = torch.tensor(
                                value, dtype=torch.float32
                            ).to(device)
                            feature_tensor = (
                                feature_tensor.half()
                            )  # Convert to half precision
                            feature_tensor /= feature_tensor.norm(dim=-1, keepdim=True)
                            decoded = self._decode_clip_features(feature_tensor)
                            result["decoded_features"][key] = decoded
                        except Exception as e:
                            logger.warning(f"Failed to decode features for {key}: {e}")

        return result

    def _decode_clip_features(self, feature_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Decode CLIP features to text descriptions using vocabulary

        Args:
            feature_tensor: Normalized CLIP feature tensor

        Returns:
            Dictionary with decoded information
        """
        # Encode vocabulary terms
        with torch.no_grad():
            vocab_embeddings = []
            for term in self.vocabulary:
                try:
                    text_emb = self.clip_extractor.encode_text(term)
                    vocab_embeddings.append(text_emb)
                except Exception as e:
                    logger.warning(f"Failed to encode vocabulary term '{term}': {e}")
                    continue

            if not vocab_embeddings:
                return {"primary_description": "unknown", "confidence": 0.0}

            vocab_embeddings = torch.cat(vocab_embeddings, dim=0)

            # Compute similarities
            similarities = torch.mm(feature_tensor.unsqueeze(0), vocab_embeddings.T)
            similarities = similarities.cpu().numpy().flatten()

            # Get top matches
            top_indices = np.argsort(similarities)[::-1][: self.top_k_matches]

            # Filter by confidence threshold
            valid_matches = []
            for idx in top_indices:
                if similarities[idx] >= self.confidence_threshold:
                    valid_matches.append((self.vocabulary[idx], similarities[idx]))

            if not valid_matches:
                return {"primary_description": "unknown", "confidence": 0.0}

            # Return results
            primary_term, primary_confidence = valid_matches[0]
            alternative_terms = [term for term, conf in valid_matches[1:]]

            return {
                "primary_description": primary_term,
                "confidence": float(primary_confidence),
                "alternative_descriptions": alternative_terms,
                "all_matches": valid_matches,
            }

    def create_textual_description(self, processed_features: Dict[str, Any]) -> str:
        """
        Create textual description from processed features

        Args:
            processed_features: Output from process_node_features

        Returns:
            Textual description
        """
        descriptions = []

        # Add enhanced descriptions
        if processed_features.get("enhanced_descriptions"):
            descriptions.extend(processed_features["enhanced_descriptions"])

        # Add other attributes
        original_attrs = processed_features.get("original_attributes", {})
        for key, value in original_attrs.items():
            if key not in ["clip_features"] and value is not None:
                if isinstance(value, (int, float)):
                    descriptions.append(f"{key}: {value}")
                elif isinstance(value, str) and len(value) < 50:
                    descriptions.append(f"{key}: {value}")

        return ", ".join(descriptions) if descriptions else "unknown object"

    def process_scene_graph(self, scene_graph) -> Dict[str, Any]:
        """
        Process entire scene graph with consistent feature extraction

        Args:
            scene_graph: Scene graph object

        Returns:
            Dictionary with processed scene graph information
        """
        processed_nodes = []
        processed_edges = []

        # Process nodes
        for node in scene_graph.nodes:
            if hasattr(node, "attributes"):
                processed = self.process_node_features(str(node.id), node.attributes)
                processed_nodes.append(processed)
            else:
                # Handle different node formats
                node_attrs = {}
                if hasattr(node, "id"):
                    node_attrs["id"] = node.id
                if hasattr(node, "label"):
                    node_attrs["label"] = node.label

                processed = self.process_node_features(
                    str(getattr(node, "id", "unknown")), node_attrs
                )
                processed_nodes.append(processed)

        # Process edges (if needed)
        for edge in scene_graph.edges:
            edge_info = {
                "source": str(edge[0])
                if hasattr(edge, "__getitem__")
                else str(edge.source),
                "target": str(edge[1])
                if hasattr(edge, "__getitem__")
                else str(edge.target),
                "attributes": getattr(edge, "attributes", {}),
            }
            processed_edges.append(edge_info)

        return {
            "nodes": processed_nodes,
            "edges": processed_edges,
            "total_nodes": len(processed_nodes),
            "total_edges": len(processed_edges),
        }

    def test_consistency(
        self, test_image_path: str, test_objects: List[str]
    ) -> Dict[str, Any]:
        """
        Test CLIP consistency with a test image

        Args:
            test_image_path: Path to test image
            test_objects: List of objects to test for

        Returns:
            Dictionary with consistency test results
        """
        from PIL import Image

        logger.info(f"Testing CLIP consistency with {test_image_path}")

        # Load test image
        test_image = Image.open(test_image_path)

        # Compute similarities
        similarities = self.clip_extractor.compute_similarities_batch(
            test_image, test_objects
        )

        # Create results
        results = []
        for i, obj in enumerate(test_objects):
            results.append(
                {
                    "object": obj,
                    "similarity": float(similarities[i]),
                    "detected": similarities[i] >= self.confidence_threshold,
                }
            )

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "test_image": test_image_path,
            "results": results,
            "top_detections": results[:5],
            "detected_objects": [r for r in results if r["detected"]],
        }


def create_consistent_feature_processor(
    preset: str = "default",
) -> ConsistentFeatureProcessor:
    """
    Create a consistent feature processor with predefined presets

    Args:
        preset: Preset configuration ("default", "medical", "high_confidence", "fast")

    Returns:
        ConsistentFeatureProcessor instance
    """
    presets = {
        "default": {
            "clip_model_type": "ViT-B-16",
            "clip_pretrained": "laion2b_s34b_b88k",
            "enable_feature_decoding": True,
            "top_k_matches": 5,
            "confidence_threshold": 0.05,
            "vocabulary_type": "general",
        },
        "medical": {
            "clip_model_type": "ViT-B-16",
            "clip_pretrained": "laion2b_s34b_b88k",
            "enable_feature_decoding": True,
            "top_k_matches": 10,
            "confidence_threshold": 0.03,
            "vocabulary_type": "medical",
        },
        "high_confidence": {
            "clip_model_type": "ViT-B-16",
            "clip_pretrained": "laion2b_s34b_b88k",
            "enable_feature_decoding": True,
            "top_k_matches": 3,
            "confidence_threshold": 0.1,
            "vocabulary_type": "general",
        },
        "fast": {
            "clip_model_type": "ViT-B-32",
            "clip_pretrained": "openai",
            "enable_feature_decoding": True,
            "top_k_matches": 3,
            "confidence_threshold": 0.05,
            "vocabulary_type": "general",
        },
    }

    if preset not in presets:
        logger.warning(f"Unknown preset '{preset}', using default")
        preset = "default"

    config = presets[preset]
    logger.info(f"Creating ConsistentFeatureProcessor with preset: {preset}")

    return ConsistentFeatureProcessor(**config)


if __name__ == "__main__":
    # Test the consistent feature processor
    print("Testing ConsistentFeatureProcessor...")

    # Create processor
    processor = create_consistent_feature_processor("default")

    # Test with sample data
    test_attributes = {
        "clip_features": [0.1] * 512,  # Dummy features
        "pos_x": 1.0,
        "pos_y": 2.0,
        "pos_z": 3.0,
    }

    # Process features
    result = processor.process_node_features("test_node", test_attributes)
    print(f"Processed features: {result}")

    # Test consistency
    if os.path.exists("00031.jpg"):
        test_objects = ["hand", "egg", "toy", "red object"]
        consistency_result = processor.test_consistency("00031.jpg", test_objects)
        print(f"Consistency test results: {consistency_result}")
    else:
        print("Test image not found, skipping consistency test")
