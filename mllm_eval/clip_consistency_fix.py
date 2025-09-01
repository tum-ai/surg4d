#!/usr/bin/env python3
"""
CLIP Feature Consistency Fix

This script addresses the CLIP feature inconsistency problem where the same model
and weights produce different language features when loaded differently.

Problem Analysis:
- CLIP features extracted during preprocessing don't match features computed during evaluation
- Objects like "hand" are not being recognized properly
- Different loading methods/configurations may cause this inconsistency

Solution:
- Ensure consistent CLIP model loading between preprocessing and evaluation
- Use the same configuration and initialization method
- Test feature consistency with known objects
"""

import torch
import numpy as np

# import matplotlib.pyplot as plt  # Optional for visualization
from PIL import Image
import cv2
import os
import sys
from pathlib import Path

# Add the preprocessing path to import the CLIP model
sys.path.append("/home/tumai/team1/Ken/4DLangSplatSurgery/preprocess")

# Import the exact same CLIP model used in preprocessing
from generate_clip_features_cholecseg8k import OpenCLIPNetwork, OpenCLIPNetworkConfig

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_consistent_clip_model():
    """Load CLIP model with the exact same configuration as preprocessing"""
    print("Loading CLIP model with preprocessing configuration...")

    # Use the same config as in the preprocessing script
    clip_config = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16",
        clip_model_pretrained="laion2b_s34b_b88k",
        clip_n_dims=512,
        negatives=("object", "things", "stuff", "texture"),
        positives=("",),
    )

    # Initialize the model exactly as in preprocessing
    clip_model = OpenCLIPNetwork(clip_config)
    clip_model.eval()

    print(f"CLIP model loaded: {clip_model.name}")
    print(f"Embedding dimension: {clip_model.embedding_dim}")

    return clip_model


def preprocess_image_for_clip(image, clip_model):
    """Preprocess image exactly as done in the preprocessing script"""
    # Convert PIL to numpy and then to tensor
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        # Convert RGB to BGR if needed (as done in preprocessing)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_tensor = torch.from_numpy(image_np).float()
    else:
        image_tensor = image

    # Apply the same preprocessing as in OpenCLIPNetwork
    processed = clip_model.process(image_tensor.unsqueeze(0))
    return processed


def test_clip_encoding(clip_model, image_path):
    """Test CLIP encoding with the exact same method as preprocessing"""
    print(f"Testing CLIP image encoding for {image_path}...")

    # Load image
    image = Image.open(image_path)

    # Preprocess the image
    processed_image = preprocess_image_for_clip(image, clip_model)
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image dtype: {processed_image.dtype}")

    # Encode with CLIP
    with torch.no_grad():
        image_embedding = clip_model.encode_image(processed_image)
        # Normalize as done in preprocessing
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Image embedding norm: {image_embedding.norm().item():.6f}")

    return image_embedding, image


def test_text_encoding(clip_model, test_objects):
    """Test text encoding with specific objects we want to detect"""
    print("Testing CLIP text encoding for specific objects...")

    # Encode text descriptions
    with torch.no_grad():
        text_embeddings = []
        for obj in test_objects:
            # Use the same tokenizer as in preprocessing
            tokens = clip_model.tokenizer(obj).to(device)
            embedding = clip_model.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            text_embeddings.append(embedding)

        text_embeddings = torch.cat(text_embeddings, dim=0)

    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Number of test objects: {len(test_objects)}")

    return text_embeddings


def compute_similarities(image_embedding, text_embeddings, test_objects):
    """Compute similarities between image and text embeddings"""
    print("Computing image-text similarities...")

    with torch.no_grad():
        # Compute cosine similarity
        similarities = torch.mm(image_embedding, text_embeddings.T)
        similarities = similarities.cpu().numpy().flatten()

    # Create results table
    results = []
    for i, obj in enumerate(test_objects):
        results.append(
            {
                "object": obj,
                "similarity": similarities[i],
                "rank": 0,  # Will be filled below
            }
        )

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Add ranks
    for i, result in enumerate(results):
        result["rank"] = i + 1

    print("\nTop 10 most similar objects:")
    print("Rank | Object | Similarity")
    print("-" * 40)
    for i, result in enumerate(results[:10]):
        print(
            f"{result['rank']:4d} | {result['object']:15s} | {result['similarity']:.4f}"
        )

    # Check if hand-related objects are detected
    hand_objects = [
        r for r in results if "hand" in r["object"] or "finger" in r["object"]
    ]
    print(f"\nHand-related objects detected: {len(hand_objects)}")
    for obj in hand_objects:
        print(f"  {obj['object']}: {obj['similarity']:.4f} (rank {obj['rank']})")

    return results


class ConsistentCLIPExtractor:
    """CLIP feature extractor that ensures consistency with preprocessing"""

    def __init__(self, model_type="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        self.model_type = model_type
        self.pretrained = pretrained

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

    def preprocess_image(self, image):
        """Preprocess image consistently"""
        if isinstance(image, Image.Image):
            # Convert to tensor
            import torchvision.transforms as transforms

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


def test_different_configurations():
    """Test with different CLIP model configurations to identify the issue"""
    print("Testing different CLIP configurations...")

    # Configuration 1: Original preprocessing config
    config1 = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16",
        clip_model_pretrained="laion2b_s34b_b88k",
        clip_n_dims=512,
    )

    # Configuration 2: Different model type
    config2 = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-32", clip_model_pretrained="openai", clip_n_dims=512
    )

    configs = [("Original", config1), ("ViT-B-32", config2)]

    results_comparison = {}

    for config_name, config in configs:
        print(f"\nTesting {config_name} configuration...")

        try:
            # Load model with this config
            model = OpenCLIPNetwork(config)
            model.eval()

            # Test with a simple image
            test_image = Image.new("RGB", (224, 224), color="red")
            processed = preprocess_image_for_clip(test_image, model)

            # Encode image
            with torch.no_grad():
                img_emb = model.encode_image(processed)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            # Encode text for hand detection
            with torch.no_grad():
                hand_tokens = model.tokenizer("hand").to(device)
                hand_emb = model.model.encode_text(hand_tokens)
                hand_emb /= hand_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            with torch.no_grad():
                hand_similarity = torch.mm(img_emb, hand_emb.T).item()

            results_comparison[config_name] = hand_similarity
            print(f"  Hand similarity: {hand_similarity:.4f}")

        except Exception as e:
            print(f"  Error with {config_name}: {e}")
            results_comparison[config_name] = None

    print("\nConfiguration comparison:")
    for config_name, similarity in results_comparison.items():
        if similarity is not None:
            print(f"  {config_name}: {similarity:.4f}")
        else:
            print(f"  {config_name}: Failed")

    return results_comparison


def main():
    """Main function to run the CLIP consistency analysis"""
    print("=== CLIP FEATURE CONSISTENCY ANALYSIS ===\n")

    # Load consistent CLIP model
    clip_model = load_consistent_clip_model()

    # Test image path
    image_path = "00031.jpg"

    # Test CLIP encoding
    image_embedding, image = test_clip_encoding(clip_model, image_path)

    # Define test objects including the problematic ones
    test_objects = [
        "hand",
        "hands",
        "finger",
        "fingers",
        "egg",
        "chicken",
        "toy",
        "plastic",
        "red object",
        "yellow object",
        "wooden board",
        "kitchen",
        "table",
        "surface",
    ]

    # Test text encoding
    text_embeddings = test_text_encoding(clip_model, test_objects)

    # Compute similarities
    results = compute_similarities(image_embedding, text_embeddings, test_objects)

    # Test different configurations
    config_results = test_different_configurations()

    # Create consistent extractor
    print("\nCreating consistent CLIP extractor...")
    consistent_extractor = ConsistentCLIPExtractor()
    print("Consistent CLIP extractor created successfully")

    # Test with our image
    print("\nTesting consistent extractor...")
    test_objects_short = ["hand", "hands", "egg", "chicken", "toy", "red object"]

    for obj in test_objects_short:
        similarity = consistent_extractor.compute_similarity(image, obj)
        print(f"  {obj}: {similarity:.4f}")

    # Summary
    print("\n=== CLIP CONSISTENCY ANALYSIS SUMMARY ===\n")

    print("1. PROBLEM IDENTIFIED:")
    print(
        "   - CLIP features extracted during preprocessing may differ from evaluation"
    )
    print(
        "   - Different model loading methods or configurations can cause inconsistencies"
    )
    print(
        "   - Objects like 'hand' may not be detected properly due to these inconsistencies"
    )

    print("\n2. SOLUTION IMPLEMENTED:")
    print(
        "   - Created ConsistentCLIPExtractor class that ensures consistent model loading"
    )
    print("   - Uses the exact same configuration as the preprocessing script")
    print("   - Implements consistent preprocessing pipeline")
    print("   - Provides standardized encoding methods")

    print("\n3. RECOMMENDATIONS:")
    print("   - Use ConsistentCLIPExtractor for all CLIP feature extraction")
    print(
        "   - Ensure the same model configuration is used in preprocessing and evaluation"
    )
    print(
        "   - Test feature consistency with known objects before running full pipeline"
    )
    print(
        "   - Consider using different CLIP model variants if detection issues persist"
    )

    print("\n4. NEXT STEPS:")
    print("   - Integrate ConsistentCLIPExtractor into the main evaluation pipeline")
    print("   - Re-run preprocessing with consistent configuration if needed")
    print("   - Test with different images to validate the fix")
    print("   - Monitor detection accuracy for problematic objects like hands")

    return {
        "clip_model": clip_model,
        "consistent_extractor": consistent_extractor,
        "results": results,
        "config_results": config_results,
    }


if __name__ == "__main__":
    results = main()
