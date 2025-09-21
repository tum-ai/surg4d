#!/usr/bin/env python3
"""
Feature map generator for Qwen2.5-VL - streamlined version
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import argparse
from tqdm import tqdm
import time
from matplotlib.backends.backend_pdf import PdfPages
import json
import gc


def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0


def load_model():
    """Load the Qwen2.5-VL model and processor"""
    model_path = "/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct"
    print("Loading Qwen2.5-VL model...")
    start_time = time.time()

    # Clear memory before loading
    clear_memory()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model.eval()

    load_time = time.time() - start_time
    allocated, reserved = get_memory_usage()
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    return model, processor


def qwen_encode_image(model, processor, image):
    """Extract patch features from image"""
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)
    with torch.no_grad():
        patch_features = model.visual(pixel_values, image_grid_thw)

    # Clear intermediate tensors
    del pixel_values, image_grid_thw
    clear_memory()

    return patch_features


def patches_to_2d(patch_features, src_image):
    """Convert patch features to 2D spatial layout using correct patch size calculation"""
    # Use the same approach as in qwen_vl.py
    patch_size, spatial_merge = 14, 2  # model hyperparams
    effective_patch_size = patch_size * spatial_merge
    patches_width = src_image.width // effective_patch_size
    patches_height = src_image.height // effective_patch_size

    # Ensure the reshape dimensions match the actual number of patches
    num_patches = patch_features.shape[0]
    expected_patches = patches_height * patches_width

    if num_patches != expected_patches:
        # If dimensions don't match, try to find the closest factor pair
        import math

        root = int(math.isqrt(num_patches))
        for a in range(root, 0, -1):
            if num_patches % a == 0:
                patches_height = a
                patches_width = num_patches // a
                break

    return patch_features.reshape(patches_height, patches_width, -1)


def process_single_image(model, processor, image_path, preserve_aspect=True):
    """Process a single image and return features"""
    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Resize while preserving aspect ratio
    if preserve_aspect:
        max_size = 896
        if image.width > image.height:
            new_width = max_size
            new_height = int((max_size * image.height) / image.width)
        else:
            new_height = max_size
            new_width = int((max_size * image.width) / image.height)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        resized_image = image.resize((896, 896), Image.Resampling.LANCZOS)

    # Extract features
    patch_features = qwen_encode_image(model, processor, resized_image)

    # Apply PCA for visualization
    pca = PCA(n_components=3)
    patch_features_pca = pca.fit_transform(patch_features.cpu().float().numpy())
    scaler = MinMaxScaler()
    patch_features_pca = scaler.fit_transform(patch_features_pca)

    # Convert to 2D spatial layout
    patch_features_pca_2d = patches_to_2d(
        torch.tensor(patch_features_pca), resized_image
    )

    return resized_image, patch_features_pca_2d


def generate_feature_map_pdf(model, processor, image_paths, output_path, test_set_name):
    """Generate PDF with feature maps for multiple images - memory optimized"""
    print(f"Generating feature map PDF for {test_set_name}...")
    allocated, reserved = get_memory_usage()
    print(f"Starting memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # Process images one by one and write to PDF immediately (streaming approach)
    with PdfPages(output_path) as pdf:
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Process single image
                resized_image, patch_features_pca_2d = process_single_image(
                    model, processor, image_path, preserve_aspect=True
                )
                base_name = os.path.splitext(os.path.basename(image_path))[0]

                # Create and save plot immediately
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Original image
                axes[0].imshow(resized_image)
                axes[0].set_title(
                    f"{base_name}\n{resized_image.width}x{resized_image.height}"
                )
                axes[0].axis("off")

                # Feature map
                axes[1].imshow(patch_features_pca_2d)
                axes[1].set_title(
                    f"Feature Map\n{patch_features_pca_2d.shape[0]}x{patch_features_pca_2d.shape[1]} patches"
                )
                axes[1].axis("off")

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)  # Close figure to free memory

                # Clear variables and memory
                del resized_image, patch_features_pca_2d
                clear_memory()

                # Print memory usage every few images
                if (i + 1) % 2 == 0:
                    allocated, reserved = get_memory_usage()
                    print(
                        f"After image {i+1} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                    )

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    print(f"Saved feature map PDF: {output_path}")
    allocated, reserved = get_memory_usage()
    print(f"Final memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def test_text_visual_alignment(model, processor, image, text_query):
    """Test text-visual alignment by asking the model about the image"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": text_query},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output_text[0]


def compute_query_patch_similarity(model, processor, patch_features, query):
    """Compute cosine similarity between query tokens and patch features"""
    # Get query token embeddings
    query_token_ids = processor.tokenizer([query], return_tensors="pt")["input_ids"]
    query_tokens = model.get_input_embeddings()(query_token_ids.to(model.device))
    query_tokens = query_tokens[:, 0, :]  # Take first token

    # Compute cosine similarity between query and patch features
    query_patch_similarity = torch.cosine_similarity(
        query_tokens[:, None, :], patch_features[None, :, :], dim=-1
    )

    return query_patch_similarity.squeeze(0)


def process_single_image_with_query(
    model, processor, image_path, query, preserve_aspect=True
):
    """Process a single image with a text query and return features with query similarity"""
    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Resize while preserving aspect ratio
    if preserve_aspect:
        max_size = 896
        if image.width > image.height:
            new_width = max_size
            new_height = int((max_size * image.height) / image.width)
        else:
            new_height = max_size
            new_width = int((max_size * image.width) / image.height)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        resized_image = image.resize((896, 896), Image.Resampling.LANCZOS)

    # Extract features
    patch_features = qwen_encode_image(model, processor, resized_image)

    # Compute query similarity
    query_similarity = compute_query_patch_similarity(
        model, processor, patch_features, query
    )

    # Apply PCA for visualization
    pca = PCA(n_components=3)
    patch_features_pca = pca.fit_transform(patch_features.cpu().float().numpy())
    scaler = MinMaxScaler()
    patch_features_pca = scaler.fit_transform(patch_features_pca)

    # Convert to 2D spatial layout
    patch_features_pca_2d = patches_to_2d(
        torch.tensor(patch_features_pca), resized_image
    )

    # Convert query similarity to 2D spatial layout
    query_similarity_2d = (
        patches_to_2d(query_similarity.unsqueeze(-1), resized_image)
        .squeeze(-1)
        .detach()
        .cpu()
        .float()
    )

    return resized_image, patch_features_pca_2d, query_similarity_2d


def generate_query_feature_map_pdf(
    model, processor, image_paths, queries, output_path, test_set_name
):
    """Generate PDF with query-based feature maps for multiple images"""
    print(f"Generating query feature map PDF for {test_set_name}...")

    # Process all images with all queries first and collect the data
    processed_data = []
    responses_data = {}

    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        responses_data[base_name] = {}

        for j, query in enumerate(queries):
            try:
                resized_image, patch_features_pca_2d, query_similarity_2d = (
                    process_single_image_with_query(
                        model, processor, image_path, query, preserve_aspect=True
                    )
                )

                # Get model response for this query
                response = test_text_visual_alignment(
                    model, processor, resized_image, query
                )
                responses_data[base_name][query] = response

                processed_data.append(
                    {
                        "resized_image": resized_image,
                        "patch_features_pca_2d": patch_features_pca_2d,
                        "query_similarity_2d": query_similarity_2d,
                        "base_name": base_name,
                        "query": query,
                    }
                )

            except Exception as e:
                print(f"Error processing {image_path} with query '{query}': {e}")
                continue

    # Save responses to JSON
    json_path = output_path.replace(".pdf", "_responses.json")
    with open(json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved responses JSON: {json_path}")

    # Now create the PDF with all processed images
    with PdfPages(output_path) as pdf:
        for data in processed_data:
            # Create 2x2 subplot (original, feature map, query similarity, combined)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Original image
            axes[0, 0].imshow(data["resized_image"])
            axes[0, 0].set_title(
                f"{data['base_name']}\n{data['resized_image'].width}x{data['resized_image'].height}"
            )
            axes[0, 0].axis("off")

            # Feature map
            axes[0, 1].imshow(data["patch_features_pca_2d"])
            axes[0, 1].set_title(
                f"Feature Map\n{data['patch_features_pca_2d'].shape[0]}x{data['patch_features_pca_2d'].shape[1]} patches"
            )
            axes[0, 1].axis("off")

            # Query similarity map
            im = axes[1, 0].imshow(data["query_similarity_2d"], cmap="hot")
            axes[1, 0].set_title(f"Query: '{data['query']}'")
            axes[1, 0].axis("off")
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # Combined: overlay query similarity on feature map
            axes[1, 1].imshow(data["patch_features_pca_2d"])
            axes[1, 1].imshow(data["query_similarity_2d"], alpha=0.6, cmap="hot")
            axes[1, 1].set_title(f"Combined: Feature Map + Query")
            axes[1, 1].axis("off")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"Saved query feature map PDF: {output_path}")


def get_test_sets():
    """Define the three test sets"""
    base_path = "/home/tumai/pawere/surgery-scene-graphs/data"

    test_sets = {
        "test": {
            "images": [
                f"{base_path}/hypernerf/americano/rgb/1x/000230.png",
                f"{base_path}/hypernerf/chickchicken/rgb/2x/000001.png",
            ],
            "queries": ["coffee"],
        },
        "americano": {
            "images": [
                f"{base_path}/hypernerf/americano/rgb/1x/000230.png",
                f"{base_path}/hypernerf/americano/rgb/1x/000148.png",
                f"{base_path}/hypernerf/americano/rgb/1x/000128.png",
                f"{base_path}/hypernerf/americano/rgb/1x/000112.png",
            ],
            "queries": ["coffee", "hand", "water", "glass", "board", "table", "liquid"],
        },
        "chickchicken": {
            "images": [
                f"{base_path}/hypernerf/chickchicken/rgb/2x/000001.png",
                f"{base_path}/hypernerf/chickchicken/rgb/2x/000032.png",
                f"{base_path}/hypernerf/chickchicken/rgb/2x/000051.png",
                f"{base_path}/hypernerf/chickchicken/rgb/2x/000451.png",
            ],
            "queries": ["hand", "board", "chicken", "egg", "table"],
        },
        "cholecseg": {
            "images": [
                f"{base_path}/cholecseg8k/preprocessed/video01_00080/images/frame_000068.png",
                f"{base_path}/cholecseg8k/preprocessed/video01_00080/images/frame_000076.png",
                f"{base_path}/cholecseg8k/preprocessed/video01_00080/images/frame_000078.png",
                f"{base_path}/cholecseg8k/preprocessed/video01_00080/images/frame_000080.png",
            ],
            "queries": [
                "grasper",
                "gallbladder",
                "liver",
                "instrument",
                "gastrointestinal tract",
            ],
        },
    }

    return test_sets


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature maps using Qwen2.5-VL"
    )
    parser.add_argument(
        "--test_sets",
        nargs="+",
        choices=["americano", "chickchicken", "cholecseg", "all", "test"],
        default=["test"],
        help="Test sets to run",
    )
    parser.add_argument(
        "--output_dir", default="../feature_maps", help="Output directory for results"
    )
    parser.add_argument(
        "--query_maps", action="store_true", help="Generate query-based feature maps"
    )

    args = parser.parse_args()

    # Load model
    model, processor = load_model()

    # Get test sets
    all_test_sets = get_test_sets()

    if "all" in args.test_sets:
        test_sets_to_run = list(all_test_sets.keys())
    else:
        test_sets_to_run = args.test_sets

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each test set
    for test_set_name in test_sets_to_run:
        if test_set_name not in all_test_sets:
            print(f"Warning: Unknown test set '{test_set_name}', skipping...")
            continue

        test_set = all_test_sets[test_set_name]
        image_paths = test_set["images"]
        text_queries = test_set["queries"]

        print(f"\n{'='*60}")
        print(f"Processing test set: {test_set_name}")
        print(f"Images: {len(image_paths)}")
        print(f"{'='*60}")

        # Generate regular feature map PDF
        feature_map_path = os.path.join(
            args.output_dir, f"{test_set_name}_feature_maps.pdf"
        )
        generate_feature_map_pdf(
            model, processor, image_paths, feature_map_path, test_set_name
        )

        # Generate query-based feature map PDF if requested
        if args.query_maps and text_queries:
            query_feature_map_path = os.path.join(
                args.output_dir, f"{test_set_name}_query_feature_maps.pdf"
            )
            generate_query_feature_map_pdf(
                model,
                processor,
                image_paths,
                text_queries,
                query_feature_map_path,
                test_set_name,
            )

    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
