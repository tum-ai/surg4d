#!/usr/bin/env python3
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from qwen_vl_qwen3 import get_patched_qwen3, qwen3_encode_image, patches_to_2d


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def process_single_image(model, processor, image_path, preserve_aspect=True):
    image = Image.open(image_path).convert("RGB")

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

    patch_features = qwen3_encode_image(resized_image, model, processor)

    pca = PCA(n_components=3)
    patch_features_pca = pca.fit_transform(patch_features.cpu().float().numpy())
    scaler = MinMaxScaler()
    patch_features_pca = scaler.fit_transform(patch_features_pca)

    patch_features_pca_2d = patches_to_2d(
        torch.tensor(patch_features_pca), resized_image
    )
    return resized_image, patch_features_pca_2d


def generate_feature_map_pdf(model, processor, image_paths, output_path, title):
    print(f"Generating feature map PDF for {title}...")
    with PdfPages(output_path) as pdf:
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                resized_image, patch_features_pca_2d = process_single_image(
                    model, processor, image_path, preserve_aspect=True
                )
                base_name = os.path.splitext(os.path.basename(image_path))[0]

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(resized_image)
                axes[0].set_title(
                    f"{base_name}\n{resized_image.width}x{resized_image.height}"
                )
                axes[0].axis("off")

                axes[1].imshow(patch_features_pca_2d)
                axes[1].set_title(
                    f"Feature Map\n{patch_features_pca_2d.shape[0]}x{patch_features_pca_2d.shape[1]} patches"
                )
                axes[1].axis("off")

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                clear_memory()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate feature maps using Qwen3-VL")
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="feature_maps_qwen3.pdf")
    parser.add_argument("--bnb4", action="store_true")
    parser.add_argument("--bnb8", action="store_true")
    args = parser.parse_args()

    model, processor = get_patched_qwen3(
        model_path=args.model_path,
        use_bnb_4bit=args.bnb4,
        use_bnb_8bit=args.bnb8,
    )
    generate_feature_map_pdf(model, processor, args.images, args.output, "Qwen3")


if __name__ == "__main__":
    main()
