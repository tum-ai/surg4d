"""
Compare global vs clip-specific autoencoder quality by prompting Qwen with reconstructed features.

For each clip:
1. Load mid frame's GT language features (full dim)
2. Prompt Qwen with GT features
3. Prompt Qwen with clip-specific AE reconstructed features
4. Prompt Qwen with global AE reconstructed features
5. Save midframe image and 3 answers as MD files
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from llm.qwen_utils import (
    get_patched_qwen,
    ask_qwen_about_image_features,
)
from autoencoder.model_qwen import QwenAutoencoder

# Random seed for reproducible permutation
PERMUTATION_SEED = 42


# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# Prompts for Qwen
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Describe what you see in detail. Use bullet points."

# Output directory for results
OUTPUT_DIR = "output/tmp_check_feats"

# Preprocessed data root directory
PREPROCESSED_ROOT = "data/preprocessed/qwen3_da3_subsampled_assigncluster"

# Clip-specific autoencoder checkpoint subdirectory (relative to clip dir)
CLIP_AE_CHECKPOINT_SUBDIR = "autoencoder_me"

# Global autoencoder checkpoint directory (relative to preprocessed root)
GLOBAL_AE_CHECKPOINT_DIR = "global_autoencoder"

# Autoencoder dimensions
FULL_DIM = 16384  # 4096 * 4 (main + 3 deepstack layers concatenated)
LATENT_DIM = 3

# Qwen version to use
QWEN_VERSION = "qwen3"

# ============================================================================


def load_autoencoder(checkpoint_path: Path, device: torch.device) -> QwenAutoencoder:
    """Load autoencoder from checkpoint."""
    ae = QwenAutoencoder(input_dim=FULL_DIM, latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ae.eval()
    return ae


def get_mid_frame_index(clip: DictConfig) -> tuple[int, int]:
    """Get the offset and actual frame number of the middle frame for a clip."""
    num_frames = clip.last_frame - clip.first_frame + 1
    mid_offset = num_frames // 2
    mid_frame_num = clip.first_frame + mid_offset
    return mid_offset, mid_frame_num


def load_mid_frame_features(clip_dir: Path, mid_offset: int) -> np.ndarray:
    """Load GT language features for middle frame (full dim)."""
    # Load full-dim features from original extraction directory
    feat_file = clip_dir / "qwen3_patch_features" / f"{mid_offset:06d}_f.npy"
    if not feat_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_file}")
    return np.load(feat_file)


def load_mid_frame_image(clip_dir: Path, mid_offset: int) -> Image.Image:
    """Load middle frame image."""
    img_file = clip_dir / "images" / f"frame_{mid_offset:06d}.png"
    if not img_file.exists():
        raise FileNotFoundError(f"Image file not found: {img_file}")
    return Image.open(img_file)


def reconstruct_features(
    features: np.ndarray,
    autoencoder: QwenAutoencoder,
    device: torch.device,
) -> np.ndarray:
    """Reconstruct features using autoencoder (encode + decode)."""
    with torch.no_grad():
        feats_tensor = torch.from_numpy(features).float().to(device)
        # Encode then decode
        latents = autoencoder.encode(feats_tensor)
        reconstructed = autoencoder.decode(latents)
        return reconstructed.cpu().numpy()


def permute_features(features: np.ndarray, seed: int = PERMUTATION_SEED) -> np.ndarray:
    """Randomly permute feature tokens along the first axis.
    
    This ablation tests whether positional encodings matter for custom vision features.
    If the model relies on positional info, permuted features should produce worse/different outputs.
    """
    rng = np.random.default_rng(seed)
    perm_indices = rng.permutation(features.shape[0])
    return features[perm_indices]


def generate_markdown_report(
    clip_name: str,
    mid_frame: Image.Image,
    mid_offset: int,
    mid_frame_num: int,
    gt_answer: str,
    gt_permuted_answer: str,
    clip_ae_answer: str,
    global_ae_answer: str,
    output_path: Path,
):
    """Generate and save markdown report."""
    # Save image
    img_path = output_path / f"{clip_name}_midframe.png"
    mid_frame.save(img_path)

    # Generate markdown
    md_content = f"""# Autoencoder Comparison: {clip_name}

**Mid Frame:** Offset {mid_offset} (Frame {mid_frame_num})

# Mid Frame Image

![Mid Frame]({clip_name}_midframe.png)

---

# Ground Truth Features Answer

{gt_answer}

---

# Ground Truth Features (Randomly Permuted) Answer

*Ablation: tokens randomly shuffled to test if positional encoding matters*

{gt_permuted_answer}

---

# Clip-Specific AE Reconstructed Features Answer

{clip_ae_answer}

---

# Global AE Reconstructed Features Answer

{global_ae_answer}
"""

    # Save markdown
    md_path = output_path / f"{clip_name}_comparison.md"
    with open(md_path, "w") as f:
        f.write(md_content)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Qwen model and processor
    print("Loading Qwen model...")
    model, processor = get_patched_qwen(qwen_version=QWEN_VERSION)
    device = model.device
    print(f"Qwen model loaded on {device}")

    # Load global autoencoder
    global_ae_path = Path(PREPROCESSED_ROOT) / GLOBAL_AE_CHECKPOINT_DIR / "best_ckpt.pth"
    if not global_ae_path.exists():
        raise FileNotFoundError(f"Global AE checkpoint not found: {global_ae_path}")
    print(f"Loading global autoencoder from {global_ae_path}")
    global_ae = load_autoencoder(global_ae_path, device)

    # Process each clip
    for clip in tqdm(cfg.clips, desc="Processing clips"):
        clip_name = clip.name
        print(f"\n{'='*80}")
        print(f"Processing clip: {clip_name}")
        print(f"{'='*80}")

        # Get clip directory
        clip_dir = Path(PREPROCESSED_ROOT) / clip_name
        if not clip_dir.exists():
            print(f"Warning: Clip directory not found: {clip_dir}, skipping")
            continue

        # Get mid frame offset and actual frame number
        mid_offset, mid_frame_num = get_mid_frame_index(clip)
        print(f"Mid frame offset: {mid_offset}, actual frame number: {mid_frame_num}")

        # Load mid frame image
        try:
            mid_frame = load_mid_frame_image(clip_dir, mid_offset)
            print(f"Loaded mid frame: {mid_frame.size}")
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping clip")
            continue

        # Load GT features (full dim)
        try:
            gt_features = load_mid_frame_features(clip_dir, mid_offset)
            print(f"Loaded GT features: shape={gt_features.shape}")
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping clip")
            continue

        # Generate answer with GT features
        print("Generating answer with GT features...")
        gt_answer = ask_qwen_about_image_features(
            image_features=torch.from_numpy(gt_features),
            prompt=USER_PROMPT,
            model=model,
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            qwen_version=QWEN_VERSION,
        )
        print(f"GT answer: {gt_answer[:200]}...")

        # Generate answer with permuted GT features (ablation)
        print("Generating answer with permuted GT features...")
        gt_permuted = permute_features(gt_features)
        gt_permuted_answer = ask_qwen_about_image_features(
            image_features=torch.from_numpy(gt_permuted),
            prompt=USER_PROMPT,
            model=model,
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            qwen_version=QWEN_VERSION,
        )
        print(f"GT permuted answer: {gt_permuted_answer[:200]}...")

        # Load clip-specific autoencoder
        clip_ae_path = clip_dir / CLIP_AE_CHECKPOINT_SUBDIR / "best_ckpt.pth"
        if not clip_ae_path.exists():
            print(f"Warning: Clip-specific AE not found: {clip_ae_path}, skipping clip AE")
            clip_ae_answer = "N/A: Clip-specific AE not found"
        else:
            print(f"Loading clip-specific autoencoder from {clip_ae_path}")
            clip_ae = load_autoencoder(clip_ae_path, device)

            # Reconstruct with clip-specific AE (encode + decode on full features)
            clip_ae_reconstructed = reconstruct_features(gt_features, clip_ae, device)
            print(f"Clip-specific AE reconstructed features: shape={clip_ae_reconstructed.shape}")

            # Generate answer with clip-specific AE reconstructed features
            print("Generating answer with clip-specific AE reconstructed features...")
            clip_ae_answer = ask_qwen_about_image_features(
                image_features=torch.from_numpy(clip_ae_reconstructed),
                prompt=USER_PROMPT,
                model=model,
                processor=processor,
                system_prompt=SYSTEM_PROMPT,
                qwen_version=QWEN_VERSION,
            )
            print(f"Clip-specific AE answer: {clip_ae_answer[:200]}...")

        # Reconstruct with global AE (encode + decode on full features)
        global_ae_reconstructed = reconstruct_features(gt_features, global_ae, device)
        print(f"Global AE reconstructed features: shape={global_ae_reconstructed.shape}")

        # Generate answer with global AE reconstructed features
        print("Generating answer with global AE reconstructed features...")
        global_ae_answer = ask_qwen_about_image_features(
            image_features=torch.from_numpy(global_ae_reconstructed),
            prompt=USER_PROMPT,
            model=model,
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            qwen_version=QWEN_VERSION,
        )
        print(f"Global AE answer: {global_ae_answer[:200]}...")

        # Save markdown report
        generate_markdown_report(
            clip_name=clip_name,
            mid_frame=mid_frame,
            mid_offset=mid_offset,
            mid_frame_num=mid_frame_num,
            gt_answer=gt_answer,
            gt_permuted_answer=gt_permuted_answer,
            clip_ae_answer=clip_ae_answer,
            global_ae_answer=global_ae_answer,
            output_path=output_dir,
        )
        print(f"Saved report to {output_dir / f'{clip_name}_comparison.md'}")

    print(f"\n{'='*80}")
    print(f"All done! Results saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
