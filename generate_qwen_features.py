from pathlib import Path
from PIL import Image
import hydra
import random
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from llm.qwen_utils import get_patched_qwen, qwen_encode_image, get_patch_segmasks


def extract_qwen_features(
    clip: DictConfig,
    cfg: DictConfig,
    model,
    processor,
):
    qwen_version = cfg.feature_extraction.qwen_version

    clip_dir = Path(cfg.preprocessed_root) / clip.name
    img_dir = clip_dir / "images"
    seg_dir = clip_dir / (
        cfg.feature_extraction.instance_mask_subdir
        if cfg.feature_extraction.aggregate_with_instance_masks
        else cfg.feature_extraction.semantic_mask_subdir
    )
    # Support both JPG and PNG images
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    frame_data = [(f.stem, f.suffix) for f in image_files]

    patch_dir = clip_dir / cfg.feature_extraction.patch_feat_subdir
    instance_dir = clip_dir / cfg.feature_extraction.instance_feat_subdir
    patch_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    for frame_stem, ext in frame_data:
        image = Image.open(img_dir / f"{frame_stem}{ext}")
        frame_stem_just_number = frame_stem.replace("frame_", "")

        feats = qwen_encode_image(image, model, processor, qwen_version)
        qwen_feats = feats.detach().float().cpu().numpy()
        patch_map = get_patch_segmasks(image.height, image.width, qwen_version).unsqueeze(0).numpy()

        np.save(patch_dir / f"{frame_stem_just_number}_f.npy", qwen_feats)
        np.save(patch_dir / f"{frame_stem_just_number}_s.npy", patch_map)

        # instance wise
        seg = np.load(seg_dir / f"{frame_stem}.npy")
        instance_map = np.full_like(patch_map, -1)
        instance_ids = np.unique(seg)
        instance_qwen_feats = []
        for i, id in enumerate(instance_ids):
            instance_mask = seg == id

            # consecutive 0-indexed instance id for _s file
            instance_map[0, instance_mask] = i

            # instance feature is mean
            instance_feat_indices, instance_feat_counts = np.unique(
                patch_map[0, instance_mask], return_counts=True
            )
            instance_feats = qwen_feats[instance_feat_indices.astype(np.int32)]
            mean_feat = (instance_feats * instance_feat_counts[:, None]).sum(
                axis=0
            ) / instance_feat_counts.sum()
            instance_qwen_feats.append(mean_feat)

        instance_qwen_feats = np.stack(instance_qwen_feats)

        np.save(instance_dir / f"{frame_stem_just_number}_f.npy", instance_qwen_feats)
        np.save(instance_dir / f"{frame_stem_just_number}_s.npy", instance_map)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model, processor = get_patched_qwen(
        qwen_version=cfg.feature_extraction.qwen_version,
        use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
        use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
    )

    for clip in tqdm(cfg.clips, desc=f"Generating {cfg.feature_extraction.qwen_version} feats", unit="clip"):
        extract_qwen_features(clip, cfg, model, processor)


if __name__ == "__main__":
    main()
