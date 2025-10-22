from pathlib import Path
from PIL import Image
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from qwen_vl import get_patched_qwen, qwen_encode_image, get_patch_segmasks


def extract_qwen_features(
    clip: DictConfig,
    cfg: DictConfig,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    clip_dir = Path(clip.dir)
    img_dir = clip_dir / "images"
    seg_dir = clip_dir / (
        cfg.preprocessing.instance_mask_subdir
        if cfg.feature_extraction.aggregate_with_instance_masks
        else cfg.preprocessing.semantic_mask_subdir
    )
    frame_stems = [f.stem for f in img_dir.glob("*.jpg")]

    patch_dir = clip_dir / cfg.feature_extraction.patch_feat_subdir
    instance_dir = clip_dir / cfg.feature_extraction.instance_feat_subdir
    patch_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    for frame_stem in frame_stems:
        image = Image.open(img_dir / f"{frame_stem}.jpg")
        frame_stem_just_number = frame_stem.replace("frame_", "")

        # patch wise
        qwen_feats = (
            qwen_encode_image(image, model, processor).detach().float().cpu().numpy()
        )
        patch_map = get_patch_segmasks(image.height, image.width).unsqueeze(0).numpy()
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
    # load qwen
    model, processor = get_patched_qwen(
        use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
        use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
    )

    for clip in tqdm(cfg.clips, desc="Generating qwen feats", unit="clip"):
        extract_qwen_features(clip, cfg, model, processor)


if __name__ == "__main__":
    main()
