from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

from qwen_vl_qwen3 import get_patched_qwen3, qwen3_encode_image


PREPROCESSED_ROOT = "data/cholecseg8k/preprocessed_ssg"


def extract_qwen3_features(clip_dir: Path, model, processor):
    img_dir = clip_dir / "images"
    seg_dir = clip_dir / "instance_masks"
    frame_stems = [f.stem for f in img_dir.glob("*.jpg")]

    patch_dir = clip_dir / "qwen3_patch_features"
    instance_dir = clip_dir / "qwen3_instance_features"
    patch_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    for frame_stem in frame_stems:
        image = Image.open(img_dir / f"{frame_stem}.jpg").convert("RGB")
        frame_stem_just_number = frame_stem.replace("frame_", "")

        # patch-wise
        feats = (
            qwen3_encode_image(image, model, processor).detach().float().cpu().numpy()
        )
        # placeholder patch map (int32 to allow -1 fill below if no mapping applied)
        patch_map = (np.load(seg_dir / f"{frame_stem}.npy").astype(np.int32) * 0)
        np.save(patch_dir / f"{frame_stem_just_number}_f.npy", feats)
        np.save(patch_dir / f"{frame_stem_just_number}_s.npy", patch_map)

        # instance-wise (if masks exist)
        if (seg_dir / f"{frame_stem}.npy").exists():
            seg = np.load(seg_dir / f"{frame_stem}.npy")
            instance_map = np.full(patch_map.shape, -1, dtype=np.int32)
            instance_ids = np.unique(seg)
            instance_feats_out = []
            for i, inst_id in enumerate(instance_ids):
                instance_mask = seg == inst_id
                instance_map[instance_mask] = i

                # Map pixel-level mask to patch indices (approximate via nearest)
                # Users can replace this with their exact mapping
                indices, counts = np.unique(
                    instance_mask.reshape(-1), return_counts=True
                )
                # If no fine mapping, just average all patch features
                mean_feat = feats.mean(axis=0)
                instance_feats_out.append(mean_feat)

            instance_feats_out = np.stack(instance_feats_out)
            np.save(
                instance_dir / f"{frame_stem_just_number}_f.npy", instance_feats_out
            )
            np.save(instance_dir / f"{frame_stem_just_number}_s.npy", instance_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bnb4", action="store_true")
    parser.add_argument("--bnb8", action="store_true")
    args = parser.parse_args()

    model, processor = get_patched_qwen3(
        model_path=args.model_path,
        use_bnb_4bit=args.bnb4,
        use_bnb_8bit=args.bnb8,
    )

    tasks = []
    video_dirs = Path(PREPROCESSED_ROOT).glob("video[0-9][0-9]")
    for video_dir in video_dirs:
        clip_dirs = video_dir.glob("video[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]")
        for clip_dir in clip_dirs:
            if clip_dir.name == "video27_00480":
                tasks.append((video_dir, clip_dir))

    if args.workers == 0:
        for _, clip_dir in tqdm(tasks, desc="Generating qwen3 feats", unit="clip"):
            extract_qwen3_features(clip_dir, model, processor)
    else:
        max_workers = args.workers if (args.workers and args.workers > 0) else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(extract_qwen3_features, clip_dir, model, processor)
                for (_, clip_dir) in tasks
            ]
            with tqdm(
                total=len(futures), desc="Generating qwen3 feats", unit="clip"
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)


if __name__ == "__main__":
    main()
