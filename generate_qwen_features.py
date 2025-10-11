from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from qwen_vl import get_patched_qwen, qwen_encode_image, get_patch_segmasks


PREPROCESSED_ROOT = "data/cholecseg8k/preprocessed_ssg"


def extract_qwen_features(
    clip_dir: Path,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
):
    img_dir = clip_dir / "images"
    seg_dir = clip_dir / "instance_masks"
    frame_stems = [f.stem for f in img_dir.glob("*.jpg")]

    patch_dir = clip_dir / "qwen_patch_features"
    instance_dir = clip_dir / "qwen_instance_features"
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


def main():
    # cmdine args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker threads to use (default: 0 - serial in main process).",
    )
    args = parser.parse_args()
    assert args.workers >= 0, "Number of workers must be >= 0"

    # load qwen
    model, processor = get_patched_qwen(use_bnb_4bit=False)

    # generate list of task arguments
    tasks = []
    tasks_tmp = [] # ! tmp
    video_dirs = Path(PREPROCESSED_ROOT).glob("video[0-9][0-9]")
    for video_dir in video_dirs:
        clip_dirs = video_dir.glob("video[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]")
        for clip_dir in clip_dirs:
            tasks.append((video_dir, clip_dir))
            # if clip_dir.stem == "video27_00480": # ! tmp
            #     tasks_tmp.append((video_dir, clip_dir))
            # if clip_dir.stem == "video01_00080": # ! tmp
            #     tasks_tmp.append((video_dir, clip_dir))
            if clip_dir.stem == "video25_00402": # ! tmp
                tasks_tmp.append((video_dir, clip_dir))
    
    tasks = tasks_tmp # ! tmp

    # run in serial for debugging or threaded per default
    if args.workers == 0:
        for video_dir, clip_dir in tqdm(
            tasks, desc="Generating qwen feats", unit="clip"
        ):
            extract_qwen_features(clip_dir, model, processor)
    else:
        max_workers = args.workers if (args.workers and args.workers > 0) else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(extract_qwen_features, clip_dir, model, processor)
                for (_, clip_dir) in tasks
            ]
            with tqdm(
                total=len(futures), desc="Generating qwen feats", unit="clip"
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)


if __name__ == "__main__":
    main()
