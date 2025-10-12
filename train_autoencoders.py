from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import shutil

from autoencoder.train_qwen import train
from autoencoder.model_qwen import QwenAutoencoder

PREPROCESSED_ROOT = "data/cholecseg8k/preprocessed_ssg"


def save_dim_reduced(clip_dir: Path, latent_dim: int):
    ae = QwenAutoencoder(input_dim=3584, latent_dim=latent_dim).to("cuda")
    ae.load_state_dict(torch.load(clip_dir / "autoencoder" / "best_ckpt.pth", map_location="cuda"))
    ae.eval()

    patch_dir = clip_dir / "qwen_patch_features"
    instance_dir = clip_dir / "qwen_instance_features"
    reduced_patch_dir = clip_dir / f"qwen_patch_features_dim{latent_dim}"
    reduced_instance_dir = clip_dir / f"qwen_instance_features_dim{latent_dim}"
    reduced_cat_dir = clip_dir / f"qwen_cat_features_dim{latent_dim*2}"

    for lf_dir, reduced_lf_dir in (patch_dir, reduced_patch_dir), (instance_dir, reduced_instance_dir):
        reduced_lf_dir.mkdir(parents=True, exist_ok=True)

        # just copy over segmentation maps
        for seg_map_file in lf_dir.glob("*_s.npy"):
            shutil.copy(seg_map_file, reduced_lf_dir / seg_map_file.name)
    


    # encode features
    for lf_file in patch_dir.glob("*_f.npy"):
        patch_lf = np.load(lf_file)
        with torch.no_grad():
            patch_lf = (
                ae.encode(torch.as_tensor(patch_lf, device="cuda", dtype=torch.float32))
                .detach()
                .cpu()
                .numpy()
            )
        np.save(reduced_patch_dir / lf_file.name, patch_lf)
        
        # aggregate and normalize features in latent space
        patch_s = np.load(patch_dir / lf_file.name.replace("f", "s"))[0]
        s = np.load(instance_dir / lf_file.name.replace("f", "s"))[0]
        assert (np.unique(s) == np.arange(np.unique(s).max()+1)).all(), "Segmentation masks don't contain clean indices from 0-n"
        latent_mean_f = np.stack([
            np.mean(patch_lf[patch_s].reshape(-1, 3)[s.reshape(-1) == mask_id], axis=0)
            for mask_id in np.unique(s)
        ])
        latent_mean_f /= np.linalg.norm(latent_mean_f, axis=-1, keepdims=True)
        np.save(reduced_instance_dir / lf_file.name, latent_mean_f)

    # concatenate reduced features for shared splat
    reduced_cat_dir.mkdir(parents=True, exist_ok=True)
    for sm_file in patch_dir.glob("*_s.npy"):
        patch_sm = np.load(sm_file)[0] # remove leading dimension
        inst_sm = np.load(instance_dir / sm_file.name)[0] # remove leading dimension
        paired_sm = np.stack((patch_sm.ravel(), inst_sm.ravel()), axis=-1)
        patch_lf = np.load(reduced_patch_dir / sm_file.name.replace("s", "f"))
        inst_lf = np.load(reduced_instance_dir / sm_file.name.replace("s", "f"))

        combs, indices = np.unique(paired_sm, axis=0, return_inverse=True)

        combined_feat = np.stack([
            np.concatenate((patch_lf[patch_mask_id], inst_lf[instance_mask_id]), axis=-1)
            for patch_mask_id, instance_mask_id in combs
        ])
        combined_sm = indices.reshape(patch_sm.shape)

        # add back leading dimension!
        np.save(reduced_cat_dir / sm_file.name, np.expand_dims(combined_sm, 0))
        np.save(reduced_cat_dir / sm_file.name.replace("s", "f"), combined_feat)

def train_ae(
    clip_dir: Path,
):
    train(
        clip_path=clip_dir,
        lf_dir_names=["qwen_patch_features", "qwen_instance_features"],
        latent_dim=3,
    )
    save_dim_reduced(
        clip_dir=clip_dir,
        latent_dim=3,
    )


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

    # generate list of task arguments
    tasks = []
    tasks_tmp = [] # ! tmp
    video_dirs = Path(PREPROCESSED_ROOT).glob("video[0-9][0-9]")
    for video_dir in video_dirs:
        clip_dirs = video_dir.glob("video[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]")
        for clip_dir in clip_dirs:
            tasks.append((video_dir, clip_dir))
            if clip_dir.stem == "video27_00480": # ! tmp
                tasks_tmp.append((video_dir, clip_dir))
            # if clip_dir.stem == "video01_00080": # ! tmp
            #     tasks_tmp.append((video_dir, clip_dir))
            # if clip_dir.stem == "video25_00402": # ! tmp
            #     tasks_tmp.append((video_dir, clip_dir))
    
    tasks = tasks_tmp # ! tmp

    # run in serial for debugging or threaded per default
    if args.workers == 0:
        for video_dir, clip_dir in tqdm(
            tasks, desc="Training autoencoders", unit="clip"
        ):
            train_ae(clip_dir)
    else:
        max_workers = args.workers if (args.workers and args.workers > 0) else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(train_ae, clip_dir) for (_, clip_dir) in tasks]
            with tqdm(
                total=len(futures), desc="Training autoencoders", unit="clip"
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)


if __name__ == "__main__":
    main()
