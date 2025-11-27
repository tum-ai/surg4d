from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
import random
from tqdm import tqdm
import shutil

from autoencoder.train_qwen import train
from autoencoder.model_qwen import QwenAutoencoder


def save_dim_reduced(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    ae = QwenAutoencoder(
        input_dim=cfg.autoencoder.full_dim, latent_dim=cfg.autoencoder.latent_dim
    ).to("cuda")
    ae.load_state_dict(
        torch.load(
            clip_dir / cfg.autoencoder.checkpoint_subdir / "best_ckpt.pth",
            map_location="cuda",
        )
    )
    ae.eval()

    patch_dir = clip_dir / cfg.autoencoder.patch_feat_subdir
    instance_dir = clip_dir / cfg.autoencoder.instance_feat_subdir
    reduced_patch_dir = clip_dir / cfg.autoencoder.latent_patch_feat_subdir
    reduced_instance_dir = clip_dir / cfg.autoencoder.latent_instance_feat_subdir
    reduced_cat_dir = clip_dir / cfg.autoencoder.latent_cat_feat_subdir

    for lf_dir, reduced_lf_dir in (
        (patch_dir, reduced_patch_dir),
        (instance_dir, reduced_instance_dir),
    ):
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
        assert (np.unique(s) == np.arange(np.unique(s).max() + 1)).all(), (
            "Segmentation masks don't contain clean indices from 0-n"
        )
        latent_mean_f = np.stack(
            [
                np.mean(
                    patch_lf[patch_s].reshape(-1, 3)[s.reshape(-1) == mask_id], axis=0
                )
                for mask_id in np.unique(s)
            ]
        )
        latent_mean_f /= np.linalg.norm(latent_mean_f, axis=-1, keepdims=True)
        np.save(reduced_instance_dir / lf_file.name, latent_mean_f)

    # concatenate reduced features for shared splat
    reduced_cat_dir.mkdir(parents=True, exist_ok=True)
    for sm_file in patch_dir.glob("*_s.npy"):
        patch_sm = np.load(sm_file)[0]  # remove leading dimension
        inst_sm = np.load(instance_dir / sm_file.name)[0]  # remove leading dimension
        paired_sm = np.stack((patch_sm.ravel(), inst_sm.ravel()), axis=-1)
        patch_lf = np.load(reduced_patch_dir / sm_file.name.replace("s", "f"))
        inst_lf = np.load(reduced_instance_dir / sm_file.name.replace("s", "f"))

        combs, indices = np.unique(paired_sm, axis=0, return_inverse=True)

        combined_feat = np.stack(
            [
                np.concatenate(
                    (patch_lf[patch_mask_id], inst_lf[instance_mask_id]), axis=-1
                )
                for patch_mask_id, instance_mask_id in combs
            ]
        )
        combined_sm = indices.reshape(patch_sm.shape)

        # add back leading dimension!
        np.save(reduced_cat_dir / sm_file.name, np.expand_dims(combined_sm, 0))
        np.save(reduced_cat_dir / sm_file.name.replace("s", "f"), combined_feat)


def train_ae(
    clip: DictConfig,
    cfg: DictConfig,
):
    train(
        clip_path=str(Path(cfg.preprocessed_root) / clip.name),
        checkpoint_subdir=cfg.autoencoder.checkpoint_subdir,
        lf_dir_names=[
            cfg.autoencoder.patch_feat_subdir,
            cfg.autoencoder.instance_feat_subdir,
        ],
        epochs=cfg.autoencoder.epochs,
        lr=cfg.autoencoder.lr,
        batch_size=cfg.autoencoder.batch_size,
        full_dim=cfg.autoencoder.full_dim,
        latent_dim=cfg.autoencoder.latent_dim,
    )
    save_dim_reduced(
        clip=clip,
        cfg=cfg,
    )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Deterministic Torch/CUDA setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for clip in tqdm(cfg.clips, desc="Training autoencoders", unit="clip"):
        train_ae(clip, cfg)


if __name__ == "__main__":
    main()
