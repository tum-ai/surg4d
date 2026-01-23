from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import random
from tqdm import tqdm
import shutil

from autoencoder.train_qwen import train
from autoencoder.model_qwen import QwenAutoencoder


def save_dim_reduced(clip: DictConfig, cfg: DictConfig, ae: QwenAutoencoder = None):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    
    # Load autoencoder if not provided (per-clip mode)
    if ae is None:
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


def train_global_ae(clips, cfg: DictConfig):
    """Train a single global autoencoder on features from all clips."""
    preprocessed_root = Path(cfg.preprocessed_root)
    checkpoint_dir = preprocessed_root / cfg.autoencoder.global_checkpoint_dir

    # Collect all feature directories from all clips
    all_data_dirs = []
    for clip in clips:
        clip_dir = preprocessed_root / clip.name
        all_data_dirs.extend([
            clip_dir / cfg.autoencoder.patch_feat_subdir,
            clip_dir / cfg.autoencoder.instance_feat_subdir,
        ])

    # Train on all features from all clips
    train(
        data_dirs=all_data_dirs,
        checkpoint_dir=checkpoint_dir,
        epochs=cfg.autoencoder.epochs,
        lr=cfg.autoencoder.lr,
        batch_size=cfg.autoencoder.batch_size,
        full_dim=cfg.autoencoder.full_dim,
        latent_dim=cfg.autoencoder.latent_dim,
    )


def train_ae(
    clip: DictConfig,
    cfg: DictConfig,
):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    data_dirs = [
        clip_dir / cfg.autoencoder.patch_feat_subdir,
        clip_dir / cfg.autoencoder.instance_feat_subdir,
    ]
    checkpoint_dir = clip_dir / cfg.autoencoder.checkpoint_subdir
    
    train(
        data_dirs=data_dirs,
        checkpoint_dir=checkpoint_dir,
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

    if cfg.autoencoder.global_mode:
        # Global mode: train one autoencoder on all clips
        train_global_ae(cfg.clips, cfg)
        
        # Load global autoencoder once
        global_checkpoint_path = Path(cfg.preprocessed_root) / cfg.autoencoder.global_checkpoint_dir / "best_ckpt.pth"
        global_ae = QwenAutoencoder(
            input_dim=cfg.autoencoder.full_dim, latent_dim=cfg.autoencoder.latent_dim
        ).to("cuda")
        global_ae.load_state_dict(torch.load(global_checkpoint_path, map_location="cuda"))
        global_ae.eval()
        
        # Apply global autoencoder to each clip
        for clip in tqdm(cfg.clips, desc="Encoding with global AE", unit="clip"):
            save_dim_reduced(clip, cfg, ae=global_ae)
    else:
        # Per-clip mode: train separate autoencoder for each clip
        for clip in tqdm(cfg.clips, desc="Training autoencoders", unit="clip"):
            train_ae(clip, cfg)


if __name__ == "__main__":
    main()
