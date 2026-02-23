from pathlib import Path
import random
import shutil
import subprocess
import sys

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from utils.cholec_utils import get_clip_seg8k, seg8k_endo_watershed_to_class_ids


REPO_ROOT = Path(__file__).resolve().parent
SASVI_ROOT = REPO_ROOT / "submodules" / "SASVi"
if str(SASVI_ROOT) not in sys.path:
    sys.path.insert(0, str(SASVI_ROOT))

from train_scripts.train_mask2former_cholecseg import train as train_mask2former_cholecseg
from train_scripts.train_maskrcnn_cholecseg import train as train_maskrcnn_cholecseg
from train_scripts.train_DETR_cholecseg import train as train_detr_cholecseg


def _as_absolute(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _collect_cholecseg8k_clip_dirs(cholecseg8k_root: Path) -> list[Path]:
    clip_dirs = []
    for video_dir in sorted(cholecseg8k_root.glob("video*")):
        if not video_dir.is_dir():
            continue
        for clip_dir in sorted(video_dir.glob("video*_*")):
            if clip_dir.is_dir():
                clip_dirs.append(clip_dir)
    return clip_dirs


def _build_training_root(cfg: DictConfig):
    source_root = Path(cfg.cholecseg8k_root)
    train_root = _as_absolute(cfg.segment.train_data_root)
    excluded_clip_names = {clip.name for clip in cfg.clips}

    train_root.mkdir(parents=True, exist_ok=True)

    all_clip_dirs = _collect_cholecseg8k_clip_dirs(source_root)
    included_clip_dirs = [
        clip_dir for clip_dir in all_clip_dirs if clip_dir.name not in excluded_clip_names
    ]

    for clip_dir in tqdm(included_clip_dirs, desc="Preparing segment training clips", unit="clip"):
        rel_parent = clip_dir.parent.relative_to(source_root)
        target_clip_dir = train_root / rel_parent / clip_dir.name
        target_clip_dir.mkdir(parents=True, exist_ok=True)

        source_files = [path for path in clip_dir.iterdir() if path.is_file()]
        for source_file in tqdm(source_files, desc=f"Copying files for {clip_dir.name}", unit="file", leave=False):
            target_file = target_clip_dir / source_file.name
            if target_file.exists():
                continue
            shutil.copy2(source_file, target_file)

        watershed_files = sorted(clip_dir.glob("frame_*_endo_watershed_mask.png"))
        for watershed_file in tqdm(watershed_files, desc=f"Generating id masks for {clip_dir.name}", unit="mask", leave=False):
            frame_id = watershed_file.stem.split("_")[1]
            id_mask_file = target_clip_dir / f"frame_{frame_id}_endo_id_mask.png"
            if id_mask_file.exists():
                continue

            class_ids = seg8k_endo_watershed_to_class_ids(Image.open(watershed_file))
            Image.fromarray(class_ids.astype(np.uint8)).save(id_mask_file)


def _sync_tree_missing_files(source_root: Path, target_root: Path):
    target_root.mkdir(parents=True, exist_ok=True)

    source_files = [path for path in source_root.rglob("*") if path.is_file()]
    for source_file in tqdm(source_files, desc=f"Syncing files to {target_root}", unit="file"):
        rel_path = source_file.relative_to(source_root)
        target_file = target_root / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists():
            continue
        shutil.copy2(source_file, target_file)


def _resolve_training_data_dir(cfg: DictConfig) -> Path:
    tmp_train_root = _as_absolute(cfg.segment.train_data_root)
    if not cfg.segment.data_staging.use_ram:
        return tmp_train_root

    ram_train_root = Path(cfg.segment.data_staging.ram_train_data_root)
    _sync_tree_missing_files(tmp_train_root, ram_train_root)
    return ram_train_root


def _latest_experiment_dir(log_dir: Path) -> Path:
    candidates = [d for d in log_dir.iterdir() if d.is_dir()]
    candidates.sort(key=lambda d: d.stat().st_mtime)
    return candidates[-1]


def _resolve_checkpoint(cfg: DictConfig) -> Path:
    if cfg.segment.run_training:
        checkpoint_path = _latest_experiment_dir(Path(cfg.segment.log_dir)) / cfg.segment.checkpoint_filename
        checkpoint_path = _as_absolute(str(checkpoint_path))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resolved training checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoint_path = Path(cfg.segment.checkpoint_path)
    if "latest" in checkpoint_path.parts or "[latest]" in checkpoint_path.parts:
        latest_dir = _latest_experiment_dir(Path(cfg.segment.log_dir))
        latest_token = "latest" if "latest" in checkpoint_path.parts else "[latest]"
        suffix_after_latest = checkpoint_path.parts[checkpoint_path.parts.index(latest_token) + 1 :]
        checkpoint_path = latest_dir / Path(*suffix_after_latest)

    checkpoint_path = _as_absolute(str(checkpoint_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resolved inference checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def _train_overseer(cfg: DictConfig):
    _build_training_root(cfg)
    training_data_dir = _resolve_training_data_dir(cfg)

    common_kwargs = dict(
        epochs=cfg.segment.train.epochs,
        steps=cfg.segment.train.steps,
        val_freq=cfg.segment.train.val_freq,
        batch_size=cfg.segment.train.batch_size,
        num_workers=cfg.segment.train.num_workers,
        weighted_sampling=cfg.segment.train.weighted_sampling,
        initial_lr=cfg.segment.train.initial_lr,
        betas=tuple(cfg.segment.train.betas),
        weight_decay=cfg.segment.train.weight_decay,
        scheduler_step_size=cfg.segment.train.scheduler_step_size,
        scheduler_gamma=cfg.segment.train.scheduler_gamma,
        img_size=tuple(cfg.segment.train.img_size),
        img_norm=tuple(cfg.segment.train.img_norm),
        ignore_ids=list(cfg.segment.train.ignore_ids),
        shift_ids_by_1=cfg.segment.train.shift_ids_by_1,
        components=cfg.segment.train.components,
        min_comp_fraction=cfg.segment.train.min_comp_fraction,
        data_dir=training_data_dir,
        log_dir=Path(cfg.segment.log_dir),
        device=cfg.segment.device,
    )

    if cfg.segment.sasvi.overseer_type == "Mask2Former":
        train_mask2former_cholecseg(
            **common_kwargs,
            weighted_loss=cfg.segment.train.weighted_loss,
            backbone=cfg.segment.backbone,
            num_queries=cfg.segment.num_queries,
        )
    elif cfg.segment.sasvi.overseer_type == "MaskRCNN":
        train_maskrcnn_cholecseg(
            **common_kwargs,
            hidden_ft=cfg.segment.hidden_ft,
            backbone=cfg.segment.backbone,
            trainable_backbone_layers=cfg.segment.trainable_backbone_layers,
        )
    elif cfg.segment.sasvi.overseer_type == "DETR":
        train_detr_cholecseg(
            **common_kwargs,
            weighted_loss=cfg.segment.train.weighted_loss,
            backbone=cfg.segment.backbone,
            num_queries=cfg.segment.num_queries,
        )
    else:
        raise ValueError(f"Unsupported overseer type for training: {cfg.segment.sasvi.overseer_type}")


def _prepare_sasvi_base_video_dir(cfg: DictConfig):
    base_video_dir = _as_absolute(cfg.segment.sasvi.base_video_dir)
    if base_video_dir.exists():
        shutil.rmtree(base_video_dir)
    base_video_dir.mkdir(parents=True, exist_ok=False)

    for clip in cfg.clips:
        frame_files, _, _ = get_clip_seg8k(
            seg8k_root=Path(cfg.cholecseg8k_root),
            seg8k_video_id=clip.video_id,
            first_frame=clip.first_frame,
            last_frame=clip.last_frame,
            frame_stride=clip.frame_stride,
        )

        clip_video_dir = base_video_dir / clip.name
        clip_video_dir.mkdir(parents=True, exist_ok=False)
        for frame_idx, frame_file in enumerate(frame_files):
            rgb = Image.open(frame_file).convert("RGB")
            rgb.save(clip_video_dir / f"{frame_idx:06d}.jpg", format="JPEG", quality=95)


def _run_sasvi_inference(cfg: DictConfig, checkpoint_path: Path):
    output_mask_dir = _as_absolute(cfg.segment.sasvi.output_mask_dir)
    if output_mask_dir.exists():
        shutil.rmtree(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=False)

    sam2_checkpoint = _as_absolute(cfg.segment.sasvi.sam2_checkpoint)
    base_video_dir = _as_absolute(cfg.segment.sasvi.base_video_dir)

    command = [
        sys.executable,
        "eval_sasvi.py",
        "--device",
        cfg.segment.device,
        "--sam2_cfg",
        cfg.segment.sasvi.sam2_cfg,
        "--sam2_checkpoint",
        str(sam2_checkpoint),
        "--overseer_checkpoint",
        str(checkpoint_path),
        "--overseer_type",
        cfg.segment.sasvi.overseer_type,
        "--dataset_type",
        cfg.segment.sasvi.dataset_type,
        "--base_video_dir",
        str(base_video_dir),
        "--output_mask_dir",
        str(output_mask_dir),
        "--score_thresh",
        str(cfg.segment.sasvi.score_thresh),
    ]

    if cfg.segment.sasvi.apply_postprocessing:
        command.append("--apply_postprocessing")
    if cfg.segment.sasvi.save_binary_mask:
        command.append("--save_binary_mask")
    if cfg.segment.sasvi.dump_overseer_masks:
        overseer_mask_dir = _as_absolute(cfg.segment.sasvi.overseer_mask_dir)
        if overseer_mask_dir.exists():
            shutil.rmtree(overseer_mask_dir)
        overseer_mask_dir.mkdir(parents=True, exist_ok=False)
        command.extend(["--overseer_mask_dir", str(overseer_mask_dir)])

    subprocess.run(
        command,
        check=True,
        cwd=SASVI_ROOT / "src" / "sam2",
    )


def _convert_sasvi_outputs_to_numpy_masks(cfg: DictConfig):
    output_mask_dir = _as_absolute(cfg.segment.sasvi.output_mask_dir)

    for clip in cfg.clips:
        frame_files, _, _ = get_clip_seg8k(
            seg8k_root=Path(cfg.cholecseg8k_root),
            seg8k_video_id=clip.video_id,
            first_frame=clip.first_frame,
            last_frame=clip.last_frame,
            frame_stride=clip.frame_stride,
        )

        sasvi_clip_dir = output_mask_dir / clip.name
        preprocessed_sem_dir = Path(cfg.preprocessed_root) / clip.name / cfg.segment.prediction_subdir
        if preprocessed_sem_dir.exists():
            shutil.rmtree(preprocessed_sem_dir)
        preprocessed_sem_dir.mkdir(parents=True, exist_ok=False)

        for frame_idx in range(len(frame_files)):
            sasvi_mask_path = sasvi_clip_dir / f"{frame_idx:06d}_rgb_mask.png"
            semantic_mask = np.asarray(Image.open(sasvi_mask_path), dtype=np.uint8)
            if semantic_mask.ndim == 3:
                semantic_mask = semantic_mask[:, :, 0]

            orig_w, orig_h = Image.open(frame_files[frame_idx]).size
            if semantic_mask.shape != (orig_h, orig_w):
                semantic_mask = cv2.resize(
                    semantic_mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            np.save(preprocessed_sem_dir / f"frame_{frame_idx:06d}.npy", semantic_mask)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    Path(cfg.preprocessed_root).mkdir(parents=True, exist_ok=True)
    Path(cfg.segment.log_dir).mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    if cfg.segment.run_training:
        _train_overseer(cfg)

    if cfg.segment.run_inference:
        checkpoint_path = _resolve_checkpoint(cfg)
        _prepare_sasvi_base_video_dir(cfg)
        _run_sasvi_inference(cfg, checkpoint_path)
        _convert_sasvi_outputs_to_numpy_masks(cfg)


if __name__ == "__main__":
    main()
