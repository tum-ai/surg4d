from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import hydra
from hydra.core.global_hydra import GlobalHydra, OmegaConf
import subprocess
import shutil
from vipe import make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.io import ArtifactPath
from scripts.vipe_to_colmap_local import convert_vipe_to_colmap
import zipfile
from openexr_numpy import imread
from scipy.ndimage import label

from cholec_utils import get_clip_seg8k, parse_cholecseg8k_instance_mask


def center_crop_divisible(
    arr: np.ndarray, k: int, skip_last_dim: bool = False
) -> np.ndarray:
    """
    Center crop an array so that each dimension's shape is divisible by the specified number (with no remainder).

    Args:
        arr (np.ndarray): Input array of any shape (all dimensions must be >= k).
        k (int): The number that each dimension should be divisible by
        skip_last_dim (bool): Whether to skip the last dimension when cropping

    Returns:
        np.ndarray: Center cropped array with each dimension being divisible by 'k'
    """
    assert k > 0, "k must be > 0"
    assert (np.array(arr.shape[: -1 if skip_last_dim else None]) >= k).all(), (
        "All dimensions must be at least k"
    )

    crop_slices = []
    for dim_size in arr.shape[: -1 if skip_last_dim else None]:
        new_dim_size = (dim_size // k) * k

        total_crop = dim_size - new_dim_size
        start_crop = total_crop // 2
        end_crop = total_crop - start_crop

        crop_slices.append(slice(start_crop, dim_size - end_crop))

    cropped_arr = arr[tuple(crop_slices)]
    return cropped_arr


def estimate_crop_box(class_ids: np.ndarray):
    """Get the top, bottom, left, right indices to
    crop the black camera borders of this clip.

    Args:
        class_ids (np.ndarray): input class ids (any frame of the clip)

    Returns:
        top (int): top index to crop
        bottom (int): bottom index to crop
        left (int): left index to crop
        right (int): right index to crop
    """
    # crop top and bottom black pixel rows by using mid vertical
    mid_col = class_ids[:, class_ids.shape[1] // 2]
    vertical_keep = mid_col != 0
    top = vertical_keep.argmax()
    bottom = vertical_keep.size - np.flip(vertical_keep).argmax()

    # we need the vertically cropped class ids to find left and right
    class_ids = class_ids[top:bottom, :]

    # crop black camera borders horizontally by cutting of first row's background annotation
    horizontal_keep = class_ids[0] != 0
    assert horizontal_keep.sum() != 0
    left = horizontal_keep.argmax()
    right = horizontal_keep.size - np.flip(horizontal_keep).argmax()

    return top, bottom, left, right


def get_cholecseg8k_frames(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    out_rgb = clip_dir / "rgb"
    out_rgb.mkdir(parents=True, exist_ok=True)

    out_sem_masks = clip_dir / cfg.preprocessing.semantic_mask_subdir
    out_sem_masks.mkdir(parents=True, exist_ok=True)

    out_inst_masks = clip_dir / cfg.preprocessing.instance_mask_subdir
    out_inst_masks.mkdir(parents=True, exist_ok=True)

    frame_files, semantic_mask_files = get_clip_seg8k(
        seg8k_root=Path(cfg.cholecseg8k_root),
        seg8k_video_id=clip.video_id,
        first_frame=clip.first_frame,
        last_frame=clip.last_frame,
        frame_stride=clip.frame_stride,
    )

    top, bottom, left, right = estimate_crop_box(
        parse_cholecseg8k_instance_mask(Image.open(semantic_mask_files[0]))
    )

    for new_frame_id, (frame_file, semantic_mask_file) in enumerate(
        zip(frame_files, semantic_mask_files)
    ):
        rgb = Image.open(frame_file)
        instance_mask = Image.open(semantic_mask_file)

        class_ids = parse_cholecseg8k_instance_mask(instance_mask)

        rgb = np.asarray(rgb)[top:bottom, left:right]
        class_ids = class_ids[top:bottom, left:right]
        rgb = center_crop_divisible(
            rgb, cfg.preprocessing.frames_divisor, skip_last_dim=True
        )  # required for qwen encoder
        class_ids = center_crop_divisible(class_ids, cfg.preprocessing.frames_divisor)

        # Generate instance masks from semantic masks using connected components
        # 2d arrays, 0 is background, > 0 are instance ids
        instance_ids = np.zeros_like(class_ids, dtype=np.int32)
        instance_counter = 1
        unique_classes = np.unique(class_ids)
        unique_classes = unique_classes[unique_classes != 0]
        for class_id in unique_classes:
            class_binary = (class_ids == class_id).astype(np.uint8)
            labeled_components, num_components = label(class_binary)
            for component_id in range(1, num_components + 1):
                component_mask = labeled_components == component_id
                instance_ids[component_mask] = instance_counter
                instance_counter += 1

        Image.fromarray(rgb).save(out_rgb / f"frame_{new_frame_id:06d}.png")
        np.save(out_sem_masks / f"frame_{new_frame_id:06d}.npy", class_ids)
        np.save(out_inst_masks / f"frame_{new_frame_id:06d}.npy", instance_ids)


def vipe(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    image_dir = clip_dir / "rgb"
    assert image_dir.exists(), f"RGB frames directory not found: {image_dir}"

    overrides = [
        "pipeline=" + ("default" if cfg.preprocessing.vipe_vda else "no_vda"),
        f"pipeline.output.path={str(clip_dir.resolve())}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz="
        + ("true" if cfg.preprocessing.vipe_save_viz else "false"),
        "pipeline.output.save_slam_map="
        + ("true" if cfg.preprocessing.vipe_slam_map else "false"),
        "streams=frame_dir_stream",
        f"streams.base_path={str(image_dir.resolve())}",
    ]

    local_config_dir = Path(__file__).parent / "submodules" / "vipe" / "configs"
    with hydra.initialize_config_dir(
        config_dir=str(local_config_dir.resolve()), version_base=None
    ):
        args = hydra.compose("default", overrides=overrides)

    pipeline = make_pipeline(args.pipeline)
    video_stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(
        desc="Reading image frames"
    )

    pipeline.run(video_stream)
    GlobalHydra.instance().clear()


def vipe_to_colmap(
    clip: DictConfig,
    cfg: DictConfig,
):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    artifacts = list(ArtifactPath.glob_artifacts(clip_dir, use_video=True))
    for artifact in artifacts:
        convert_vipe_to_colmap(
            artifact=artifact,
            output_path=clip_dir,
            depth_step=cfg.preprocessing.vipe_depth_step,
            use_slam_map=cfg.preprocessing.vipe_slam_map,
            use_single_depth=cfg.preprocessing.pc_single_depth_projection,
        )


def colmap_txt_to_bin(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    output_dir = clip_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap",
        "model_converter",
        "--input_path",
        str(clip_dir.resolve()),
        "--output_path",
        str(output_dir.resolve()),
        "--output_type",
        "BIN",
    ]
    subprocess.run(cmd, check=True)


def extract_depth_maps(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    depth_dir = clip_dir / "depth"
    if not cfg.preprocessing.depth_subdir == "depth":
        shutil.move(depth_dir, clip_dir / cfg.preprocessing.depth_subdir)
        depth_dir = clip_dir / cfg.preprocessing.depth_subdir

    rgb_zip = depth_dir / "rgb.zip"
    assert rgb_zip.exists()

    # unzip zip created by vipe
    with zipfile.ZipFile(rgb_zip, "r") as zip_ref:
        zip_ref.extractall(depth_dir)
    rgb_zip.unlink()
    
    # rename to 6 digit format
    for exr_file in depth_dir.glob("*.exr"):
        frame_idx = int(exr_file.name[:5])
        exr_file.rename(exr_file.with_name(f"{frame_idx:06d}.exr"))

    # convert to np array
    for exr_file in depth_dir.glob("*.exr"):
        depth = imread(str(exr_file), "Z")
        np.save(exr_file.with_name(f"{exr_file.stem}.npy"), depth)


def delete_unused_files(clip: DictConfig, cfg: DictConfig):
    clip_dir = Path(cfg.preprocessed_root) / clip.name

    # vipe internals
    if (clip_dir / "intrinsics").exists():
        shutil.rmtree(clip_dir / "intrinsics")
    if (clip_dir / "mask").exists():
        shutil.rmtree(clip_dir / "mask")
    if (clip_dir / "pose").exists():
        shutil.rmtree(clip_dir / "pose")
    if (clip_dir / "vipe").exists():
        shutil.rmtree(clip_dir / "vipe")
    if (clip_dir / "vipe" / "rgb_info.pkl").exists():
        (clip_dir / "vipe" / "rgb_info.pkl").unlink()
    if (clip_dir / "vipe" / "rgb_slam_map.pt").exists():
        (clip_dir / "vipe" / "rgb_slam_map.pt").unlink()

    # vipe aux vis
    if (clip_dir / "vipe_aux_vis").exists():
        shutil.rmtree(clip_dir / "vipe_aux_vis")

    # colmap txt version
    if (clip_dir / "cameras.txt").exists():
        (clip_dir / "cameras.txt").unlink()
    if (clip_dir / "images.txt").exists():
        (clip_dir / "images.txt").unlink()
    if (clip_dir / "points3D.txt").exists():
        (clip_dir / "points3D.txt").unlink()

    # rgb dir becomes images dir
    if (clip_dir / "rgb").exists():
        shutil.rmtree(clip_dir / "rgb")


def process_clip(clip: DictConfig, cfg: DictConfig):
    get_cholecseg8k_frames(clip, cfg)
    vipe(clip, cfg)
    vipe_to_colmap(clip, cfg)
    colmap_txt_to_bin(clip, cfg)
    extract_depth_maps(clip, cfg)
    if not cfg.preprocessing.verbose_output:
        delete_unused_files(clip, cfg)


def main():
    # do hydra init manually here to avoid conflicts with vipe hydra
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        cfg = hydra.compose("config.yaml")
    
    # Clear after composing the main config so vipe can initialize its own
    GlobalHydra.instance().clear()
    
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        process_clip(clip, cfg)


if __name__ == "__main__":
    main()
