from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from loguru import logger
import re
from depth_anything_3.api import DepthAnything3

from utils.da3_utils import filter_prediction_edge_artifacts


def extract_frame_number(filepath: Path) -> int:
    """Extract frame number from filename for proper numerical sorting."""
    match = re.search(r"frame_(\d+)", filepath.stem)
    if match:
        return int(match.group(1))
    return 0


def extract_geometry(clip: DictConfig, cfg: DictConfig, model: DepthAnything3):
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    images_dir = clip_dir / cfg.extract_geometry.image_subdir

    # construct image paths and determine processing resolution close to orig
    image_filenames = sorted(
        list(images_dir.glob("*.png")), key=extract_frame_number
    )
    image_filenames = [str(img_file) for img_file in image_filenames]
    orig_w, orig_h = Image.open(image_filenames[0]).size
    processing_res = max(orig_w, orig_h) # by setting this + upper bound resize we ensure processing res is equal to original image size, since we already make it divisible by da3 vit patch size in preprocess step

    # da3 inference
    prediction = model.inference(
        image=image_filenames,
        ref_view_strategy="middle",  # good for video according to docs
        process_res=processing_res,
        process_res_method="upper_bound_resize",
    )

    # Apply edge filtering to prediction at processed resolution (if configured)
    edge_gradient_threshold = cfg.extract_geometry.da3_edge_gradient_threshold
    if edge_gradient_threshold is not None:
        logger.info(f"Applying depth edge filtering with threshold: {edge_gradient_threshold}")
        prediction = filter_prediction_edge_artifacts(
            prediction,
            gradient_threshold=edge_gradient_threshold,
        )

    # export manually since we might apply depth edge filtering
    # export format matches da3 format mini_npz
    export_dir = clip_dir / cfg.extract_geometry.da3_subdir
    export_dir.mkdir(parents=True, exist_ok=True)
    export_dict = {
        "depth": prediction.depth, # (T, H, W)
        "conf": prediction.conf, # (T, H, W)
        "intrinsics": prediction.intrinsics, # (T, 3, 3)
        "extrinsics": prediction.extrinsics # (T, 3, 4)
    }

    np.savez(export_dir / "results.npz", **export_dict)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(cfg.preprocessed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to("cuda:0")

    for clip in tqdm(cfg.clips, desc="Processing clips", unit="clip"):
        extract_geometry(clip, cfg, model)


if __name__ == "__main__":
    main()
