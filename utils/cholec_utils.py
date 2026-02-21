from pathlib import Path
from PIL import Image
import numpy as np


def get_clip_seg8k(
    seg8k_root: Path,
    seg8k_video_id: int,
    first_frame: int,
    last_frame: int,
    frame_stride: int,
):
    """Get clip from CholecSeg8K dataset

    Args:
        seg8k_root (Path): Root directory of CholecSeg8K dataset
        seg8k_video_id (int): Video ID (constant over cholec80 derivatives)
        first_frame (int): First frame of the clip (inclusive)
        last_frame (int): Last frame of the clip (exclusive)
        frame_stride (int): Frame stride

    Returns:
        List[Path]: Sorted list of frame files
        List[Path]: Sorted list of semantic mask files
    """
    vid_dir = seg8k_root / f"video{seg8k_video_id:02d}"
    if not vid_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {vid_dir}")

    clip_dir = vid_dir / f"video{seg8k_video_id:02d}_{first_frame:05d}"
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")

    all_frames = {
        int(i.stem.split("_")[1]): i for i in clip_dir.glob("frame_*_endo.png")
    }
    all_semantic_masks = {
        int(i.stem.split("_")[1]): i
        for i in clip_dir.glob("frame_*_endo_watershed_mask.png")
    }

    if last_frame - 1 > max(all_frames.keys()):
        raise FileNotFoundError(
            f"Last frame {last_frame - 1} not found in clip directory: {clip_dir}"
        )

    frame_files = [all_frames[i] for i in range(first_frame, last_frame, frame_stride)]
    semantic_mask_files = [
        all_semantic_masks[i] for i in range(first_frame, last_frame, frame_stride)
    ]

    return frame_files, semantic_mask_files


def get_clip_t50(
    t50_root: Path,
    t50_video_id: int,
    first_frame: int,
    last_frame: int,
    frame_stride: int,
):
    """Get clip from CholecT50 dataset

    Args:
        t50_root (Path): Root directory of CholecT50 dataset
        t50_video_id (int): Video ID (constant over cholec80 derivatives)
        first_frame (int): First frame of the clip (inclusive)
        last_frame (int): Last frame of the clip (exclusive)
        frame_stride (int): Frame stride

    Returns:
        _type_: _description_
    """
    vid_dir = t50_root / "videos" / f"VID{t50_video_id:02d}"
    if not vid_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {vid_dir}")

    all_frames = {int(i.stem): i for i in vid_dir.glob("*.png")}

    if first_frame < min(all_frames.keys()):
        raise FileNotFoundError(
            f"First frame {first_frame} not found in video directory: {vid_dir}"
        )

    if last_frame - 1 > max(all_frames.keys()):
        raise FileNotFoundError(
            f"Last frame {last_frame - 1} not found in video directory: {vid_dir}"
        )

    frame_files = [all_frames[i] for i in range(first_frame, last_frame, frame_stride)]

    return frame_files


def parse_cholecseg8k_instance_mask(instance_mask: Image.Image):
    """convert instance watershed mask to zero indexed class ids numpy array"""
    rgb_to_class = {  # taken from https://www.kaggle.com/datasets/newslab/cholecseg8k
        255: {  # they are not defined but present, just map to background as well
            "class_id": 0,
            "class_name": "Black Background",
        },
        50: {
            "class_id": 0,
            "class_name": "Black Background",
        },
        11: {
            "class_id": 1,
            "class_name": "Abdominal Wall",
        },
        21: {
            "class_id": 2,
            "class_name": "Liver",
        },
        13: {
            "class_id": 3,
            "class_name": "Gastrointestinal Tract",
        },
        12: {
            "class_id": 4,
            "class_name": "Fat",
        },
        31: {
            "class_id": 5,
            "class_name": "Grasper",
        },
        23: {
            "class_id": 6,
            "class_name": "Connective Tissue",
        },
        24: {
            "class_id": 7,
            "class_name": "Blood",
        },
        25: {
            "class_id": 8,
            "class_name": "Cystic Duct",
        },
        32: {
            "class_id": 9,
            "class_name": "L-hook Electrocautery",
        },
        22: {
            "class_id": 10,
            "class_name": "Gallbladder",
        },
        33: {
            "class_id": 11,
            "class_name": "Hepatic Vein",
        },
        5: {
            "class_id": 12,
            "class_name": "Liver Ligament",
        },
    }
    arr = np.asarray(instance_mask)
    single_channel = arr[:, :, 0]  # the rgb vals just repeat for each id mapping
    class_ids = np.empty(shape=single_channel.shape, dtype=np.uint8)
    test = np.zeros(shape=single_channel.shape, dtype=np.uint8)
    for rgb_val, class_info in rgb_to_class.items():
        class_ids[single_channel == rgb_val] = class_info["class_id"]
        test[single_channel == rgb_val] = 1

    return class_ids


def get_semantic_label_from_class_id(class_id: int) -> str:
    """Get semantic label string from class ID.
    
    Args:
        class_id: Semantic class ID (0-12)
    
    Returns:
        Semantic label string
    """
    class_id_to_name = {
        0: "Black Background",
        1: "Abdominal Wall",
        2: "Liver",
        3: "Gastrointestinal Tract",
        4: "Fat",
        5: "Grasper",
        6: "Connective Tissue",
        7: "Blood",
        8: "Cystic Duct",
        9: "L-hook Electrocautery",
        10: "Gallbladder",
        11: "Hepatic Vein",
        12: "Liver Ligament",
    }
    return class_id_to_name.get(class_id, f"Unknown-{class_id}")
