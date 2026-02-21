from pathlib import Path
from typing import List, Tuple

import numpy as np


def homogenize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[1:] == (3, 4):
        extrinsics_h = np.tile(np.eye(4, dtype=np.float32), (extrinsics.shape[0], 1, 1))
        extrinsics_h[:, :3, :4] = extrinsics.astype(np.float32)
        return extrinsics_h
    if extrinsics.shape[1:] == (4, 4):
        return extrinsics.astype(np.float32)
    raise AssertionError(
        f"Expected extrinsics shape (T,3,4) or (T,4,4), got {extrinsics.shape}"
    )


def load_da3_geometry(
    geometry_npz_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    geometry = np.load(str(geometry_npz_path))
    depth_all = geometry["depth"]
    conf_all = geometry["conf"]
    intrinsics_all = geometry["intrinsics"]
    extrinsics_all = geometry["extrinsics"]

    w2c_h = homogenize_extrinsics(extrinsics_all)
    intrinsics_list = [
        intrinsics_all[t].astype(np.float32) for t in range(intrinsics_all.shape[0])
    ]
    w2c_list = [w2c_h[t].astype(np.float32) for t in range(w2c_h.shape[0])]
    c2w_list = [
        np.linalg.inv(w2c_h[t]).astype(np.float32)
        for t in range(w2c_h.shape[0])
    ]

    return depth_all, conf_all, intrinsics_all, intrinsics_list, w2c_list, c2w_list
