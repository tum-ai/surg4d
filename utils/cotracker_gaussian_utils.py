"""
Utilities for integrating CoTracker3 control points with GaussianModel.
This module handles loading control point data and managing control-point-driven Gaussians.
"""
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Dict
from loguru import logger


def load_cotracker_data(clip_dir: Path, cotracker_subdir: str) -> Dict[str, torch.Tensor]:
    """
    Load CoTracker3 data from preprocessed directory.
    
    Args:
        clip_dir: Directory containing preprocessed data for a clip
    
    Returns:
        Dictionary containing torch tensors:
            - control_points_3d: (T, N_control_points, 3)
            - gaussian_control_point_indices: (N_gaussians, K)
            - gaussian_control_point_weights: (N_gaussians, K)
            - gaussian_positions_precomputed: (T, N_gaussians, 3)
    """
    cotracker_dir = clip_dir / cotracker_subdir
    
    if not cotracker_dir.exists():
        return None
    
    data = {
        "control_points_3d": torch.from_numpy(np.load(cotracker_dir / "control_points_3d.npy")),
        "gaussian_control_point_indices": torch.from_numpy(np.load(cotracker_dir / "point_control_point_indices.npy")),
        "gaussian_control_point_weights": torch.from_numpy(np.load(cotracker_dir / "point_control_point_weights.npy")),
        "gaussian_positions_precomputed": torch.from_numpy(np.load(cotracker_dir / "point_positions_precomputed.npy")),
    }
    
    logger.info(f"Loaded CoTracker data from {cotracker_dir}")
    return data


def initialize_control_point_driven_mask(
    n_gaussians: int,
    gaussian_control_point_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Initialize mask indicating which Gaussians are control-point-driven.
    
    Initially, all Gaussians that have control point associations are control-point-driven.
    This mask will be updated if control points fail.
    
    Args:
        n_gaussians: Total number of Gaussians
        gaussian_control_point_indices: (N_associated, K) indices (only for associated Gaussians)
    
    Returns:
        is_control_point_driven: (n_gaussians,) boolean tensor
    """
    # For now, we assume all Gaussians that have associations are control-point-driven
    # The number of associated Gaussians should match the first dimension of indices
    n_associated = gaussian_control_point_indices.shape[0]
    
    # Create mask: first n_associated Gaussians are control-point-driven
    is_control_point_driven = torch.zeros(n_gaussians, dtype=torch.bool)
    is_control_point_driven[:n_associated] = True
    
    return is_control_point_driven


def get_gaussian_positions_at_time(
    time_idx: int,
    is_control_point_driven: torch.Tensor,
    gaussian_positions_precomputed: torch.Tensor,
    gaussian_xyz_optimizable: torch.Tensor,
) -> torch.Tensor:
    """
    Get Gaussian positions at a specific timestep.
    
    For control-point-driven Gaussians, use precomputed positions.
    For optimizable Gaussians, use the optimizable xyz tensor.
    
    Args:
        time_idx: Time index (0-based)
        is_control_point_driven: (N_gaussians,) boolean mask
        gaussian_positions_precomputed: (T, N_control_driven, 3) precomputed positions
        gaussian_xyz_optimizable: (N_optimizable, 3) optimizable positions (base positions)
    
    Returns:
        positions: (N_gaussians, 3) positions at this timestep
    """
    N_gaussians = is_control_point_driven.shape[0]
    positions = torch.zeros(N_gaussians, 3, device=is_control_point_driven.device)
    
    # Set control-point-driven positions
    control_driven_mask = is_control_point_driven
    n_control_driven = control_driven_mask.sum().item()
    if n_control_driven > 0:
        positions[control_driven_mask] = gaussian_positions_precomputed[time_idx, :n_control_driven]
    
    # Set optimizable positions (base positions, will be modified by hexplane)
    optimizable_mask = ~control_driven_mask
    n_optimizable = optimizable_mask.sum().item()
    if n_optimizable > 0:
        positions[optimizable_mask] = gaussian_xyz_optimizable[:n_optimizable]
    
    return positions

