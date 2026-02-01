"""
Utilities for interpolating Gaussian positions from control points using IDW.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from loguru import logger
import rerun as rr


def interpolate_gaussian_positions_from_control_points(
    control_points_3d: torch.Tensor,
    control_point_indices: torch.Tensor,
    control_point_weights: torch.Tensor,
    save_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate Gaussian 3D positions from control points using IDW weights.
    
    Args:
        control_points_3d: (T, N_control_points, 3) 3D control point positions
        control_point_indices: (N_gaussians, K) indices of associated control points
        control_point_weights: (N_gaussians, K) IDW weights
    
    Returns:
        gaussian_positions: (T, N_gaussians, 3) interpolated Gaussian positions
    """
    T, N_control_points, _ = control_points_3d.shape
    N_gaussians, K = control_point_indices.shape
    
    device = control_points_3d.device
    
    # Index control points for each Gaussian using advanced indexing
    # control_points_3d: (T, N_control_points, 3)
    # control_point_indices: (N_gaussians, K)
    # We need: (T, N_gaussians, K, 3)
    
    # Use advanced indexing: control_points_3d[t, indices] for each t
    control_points_selected = control_points_3d[:, control_point_indices]  # (T, N_gaussians, K, 3)
    weights_expanded = control_point_weights.unsqueeze(0).expand(T, -1, -1)  # (T, N_gaussians, K)
    
    # Weighted sum: (T, N_gaussians, K, 3) * (T, N_gaussians, K, 1) -> (T, N_gaussians, 3)
    gaussian_positions = (control_points_selected * weights_expanded.unsqueeze(-1)).sum(dim=2)

    # Generate X% random indices of the total number of gaussians
    n_samples = int(N_gaussians * 0.1)
    random_indices = torch.randint(0, N_gaussians, (n_samples,), device=device)
    sampled_gaussians = gaussian_positions[:, random_indices, :] # (T, N_sampled_gaussians, 3)

    # Get the control points for each sampled Gaussian
    sampled_control_point_indices = control_point_indices[random_indices, :] # (N_sampled_gaussians, K)
    sampled_control_points = control_points_3d[:, sampled_control_point_indices] # (T, N_sampled_gaussians, K, 3)
    # Collapse n samples and k into T, control_points, 3
    sampled_control_points = sampled_control_points.view(T, -1, 3)

    # Generate # samples random colors
    colors_gaussians = torch.rand(n_samples, 3, device=device) # (N_sampled_gaussians, 3)
    colors_control_points = colors_gaussians.repeat_interleave(K, dim=0) # (N_sampled_gaussians * K, 3)

    # Make sure everything is on cpu and numpy
    sampled_gaussians = sampled_gaussians.cpu().numpy()
    sampled_control_points = sampled_control_points.cpu().numpy()
    colors_gaussians = colors_gaussians.cpu().numpy()
    colors_control_points = colors_control_points.cpu().numpy()

    rr.init("cotracker_interpolation")
    rr.save(save_dir / "cotracker_interpolation.rrd")
    for t in range(T):
        rr.set_time_sequence("frame", t)
        rr.log("world/sampled_gaussians", rr.Points3D(
            positions=sampled_gaussians[t],
            colors=colors_gaussians,
            radii=0.003,
        ))
        rr.log("world/sampled_control_points", rr.Points3D(
            positions=sampled_control_points[t],
            colors=colors_control_points,
            radii=0.003,
        ))

    return gaussian_positions


def precompute_control_point_positions(
    control_points_3d: torch.Tensor,
    control_point_indices: torch.Tensor,
    control_point_weights: torch.Tensor,
    save_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Precompute Gaussian positions for all timesteps.
    
    This is called during preprocessing to compute and store positions.
    
    Args:
        control_points_3d: (T, N_control_points, 3) 3D control point positions (torch.Tensor)
        control_point_indices: (N_gaussians, K) indices of associated control points (torch.Tensor)
        control_point_weights: (N_gaussians, K) IDW weights (torch.Tensor)
    
    Returns:
        gaussian_positions: (T, N_gaussians, 3) interpolated positions (torch.Tensor)
    """
    # Ensure correct dtypes
    control_points_3d_torch = control_points_3d.float()
    control_point_indices_torch = control_point_indices.long()
    control_point_weights_torch = control_point_weights.float()
        
    # Compute positions
    positions = interpolate_gaussian_positions_from_control_points(
        control_points_3d_torch,
        control_point_indices_torch,
        control_point_weights_torch,
        save_dir,
    )
    
    return positions


def mark_failed_control_points_as_optimizable(
    control_point_validity: torch.Tensor,
    control_point_indices: torch.Tensor,
    is_control_point_driven: torch.Tensor,
) -> torch.Tensor:
    """
    Mark Gaussians as optimizable if any of their associated control points fail.
    
    Args:
        control_point_validity: (T, N_control_points) boolean mask
        control_point_indices: (N_gaussians, K) indices of associated control points
        is_control_point_driven: (N_gaussians,) boolean mask of control-point-driven Gaussians
    
    Returns:
        Updated is_control_point_driven mask (Gaussians with failed control points become False)
    """
    # Check if any control point fails at any timestep
    # validity_selected: (T, N_gaussians, K)
    validity_selected = control_point_validity[:, control_point_indices]
    
    # A Gaussian becomes optimizable if ANY of its control points fail at ANY time
    # (T, N_gaussians, K) -> (N_gaussians,)
    any_failed = (~validity_selected).any(dim=(0, 2))
    
    # Update mask: control-point-driven Gaussians with failed control points become optimizable
    is_control_point_driven = is_control_point_driven & (~any_failed)
    
    return is_control_point_driven

