from pathlib import Path
import numpy as np
import torch
import argparse
import logging
import mmcv
import os
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import GaussianModel, Scene
from utils.params_utils import merge_hparams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_TIMESTEPS = 20  # sample 20 timesteps for integration

# Set environment variables for language feature dimensions (needed for model loading)
os.environ['language_feature_hiddendim'] = '6'
os.environ['use_discrete_lang_f'] = 'f'


def init_params(model_path: str):
    """Setup parameters for loading a trained model"""
    parser = argparse.ArgumentParser()
    
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    
    # Create args with model_path
    args = parser.parse_args([
        "--model_path", model_path,
        "--iteration", "-1",
        "--load_stage", "fine-lang"
    ])
    
    # Load stored config file and merge (similar to get_combined_args)
    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfgfile_string = f.read()
            # Provide Namespace in eval context
            from argparse import Namespace
            args_cfgfile = eval(cfgfile_string)
            
            # Merge: start with stored config, override with our specific args
            merged_dict = vars(args_cfgfile).copy()
            # Override model_path, iteration, and load_stage with our values
            merged_dict['model_path'] = model_path
            merged_dict['iteration'] = -1
            merged_dict['load_stage'] = 'fine-lang'
            
            args = argparse.Namespace(**merged_dict)
        except Exception as e:
            logger.warning(f"Could not load cfg_args: {e}")
            # Continue with default args
    
    return args, model_params, pipeline, hyperparam


def load_model(model_path: str):
    """Load a trained gaussian model"""
    args, model_params, pipeline, hyperparam = init_params(model_path)
    
    hyper = hyperparam.extract(args)
    dataset = model_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        load_stage=args.load_stage,
    )
    
    return gaussians, scene, args, dataset


def compute_deformation_at_timestep(gaussians: GaussianModel, timestep: float):
    """Compute all deformed properties at a given timestep
    
    Returns:
        dict with keys: 'positions', 'scales', 'rotations', 'opacity', 'shs', 'lang'
        Each value is the deformed property as a numpy array
    """
    with torch.no_grad():
        means3D = gaussians.get_xyz
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        lang = gaussians.get_language_feature
        
        time = torch.full(
            (means3D.shape[0], 1),
            float(timestep),
            device=means3D.device,
            dtype=means3D.dtype,
        )
        
        # Disable language deformation for consistent measurement
        try:
            orig_no_dlang = gaussians._deformation.deformation_net.args.no_dlang
            gaussians._deformation.deformation_net.args.no_dlang = 1
        except Exception:
            orig_no_dlang = None
        
        means3D_deformed, scales_deformed, rotations_deformed, opacity_deformed, shs_deformed, lang_deformed, _ = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, lang, time
        )
        
        # Restore original setting
        if orig_no_dlang is not None:
            try:
                gaussians._deformation.deformation_net.args.no_dlang = orig_no_dlang
            except Exception:
                pass
        
    return {
        'positions': means3D_deformed.detach().cpu().numpy(),
        'scales': scales_deformed.detach().cpu().numpy(),
        'rotations': rotations_deformed.detach().cpu().numpy(),
        'opacity': opacity_deformed.detach().cpu().numpy(),
        'shs': shs_deformed.detach().cpu().numpy(),
        'lang': lang_deformed.detach().cpu().numpy() if lang_deformed is not None else None,
    }


def compute_original_properties(gaussians: GaussianModel):
    """Get the original (canonical) properties"""
    with torch.no_grad():
        return {
            'positions': gaussians.get_xyz.detach().cpu().numpy(),
            'scales': gaussians._scaling.detach().cpu().numpy(),
            'rotations': gaussians._rotation.detach().cpu().numpy(),
            'opacity': gaussians._opacity.detach().cpu().numpy(),
            'shs': gaussians.get_features.detach().cpu().numpy(),
            'lang': gaussians.get_language_feature.detach().cpu().numpy(),
        }


def analyze_scene_deformations(model_path: str):
    """Analyze deformations for a single scene
    
    Returns:
        dict with statistics about the amount of deformation
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        gaussians, scene, args, dataset = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None
    
    n_gaussians = gaussians.get_xyz.shape[0]
    logger.info(f"Loaded {n_gaussians} gaussians")
    
    # Sample timesteps uniformly
    timesteps = np.linspace(0, 1, N_TIMESTEPS)
    
    # Get original properties
    original = compute_original_properties(gaussians)
    
    # Compute properties at all timesteps
    logger.info(f"Computing deformations at {N_TIMESTEPS} timesteps...")
    properties_over_time = []
    for t in tqdm(timesteps, desc="Timesteps"):
        props = compute_deformation_at_timestep(gaussians, float(t))
        properties_over_time.append(props)
    
    # Get opacity weights (convert from logit space to [0,1])
    opacity_weights = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    opacity_weights = opacity_weights / opacity_weights.sum()  # normalize to sum to 1
    
    # Compute integral of absolute changes
    # For each property, we compute sum over time of |prop(t) - prop(0)|
    stats = {}
    
    # Position changes (3D)
    pos_changes = []
    for props in properties_over_time:
        diff = np.abs(props['positions'] - original['positions'])
        pos_changes.append(diff)
    pos_changes = np.stack(pos_changes, axis=0)  # (T, N, 3)
    # Integrate over time (sum), then take mean over gaussians
    pos_integral = pos_changes.sum(axis=0)  # (N, 3)
    stats['position_mean_total_change'] = pos_integral.mean()
    stats['position_std_total_change'] = pos_integral.std()
    stats['position_max_total_change'] = pos_integral.max()
    stats['position_mean_per_axis'] = pos_integral.mean(axis=0)  # per x,y,z
    # Opacity-weighted version
    stats['position_mean_total_change_weighted'] = (pos_integral.sum(axis=-1) * opacity_weights).sum()
    stats['position_mean_opacity'] = opacity_weights.mean()
    stats['position_median_opacity'] = np.median(opacity_weights * n_gaussians)  # rescale for interpretability
    
    # Scale changes (3D)
    scale_changes = []
    for props in properties_over_time:
        diff = np.abs(props['scales'] - original['scales'])
        scale_changes.append(diff)
    scale_changes = np.stack(scale_changes, axis=0)
    scale_integral = scale_changes.sum(axis=0)
    stats['scale_mean_total_change'] = scale_integral.mean()
    stats['scale_std_total_change'] = scale_integral.std()
    stats['scale_max_total_change'] = scale_integral.max()
    # Opacity-weighted version
    stats['scale_mean_total_change_weighted'] = (scale_integral.sum(axis=-1) * opacity_weights).sum()
    
    # Rotation changes (4D quaternion)
    rotation_changes = []
    for props in properties_over_time:
        diff = np.abs(props['rotations'] - original['rotations'])
        rotation_changes.append(diff)
    rotation_changes = np.stack(rotation_changes, axis=0)
    rotation_integral = rotation_changes.sum(axis=0)
    stats['rotation_mean_total_change'] = rotation_integral.mean()
    stats['rotation_std_total_change'] = rotation_integral.std()
    stats['rotation_max_total_change'] = rotation_integral.max()
    # Opacity-weighted version
    stats['rotation_mean_total_change_weighted'] = (rotation_integral.sum(axis=-1) * opacity_weights).sum()
    
    # Opacity changes (1D)
    opacity_changes = []
    for props in properties_over_time:
        diff = np.abs(props['opacity'] - original['opacity'])
        opacity_changes.append(diff)
    opacity_changes = np.stack(opacity_changes, axis=0)
    opacity_integral = opacity_changes.sum(axis=0)
    stats['opacity_mean_total_change'] = opacity_integral.mean()
    stats['opacity_std_total_change'] = opacity_integral.std()
    stats['opacity_max_total_change'] = opacity_integral.max()
    # Opacity-weighted version
    stats['opacity_mean_total_change_weighted'] = (opacity_integral.squeeze() * opacity_weights).sum()
    
    # SHS (spherical harmonics) changes
    shs_changes = []
    for props in properties_over_time:
        diff = np.abs(props['shs'] - original['shs'])
        shs_changes.append(diff)
    shs_changes = np.stack(shs_changes, axis=0)
    shs_integral = shs_changes.sum(axis=0)
    stats['shs_mean_total_change'] = shs_integral.mean()
    stats['shs_std_total_change'] = shs_integral.std()
    stats['shs_max_total_change'] = shs_integral.max()
    # Opacity-weighted version
    stats['shs_mean_total_change_weighted'] = (shs_integral.reshape(n_gaussians, -1).sum(axis=-1) * opacity_weights).sum()
    
    # Language feature changes (if present)
    if original['lang'] is not None and properties_over_time[0]['lang'] is not None:
        lang_changes = []
        for props in properties_over_time:
            diff = np.abs(props['lang'] - original['lang'])
            lang_changes.append(diff)
        lang_changes = np.stack(lang_changes, axis=0)
        lang_integral = lang_changes.sum(axis=0)
        stats['lang_mean_total_change'] = lang_integral.mean()
        stats['lang_std_total_change'] = lang_integral.std()
        stats['lang_max_total_change'] = lang_integral.max()
        # Opacity-weighted version
        stats['lang_mean_total_change_weighted'] = (lang_integral.sum(axis=-1) * opacity_weights).sum()
    
    # Also compute actual position displacement magnitudes
    pos_displacements = []
    for props in properties_over_time:
        displacement = np.linalg.norm(props['positions'] - original['positions'], axis=-1)
        pos_displacements.append(displacement)
    pos_displacements = np.stack(pos_displacements, axis=0)  # (T, N)
    stats['position_displacement_mean'] = pos_displacements.mean()
    stats['position_displacement_max'] = pos_displacements.max()
    stats['position_displacement_over_time_mean'] = pos_displacements.mean(axis=1)  # mean per timestep
    # Opacity-weighted version
    pos_displacement_integral = pos_displacements.sum(axis=0)  # (N,) - integrated over time
    stats['position_displacement_mean_weighted'] = (pos_displacement_integral * opacity_weights).sum()
    
    stats['n_gaussians'] = n_gaussians
    stats['scene_name'] = Path(model_path).name
    
    return stats


def main():
    output_dir = Path("/home/students/lmu_proj/surgery-scene-graphs/output/cholecseg8k")
    
    # Find all scene directories
    scene_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(scene_dirs)} scene directories")
    
    all_stats = {}
    
    for scene_dir in sorted(scene_dirs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scene: {scene_dir.name}")
        logger.info(f"{'='*60}")
        
        stats = analyze_scene_deformations(str(scene_dir))
        
        if stats is not None:
            all_stats[scene_dir.name] = stats
            
            # Print summary for this scene
            logger.info(f"\nResults for {scene_dir.name}:")
            logger.info(f"  Number of gaussians: {stats['n_gaussians']}")
            logger.info(f"  Mean opacity: {stats['position_mean_opacity']:.6f}")
            logger.info(f"  Position mean total change: {stats['position_mean_total_change']:.6f} (weighted: {stats['position_mean_total_change_weighted']:.6f})")
            logger.info(f"  Position displacement mean: {stats['position_displacement_mean']:.6f} (weighted: {stats['position_displacement_mean_weighted']:.6f})")
            logger.info(f"  Position displacement max: {stats['position_displacement_max']:.6f}")
            logger.info(f"  Scale mean total change: {stats['scale_mean_total_change']:.6f} (weighted: {stats['scale_mean_total_change_weighted']:.6f})")
            logger.info(f"  Rotation mean total change: {stats['rotation_mean_total_change']:.6f} (weighted: {stats['rotation_mean_total_change_weighted']:.6f})")
            logger.info(f"  Opacity mean total change: {stats['opacity_mean_total_change']:.6f} (weighted: {stats['opacity_mean_total_change_weighted']:.6f})")
            logger.info(f"  SHS mean total change: {stats['shs_mean_total_change']:.6f} (weighted: {stats['shs_mean_total_change_weighted']:.6f})")
            if 'lang_mean_total_change' in stats:
                logger.info(f"  Lang mean total change: {stats['lang_mean_total_change']:.6f} (weighted: {stats['lang_mean_total_change_weighted']:.6f})")
    
    # Save results
    results_file = output_dir / "deformation_analysis.npz"
    np.savez(results_file, **all_stats)
    logger.info(f"\nSaved results to {results_file}")
    
    # Print summary across all scenes
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY ACROSS ALL SCENES")
    logger.info(f"{'='*60}")
    
    if all_stats:
        # Unweighted metrics
        for metric in ['position_mean_total_change', 'position_displacement_mean', 
                      'scale_mean_total_change', 'rotation_mean_total_change', 
                      'opacity_mean_total_change', 'shs_mean_total_change']:
            values = [s[metric] for s in all_stats.values()]
            logger.info(f"{metric}:")
            logger.info(f"  Mean across scenes: {np.mean(values):.6f}")
            logger.info(f"  Std across scenes: {np.std(values):.6f}")
            logger.info(f"  Min: {np.min(values):.6f}")
            logger.info(f"  Max: {np.max(values):.6f}")
        
        logger.info(f"\n{'='*60}")
        logger.info("OPACITY-WEIGHTED METRICS")
        logger.info(f"{'='*60}")
        
        # Weighted metrics
        for metric in ['position_mean_total_change_weighted', 'position_displacement_mean_weighted',
                      'scale_mean_total_change_weighted', 'rotation_mean_total_change_weighted',
                      'opacity_mean_total_change_weighted', 'shs_mean_total_change_weighted']:
            values = [s[metric] for s in all_stats.values()]
            logger.info(f"{metric}:")
            logger.info(f"  Mean across scenes: {np.mean(values):.6f}")
            logger.info(f"  Std across scenes: {np.std(values):.6f}")
            logger.info(f"  Min: {np.min(values):.6f}")
            logger.info(f"  Max: {np.max(values):.6f}")


if __name__ == "__main__":
    main()