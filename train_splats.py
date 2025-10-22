from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
from argparse import ArgumentParser
import time as time_module  # Rename to avoid collision with train.py's "from time import time"

from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams
from utils.params_utils import merge_hparams
from train import training
from render import render_sets


def train_splat(clip: DictConfig, cfg: DictConfig):
    """Train Gaussian Splatting for a single clip."""

    clip_dir = Path(clip.dir)

    # Set up environment variables
    os.environ["language_feature_hiddendim"] = str(cfg.splat.language_feature_hiddendim)
    os.environ["use_discrete_lang_f"] = cfg.splat.use_discrete_lang_f

    # Experiment name is just the clip directory name
    exp_name = clip_dir.name

    # Create base argument parser
    parser = ArgumentParser(description="Training script parameters")

    # Initialize param groups
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    # Add additional arguments (from train.py)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[2000, 10000, 20000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[2000, 10000, 20000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--joint_coarse", action="store_true")
    parser.add_argument("--joint_fine", action="store_true")
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--resume_from_final_stage", type=int, default=0)
    parser.add_argument("--resume_from_final_stage_load_iter", type=int, default=10000)
    parser.add_argument(
        "--init_from_stage", choices=["fine-lang", "fine-base"], default="fine-base"
    )
    parser.add_argument("--coff_time_smooth_loss_weight", type=float, default=1e-1)
    parser.add_argument("--resume_from_stage", type=str, default="")
    parser.add_argument("--resume_from_iter", type=int, default=-1)
    parser.add_argument("--depth_loss_weight", type=float, default=0.0)
    parser.add_argument("--opacity_loss_weight", type=float, default=0.0)

    # Build command line args
    cmd_args = [
        "-s",
        str(clip_dir),
        "--port",
        "6021",
        "--expname",
        exp_name,
        "--configs",
        cfg.splat.config_path,
        "--include_feature",
        "--language_features_name",
        cfg.autoencoder.latent_cat_feat_subdir,
        "--feature_level",
        "0",
        "--joint_coarse",
        "--no_dlang",
        "0" if cfg.splat.dynamic_language else "1",
        "--depth_loss_weight",
        str(cfg.splat.depth_loss_weight),
        "--opacity_loss_weight",
        str(cfg.splat.opacity_loss_weight),
    ]

    # Add no_ds flag if dynamic_scale is enabled
    if not cfg.splat.dynamic_scale:
        cmd_args.append("--no_ds")

    # Parse arguments
    args = parser.parse_args(cmd_args)

    # Load and merge config from the specified config file
    if args.configs:
        config = None
        try:
            import mmcv

            if hasattr(mmcv, "Config"):
                config = mmcv.Config.fromfile(args.configs)
        except Exception:
            pass
        
        if config is None:
            try:
                from mmengine.config import Config as MMEngineConfig

                config = MMEngineConfig.fromfile(args.configs)
            except Exception:
                raise ImportError(
                    "Neither mmcv.Config nor mmengine.config.Config is available; install mmcv<=1.x or mmengine."
                )
        args = merge_hparams(args, config)

    # Set model path
    output_root = Path(cfg.get("output_root", "output"))
    args.model_path = str(output_root / exp_name)
    
    # Override training iteration settings from Hydra config
    args.test_iterations = cfg.splat.test_iterations
    args.save_iterations = cfg.splat.save_iterations
    args.checkpoint_iterations = cfg.splat.checkpoint_iterations
    args.coarse_base_iterations = cfg.splat.coarse_base_iterations
    args.coarse_lang_iterations = cfg.splat.coarse_lang_iterations
    args.fine_base_iterations = cfg.splat.fine_base_iterations
    args.fine_lang_iterations = cfg.splat.fine_lang_iterations

    # Get timestamp
    timestamp = time_module.strftime("%Y%m%d_%H%M%S")

    # Inject args into train module's global namespace (train.py expects it)
    import train as train_mod
    train_mod.args = args
    
    # Fix time module conflict: train.py has "from time import time" but uses "time.time()"
    # The "import time" happens in __main__ block, so we need to do it here
    import time as time_module_fix
    train_mod.time = time_module_fix

    # Call the training function
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
        timestamp,
        args,
    )

    # Render outputs for each enabled stage
    stage_map = {
        "coarse_base": "coarse-base",
        "coarse_lang": "coarse-lang",
        "fine_base": "fine-base",
        "fine_lang": "fine-lang",
    }

    for stage_key, stage_name in stage_map.items():
        if cfg.splat.render_outputs.get(stage_key, False):
            render_splat(clip, cfg, args.model_path, stage_name)


def render_splat(clip: DictConfig, cfg: DictConfig, model_path: str, stage: str):
    """Render RGB and language outputs after training for a specific stage.

    Args:
        clip_cfg: Clip configuration
        cfg: Full hydra configuration
        model_path: Path to the trained model
        stage: Training stage name (e.g., "fine-lang", "coarse-base")
    """

    clip_dir = Path(clip.dir)

    # Determine which modes to render based on stage
    if "lang" in stage:
        modes = ["lang", "rgb"]
    else:
        modes = ["rgb"]

    for mode in modes:
        print(f"\nRendering {mode} outputs for stage {stage}...")

        # Create parser
        parser = ArgumentParser(description="Rendering script parameters")

        # Initialize param groups
        mp = ModelParams(parser, sentinel=True)
        pp = PipelineParams(parser)
        hp = ModelHiddenParams(parser)

        # Add render-specific arguments
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--skip_video", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--configs", type=str, default="")
        parser.add_argument("--mode", type=str, default="rgb")
        parser.add_argument("--novideo", type=int, default=0)
        parser.add_argument("--noimage", type=int, default=0)
        parser.add_argument("--nonpy", type=int, default=0)
        parser.add_argument("--load_stage", type=str, default="fine-lang")

        # Build command line args (use same flags as training)
        cmd_args = [
            "-s",
            str(clip_dir),
            "--language_features_name",
            cfg.autoencoder.latent_cat_feat_subdir,
            "--model_path",
            model_path,
            "--feature_level",
            "0",
            "--skip_train",
            "--skip_test",
            "--configs",
            cfg.splat.config_path,
            "--mode",
            mode,
            "--no_dlang",
            "0" if cfg.splat.dynamic_language else "1",
            "--load_stage",
            stage,
        ]

        # Add no_ds flag if dynamic_scale is disabled (same as training)
        if not cfg.splat.dynamic_scale:
            cmd_args.append("--no_ds")

        # Parse arguments
        args = parser.parse_args(cmd_args)

        # Load and merge config
        if args.configs:
            config = None
            try:
                import mmcv

                if hasattr(mmcv, "Config"):
                    config = mmcv.Config.fromfile(args.configs)
            except Exception:
                pass
            
            if config is None:
                try:
                    from mmengine.config import Config as MMEngineConfig

                    config = MMEngineConfig.fromfile(args.configs)
                except Exception:
                    raise ImportError(
                        "Neither mmcv.Config nor mmengine.config.Config is available"
                    )
            args = merge_hparams(args, config)

        # Call render function
        render_sets(
            mp.extract(args),
            hp.extract(args),
            args.iteration,
            pp.extract(args),
            args.skip_train,
            args.skip_test,
            args.skip_video,
            args.mode,
            args,
        )


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """Main training loop for all clips."""
    for clip in tqdm(cfg.clips, desc="Training splats", unit="clip"):
        train_splat(clip, cfg)


if __name__ == "__main__":
    main()
