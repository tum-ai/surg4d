#!/usr/bin/env python
"""Render language feature numpy arrays for debugging."""
import os
import sys
from pathlib import Path

# CRITICAL: Set env vars BEFORE any imports
# Match checkpoint: lang_deform output=128 (64*2), lang_deforms output=64
os.environ["language_feature_hiddendim"] = "32"
os.environ["use_discrete_lang_f"] = "f"
os.environ["num_lang_features"] = "2"
os.environ["lang_feature_dim"] = "16"
os.environ["lang_deform_width"] = "256"
os.environ["lang_timebase_pe"] = "4"
os.environ["centers_num"] = "10"

from argparse import ArgumentParser
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from render import render_sets
from utils.params_utils import merge_hparams
from utils.gaussian_loading_utils import get_latest_model_iteration


def render_splat(clip: DictConfig, cfg: DictConfig, model_path: str, stage: str):
    """Render language features as numpy arrays."""
    
    clip_dir = Path(cfg.preprocessed_root) / clip.name
    
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--mode", type=str, default="rgb")
    parser.add_argument("--novideo", type=int, default=1)
    parser.add_argument("--noimage", type=int, default=1)
    parser.add_argument("--nonpy", type=int, default=0)  # Save npy files
    parser.add_argument("--load_stage", type=str, default="fine-lang")
    
    cmd_args = [
        "-s", str(clip_dir),
        "--model_path", model_path,
        "--language_features_name", cfg.splat.latent_cat_feat_subdir,
        "--feature_level", "0",
        "--skip_train",
        "--skip_test",
        "--configs", cfg.splat.config_path,
        "--mode", "lang",
        "--load_stage", stage,
    ]
    
    args = parser.parse_args(cmd_args)
    
    # Load config
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
                raise ImportError("Config loading failed")
        args = merge_hparams(args, config)
    
    # Match training config
    args.no_dshs = not cfg.splat.dynamic_color
    args.no_ds = not cfg.splat.dynamic_scale
    args.rezero_init = cfg.splat.rezero_init
    
    if args.iteration == -1:
        args.iteration = get_latest_model_iteration(cfg)
    
    print(f"\nRendering {stage} language features to numpy...")
    print(f"  Model: {model_path}")
    print(f"  Output: {model_path}/video_lang/ours_{args.iteration}/renders_npy/")
    
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
    print("Done!")


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    """Render numpy outputs for specified clips."""
    
    clips_to_render = ["video18_00979"]#, "video17_01803"]
    
    for clip_cfg in cfg.clips:
        if clip_cfg.name in clips_to_render:
            output_root = Path(cfg.get("output_root", "output"))
            model_path = str(output_root / clip_cfg.name)
            
            # Render only fine-lang stage (has language features)
            if cfg.splat.render_outputs.get("fine_lang", True):
                render_splat(clip_cfg, cfg, model_path, "fine-lang")


if __name__ == "__main__":
    main()
