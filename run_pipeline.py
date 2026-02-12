import hydra
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from omegaconf import OmegaConf
import torch
import gc
import random
import numpy as np
from tqdm import tqdm

from benchmark.spatial import get_patched_qwen_for_spatial_grounding
from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae, train_global_ae, save_dim_reduced
from autoencoder.model_qwen import QwenAutoencoder
from train_splats import train_splat
from extract_graphs import extract_graph
from evaluate_benchmark import evaluate_triplets, evaluate_temporal, evaluate_spatial
from compute_metrics import (
    compute_spatial_metrics,
    compute_temporal_metrics,
    compute_triplets_metrics,
)
from llm.qwen_utils import get_patched_qwen


import sys


def main():
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # CUDA seeds and deterministic flags for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # do hydra init manually here to avoid conflicts with vipe hydra
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        overrides = sys.argv[1:]
        cfg = hydra.compose("config.yaml", overrides=overrides)

    # Clear after composing the main config so vipe can initialize its own
    GlobalHydra.instance().clear()

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    if cfg.whole_pipeline_clips_sequentially:
        global_ae_mode = not cfg.skip_autoencoder and cfg.autoencoder.global_mode

        # Global AE pre-pass: need features from ALL clips before training AE
        if global_ae_mode:
            if not cfg.skip_feature_extraction:
                model, processor = get_patched_qwen(
                    qwen_version=cfg.feature_extraction.qwen_version,
                    use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                    use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
                )
            for clip in tqdm(cfg.clips, desc="Prep for global AE", unit="clip"):
                if not cfg.skip_preprocessing:
                    process_clip(clip, cfg)
                if not cfg.skip_feature_extraction:
                    extract_qwen_features(clip, cfg, model, processor)
            if not cfg.skip_feature_extraction:
                del model
                del processor
                gc.collect()
                torch.cuda.empty_cache()

            train_global_ae(cfg.clips, cfg)
            global_checkpoint_path = (
                Path(cfg.preprocessed_root)
                / cfg.autoencoder.global_checkpoint_dir
                / "best_ckpt.pth"
            )
            global_ae = QwenAutoencoder(
                input_dim=cfg.autoencoder.full_dim,
                latent_dim=cfg.autoencoder.latent_dim,
            ).to("cuda")
            global_ae.load_state_dict(
                torch.load(global_checkpoint_path, map_location="cuda")
            )
            global_ae.eval()
            for clip in tqdm(cfg.clips, desc="Encoding with global AE", unit="clip"):
                save_dim_reduced(clip, cfg, ae=global_ae)
            del global_ae
            gc.collect()
            torch.cuda.empty_cache()

        for clip in tqdm(cfg.clips, desc="Full Pipeline", unit="clip"):
            # Skip preprocessing + features if already done in global AE pre-pass
            if not cfg.skip_preprocessing and not global_ae_mode:
                process_clip(clip, cfg)

            if not cfg.skip_feature_extraction and not global_ae_mode:
                model, processor = get_patched_qwen(
                    qwen_version=cfg.feature_extraction.qwen_version,
                    use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                    use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
                )
                extract_qwen_features(clip, cfg, model, processor)
                del model
                del processor
                gc.collect()
                torch.cuda.empty_cache()

            # Per-clip AE (only in non-global mode)
            if not cfg.skip_autoencoder and not global_ae_mode:
                train_ae(clip, cfg)

            if not cfg.skip_splat:
                train_splat(clip, cfg)

            if not cfg.skip_graph_extraction:
                extract_graph(clip, cfg)

            # TODO pass model and processor from outside
            if not cfg.skip_eval:
                # Load models for triplets and temporal eval
                model, processor = get_patched_qwen(
                    qwen_version=cfg.eval.qwen_version,
                    use_bnb_4bit=cfg.eval.get("use_bnb_4bit", False),
                    use_bnb_8bit=cfg.eval.get("use_bnb_8bit", False),
                )
                evaluate_triplets(clip, cfg, model=model, processor=processor)
                evaluate_temporal(clip, cfg, model=model, processor=processor)

                # Only load spatial attention model if spatial evaluation is enabled
                model_spatial = None
                processor_spatial = None
                if cfg.eval is not None and cfg.eval.get("spatial") is not None:
                    model_spatial, processor_spatial = (
                        get_patched_qwen_for_spatial_grounding(
                            qwen_version=cfg.eval.qwen_version,
                            use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                            use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
                        )
                    )
                    evaluate_spatial(
                        clip=clip,
                        cfg=cfg,
                        model_spatial=model_spatial,
                        processor_spatial=processor_spatial,
                        model=model,
                        processor=processor,
                    )
                
                del model
                del processor
                if model_spatial is not None:
                    del model_spatial
                    del processor_spatial
                gc.collect()
                torch.cuda.empty_cache()
    else:
        if not cfg.skip_preprocessing:
            for clip in tqdm(cfg.clips, desc="Preprocessing", unit="clip"):
                process_clip(clip, cfg)

        if not cfg.skip_feature_extraction:
            model, processor = get_patched_qwen(
                qwen_version=cfg.feature_extraction.qwen_version,
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )
            for clip in tqdm(cfg.clips, desc="Feature Extraction", unit="clip"):
                extract_qwen_features(clip, cfg, model, processor)
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()

        if not cfg.skip_autoencoder:
            if cfg.autoencoder.global_mode:
                train_global_ae(cfg.clips, cfg)
                global_checkpoint_path = (
                    Path(cfg.preprocessed_root)
                    / cfg.autoencoder.global_checkpoint_dir
                    / "best_ckpt.pth"
                )
                global_ae = QwenAutoencoder(
                    input_dim=cfg.autoencoder.full_dim,
                    latent_dim=cfg.autoencoder.latent_dim,
                ).to("cuda")
                global_ae.load_state_dict(
                    torch.load(global_checkpoint_path, map_location="cuda")
                )
                global_ae.eval()
                for clip in tqdm(cfg.clips, desc="Encoding with global AE", unit="clip"):
                    save_dim_reduced(clip, cfg, ae=global_ae)
                del global_ae
                gc.collect()
                torch.cuda.empty_cache()
            else:
                for clip in tqdm(cfg.clips, desc="Autoencoder Training", unit="clip"):
                    train_ae(clip, cfg)

        if not cfg.skip_splat:
            for clip in tqdm(cfg.clips, desc="Splat Training", unit="clip"):
                train_splat(clip, cfg)

        if not cfg.skip_graph_extraction:
            for clip in tqdm(cfg.clips, desc="Graph Extraction", unit="clip"):
                extract_graph(clip, cfg)

        if not cfg.skip_eval:
            # Load models for eval
            model, processor = get_patched_qwen(
                qwen_version=cfg.eval.qwen_version,
                use_bnb_4bit=cfg.eval.get("use_bnb_4bit", False),
                use_bnb_8bit=cfg.eval.get("use_bnb_8bit", False),
            )

            # triplets and temporal eval
            for clip in tqdm(cfg.clips, desc="Triplets Eval", unit="clip"):
                evaluate_triplets(clip, cfg, model=model, processor=processor)
            
            for clip in tqdm(cfg.clips, desc="Temporal Eval", unit="clip"):
                evaluate_temporal(clip, cfg, model=model, processor=processor)

            # Only load spatial attention model if spatial evaluation is enabled
            model_spatial = None
            processor_spatial = None
            if cfg.eval is not None and cfg.eval.get("spatial") is not None:
                model_spatial, processor_spatial = get_patched_qwen_for_spatial_grounding(
                    qwen_version=cfg.eval.qwen_version,
                    use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                    use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
                )
                
                # spatial eval
                for clip in tqdm(cfg.clips, desc="Spatial Eval", unit="clip"):
                    evaluate_spatial(
                        clip=clip,
                        cfg=cfg,
                        model_spatial=model_spatial,
                        processor_spatial=processor_spatial,
                        model=model,
                        processor=processor,
                    )
                    # Clear VRAM after each clip to prevent OOM
                    gc.collect()
                    torch.cuda.empty_cache()

            del model
            del processor
            if model_spatial is not None:
                del model_spatial
                del processor_spatial
            gc.collect()
            torch.cuda.empty_cache()

    if not cfg.skip_compute_metrics:
        compute_spatial_metrics(cfg)
        compute_temporal_metrics(cfg)
        compute_triplets_metrics(cfg)


if __name__ == "__main__":
    main()
