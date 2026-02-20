import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import torch
import gc
import random
import numpy as np
from tqdm import tqdm

from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae, train_global_ae, save_dim_reduced
from autoencoder.model_qwen import QwenAutoencoder
from train_splats import train_splat
from extract_graphs import extract_graph
from evaluate_benchmark import evaluate_temporal, evaluate_spatial
from compute_metrics import (
    compute_spatial_metrics,
    compute_temporal_metrics,
)
from llm.qwen_utils import get_qwen3

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # CUDA seeds and deterministic flags for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for config_dump in cfg.config_dumps or []:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    if cfg.whole_pipeline_clips_sequentially:
        global_ae_mode = not cfg.skip_autoencoder and cfg.autoencoder.global_mode

        # Global AE pre-pass: need features from ALL clips before training AE
        if global_ae_mode:
            if not cfg.skip_feature_extraction:
                model, processor = get_qwen3(
                    size=cfg.feature_extraction.qwen3_size,
                    use_fp8=cfg.feature_extraction.qwen3_use_fp8,
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
                model, processor = get_qwen3(
                    size=cfg.feature_extraction.qwen3_size,
                    use_fp8=cfg.feature_extraction.qwen3_use_fp8,
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
                # load normal qwen
                model, processor = get_qwen3(
                    size=cfg.eval.qwen3_size,
                    use_fp8=cfg.eval.qwen3_use_fp8,
                )
                evaluate_temporal(clip, cfg, model=model, processor=processor)

                # Only load spatial attention model if spatial evaluation is enabled
                model_spatial = None
                processor_spatial = None
                if cfg.eval is not None and cfg.eval.get("spatial") is not None:
                    evaluate_spatial(
                        clip=clip,
                        cfg=cfg,
                        model_spatial=model_spatial,
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
            model, processor = get_qwen3(
                size=cfg.feature_extraction.qwen3_size,
                use_fp8=cfg.feature_extraction.qwen3_use_fp8,
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
            model, processor = get_qwen3(
                size=cfg.eval.qwen3_size,
                use_fp8=cfg.eval.qwen3_use_fp8,
            )

            # temporal eval
            for clip in tqdm(cfg.clips, desc="Temporal Eval", unit="clip"):
                evaluate_temporal(clip, cfg, model=model, processor=processor)

            for clip in tqdm(cfg.clips, desc="Spatial Eval", unit="clip"):
                evaluate_spatial(
                    clip=clip,
                    cfg=cfg,
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

    if not cfg.skip_compute_metrics:
        compute_spatial_metrics(cfg)
        compute_temporal_metrics(cfg)


if __name__ == "__main__":
    main()
