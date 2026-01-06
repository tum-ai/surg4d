import hydra
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from omegaconf import OmegaConf
import torch
import gc
import random
import numpy as np

from benchmark.spatial import get_patched_qwen_for_spatial_grounding
from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae
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

    for config_dump in cfg.config_dumps:
        Path(config_dump).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump)

    if cfg.whole_pipeline_clips_sequentially:
        for clip in cfg.clips:
            if not cfg.skip_preprocessing:
                process_clip(clip, cfg)

            if not cfg.skip_feature_extraction:
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

            if not cfg.skip_autoencoder:
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
                del model_spatial
                del processor_spatial
                gc.collect()
                torch.cuda.empty_cache()
    else:
        if not cfg.skip_preprocessing:
            for clip in cfg.clips:
                process_clip(clip, cfg)

        if not cfg.skip_feature_extraction:
            model, processor = get_patched_qwen(
                qwen_version=cfg.feature_extraction.qwen_version,
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )
            for clip in cfg.clips:
                extract_qwen_features(clip, cfg, model, processor)
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()

        if not cfg.skip_autoencoder:
            for clip in cfg.clips:
                train_ae(clip, cfg)

        if not cfg.skip_splat:
            for clip in cfg.clips:
                train_splat(clip, cfg)

        if not cfg.skip_graph_extraction:
            for clip in cfg.clips:
                extract_graph(clip, cfg)

        if not cfg.skip_eval:
            # Load models for eval
            model, processor = get_patched_qwen(
                qwen_version=cfg.eval.qwen_version,
                use_bnb_4bit=cfg.eval.get("use_bnb_4bit", False),
                use_bnb_8bit=cfg.eval.get("use_bnb_8bit", False),
            )
            model_spatial, processor_spatial = get_patched_qwen_for_spatial_grounding(
                qwen_version=cfg.eval.qwen_version,
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )

            # triplets and temporal eval
            for clip in cfg.clips:
                evaluate_triplets(clip, cfg, model=model, processor=processor)
            for clip in cfg.clips:
                evaluate_temporal(clip, cfg, model=model, processor=processor)

            # spatial eval
            for clip in cfg.clips:
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
