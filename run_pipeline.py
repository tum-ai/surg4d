import hydra
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from omegaconf import OmegaConf
import torch
import os
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


import sys
from typing import Callable


def _get_qwen_loader(cfg) -> Callable[[], tuple]:
    if cfg.feature_extraction.get("use_qwen3", False):
        from qwen_vl_qwen3 import get_patched_qwen3

        def loader():
            return get_patched_qwen3(
                model_path=cfg.feature_extraction.get(
                    "model_id", "Qwen/Qwen3-VL-8B-Instruct"
                ),
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )

        return loader
    else:
        from qwen_vl import get_patched_qwen

        def loader():
            return get_patched_qwen(
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )

        return loader


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
                get_patched = _get_qwen_loader(cfg)
                model, processor = get_patched()
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
                evaluate_triplets(clip, cfg)
                evaluate_temporal(clip, cfg)

                model_spatial, processor_spatial = (
                    get_patched_qwen_for_spatial_grounding(
                        use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                        use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
                    )
                )
                evaluate_spatial(clip, cfg, model_spatial, processor_spatial)
                del model_spatial
                del processor_spatial
                gc.collect()
                torch.cuda.empty_cache()
    else:
        if not cfg.skip_preprocessing:
            for clip in cfg.clips:
                process_clip(clip, cfg)

        if not cfg.skip_feature_extraction:
            get_patched = _get_qwen_loader(cfg)
            model, processor = get_patched()
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
            # triplets and temporal eval
            # TODO pass model and processor from outside
            for clip in cfg.clips:
                evaluate_triplets(clip, cfg)
            for clip in cfg.clips:
                evaluate_temporal(clip, cfg)

            # spatial eval
            model_spatial, processor_spatial = get_patched_qwen_for_spatial_grounding(
                use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
                use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
            )
            for clip in cfg.clips:
                evaluate_spatial(clip, cfg, model_spatial, processor_spatial)
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
