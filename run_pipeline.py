import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from pathlib import Path

from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae
from train_splats import train_splat
from extract_graphs import extract_graph


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
    # do hydra init manually here to avoid conflicts with vipe hydra
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        overrides = sys.argv[1:]
        cfg = hydra.compose("config.yaml", overrides=overrides)

    # Clear after composing the main config so vipe can initialize its own
    GlobalHydra.instance().clear()

    for clip in cfg.clips:
        process_clip(clip, cfg)
        get_patched = _get_qwen_loader(cfg)
        model, processor = get_patched()
        extract_qwen_features(clip, cfg, model, processor)
        del model
        del processor
        train_ae(clip, cfg)
        train_splat(clip, cfg)
        extract_graph(clip, cfg)

if __name__ == "__main__":
    main()