import hydra
from omegaconf import DictConfig

from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae
from train_splats import train_splat


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    for clip in cfg.clips:
        process_clip(clip, cfg)
        extract_qwen_features(clip, cfg)
        train_ae(clip, cfg)
        train_splat(clip, cfg)
