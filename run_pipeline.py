import hydra
from omegaconf import DictConfig

from preprocess import process_clip
from generate_qwen_features import extract_qwen_features
from train_autoencoders import train_ae
from train_splats import train_splat
from extract_graphs import extract_graph
from qwen_vl import get_patched_qwen


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Load qwen model once for all clips
    model, processor = get_patched_qwen(
        use_bnb_4bit=cfg.feature_extraction.bnb_4bit,
        use_bnb_8bit=cfg.feature_extraction.bnb_8bit,
    )
    
    for clip in cfg.clips:
        process_clip(clip, cfg)
        extract_qwen_features(clip, cfg, model, processor)
        train_ae(clip, cfg)
        train_splat(clip, cfg)
        extract_graph(clip, cfg)
