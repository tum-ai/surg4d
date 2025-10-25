"""
Configuration for the surgical VQA benchmark
"""
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""

    # Either none or a dict witht the config for that task
    triplets_config: Optional[dict] = None
    temporal_config: Optional[dict] = None
    spatial_config: Optional[dict] = None
    spatiotemporal_config: Optional[dict] = None

    # Data paths
    # Root to triplets
    cholect50_root: Path = Path("/workspace/nico/surgery-scene-graphs/data/cholect50")
    # Root to data
    preprocessed_root: Path = Path("/workspace/nico/surgery-scene-graphs/data/preprocessed_dyn_scene")
    # Root to output
    output_root: Path = Path("/workspace/nico/surgery-scene-graphs/output/dyn_scene")
    results_dir: Path = output_root / "benchmark"

    # Path to specific video to evaluate
    video_dir: Path = preprocessed_root / "video01_16345"
    graph_dir: Path = output_root / "video01_16345/graph"
    
    # Model settings
    model_name: Literal["qwen", "gpt4"] = "qwen"
    qwen_version: Literal["qwen2.5", "qwen3"] = "qwen2.5"  # Differentiate between Qwen versions
    use_4bit_quantization: bool = False
    # Should always be run on cuda
    device: str = "cuda"
    
    # Evaluation settings
    seed: int = 42


# Note: Synonym matching has been removed - using exact string matching only
# The model is provided with exact options in the system prompt
def normalize_for_matching(text: str) -> str:
    """Normalize text for comparison (basic normalization only)"""
    return text.lower().strip().replace("_", " ").replace("-", " ")




