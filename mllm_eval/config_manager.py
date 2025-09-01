"""
Configuration management for 4D Scene Graph VQA experiments
"""
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from feature_processor import FeatureProcessorConfig


@dataclass
class ModelConfig:
    """Configuration for MLLM models"""
    name: str
    type: str  # "local" or "api"
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphConfig:
    """Configuration for scene graph processing"""
    source_type: str  # "2d", "3d", "4d"
    encoding_format: str  # "json", "xml", "text", "som"
    file_format: str = "auto"  # "json", "pickle", "graphml", "auto"
    temporal_strategy: str = "single_file"  # "single_file", "multi_file"
    temporal_frames: int = 1
    graph_dir: str = ""
    include_spatial_info: bool = True
    include_temporal_info: bool = False

    # Feature processing configuration
    feature_processing: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def get_feature_processor_config(self) -> FeatureProcessorConfig:
        """Create FeatureProcessorConfig from scene graph config"""
        if not self.feature_processing:
            # Return default config
            return FeatureProcessorConfig()
        
        # Handle preset-based configuration
        if "preset" in self.feature_processing:
            config = FeatureProcessorConfig(preset=self.feature_processing["preset"])
            
            # Apply any additional overrides
            overrides = {k: v for k, v in self.feature_processing.items() if k != "preset"}
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
        else:
            # Create config from individual parameters
            return FeatureProcessorConfig(**self.feature_processing)


@dataclass
class DataConfig:
    """Configuration for dataset"""
    data_root: str
    train_sequences: List[str] = field(default_factory=list)
    val_sequences: List[str] = field(default_factory=list)
    test_sequences: List[str] = field(default_factory=list)
    question_types: List[str] = field(default_factory=list)
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    name: str
    model: ModelConfig
    scene_graph: SceneGraphConfig
    data: DataConfig
    output_dir: str
    save_predictions: bool = True
    compute_baseline: bool = False
    novel_view_synthesis: bool = False


class ConfigManager:
    """Manages experiment configurations"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> ExperimentConfig:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclasses
        model_config = ModelConfig(**config_dict['model'])
        scene_graph_config = SceneGraphConfig(**config_dict['scene_graph'])
        data_config = DataConfig(**config_dict['data'])
        
        experiment_config = ExperimentConfig(
            name=config_dict['name'],
            model=model_config,
            scene_graph=scene_graph_config,
            data=data_config,
            output_dir=config_dict['output_dir'],
            save_predictions=config_dict.get('save_predictions', True),
            compute_baseline=config_dict.get('compute_baseline', False),
            novel_view_synthesis=config_dict.get('novel_view_synthesis', False)
        )
        
        return experiment_config
    
    def get_config(self) -> ExperimentConfig:
        """Get the loaded configuration"""
        return self.config
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        config = self.config
        
        # Validate paths exist
        if not Path(config.data.data_root).exists():
            raise ValueError(f"Data root does not exist: {config.data.data_root}")
        
        if config.scene_graph.graph_dir and not Path(config.scene_graph.graph_dir).exists():
            raise ValueError(f"Scene graph directory does not exist: {config.scene_graph.graph_dir}")
        
        # Validate model configuration
        if config.model.type == "local" and not config.model.model_path:
            raise ValueError("Local models require model_path")
        
        if config.model.type == "api" and not config.model.api_key:
            raise ValueError("API models require api_key")
        
        # Validate scene graph configuration
        valid_formats = ["json", "xml", "text", "som", "graphtext"]
        if config.scene_graph.encoding_format not in valid_formats:
            raise ValueError(f"Invalid encoding format. Must be one of: {valid_formats}")
        
        valid_file_formats = ["json", "pickle", "graphml", "auto"]
        if config.scene_graph.file_format not in valid_file_formats:
            raise ValueError(f"Invalid file format. Must be one of: {valid_file_formats}")
        
        valid_sources = ["2d", "3d", "4d"]
        if config.scene_graph.source_type not in valid_sources:
            raise ValueError(f"Invalid source type. Must be one of: {valid_sources}")
        
        return True
    
    @staticmethod
    def create_example_config(output_path: str):
        """Create an example configuration file with feature processing options"""
        example_config = {
            'name': 'example_4d_scene_graph_experiment',
            'model': {
                'name': 'qwen2.5-vl-7b-instruct',
                'type': 'local',
                'model_path': '/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct',
                'max_tokens': 500,
                'temperature': 0.1
            },
            'scene_graph': {
                'source_type': '4d',
                'encoding_format': 'json',
                'temporal_frames': 3,
                'graph_dir': './data/scene_graphs_4d/',
                'include_spatial_info': True,
                'include_temporal_info': True,
                # Feature processing configuration
                'feature_processing': {
                    'preset': 'surgical_optimized'  # or 'default', 'fast_inference', 'research_mode'
                }
            },
            'data': {
                'data_root': './data/',
                'train_sequences': ['VID73', 'VID40', 'VID62'],
                'val_sequences': ['VID18', 'VID48', 'VID01'],
                'test_sequences': ['VID22', 'VID74', 'VID60'],
                'question_types': ['all'],
                'batch_size': 16
            },
            'output_dir': './experiments/example_run/',
            'save_predictions': True,
            'compute_baseline': True,
            'novel_view_synthesis': False
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        
        print(f"Example configuration saved to: {output_path}")