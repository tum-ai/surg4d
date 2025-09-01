# MLLM Evaluation Pipeline

A comprehensive evaluation framework for Multimodal Large Language Models (MLLMs) on surgical Video Question Answering (VQA) tasks, with integrated scene graph support for enhanced reasoning capabilities.

## Overview

This pipeline evaluates how well MLLMs can leverage structured scene graph information alongside visual and textual cues to answer questions about surgical procedures. It extends the SSG-VQA dataset structure with flexible scene graph encoding strategies and supports both local HuggingFace models and API-based models.

## Architecture

### Core Components

- **Configuration Management**: Centralized configuration system for experiments, models, and data
- **Data Loading & Processing**: Multi-modal dataset integration with scene graph support
- **Scene Graph Processing**: Flexible encoding strategies for structured surgical scene information
- **Model Interface**: Unified interface supporting local and API-based MLLMs
- **Experiment Orchestration**: Automated experiment running and result collection
- **Results Analysis**: Comprehensive evaluation metrics and analysis tools

### Data Flow

```
Raw Data Sources → Dataset Loader → Scene Graph Encoder → MLLM Interface → Results Analyzer
     ↓                    ↓                ↓                    ↓              ↓
- VQA Questions      - Image Loading   - JSON Encoding    - Local Models   - Accuracy Metrics
- Scene Graphs       - Visual Features - NL Encoding      - API Models     - Question Analysis
- Visual Data        - Graph Loading   - Temporal Context - Response Proc. - Error Analysis
```

## Quick Start

### 1. Setup Configuration

Create a configuration file based on `example_config.yaml`:

```yaml
# Data configuration
data:
  data_root: "./data/"
  train_sequences: ["VID01", "VID02"]
  val_sequences: ["VID03"]
  test_sequences: ["VID04", "VID05"]
  question_types: ["all"]  # or specific types like ["exist", "count", "query_color"]

# Scene graph configuration
scene_graph:
  source_type: "2d"  # "2d", "3d", or "4d"
  encoding_format: "natural_language"  # "json" or "natural_language"
  graph_dir: "./data/scene_graphs/"
  temporal_frames: 1
  include_spatial_info: true
  include_temporal_info: false

# Model configuration
model:
  name: "qwen2.5-vl-7b-instruct"
  type: "local"  # "local" or "api"
  model_path: "/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct"
  max_tokens: 500
  temperature: 0.1
```

### 2. Run Evaluation

```python
python run_experiment.py --config example_config.yaml
```

### 3. Analyze Results

```python
from results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer("./results/experiment_results.json")
analyzer.generate_comprehensive_report()
```

## Detailed Component Guide

### Dataset Loader (`dataset_loader.py`)

The dataset loader integrates multiple data modalities while maintaining SSG-VQA compatibility:

#### Key Features:
- **Multi-modal Integration**: Combines images, visual features, text, and scene graphs
- **SSG-VQA Compatibility**: Maintains original dataset structure and evaluation metrics
- **Flexible Loading**: Supports various image formats and feature representations
- **Question Analysis**: Built-in question type categorization and complexity analysis

#### Usage:
```python
from dataset_loader import DatasetFactory
from config_manager import SceneGraphConfig

config = SceneGraphConfig(
    source_type="2d",
    encoding_format="natural_language",
    temporal_frames=1,
    graph_dir="./data/scene_graphs/"
)

dataset = DatasetFactory.create_dataset(
    sequences=["VID01", "VID02"],
    data_root="./data/",
    scene_graph_config=config,
    question_types=["exist", "count", "query_color"]
)
```

### Scene Graph Processing

#### Scene Graph Loader (`scene_graph_loader.py`)

Loads and processes scene graphs from various sources:

- **2D Scene Graphs**: Frame-level object detection and relationships
- **3D Scene Graphs**: Spatial relationships in 3D surgical scenes
- **4D Scene Graphs**: Temporal evolution of surgical scenes
- **Temporal Context**: Multi-frame temporal relationship extraction

#### Scene Graph Encoder (`scene_graph_encoder.py`)

Provides flexible encoding strategies for different model types:

##### JSON Encoding
Structured representation preserving all graph information:

```json
{
  "objects": [
    {"id": "obj_1", "type": "grasper", "attributes": {"color": "silver"}},
    {"id": "obj_2", "type": "gallbladder", "attributes": {"state": "inflated"}}
  ],
  "relationships": [
    {"subject": "obj_1", "predicate": "grasping", "object": "obj_2"}
  ],
  "spatial_info": {...},
  "temporal_context": {...}
}
```

##### Natural Language Encoding
Human-readable descriptions optimized for language models:

```
Scene Description:
- A silver grasper is present in the scene
- The gallbladder appears inflated
- The grasper is currently grasping the gallbladder
- Spatial relationship: The grasper is positioned above the gallbladder
```

#### Configuration Options:

```yaml
scene_graph:
  # Source type determines which scene graph data to load
  source_type: "2d"  # Options: "2d", "3d", "4d"
  
  # Encoding format affects how graphs are presented to models
  encoding_format: "natural_language"  # Options: "json", "natural_language"
  
  # Temporal processing
  temporal_frames: 3  # Number of frames for temporal context
  include_temporal_info: true
  
  # Spatial information inclusion
  include_spatial_info: true
  
  # Advanced options
  relationship_types: ["spatial", "functional", "temporal"]
  attribute_filtering: ["color", "state", "position"]
  context_window: 5  # For temporal scene graphs
```

### MLLM Interface (`mllm_interface.py`)

Unified interface supporting multiple model types and deployment strategies:

#### Local Models
Supports HuggingFace models with local inference:

```python
local_config = ModelConfig(
    name="qwen2.5-vl-7b-instruct",
    type="local",
    model_path="/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct",
    max_tokens=500,
    temperature=0.1
)

interface = MLLMFactory.create_interface(local_config)
```

#### API Models
Supports OpenAI, OpenRouter, and other API services:

```python
api_config = ModelConfig(
    name="gpt-4-vision-preview",
    type="api",
    api_key="your-api-key",
    api_base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
    temperature: 0.1
)

interface = MLLMFactory.create_interface(api_config)
```

#### Response Processing
Automatic response normalization and mapping to SSG-VQA labels:

- **Label Mapping**: Converts various response formats to standardized labels
- **Confidence Extraction**: Extracts confidence scores when available
- **Error Handling**: Robust handling of API errors and model failures

### Experiment Runner (`experiment_runner.py`, `run_experiment.py`)

Orchestrates comprehensive evaluation experiments:

#### Features:
- **Batch Processing**: Efficient processing of large datasets
- **Progress Tracking**: Real-time progress monitoring
- **Error Recovery**: Automatic retry mechanisms for failed queries
- **Result Persistence**: Automatic saving and loading of experiment results
- **Multi-configuration Support**: Compare multiple models and settings

#### Usage:
```bash
# Run single experiment
python run_experiment.py --config config.yaml

# Run multiple configurations
python run_experiment.py --config-dir ./configs/ --output-dir ./results/

# Resume interrupted experiment
python run_experiment.py --config config.yaml --resume ./results/partial_results.json
```

### Results Analyzer (`results_analyzer.py`)

Comprehensive analysis and reporting tools:

#### Analysis Types:
- **Overall Performance**: Accuracy, F1-score, precision, recall
- **Question Type Analysis**: Performance breakdown by question categories
- **Complexity Analysis**: Zero-hop, one-hop, multi-hop reasoning evaluation
- **Scene Graph Impact**: Comparison with and without scene graph information
- **Error Analysis**: Common failure modes and error patterns

#### Report Generation:
```python
analyzer = ResultsAnalyzer("results.json")

# Generate full report
report = analyzer.generate_comprehensive_report()

# Specific analyses
question_stats = analyzer.analyze_by_question_type()
complexity_stats = analyzer.analyze_by_complexity()
scene_graph_impact = analyzer.analyze_scene_graph_impact()
```

## Configuration Reference

### Complete Configuration Example

```yaml
# Experiment metadata
experiment:
  name: "scene_graph_evaluation"
  description: "Evaluating MLLM performance with scene graph integration"
  output_dir: "./results/"

# Data configuration
data:
  data_root: "./data/"
  train_sequences: ["VID01", "VID02", "VID03", "VID04", "VID05"]
  val_sequences: ["VID06", "VID07"]
  test_sequences: ["VID08", "VID09", "VID10"]
  question_types: ["all"]  # or ["exist", "count", "query_color", "query_location", "query_type"]
  load_images: true
  batch_size: 16

# Scene graph configuration
scene_graph:
  source_type: "2d"  # "2d": frame-level, "3d": spatial, "4d": temporal
  encoding_format: "natural_language"  # "json": structured, "natural_language": descriptive
  graph_dir: "./data/scene_graphs/"
  
  # Temporal settings
  temporal_frames: 3
  include_temporal_info: true
  context_window: 5
  
  # Spatial settings
  include_spatial_info: true
  spatial_resolution: "high"  # "low", "medium", "high"
  
  # Content filtering
  relationship_types: ["spatial", "functional", "temporal", "semantic"]
  attribute_filtering: ["color", "state", "position", "size", "orientation"]
  
  # Advanced options
  graph_simplification: false
  max_objects: 20
  max_relationships: 50

# Model configuration
model:
  name: "qwen2.5-vl-7b-instruct"
  type: "local"  # "local" for HuggingFace models, "api" for external APIs
  
  # Local model settings
  model_path: "/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct"
  device: "auto"  # "auto", "cuda", "cpu"
  torch_dtype: "auto"  # "auto", "float16", "float32"
  
  # API model settings (if type: "api")
  # api_key: "your-api-key"
  # api_base_url: "https://openrouter.ai/api/v1"
  
  # Generation parameters
  max_tokens: 500
  temperature: 0.1
  top_p: 0.9
  
  # Additional parameters
  additional_params:
    do_sample: true
    num_beams: 1
    repetition_penalty: 1.1

# Evaluation settings
evaluation:
  metrics: ["accuracy", "f1", "precision", "recall"]
  save_predictions: true
  save_intermediate: true
  error_analysis: true
  
  # Analysis settings
  analyze_by_question_type: true
  analyze_by_complexity: true
  analyze_scene_graph_impact: true
  
  # Output settings
  generate_report: true
  plot_results: true
  save_visualizations: true
```

## Supported Models

### Local Models (HuggingFace)
- **Qwen2.5-VL variants**: Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct
- **LLaVA variants**: LLaVA-1.5-7B, LLaVA-1.5-13B (legacy support)
- **Generic multimodal models**: Any HuggingFace model with vision capabilities
- **Custom models**: Support for custom model architectures

### API Models
- **OpenAI**: GPT-4 Vision, GPT-4 Turbo with Vision
- **OpenRouter**: Access to various vision-language models
- **Custom APIs**: Configurable API endpoints

### Model-Specific Features

#### Qwen2.5-VL Integration
The pipeline provides native support for Qwen2.5-VL models with:

- **Optimized Chat Template**: Uses the official Qwen chat message format
- **Vision Processing**: Integrates `qwen_vl_utils` for proper image handling  
- **Automatic Detection**: Models are automatically detected based on path/name
- **Flexible Token Management**: Configurable visual token ranges for performance tuning

Example Qwen2.5-VL configuration:
```yaml
model:
  name: "qwen2.5-vl-7b-instruct"
  type: "local" 
  model_path: "/path/to/Qwen--Qwen2.5-VL-7B-Instruct"
  torch_dtype: "auto"  # Uses model's recommended dtype
  device_map: "auto"   # Automatic device placement
  max_tokens: 500
  temperature: 0.1
```

## Question Types and Complexity Levels

### Question Types (SSG-VQA Compatible)
- **Existence**: "Is there a grasper in the scene?"
- **Counting**: "How many instruments are visible?"
- **Color Query**: "What color is the grasper?"
- **Location Query**: "Where is the gallbladder located?"
- **Type Query**: "What type of instrument is this?"
- **Action Query**: "What action is being performed?"

### Complexity Levels
- **Zero-hop**: Direct observation questions
- **One-hop**: Single relationship reasoning
- **Multi-hop**: Complex reasoning across multiple relationships

## Development and Debugging

### Development Notebook
Use `dev_fabian.ipynb` for:
- Interactive development and testing
- Data exploration and visualization
- Model debugging and prompt engineering
- Result analysis and visualization

### Debugging Tips
1. **Start with small datasets** for rapid iteration
2. **Use `load_images: false`** for faster debugging without visual data
3. **Enable verbose logging** in configuration
4. **Test with single samples** before batch processing
5. **Validate scene graph loading** before running full experiments

## Performance Optimization

### Local Model Optimization
- Use `torch_dtype: "auto"` for optimal performance with Qwen2.5-VL
- Enable `device_map: "auto"` for multi-GPU setups
- Adjust batch size based on available memory
- Consider model quantization for resource-constrained environments

### API Model Optimization
- Implement request batching and rate limiting
- Add retry logic with exponential backoff
- Cache responses to avoid redundant API calls
- Monitor API usage and costs

## Troubleshooting

### Common Issues

1. **Scene Graph Loading Errors**
   - Verify `graph_dir` path exists and contains valid data
   - Check `source_type` matches available scene graph format
   - Ensure proper file naming conventions

2. **Model Loading Issues**
   - Verify model path and availability
   - Check CUDA availability and memory
   - Ensure proper model dependencies are installed
   - For Qwen2.5-VL: Install `qwen-vl-utils` package

3. **API Connection Problems**
   - Verify API key and base URL
   - Check network connectivity and rate limits
   - Ensure proper API endpoint configuration

4. **Memory Issues**
   - Reduce batch size
   - Use smaller models or quantization
   - Enable gradient checkpointing for local models

### Error Recovery
The pipeline includes automatic error recovery mechanisms:
- Failed API calls are retried with exponential backoff
- Corrupted data samples are skipped with logging
- Partial results are saved and can be resumed
- Configuration validation prevents common setup errors

## Contributing

When extending the pipeline:

1. **Follow the modular design**: Keep components separate and well-defined
2. **Maintain SSG-VQA compatibility**: Ensure new features don't break existing functionality
3. **Add comprehensive logging**: Include informative error messages and progress tracking
4. **Update configuration schema**: Document new configuration options
5. **Include tests**: Add test cases for new functionality

## Citation

If you use this evaluation pipeline in your research, please cite the relevant papers and acknowledge the SSG-VQA dataset that forms the foundation of this work.