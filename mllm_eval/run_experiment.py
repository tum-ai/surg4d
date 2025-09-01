#!/usr/bin/env python3
"""
Main entry point for running 4D Scene Graph VQA experiments

Usage:
    python run_experiment.py --config configs/example.yaml
    python run_experiment.py --config configs/example.yaml --dry-run
    python run_experiment.py --create-example-config configs/new_example.yaml
"""

import sys
import argparse
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from experiment_runner import ExperimentRunner
from config_manager import ConfigManager


def main():
    parser = argparse.ArgumentParser(
        description="4D Scene Graph VQA Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a full experiment
  python run_experiment.py --config configs/baseline.yaml

  # Validate configuration without running
  python run_experiment.py --config configs/baseline.yaml --dry-run

  # Create an example configuration file
  python run_experiment.py --create-example-config configs/new_config.yaml

  # Run with specific output directory
  python run_experiment.py --config configs/baseline.yaml --output-dir ./my_experiments/

Configuration file should contain:
  - model: Model configuration (local or API)
  - scene_graph: Scene graph settings
  - data: Dataset configuration
  - output_dir: Results directory
  - Additional experiment parameters
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to experiment configuration YAML file"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Validate configuration and setup without running experiments"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    # Utility arguments
    parser.add_argument(
        "--create-example-config",
        type=str,
        help="Create an example configuration file at the specified path"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations"
    )
    
    parser.add_argument(
        "--validate-data",
        type=str,
        help="Validate data directory structure"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands first
    if args.create_example_config:
        print(f"Creating example configuration file: {args.create_example_config}")
        ConfigManager.create_example_config(args.create_example_config)
        return 0
    
    if args.list_models:
        print_available_models()
        return 0
    
    if args.validate_data:
        validate_data_directory(args.validate_data)
        return 0
    
    # Main experiment logic
    if not args.config:
        print("Error: --config is required for running experiments")
        parser.print_help()
        return 1
    
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    try:
        print("="*80)
        print("4D SCENE GRAPH VQA EXPERIMENT RUNNER")
        print("="*80)
        print(f"Configuration: {args.config}")
        
        # Initialize experiment runner
        runner = ExperimentRunner(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            runner.config.output_dir = args.output_dir
            runner.setup_experiment_directory()
            print(f"Output directory overridden: {args.output_dir}")
        
        if args.dry_run:
            print("\nDRY RUN MODE - Validating configuration...")
            
            # Validate configuration
            runner.config_manager.validate_config()
            print("✓ Configuration is valid")
            
            # Check model availability
            model_info = runner.mllm_interface.get_model_info()
            print(f"✓ Model: {model_info['name']} ({model_info['type']})")
            
            # Check data availability
            data_stats = validate_data_availability(runner.config)
            print(f"✓ Data: {data_stats['total_sequences']} sequences, {data_stats['total_samples']} samples")
            
            print("\nDry run completed successfully!")
            print("Configuration is valid and ready for execution.")
            return 0
        
        # Run the full experiment
        print(f"\nStarting experiment: {runner.config.name}")
        runner.run_experiment()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {runner.experiment_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_available_models():
    """Print information about available model configurations"""
    print("Available Model Configurations:")
    print("\nLocal Models (HuggingFace):")
    local_models = [
        # TODO
    ]
    
    for model in local_models:
        print(f"  - {model}")
    
    print("\nAPI Models:")
    api_models = [
        # TODO
    ]
    
    for model in api_models:
        print(f"  - {model}")
    
    print("\nNote: Local models require sufficient GPU memory.")
    print("API models require valid API keys.")


def validate_data_directory(data_root: str):
    """Validate data directory structure"""
    data_path = Path(data_root)
    
    print(f"Validating data directory: {data_path}")
    
    if not data_path.exists():
        print(f"X Data directory does not exist: {data_path}")
        return False
    
    # Check required subdirectories
    required_dirs = ["qa_txt", "visual_feats"]
    optional_dirs = ["scene_graphs", "images", "CholecT50"]
    
    print("\nRequired directories:")
    all_required_exist = True
    for dirname in required_dirs:
        dir_path = data_path / dirname
        if dir_path.exists():
            print(f"  y {dirname}/")
        else:
            print(f"  X {dirname}/ (missing)")
            all_required_exist = False
    
    print("\nOptional directories:")
    for dirname in optional_dirs:
        dir_path = data_path / dirname
        if dir_path.exists():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  - {dirname}/ (not found)")
    
    # Check for sequence data
    qa_txt_path = data_path / "qa_txt"
    if qa_txt_path.exists():
        sequences = [d.name for d in qa_txt_path.iterdir() if d.is_dir()]
        print(f"\nFound {len(sequences)} sequences in qa_txt/")
        if sequences:
            print(f"  Examples: {', '.join(sequences[:5])}")
            if len(sequences) > 5:
                print(f"  ... and {len(sequences) - 5} more")
    
    return all_required_exist


def validate_data_availability(config):
    """Validate that required data is available for the experiment"""
    from dataset_loader import DatasetFactory
    
    stats = {
        "total_sequences": 0,
        "total_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0
    }
    
    all_sequences = (config.data.train_sequences + 
                    config.data.val_sequences + 
                    config.data.test_sequences)
    stats["total_sequences"] = len(set(all_sequences))
    
    # Check test data (most important)
    if config.data.test_sequences:
        try:
            test_dataset = DatasetFactory.create_dataset(
                config.data.test_sequences,
                config.data.data_root,
                config.scene_graph,
                question_types=config.data.question_types,
                load_images=False  # Quick check without loading images
            )
            stats["test_samples"] = len(test_dataset)
            stats["total_samples"] += stats["test_samples"]
        except Exception as e:
            print(f"Warning: Could not load test data: {e}")
    
    # Check validation data
    if config.data.val_sequences:
        try:
            val_dataset = DatasetFactory.create_dataset(
                config.data.val_sequences,
                config.data.data_root,
                config.scene_graph,
                question_types=config.data.question_types,
                load_images=False
            )
            stats["val_samples"] = len(val_dataset)
            stats["total_samples"] += stats["val_samples"]
        except Exception as e:
            print(f"Warning: Could not load validation data: {e}")
    
    return stats


if __name__ == "__main__":
    exit(main())