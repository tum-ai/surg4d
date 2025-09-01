"""
Main experiment runner for 4D Scene Graph VQA experiments
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_manager import ConfigManager, ExperimentConfig
from dataset_loader import DatasetFactory, create_data_loaders, collate_vqa_batch
from mllm_interface import MLLMFactory, ResponseProcessor
from scene_graph_encoder import SceneGraphEncoderFactory
from results_analyzer import ResultsAnalyzer


class ExperimentRunner:
    """Main class for running 4D Scene Graph VQA experiments"""
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Validate configuration
        self.config_manager.validate_config()
        
        # Setup experiment directory
        self.setup_experiment_directory()
        
        # Initialize components
        self.mllm_interface = MLLMFactory.create_interface(self.config.model)
        self.response_processor = ResponseProcessor()
        self.results_analyzer = ResultsAnalyzer(self.config.data.data_root)
        
        print(f"Experiment: {self.config.name}")
        print(f"Model: {self.config.model.name} ({self.config.model.type})")
        print(f"Scene Graph: {self.config.scene_graph.source_type} -> {self.config.scene_graph.encoding_format}")
        print(f"Output Directory: {self.experiment_dir}")
    
    def setup_experiment_directory(self):
        """Create experiment directory and save configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(self.config.output_dir) / f"{self.config.name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_save_path = self.experiment_dir / "experiment_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("\n" + "="*80)
        print("STARTING EXPERIMENT")
        print("="*80)
        
        start_time = time.time()
        
        # Create datasets
        print("\n1. Loading datasets...")
        data_loaders = self._create_data_loaders()
        
        # Run evaluation on test set
        if "test" in data_loaders:
            print("\n2. Running evaluation on test set...")
            test_results = self._evaluate_dataset(data_loaders["test"], "test")
            self._save_results(test_results, "test")
        
        # Run baseline comparison if requested
        if self.config.compute_baseline:
            print("\n3. Running baseline comparison...")
            baseline_results = self._run_baseline_comparison(data_loaders.get("test"))
            if baseline_results:
                self._save_results(baseline_results, "baseline")
        
        # Run ablation studies
        print("\n4. Running ablation studies...")
        ablation_results = self._run_ablation_studies(data_loaders.get("test"))
        if ablation_results:
            self._save_results(ablation_results, "ablation")
        
        # Generate final report
        print("\n5. Generating final report...")
        self._generate_final_report()
        
        total_time = time.time() - start_time
        print(f"\nExperiment completed in {total_time:.2f} seconds")
        print(f"Results saved to: {self.experiment_dir}")
    
    def _create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for train, validation, and test sets"""
        return create_data_loaders(
            self.config.data,
            self.config.scene_graph,
            batch_size=self.config.data.batch_size
        )
    
    def _evaluate_dataset(self, data_loader: DataLoader, split_name: str) -> Dict[str, Any]:
        """Evaluate the model on a dataset split"""
        print(f"Evaluating on {split_name} set ({len(data_loader)} batches)...")
        
        all_predictions = []
        all_ground_truth = []
        all_responses = []
        all_metadata = []
        
        # Track performance metrics
        correct_predictions = 0
        total_predictions = 0
        processing_times = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {split_name}")):
            batch_start_time = time.time()
            
            batch_size = len(batch["questions"])
            
            for i in range(batch_size):
                # Extract sample data
                sample_data = {
                    "sequence_id": batch["sequence_ids"][i],
                    "frame_id": batch["frame_ids"][i],
                    "question": batch["questions"][i],
                    "ground_truth": batch["answers"][i],
                    "question_type": batch["question_types"][i],
                    "question_category": batch["question_categories"][i],
                    "scene_graph": batch["scene_graphs"][i] if batch["scene_graphs"][i] else None,
                    "temporal_context": batch["temporal_contexts"][i] if batch["temporal_contexts"][i] else None
                }
                
                # Get image if available
                image = None
                if "images" in batch:
                    image = batch["images"][i]
                elif "visual_features" in batch:
                    # For now, skip samples without actual images
                    # TODO: Implement feature-to-image conversion if needed
                    continue
                
                # Query the model
                try:
                    response = self._query_model(sample_data, image)
                    processed_response = self.response_processor.process_response(response)
                    
                    # Store results
                    all_predictions.append(processed_response)
                    all_ground_truth.append(sample_data["ground_truth"])
                    all_responses.append(response)
                    all_metadata.append({
                        "sequence_id": sample_data["sequence_id"],
                        "frame_id": sample_data["frame_id"],
                        "question": sample_data["question"],
                        "question_type": sample_data["question_type"],
                        "question_category": sample_data["question_category"],
                        "has_scene_graph": sample_data["scene_graph"] is not None
                    })
                    
                    # Track accuracy
                    if processed_response == sample_data["ground_truth"]:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    print(f"Error processing sample {sample_data['sequence_id']}/{sample_data['frame_id']}: {e}")
                    continue
            
            batch_time = time.time() - batch_start_time
            processing_times.append(batch_time)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_time = sum(processing_times[-10:]) / len(processing_times[-10:])
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Batch {batch_idx + 1}/{len(data_loader)} - "
                      f"Accuracy: {accuracy:.3f} - "
                      f"Avg Time: {avg_time:.2f}s/batch")
        
        # Compute final metrics
        results = {
            "split": split_name,
            "total_samples": total_predictions,
            "predictions": all_predictions,
            "ground_truth": all_ground_truth,
            "raw_responses": all_responses,
            "metadata": all_metadata,
            "processing_time": {
                "total_time": sum(processing_times),
                "avg_batch_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "avg_sample_time": sum(processing_times) / total_predictions if total_predictions > 0 else 0
            }
        }
        
        # Compute detailed metrics using results analyzer
        detailed_metrics = self.results_analyzer.compute_metrics(all_ground_truth, all_predictions)
        results.update(detailed_metrics)
        
        # Compute per-question-type metrics
        question_type_metrics = self.results_analyzer.compute_question_type_metrics(
            all_ground_truth, all_predictions, all_metadata
        )
        results["question_type_metrics"] = question_type_metrics
        
        print(f"\n{split_name.upper()} RESULTS:")
        print(f"Total Samples: {total_predictions}")
        print(f"Overall Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"Mean F1 Score: {results.get('mean_f1', 0):.3f}")
        print(f"Processing Time: {results['processing_time']['total_time']:.2f}s")
        
        return results
    
    def _query_model(self, sample_data: Dict, image) -> str:
        """Query the MLLM with a single sample"""
        question = sample_data["question"]
        scene_graph = sample_data.get("scene_graph")
        temporal_context = sample_data.get("temporal_context")
        
        # Combine additional context
        additional_context = None
        if temporal_context:
            additional_context = temporal_context
        
        # Query the model
        response = self.mllm_interface.query(
            image=image,
            question=question,
            scene_graph=scene_graph,
            additional_context=additional_context
        )
        
        return response
    
    def _run_baseline_comparison(self, test_loader: Optional[DataLoader]) -> Optional[Dict]:
        """Run baseline comparison without scene graphs"""
        if not test_loader:
            print("No test data available for baseline comparison")
            return None
        
        print("Running baseline (no scene graph) evaluation...")
        
        # Temporarily disable scene graph encoding
        original_config = self.config.scene_graph
        baseline_config = type(original_config)(
            source_type=original_config.source_type,
            encoding_format=original_config.encoding_format,
            temporal_frames=1,
            graph_dir="",  # Disable scene graph loading
            include_spatial_info=False,
            include_temporal_info=False
        )
        
        # Create baseline dataset
        baseline_dataset = DatasetFactory.create_dataset(
            self.config.data.test_sequences,
            self.config.data.data_root,
            baseline_config,
            question_types=self.config.data.question_types,
            load_images=True
        )
        
        baseline_loader = DataLoader(
            baseline_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=collate_vqa_batch
        )
        
        # Run evaluation
        baseline_results = self._evaluate_dataset(baseline_loader, "baseline")
        baseline_results["config"] = "no_scene_graph"
        
        return baseline_results
    
    def _run_ablation_studies(self, test_loader: Optional[DataLoader]) -> Optional[Dict]:
        """Run ablation studies with different encoding formats"""
        if not test_loader:
            print("No test data available for ablation studies")
            return None
        
        encoding_formats = ["json", "xml", "text", "som"]
        current_format = self.config.scene_graph.encoding_format
        
        # Remove current format from ablation list
        ablation_formats = [fmt for fmt in encoding_formats if fmt != current_format]
        
        if not ablation_formats:
            print("No additional encoding formats to test")
            return None
        
        ablation_results = {}
        
        for encoding_format in ablation_formats:
            print(f"\nTesting encoding format: {encoding_format}")
            
            # Create dataset with different encoding
            ablation_config = type(self.config.scene_graph)(
                source_type=self.config.scene_graph.source_type,
                encoding_format=encoding_format,
                temporal_frames=self.config.scene_graph.temporal_frames,
                graph_dir=self.config.scene_graph.graph_dir,
                include_spatial_info=self.config.scene_graph.include_spatial_info,
                include_temporal_info=self.config.scene_graph.include_temporal_info
            )
            
            ablation_dataset = DatasetFactory.create_dataset(
                self.config.data.test_sequences,
                self.config.data.data_root,
                ablation_config,
                question_types=self.config.data.question_types,
                load_images=True
            )
            
            ablation_loader = DataLoader(
                ablation_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                collate_fn=collate_vqa_batch
            )
            
            # Run evaluation
            format_results = self._evaluate_dataset(ablation_loader, f"ablation_{encoding_format}")
            format_results["config"] = f"encoding_{encoding_format}"
            ablation_results[encoding_format] = format_results
        
        return ablation_results
    
    def _save_results(self, results: Dict, result_type: str):
        """Save results to files"""
        # Save detailed results as JSON
        results_file = self.experiment_dir / f"results_{result_type}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions in CSV format (compatible with SSG-VQA evaluation)
        if "predictions" in results and "ground_truth" in results and "metadata" in results:
            self.results_analyzer.save_predictions_csv(
                results["predictions"],
                results["ground_truth"],
                results["metadata"],
                self.experiment_dir / f"predictions_{result_type}.csv"
            )
    
    def _generate_final_report(self):
        """Generate a comprehensive final report"""
        report_path = self.experiment_dir / "final_report.md"
        
        # Load all results
        results_files = list(self.experiment_dir.glob("results_*.json"))
        all_results = {}
        
        for results_file in results_files:
            result_type = results_file.stem.replace("results_", "")
            with open(results_file, 'r') as f:
                all_results[result_type] = json.load(f)
        
        # Generate markdown report
        report_content = self._create_markdown_report(all_results)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Final report generated: {report_path}")
    
    def _create_markdown_report(self, all_results: Dict) -> str:
        """Create markdown report content"""
        report = []
        report.append(f"# 4D Scene Graph VQA Experiment Report")
        report.append(f"**Experiment:** {self.config.name}")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Configuration summary
        report.append("## Configuration")
        report.append(f"- **Model:** {self.config.model.name} ({self.config.model.type})")
        report.append(f"- **Scene Graph Source:** {self.config.scene_graph.source_type}")
        report.append(f"- **Encoding Format:** {self.config.scene_graph.encoding_format}")
        report.append(f"- **Temporal Frames:** {self.config.scene_graph.temporal_frames}")
        report.append("")
        
        # Results summary
        report.append("## Results Summary")
        
        if "test" in all_results:
            test_results = all_results["test"]
            report.append(f"### Test Set Performance")
            report.append(f"- **Total Samples:** {test_results.get('total_samples', 'N/A')}")
            report.append(f"- **Overall Accuracy:** {test_results.get('accuracy', 0):.3f}")
            report.append(f"- **Mean F1 Score:** {test_results.get('mean_f1', 0):.3f}")
            report.append(f"- **Weighted F1 Score:** {test_results.get('weighted_f1', 0):.3f}")
            report.append("")
        
        # Question type performance
        if "test" in all_results and "question_type_metrics" in all_results["test"]:
            report.append("### Performance by Question Type")
            qt_metrics = all_results["test"]["question_type_metrics"]
            
            report.append("| Question Type | Samples | Accuracy | F1 Score |")
            report.append("|---------------|---------|----------|----------|")
            
            for q_type, metrics in qt_metrics.items():
                samples = metrics.get('count', 0)
                accuracy = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)
                report.append(f"| {q_type} | {samples} | {accuracy:.3f} | {f1:.3f} |")
            
            report.append("")
        
        # Baseline comparison
        if "baseline" in all_results:
            baseline_results = all_results["baseline"]
            test_results = all_results.get("test", {})
            
            report.append("### Baseline Comparison (No Scene Graph)")
            report.append(f"- **Baseline Accuracy:** {baseline_results.get('accuracy', 0):.3f}")
            report.append(f"- **Scene Graph Accuracy:** {test_results.get('accuracy', 0):.3f}")
            
            if test_results.get('accuracy', 0) > 0:
                improvement = test_results.get('accuracy', 0) - baseline_results.get('accuracy', 0)
                report.append(f"- **Improvement:** {improvement:+.3f}")
            
            report.append("")
        
        # Ablation studies
        ablation_results = {k: v for k, v in all_results.items() if k.startswith('ablation')}
        if ablation_results:
            report.append("### Ablation Study: Encoding Formats")
            
            report.append("| Encoding Format | Accuracy | F1 Score |")
            report.append("|-----------------|----------|----------|")
            
            # Add main result
            if "test" in all_results:
                main_format = self.config.scene_graph.encoding_format
                main_acc = all_results["test"].get('accuracy', 0)
                main_f1 = all_results["test"].get('mean_f1', 0)
                report.append(f"| {main_format} (main) | {main_acc:.3f} | {main_f1:.3f} |")
            
            # Add ablation results
            for result_name, result_data in ablation_results.items():
                format_name = result_name.replace('ablation_', '')
                acc = result_data.get('accuracy', 0)
                f1 = result_data.get('mean_f1', 0)
                report.append(f"| {format_name} | {acc:.3f} | {f1:.3f} |")
            
            report.append("")
        
        # Performance analysis
        report.append("### Performance Analysis")
        if "test" in all_results:
            processing_time = all_results["test"].get("processing_time", {})
            avg_sample_time = processing_time.get("avg_sample_time", 0)
            total_time = processing_time.get("total_time", 0)
            
            report.append(f"- **Total Processing Time:** {total_time:.2f} seconds")
            report.append(f"- **Average Time per Sample:** {avg_sample_time:.3f} seconds")
            
            if avg_sample_time > 0:
                throughput = 1.0 / avg_sample_time
                report.append(f"- **Throughput:** {throughput:.2f} samples/second")
        
        report.append("")
        
        # Key findings and recommendations
        report.append("## Key Findings")
        findings = self._generate_findings(all_results)
        for finding in findings:
            report.append(f"- {finding}")
        
        report.append("")
        report.append("## Files Generated")
        report.append("- `experiment_config.json`: Experiment configuration")
        report.append("- `results_test.json`: Detailed test results")
        
        if "baseline" in all_results:
            report.append("- `results_baseline.json`: Baseline comparison results")
        
        if ablation_results:
            report.append("- `results_ablation_*.json`: Ablation study results")
        
        report.append("- `predictions_*.csv`: Prediction files for analysis")
        report.append("- `final_report.md`: This report")
        
        return "\n".join(report)
    
    def _generate_findings(self, all_results: Dict) -> List[str]:
        """Generate key findings from the results"""
        findings = []
        
        # Performance findings
        if "test" in all_results:
            test_acc = all_results["test"].get('accuracy', 0)
            
            if test_acc > 0.7:
                findings.append("Strong overall performance achieved")
            elif test_acc > 0.5:
                findings.append("Moderate performance with room for improvement")
            else:
                findings.append("Performance below expectations, requires investigation")
        
        # Scene graph impact
        if "baseline" in all_results and "test" in all_results:
            baseline_acc = all_results["baseline"].get('accuracy', 0)
            test_acc = all_results["test"].get('accuracy', 0)
            improvement = test_acc - baseline_acc
            
            if improvement > 0.05:
                findings.append(f"Scene graphs provide significant improvement (+{improvement:.3f})")
            elif improvement > 0.01:
                findings.append(f"Scene graphs provide modest improvement (+{improvement:.3f})")
            else:
                findings.append("Scene graphs show limited benefit in current setup")
        
        # Question type analysis
        if "test" in all_results and "question_type_metrics" in all_results["test"]:
            qt_metrics = all_results["test"]["question_type_metrics"]
            
            # Find best and worst performing question types
            sorted_types = sorted(qt_metrics.items(), key=lambda x: x[1].get('accuracy', 0))
            
            if len(sorted_types) > 1:
                worst_type = sorted_types[0][0]
                best_type = sorted_types[-1][0]
                findings.append(f"Best performance on '{best_type}' questions")
                findings.append(f"Challenging performance on '{worst_type}' questions")
        
        # Encoding format analysis
        ablation_results = {k: v for k, v in all_results.items() if k.startswith('ablation')}
        if ablation_results and "test" in all_results:
            main_acc = all_results["test"].get('accuracy', 0)
            main_format = self.config.scene_graph.encoding_format
            
            format_performances = [(main_format, main_acc)]
            for result_name, result_data in ablation_results.items():
                format_name = result_name.replace('ablation_', '')
                acc = result_data.get('accuracy', 0)
                format_performances.append((format_name, acc))
            
            best_format = max(format_performances, key=lambda x: x[1])
            findings.append(f"Best encoding format: {best_format[0]} ({best_format[1]:.3f})")
        
        return findings


def main():
    """Main entry point for running experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 4D Scene Graph VQA Experiments")
    parser.add_argument("--config", required=True, help="Path to experiment configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running")
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner(args.config)
        
        if args.dry_run:
            print("Configuration validated successfully!")
            print("Dry run completed - no experiments were executed.")
            return
        
        # Run the experiment
        runner.run_experiment()
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
