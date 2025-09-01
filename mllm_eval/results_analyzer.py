"""
Results analysis and evaluation metrics (adapted from SSG-VQA evaluation)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import json


class ResultsAnalyzer:
    """Analyzes and computes metrics for VQA experiments"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        
        # SSG-VQA label set for compatibility
        self.labels = [
            "0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9",
            "False", "True",
            "abdominal_wall_cavity", "adhesion", "anatomy", "aspirate", "bipolar",
            "blood_vessel", "blue", "brown", "clip", "clipper", "coagulate", "cut",
            "cystic_artery", "cystic_duct", "cystic_pedicle", "cystic_plate",
            "dissect", "fluid", "gallbladder", "grasp", "grasper", "gut", "hook",
            "instrument", "irrigate", "irrigator", "liver", "omentum", "pack",
            "peritoneum", "red", "retract", "scissors", "silver", "specimen_bag",
            "specimenbag", "white", "yellow"
        ]
    
    def compute_metrics(self, ground_truth: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        Compatible with SSG-VQA evaluation methodology
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have the same length")
        
        if len(ground_truth) == 0:
            return {"accuracy": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0}
        
        # Convert to indices for sklearn compatibility
        gt_indices = self._convert_to_indices(ground_truth)
        pred_indices = self._convert_to_indices(predictions)
        
        # Basic accuracy
        accuracy = accuracy_score(gt_indices, pred_indices)
        
        # Precision, recall, F1 (macro and weighted averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_indices, pred_indices, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            gt_indices, pred_indices, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            gt_indices, pred_indices, average=None, zero_division=0
        )
        
        # SSG-VQA style evaluation (following their eval_for_f1_et_all function)
        ssg_metrics = self._compute_ssg_vqa_metrics(gt_indices, pred_indices)
        
        return {
            "accuracy": float(accuracy),
            "mean_precision": float(precision_macro),
            "mean_recall": float(recall_macro),
            "mean_f1": float(f1_macro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1": float(f1_weighted),
            "per_class_precision": precision_per_class.tolist(),
            "per_class_recall": recall_per_class.tolist(),
            "per_class_f1": f1_per_class.tolist(),
            "per_class_support": support.tolist(),
            "ssg_map": ssg_metrics["mAP"],
            "ssg_mar": ssg_metrics["mAR"],
            "ssg_maf1": ssg_metrics["mAf1"],
            "ssg_wf1": ssg_metrics["wf1"],
            "ssg_oa": ssg_metrics["oa"]
        }
    
    def compute_question_type_metrics(self, 
                                    ground_truth: List[str], 
                                    predictions: List[str],
                                    metadata: List[Dict]) -> Dict[str, Dict]:
        """Compute metrics broken down by question type"""
        if len(ground_truth) != len(predictions) != len(metadata):
            raise ValueError("All inputs must have the same length")
        
        # Group by question type
        question_type_groups = {}
        for i, meta in enumerate(metadata):
            q_type = meta.get("question_type", "unknown")
            if q_type not in question_type_groups:
                question_type_groups[q_type] = {"gt": [], "pred": [], "meta": []}
            
            question_type_groups[q_type]["gt"].append(ground_truth[i])
            question_type_groups[q_type]["pred"].append(predictions[i])
            question_type_groups[q_type]["meta"].append(meta)
        
        # Compute metrics for each question type
        question_type_metrics = {}
        for q_type, data in question_type_groups.items():
            if len(data["gt"]) > 0:
                metrics = self.compute_metrics(data["gt"], data["pred"])
                metrics["count"] = len(data["gt"])
                question_type_metrics[q_type] = metrics
        
        return question_type_metrics
    
    def compute_complexity_metrics(self, 
                                 ground_truth: List[str], 
                                 predictions: List[str],
                                 metadata: List[Dict]) -> Dict[str, Dict]:
        """Compute metrics by question complexity (following SSG-VQA analysis)"""
        # Define complexity categories
        complexity_mapping = {
            "zero_hop": ["exist", "count", "query_component"],
            "one_hop": ["query_color", "query_type", "query_location"], 
            "single_and": ["single_and"]
        }
        
        # Group samples by complexity
        complexity_groups = {complexity: {"gt": [], "pred": []} for complexity in complexity_mapping}
        
        for i, meta in enumerate(metadata):
            q_type = meta.get("question_type", "")
            q_category = meta.get("question_category", "")
            
            # Determine complexity
            assigned_complexity = None
            for complexity, types in complexity_mapping.items():
                if q_type in types or q_category in types:
                    assigned_complexity = complexity
                    break
            
            if assigned_complexity:
                complexity_groups[assigned_complexity]["gt"].append(ground_truth[i])
                complexity_groups[assigned_complexity]["pred"].append(predictions[i])
        
        # Compute metrics for each complexity level
        complexity_metrics = {}
        for complexity, data in complexity_groups.items():
            if len(data["gt"]) > 0:
                metrics = self.compute_metrics(data["gt"], data["pred"])
                metrics["count"] = len(data["gt"])
                complexity_metrics[complexity] = metrics
        
        return complexity_metrics
    
    def _convert_to_indices(self, labels: List[str]) -> List[int]:
        """Convert string labels to indices"""
        indices = []
        for label in labels:
            try:
                indices.append(self.labels.index(str(label)))
            except ValueError:
                # Handle unknown labels by assigning to index 0
                indices.append(0)
        return indices
    
    def _compute_ssg_vqa_metrics(self, ground_truth: List[int], predictions: List[int]) -> Dict[str, float]:
        """
        Compute metrics following SSG-VQA's eval_for_f1_et_all function
        """
        cm = confusion_matrix(ground_truth, predictions, labels=range(len(self.labels)))
        
        ap = []  # average precision per class
        ar = []  # average recall per class
        af1 = [] # average f1 per class
        samples = []  # samples per class
        
        for i in range(len(cm)):
            if sum(cm[i, :]) != 0:
                samples.append(sum(cm[i, :]))
                tp = cm[i][i]
                fn = sum(cm[i, :]) - tp
                fp = sum(cm[:, i]) - tp
                
                # Precision
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                
                # Recall
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0
                
                # F1 score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                
                ap.append(precision)
                ar.append(recall)
                af1.append(f1)
        
        if len(ap) == 0:
            return {"mAP": 0.0, "mAR": 0.0, "mAf1": 0.0, "wf1": 0.0, "oa": 0.0}
        
        # Compute averages
        mAP = np.mean(ap)
        mAR = np.mean(ar)
        mAf1 = np.mean(af1)
        
        # Weighted F1
        if sum(samples) > 0:
            wf1 = sum([samples[i] * af1[i] for i in range(len(samples))]) / sum(samples)
        else:
            wf1 = 0.0
        
        # Overall accuracy
        oa = sum(cm.diagonal()) / sum(samples) if sum(samples) > 0 else 0.0
        
        return {
            "mAP": float(mAP),
            "mAR": float(mAR), 
            "mAf1": float(mAf1),
            "wf1": float(wf1),
            "oa": float(oa)
        }
    
    def analyze_error_patterns(self, 
                              ground_truth: List[str], 
                              predictions: List[str],
                              metadata: List[Dict]) -> Dict[str, Any]:
        """Analyze common error patterns in predictions"""
        errors = []
        
        for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
            if gt != pred:
                error_info = {
                    "ground_truth": gt,
                    "prediction": pred,
                    "question_type": metadata[i].get("question_type", ""),
                    "question_category": metadata[i].get("question_category", ""),
                    "sequence_id": metadata[i].get("sequence_id", ""),
                    "frame_id": metadata[i].get("frame_id", ""),
                    "has_scene_graph": metadata[i].get("has_scene_graph", False)
                }
                errors.append(error_info)
        
        # Analyze error patterns
        error_analysis = {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(ground_truth) if len(ground_truth) > 0 else 0,
            "errors_by_question_type": {},
            "errors_by_ground_truth": {},
            "errors_with_vs_without_scene_graph": {"with": 0, "without": 0},
            "most_common_errors": []
        }
        
        # Group errors by question type
        for error in errors:
            q_type = error["question_type"]
            if q_type not in error_analysis["errors_by_question_type"]:
                error_analysis["errors_by_question_type"][q_type] = 0
            error_analysis["errors_by_question_type"][q_type] += 1
        
        # Group errors by ground truth label
        for error in errors:
            gt = error["ground_truth"]
            if gt not in error_analysis["errors_by_ground_truth"]:
                error_analysis["errors_by_ground_truth"][gt] = 0
            error_analysis["errors_by_ground_truth"][gt] += 1
        
        # Scene graph impact on errors
        for error in errors:
            if error["has_scene_graph"]:
                error_analysis["errors_with_vs_without_scene_graph"]["with"] += 1
            else:
                error_analysis["errors_with_vs_without_scene_graph"]["without"] += 1
        
        # Most common error patterns
        error_patterns = {}
        for error in errors:
            pattern = f"{error['ground_truth']} -> {error['prediction']}"
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
        
        # Sort by frequency
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        error_analysis["most_common_errors"] = sorted_patterns[:10]  # Top 10
        
        return error_analysis
    
    def generate_confusion_matrix_report(self, 
                                       ground_truth: List[str], 
                                       predictions: List[str]) -> Dict[str, Any]:
        """Generate detailed confusion matrix analysis"""
        gt_indices = self._convert_to_indices(ground_truth)
        pred_indices = self._convert_to_indices(predictions)
        
        # Compute confusion matrix
        cm = confusion_matrix(gt_indices, pred_indices, labels=range(len(self.labels)))
        
        # Get active labels (labels that appear in the data)
        active_labels = set(gt_indices + pred_indices)
        active_label_names = [self.labels[i] for i in sorted(active_labels)]
        active_cm = cm[list(sorted(active_labels))][:, list(sorted(active_labels))]
        
        return {
            "confusion_matrix": cm.tolist(),
            "active_confusion_matrix": active_cm.tolist(),
            "active_labels": active_label_names,
            "all_labels": self.labels
        }
    
    def save_predictions_csv(self, 
                           predictions: List[str], 
                           ground_truth: List[str],
                           metadata: List[Dict],
                           output_path: Path):
        """Save predictions in CSV format compatible with SSG-VQA evaluation"""
        df_data = []
        
        for i, (pred, gt, meta) in enumerate(zip(predictions, ground_truth, metadata)):
            row = {
                "Sequence_ID": meta.get("sequence_id", ""),
                "Frame_ID": meta.get("frame_id", ""),
                "Question": meta.get("question", ""),
                "Ground_Truth": gt,
                "Prediction": pred,
                "Correct": "Yes" if pred == gt else "No",
                "Question_Type": meta.get("question_type", ""),
                "Question_Category": meta.get("question_category", ""),
                "Has_Scene_Graph": "Yes" if meta.get("has_scene_graph", False) else "No"
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")
    
    def save_detailed_analysis(self, 
                             ground_truth: List[str], 
                             predictions: List[str],
                             metadata: List[Dict],
                             output_dir: Path):
        """Save comprehensive analysis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Overall metrics
        metrics = self.compute_metrics(ground_truth, predictions)
        with open(output_dir / "overall_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Question type metrics
        qt_metrics = self.compute_question_type_metrics(ground_truth, predictions, metadata)
        with open(output_dir / "question_type_metrics.json", 'w') as f:
            json.dump(qt_metrics, f, indent=2)
        
        # Complexity metrics
        complexity_metrics = self.compute_complexity_metrics(ground_truth, predictions, metadata)
        with open(output_dir / "complexity_metrics.json", 'w') as f:
            json.dump(complexity_metrics, f, indent=2)
        
        # Error analysis
        error_analysis = self.analyze_error_patterns(ground_truth, predictions, metadata)
        with open(output_dir / "error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Confusion matrix
        cm_report = self.generate_confusion_matrix_report(ground_truth, predictions)
        with open(output_dir / "confusion_matrix.json", 'w') as f:
            json.dump(cm_report, f, indent=2)
        
        print(f"Detailed analysis saved to: {output_dir}")


def compare_results(results_a: Dict, results_b: Dict, name_a: str = "A", name_b: str = "B") -> Dict:
    """Compare two sets of results"""
    comparison = {
        "comparison": f"{name_a} vs {name_b}",
        "metrics_comparison": {},
        "improvements": {},
        "degradations": {}
    }
    
    # Compare key metrics
    key_metrics = ["accuracy", "mean_f1", "weighted_f1", "ssg_maf1"]
    
    for metric in key_metrics:
        if metric in results_a and metric in results_b:
            val_a = results_a[metric]
            val_b = results_b[metric]
            diff = val_b - val_a
            
            comparison["metrics_comparison"][metric] = {
                f"{name_a}": val_a,
                f"{name_b}": val_b,
                "difference": diff,
                "relative_change": diff / val_a if val_a > 0 else 0
            }
            
            if diff > 0:
                comparison["improvements"][metric] = diff
            elif diff < 0:
                comparison["degradations"][metric] = abs(diff)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    analyzer = ResultsAnalyzer("./data")
    
    # Test with dummy data
    ground_truth = ["True", "False", "grasper", "gallbladder", "True"]
    predictions = ["True", "True", "grasper", "liver", "False"]
    metadata = [
        {"question_type": "exist", "sequence_id": "VID01", "frame_id": "001"},
        {"question_type": "exist", "sequence_id": "VID01", "frame_id": "002"},
        {"question_type": "query_component", "sequence_id": "VID02", "frame_id": "001"},
        {"question_type": "query_component", "sequence_id": "VID02", "frame_id": "002"},
        {"question_type": "exist", "sequence_id": "VID03", "frame_id": "001"}
    ]
    
    metrics = analyzer.compute_metrics(ground_truth, predictions)
    print("Overall Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    qt_metrics = analyzer.compute_question_type_metrics(ground_truth, predictions, metadata)
    print("\nQuestion Type Metrics:")
    for q_type, metrics in qt_metrics.items():
        print(f"  {q_type}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['mean_f1']:.3f}")
