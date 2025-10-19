"""
Benchmark evaluator for action triplet recognition
"""
import sys
from pathlib import Path
from typing import List, Dict
import json
from PIL import Image
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.benchmark_config import (
    BenchmarkConfig, TestSample, normalize_for_matching
)
from benchmark.sample_selector import SampleSelector
import qwen_vl


class TripletEvaluator:
    """Evaluate MLLM on action triplet recognition"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        
        # Initialize model
        if config.model_name == "qwen":
            print("Loading Qwen model...")
            self.model, self.processor = qwen_vl.get_patched_qwen(
                use_bnb_4bit=config.use_4bit_quantization,
                device_map=config.device,
            )
            print("Model loaded!")
        else:
            raise NotImplementedError(f"Model {config.model_name} not implemented yet")
    
    def generate_prompt(self, sample: TestSample) -> str:
        """Generate prompt for action triplet recognition"""
        
        system_prompt = """You are an expert in laparoscopic cholecystectomy surgery. You will be shown a surgical image and asked to identify the action being performed.

Your task is to identify:
1. The surgical instrument being used
2. The action/verb being performed  
3. The target anatomical structure

Provide your answer in this exact format:
Instrument: [instrument name]
Verb: [action verb]
Target: [anatomical structure]

Be concise and specific. Use standard surgical terminology."""

        user_prompt = """What surgical action is being performed in this image?

Please identify:
- The surgical instrument (e.g., grasper, hook, bipolar, scissors, clipper, irrigator)
- The action/verb (e.g., grasp, retract, dissect, coagulate, clip, cut, aspirate)
- The target anatomy (e.g., gallbladder, cystic_plate, cystic_duct, liver, etc.)

Format your answer as:
Instrument: [name]
Verb: [action]
Target: [anatomy]"""

        return system_prompt, user_prompt
    
    def query_model(self, image_path: Path, sample: TestSample) -> Dict:
        """Query the model for a prediction"""
        
        system_prompt, user_prompt = self.generate_prompt(sample)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Query model
        response = qwen_vl.ask_qwen_about_image(
            image=image,
            prompt=user_prompt,
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt,
        )
        
        return {
            'response': response,
            'system_prompt': system_prompt if self.config.save_prompts else None,
            'user_prompt': user_prompt if self.config.save_prompts else None,
        }
    
    def parse_response(self, response: str) -> Dict:
        """Parse model response into structured format"""
        
        pred = {'instrument': None, 'verb': None, 'target': None}
        
        lines = response.lower().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('instrument:'):
                pred['instrument'] = line.split(':', 1)[1].strip()
            elif line.startswith('verb:'):
                pred['verb'] = line.split(':', 1)[1].strip()
            elif line.startswith('target:'):
                pred['target'] = line.split(':', 1)[1].strip()
        
        return pred
    
    def evaluate_prediction(self, pred: Dict, gt_triplets: List[Dict]) -> Dict:
        """Evaluate prediction against ground truth"""
        
        # For multi-triplet frames, check if prediction matches ANY ground truth
        best_match = {'instrument': False, 'verb': False, 'target': False, 'triplet': False}
        
        for gt in gt_triplets:
            inst_match = False
            verb_match = False
            targ_match = False
            
            # Use normalized exact matching
            if pred['instrument']:
                pred_inst = normalize_for_matching(pred['instrument'])
                gt_inst = normalize_for_matching(gt['instrument'])
                inst_match = pred_inst == gt_inst
            
            if pred['verb']:
                pred_verb = normalize_for_matching(pred['verb'])
                gt_verb = normalize_for_matching(gt['verb'])
                verb_match = pred_verb == gt_verb
            
            if pred['target']:
                pred_targ = normalize_for_matching(pred['target'])
                gt_targ = normalize_for_matching(gt['target'])
                targ_match = pred_targ == gt_targ
            
            # Update best match
            if inst_match:
                best_match['instrument'] = True
            if verb_match:
                best_match['verb'] = True
            if targ_match:
                best_match['target'] = True
            if inst_match and verb_match and targ_match:
                best_match['triplet'] = True
        
        return best_match
    
    def evaluate_sample(self, sample: TestSample) -> Dict:
        """Evaluate a single sample"""
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating {sample.sample_id}")
            print(f"Ground truth: {[t['triplet_name'] for t in sample.gt_triplets]}")
        
        # Query model
        model_output = self.query_model(sample.image_path, sample)
        response = model_output['response']
        
        if self.config.verbose:
            print(f"Model response:\n{response}")
        
        # Parse response
        pred = self.parse_response(response)
        
        if self.config.verbose:
            print(f"Parsed: {pred}")
        
        # Evaluate
        scores = self.evaluate_prediction(pred, sample.gt_triplets)
        
        if self.config.verbose:
            print(f"Scores: {scores}")
            if scores['triplet']:
                print("✓ CORRECT TRIPLET")
            else:
                print(f"✗ INCORRECT - Got {sum(scores.values())-1}/3 components")
        
        # Store result
        result = {
            'sample_id': sample.sample_id,
            'video_id': sample.video_id,
            'frame_num': sample.frame_num,
            'gt_triplets': [t['triplet_name'] for t in sample.gt_triplets],
            'gt_phase': sample.gt_phase,
            'prediction': pred,
            'response': response if self.config.save_responses else None,
            'prompts': {
                'system': model_output['system_prompt'],
                'user': model_output['user_prompt'],
            } if self.config.save_prompts else None,
            'scores': scores,
        }
        
        return result
    
    def evaluate_all(self, samples: List[TestSample]) -> Dict:
        """Evaluate all samples and compute metrics"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING {len(samples)} SAMPLES")
        print(f"{'='*70}")
        
        results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}]")
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # Compute aggregate metrics
        metrics = self._compute_metrics(results)
        
        return {
            'config': {
                'model': self.config.model_name,
                'num_samples': len(samples),
                'exact_match': self.config.exact_match,
                'timestamp': datetime.now().isoformat(),
            },
            'results': results,
            'metrics': metrics,
        }
    
    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics"""
        
        total = len(results)
        
        instrument_correct = sum(r['scores']['instrument'] for r in results)
        verb_correct = sum(r['scores']['verb'] for r in results)
        target_correct = sum(r['scores']['target'] for r in results)
        triplet_correct = sum(r['scores']['triplet'] for r in results)
        
        metrics = {
            'total_samples': total,
            'triplet_accuracy': triplet_correct / total if total > 0 else 0,
            'instrument_accuracy': instrument_correct / total if total > 0 else 0,
            'verb_accuracy': verb_correct / total if total > 0 else 0,
            'target_accuracy': target_correct / total if total > 0 else 0,
            'component_breakdown': {
                'instrument': f"{instrument_correct}/{total}",
                'verb': f"{verb_correct}/{total}",
                'target': f"{target_correct}/{total}",
                'full_triplet': f"{triplet_correct}/{total}",
            }
        }
        
        return metrics
    
    def save_results(self, evaluation: Dict, output_path: Path):
        """Save results to JSON file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        print('='*70)
        print(f"Results saved to: {output_path}")
        print('='*70)
    
    def print_summary(self, metrics: Dict):
        """Print summary of results"""
        
        print('='*70)
        print("EVALUATION SUMMARY")
        print('='*70)
        print(f"Total samples: {metrics['total_samples']}")
        print("\nAccuracy:")
        print(f"  Full Triplet:  {metrics['triplet_accuracy']:.1%} ({metrics['component_breakdown']['full_triplet']})")
        print(f"  Instrument:    {metrics['instrument_accuracy']:.1%} ({metrics['component_breakdown']['instrument']})")
        print(f"  Verb:          {metrics['verb_accuracy']:.1%} ({metrics['component_breakdown']['verb']})")
        print(f"  Target:        {metrics['target_accuracy']:.1%} ({metrics['component_breakdown']['target']})")
        print('='*70 + '\n')


def main():
    """Run baseline evaluation"""
    
    # Configuration
    config = BenchmarkConfig(
        num_test_frames=10,  # Start with 10 frames
        model_name="qwen",
        qwen_version="qwen2.5",
        use_4bit_quantization=False,  # Disable if bitsandbytes not available
        exact_match=True,  # Use exact string matching (default now)
        save_responses=True,
        save_prompts=True,
        verbose=True,
    )
    
    # Select samples
    print("Selecting test samples with same triplet configuration...")
    selector = SampleSelector(config)
    # Changed from "diverse" to "by_triplet_config" to group frames by triplet configuration
    samples = selector.select_samples(config.num_test_frames, strategy="by_triplet_config")
    
    if not samples:
        print("ERROR: No samples selected!")
        return
    
    selector.print_sample_summary(samples)
    
    # Evaluate
    evaluator = TripletEvaluator(config)
    evaluation = evaluator.evaluate_all(samples)
    
    # Print summary
    evaluator.print_summary(evaluation['metrics'])
    
    # Save results
    config.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.results_dir / f"baseline_qwen_{timestamp}.json"
    evaluator.save_results(evaluation, output_file)
    
    print("✓ Evaluation complete!")
    print(f"✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

