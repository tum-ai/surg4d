#!/usr/bin/env python3
"""
Multi-frame evaluator for surgical action triplet recognition.

Supports 3-condition ablation study:
1. Single Frame: Baseline using one frame
2. Multi-Frame (Video): Temporal reasoning with multiple frames
3. Multi-Frame + Graph: Spatiotemporal reasoning with scene graph

Uses Qwen2.5-VL's native video/multi-image interface.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig, normalize_for_matching
from benchmark.cholect50_utils import CholecT50Loader
from qwen_vl import get_patched_qwen, prompt_with_graph
from qwen_vl_utils import process_vision_info


# TODO: why is this defined in here, not in the frame_selectors?
@dataclass
class MultiFrameSample:
    """Sample with multiple frames for temporal evaluation"""
    video_id: int
    start_frame: int
    end_frame: int
    clip_start: int
    image_paths: List[Path]  # Multiple frames
    graph_path: Optional[Path]
    # TODO: For now, loading these also for non-triplet tasks; might change this
    gt_triplets: List[Dict]  # Ground truth for the sequence
    gt_phase: Optional[str]
    
    @property
    def sample_id(self) -> str:
        return f"v{self.video_id:02d}_f{self.start_frame:05d}-{self.end_frame:05d}"
    
    @property
    def num_frames(self) -> int:
        return len(self.image_paths)


class TripletsFrameEvaluator:
    """Evaluator for multi-frame triplet recognition with ablation study"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        
        # Load model
        print("Loading Qwen2.5-VL model...")
        self.model, self.processor = get_patched_qwen(
            use_bnb_4bit=config.use_4bit_quantization,
            device_map=config.device
        )
        print("✓ Model loaded")
    
    # TODO: the prompts should be part of the config, not hardcoded here -> seems to make more sense to have a
    #  benchmark config for each task and initialize the dataclass from there rather than passing all as cmd line args
    # TODO: if we do that, it would be clear which tasks are expected to be performed based on whether the appropriate config file
    #  was provided on top of a base benchmarking config

#     def _build_single_frame_prompt(self, gt_triplets: List[Dict]) -> str:
#         """Build prompt for single-frame condition"""
        
#         # Get unique options from ground truth for this video
#         instruments = sorted(set(t['instrument'] for t in gt_triplets))
#         verbs = sorted(set(t['verb'] for t in gt_triplets))
#         targets = sorted(set(t['target'] for t in gt_triplets))
        
#         # TODO: add the null instrument or however we call it when we handle the -1 exception case
#         prompt = """You are an expert visceral surgeon analyzing a cholecystectomy image.

# YOUR TASK: Identify instrument-verb-target triplets that are CURRENTLY ACTIVE and CLEARLY VISIBLE in this image.

# CRITICAL RULES:
# 1. ONLY report triplets where you can see BOTH the instrument AND the action occurring
# 2. If an instrument is visible but NOT actively being used, do NOT report it
# 3. If you cannot clearly identify what action is happening, output an empty triplet list
# 4. DO NOT use words like: "appears", "seems", "likely", "possibly", "might be"
# 5. DO NOT infer actions based on typical cholecystectomy procedures
# 6. DO NOT speculate about what COULD be happening or what happened before/after
# 7. An empty response {"triplets": []} is CORRECT if nothing is clearly active

# VALID CATEGORIES:
# "instrument": {"0": "grasper", "1": "bipolar", "2": "hook", "3": "scissors", "4": "clipper", "5": "irrigator"}
# "verb": {"0": "grasp", "1": "retract", "2": "dissect", "3": "coagulate", "4": "clip", "5": "cut", "6": "aspirate", "7": "irrigate", "8": "pack", "9": "null_verb"}
# "target": {"0": "gallbladder", "1": "cystic_plate", "2": "cystic_duct", "3": "cystic_artery", "4": "cystic_pedicle", "5": "blood_vessel", "6": "fluid", "7": "abdominal_wall_cavity", "8": "liver", "9": "adhesion", "10": "omentum", "11": "peritoneum", "12": "gut", "13": "specimen_bag", "14": "null_target"}

# RESPONSE FORMAT:
# First, describe ONLY what you definitively see (anatomy and instruments).
# Then state your confidence level for each potential triplet.
# Finally, output ONLY the triplets you are 100% confident about.

# If uncertain, return: {"triplets": []}

# Example of GOOD response:
# "I can see a grasper firmly holding the gallbladder. No other active instrument interactions are visible."
# {"triplets": [{"instrument": "grasper", "verb": "grasp", "target": "gallbladder"}]}

# Example of BAD response:
# "The bipolar appears to be near a vessel and might be coagulating it."
# {"triplets": [{"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel"}]}

# Output format:
# {
#   "triplets": [
#     {"instrument": "...", "verb": "...", "target": "..."}
#   ]
# }
# """
        
        # return prompt
    
#     def _build_multiframe_prompt(self, gt_triplets: List[Dict]) -> str:
#         """Build prompt for multi-frame condition"""
        
#         instruments = sorted(set(t['instrument'] for t in gt_triplets))
#         verbs = sorted(set(t['verb'] for t in gt_triplets))
#         targets = sorted(set(t['target'] for t in gt_triplets))

#         prompt = """You are an expert visceral surgeon analyzing a cholecystectomy video sequence.
# YOUR TASK: Identify the TIMING of a given action.

# CRITICAL RULES:
# 1. ONLY report the timing of the action if you are 100% confident about it
# 2. If you cannot clearly identify the timing of the action, output an empty response
# 3. DO NOT use words like: "appears", "seems", "likely", "possibly", "might be"
# 4. DO NOT infer actions based on typical cholecystectomy procedures
# 5. DO NOT speculate about what COULD be happening or what happened before/after
# 6. An empty response is CORRECT if nothing is clearly active

# RESPONSE FORMAT:
# First, describe ONLY what you definitively see (anatomy and instruments).
# Then state your confidence level for the action.
# Finally, output the timing of the action in the range that you are 100% confident about in the format: {{"begin": <frame number>, "end": <frame number>}}

# The action to identify here is the grasper grasping the gallbladder. The grasper has to be clearly visible in the image.
# """
#         return prompt
    
    def _query_single_frame(self, image_path: Path, prompt: str) -> str:
        """Query model with a single frame"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # TODO: check if we need this custom impl here of if standard impl from qwen works
        # image_inputs, video_inputs = self._process_vision_info(messages)
        print(f"image_paths: {image_path}")
        image_inputs, video_inputs = process_vision_info(messages)
        print(f"image_inputs: {image_inputs}")
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens = 2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print(f"output_text: {output_text}")
        
        return output_text
    
    def _query_multiframe(self, image_paths: List[Path], prompt: str) -> str:
        """Query model with multiple frames (using multiple images)"""
        
        # Build content with all images
        content = []
        content.append({"type": "video", "video": [str(p) for p in image_paths]})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        print(f"image_paths: {image_paths}")
        image_inputs, video_inputs = process_vision_info(messages)

        # # TODO: why are image_paths datatype path, potentially fix?
        # print(f"image_paths: {image_paths}")
        # # Convert path list to str list
        # video_paths = [str(p) for p in image_paths]
        # print(f"video_paths: {video_paths}")

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens = 2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print(f"output_text: {output_text}")
        
        return output_text
    
    # TODO: have to look into this to see if it works as expected
    def _query_with_graph(self, image_paths: List[Path], graph_path: Path, prompt: str, system_prompt: str = None) -> str:
        """Query model with multiple frames AND scene graph"""
        
        # Load graph data
        graph_data = self._load_graph_data(graph_path)

        print(f"shape of node_feats: {len(graph_data['node_feats'])}")
        
        if graph_data is None:
            # Fallback to multiframe without graph
            return self._query_multiframe(image_paths, prompt)
        
        # Use existing prompt_with_graph from qwen_vl.py
        response = prompt_with_graph(
            node_feats=graph_data['node_feats'],
            adjacency_matrices=graph_data['adjacency_matrices'],
            node_centers=graph_data['node_centers'],
            node_centroids=graph_data['node_centroids'],
            node_extents=graph_data['node_extents'],
            question=prompt,
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt
        )

        print(f"response: {response}")
        
        return response
    
    def _load_graph_data(self, graph_path: Path) -> Optional[Dict]:
        """Load precomputed 4D graph data"""
        
        try:
            # Assuming graph structure similar to video01_00080
            clip_dir = graph_path

            print(f"clip_dir: {clip_dir}")
            
            # Load qwen features
            qwen_feat_dir = clip_dir / "c_qwen_feats"
            if not qwen_feat_dir.exists():
                print(f"Warning: No qwen features found at {qwen_feat_dir}")
                return None
            
            feat_files = sorted(qwen_feat_dir.glob("*.npy"))
            node_feats = [np.load(f) for f in feat_files]
            
            # Load spatial matrices
            # TODO: this used to be called adjacency_matrices.npy, maybe this is graph.npy now? Not sure
            adjacency_matrices = np.load(clip_dir / "graph.npy")
            print(f"adjacency_matrices: {adjacency_matrices.shape}")
            centers = np.load(clip_dir / "c_centers.npy")
            centroids = np.load(clip_dir / "c_centroids.npy")
            extents = np.load(clip_dir / "c_extents.npy")
            
            return {
                'node_feats': node_feats,
                'adjacency_matrices': adjacency_matrices,
                'node_centers': centers,
                'node_centroids': centroids,
                'node_extents': extents
            }
        except Exception as e:
            print(f"Warning: Could not load graph data: {e}")
            return None
    
    # TODO: potentially remove, check first if needed
    def _process_vision_info(self, messages):
        """Process vision info from messages (helper from Qwen2.5-VL docs)"""
        image_inputs, video_inputs = [], []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele.get("type") == "image":
                        image_inputs.append(ele["image"])
                    elif ele.get("type") == "video":
                        video_inputs.append(ele["video"])
        return image_inputs if image_inputs else None, video_inputs if video_inputs else None
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse model response to extract triplets"""
        
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                triplets = data.get('triplets', [])
                return triplets
        except Exception as e:
            print(f"Warning: Could not parse JSON response: {e}")
        
        return []
    
    # TODO: adjust this
    def evaluate_sample(
        self, 
        sample: MultiFrameSample, 
        # TODO: have to change all of this; not a good way to do this
        condition: str,
        prompt: str,
        system_prompt: str = None
    ) -> Dict:
        """
        Evaluate a single sample under specified condition.
        
        Args:
            sample: MultiFrameSample to evaluate
            condition: One of "single_frame", "multiframe", "multiframe_graph"
            prompt: The main prompt/question for the task
            system_prompt: Optional system prompt (used for multiframe_graph)
        """

        print(f"calling evaluate sample for end frame {sample.end_frame}")
        
        # Build prompt
        if condition == "single_frame":
            response = self._query_single_frame(sample.image_paths[sample.end_frame], prompt)
        elif condition == "multiframe":
            response = self._query_multiframe(sample.image_paths[sample.start_frame:sample.end_frame + 1], prompt)
        elif condition == "multiframe_graph":
            if sample.graph_path is None:
                print(f"Warning: No graph available for {sample.sample_id}, using multiframe")
                # TODO: not actually using the graph here, fix this
                response = self._query_multiframe(sample.image_paths[sample.start_frame:sample.end_frame + 1], prompt)
            else:
                response = self._query_with_graph(
                    sample.image_paths[sample.start_frame:sample.end_frame + 1], 
                    sample.graph_path, 
                    prompt,
                    system_prompt=system_prompt
                )
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Parse response
        predicted_triplets = self._parse_response(response)
        
        # Evaluate
        metrics = self._evaluate_prediction(predicted_triplets, sample.gt_triplets)
        
        return {
            'sample_id': sample.sample_id,
            'condition': condition,
            'num_frames': sample.num_frames,
            'predicted_triplets': predicted_triplets,
            'gt_triplets': sample.gt_triplets,
            'response': response,
            'metrics': metrics
        }
    
    def _evaluate_prediction(self, pred_triplets: List[Dict], gt_triplets: List[Dict]) -> Dict:
        """Evaluate predicted triplets against ground truth"""
        
        # For multi-triplet ground truth, check if prediction matches ANY GT triplet
        best_match = {'instrument': False, 'verb': False, 'target': False, 'triplet': False}
        
        for gt in gt_triplets:
            for pred in pred_triplets:
                inst_match = False
                verb_match = False
                targ_match = False
                
                if pred.get('instrument'):
                    pred_inst = normalize_for_matching(pred['instrument'])
                    gt_inst = normalize_for_matching(gt['instrument'])
                    inst_match = pred_inst == gt_inst
                
                if pred.get('verb'):
                    pred_verb = normalize_for_matching(pred['verb'])
                    gt_verb = normalize_for_matching(gt['verb'])
                    verb_match = pred_verb == gt_verb
                
                if pred.get('target'):
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
    
    def run_ablation_study(
        self, 
        samples: List[MultiFrameSample],
        ablations: List[str]
    ) -> Dict:
        """
        Run ablation study across multiple conditions.
        
        Args:
            samples: List of MultiFrameSample objects
            conditions: List of conditions to test (default: all three)
        """
        
        print(f"ablations: {ablations}")
        
        results = {
            'conditions': {},
            'samples': []
        }
        
        for ablation in ablations:
            print(f"\n{'='*80}")
            print(f"ABLATION: {ablation.upper()}")
            print(f"{'='*80}\n")
            
            ablation_results = []
            
            for i, sample in enumerate(samples, 1):
                print(f"[{i}/{len(samples)}] {sample.sample_id}...")
                
                # TODO: evaluate the sample given the specific configuration
                if ablation == "single_frame":
                    prompt = self.config.triplets_config['single_frame_prompt']
                    result = self.evaluate_sample(sample, ablation, prompt)
                elif ablation == "multiframe":
                    prompt = self.config.triplets_config['multiframe_prompt']
                    result = self.evaluate_sample(sample, ablation, prompt)
                elif ablation == "multiframe_graph":
                    prompt = self.config.triplets_config['multiframe_graph_prompt']
                    system_prompt = self.config.triplets_config.get('multiframe_graph_system_prompt', None)
                    result = self.evaluate_sample(sample, ablation, prompt, system_prompt=system_prompt)
                else:
                    raise ValueError(f"Unknown ablation: {ablation}")
                
                ablation_results.append(result)
                
                # Print result
                metrics = result['metrics']
                status = "✓" if metrics['triplet'] else "✗"
                print(f"  {status} I:{int(metrics['instrument'])} V:{int(metrics['verb'])} T:{int(metrics['target'])} Full:{int(metrics['triplet'])}")
            
            # Compute aggregate metrics for this condition
            n = len(ablation_results)
            metrics = {
                'instrument_acc': sum(r['metrics']['instrument'] for r in ablation_results) / n,
                'verb_acc': sum(r['metrics']['verb'] for r in ablation_results) / n,
                'target_acc': sum(r['metrics']['target'] for r in ablation_results) / n,
                'triplet_acc': sum(r['metrics']['triplet'] for r in ablation_results) / n,
                'num_samples': n
            }
            
            results['conditions'][ablation] = {
                'metrics': metrics,
                'results': ablation_results
            }
            
            print(f"\nCondition Metrics:")
            print(f"  Instrument: {metrics['instrument_acc']:.1%}")
            print(f"  Verb: {metrics['verb_acc']:.1%}")
            print(f"  Target: {metrics['target_acc']:.1%}")
            print(f"  Full Triplet: {metrics['triplet_acc']:.1%}")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {output_path}")

