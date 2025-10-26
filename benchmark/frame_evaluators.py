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

from benchmark.frame_selectors import MultiFrameSample


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
        
        # Assuming graph structure similar to video01_00080
        clip_dir = graph_path

        print(f"clip_dir: {clip_dir}")
        
        # Load qwen features from npz file
        qwen_feat_file = clip_dir / "c_qwen_feats.npz"
        if not qwen_feat_file.exists():
            print(f"Warning: No qwen features found at {qwen_feat_file}")
            return None
        
        # Load npz file and extract features in sorted cluster ID order
        qwen_feats_dict = np.load(qwen_feat_file)
        cluster_ids = sorted([int(k) for k in qwen_feats_dict.keys()])
        node_feats = [qwen_feats_dict[str(cluster_id)][0] for cluster_id in cluster_ids]
        # TODO this is hardcoded to use features at timestep 0
        
        # Load spatial matrices
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
        # TODO: have to change all of this; not a good way to do this (fix once we have more ablations)
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
            
            # TODO: fix for quantitative metrics
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
            
            # print(f"\nCondition Metrics:")
            # print(f"  Instrument: {metrics['instrument_acc']:.1%}")
            # print(f"  Verb: {metrics['verb_acc']:.1%}")
            # print(f"  Target: {metrics['target_acc']:.1%}")
            # print(f"  Full Triplet: {metrics['triplet_acc']:.1%}")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {output_path}")

