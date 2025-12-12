#!/usr/bin/env python3
"""
Temporal action localization evaluator for surgical videos.

Supports temporal query types:
- Action Onset: When does X start?
- Action Offset: When does X end?
- Action Duration: During which frames does X happen?
- Multiple Event Ordering: Which happens first?
- Count/Frequency: How many times does X happen?
"""

import sys
import json
# regex no longer needed; strict JSON parsing
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig
from llm.qwen_utils import prompt_with_dynamic_graph, prompt_with_dynamic_descriptors
from qwen_vl_utils import process_vision_info

#TODO: change "field of view" query focus for video01_16345 (does not make sense for graph)
#TODO: not graph for every frame -> map graph to frames via timestamps in graphs!
#TODO: discussion with Nico -> ask for time frames instead of frame numbers and use stride 4 for graphs (20 graphs for 80 frames)
class TemporalFrameEvaluator:
    """Evaluator for temporal action localization"""
    
    def __init__(self, config: BenchmarkConfig, model=None, processor=None):
        self.config = config
        self.temporal_cfg = config.temporal_config
        
        # Load video frames using configured images subdirectory
        images_dir = config.video_dir / config.images_subdir
        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            print(f"Expected structure: {config.video_dir}/{config.images_subdir}/")
            self.video_frames = []
            self.num_frames = 0
        else:
            # Support both JPG and PNG images
            self.video_frames = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            self.num_frames = len(self.video_frames)
            if self.num_frames == 0:
                print(f"ERROR: No .jpg or .png images found in {images_dir}")
        
        self.graph_path = config.graph_dir
        
        print(f"Loaded {self.num_frames} frames from {config.video_dir}")
        
        # Load model
        # Reuse provided Qwen model and processor (non-attention variant used for spatial)
        self.model = model
        self.processor = processor
        
        # Increase max_pixels to allow more frames (default is ~12.8M pixels)
        # With 80 frames at 1920x1080, we need ~166M pixels
        # This will allow the model to process all 80 frames without downsampling
        if self.processor is not None and hasattr(self.processor, 'image_processor'):
            original_max_pixels = self.processor.image_processor.max_pixels
            self.processor.image_processor.max_pixels = 200_000_000  # 200M pixels
            print(f"✓ Using shared model (increased max_pixels from {original_max_pixels:,} to {self.processor.image_processor.max_pixels:,})")
        
        # Initialize parsers and evaluators
        self._init_parsers()
        self._init_evaluators()
    
    def _init_parsers(self):
        """Initialize category-specific parsers"""
        self.parsers = {
            'action_onset': self._parse_single_frame,
            'action_offset': self._parse_single_frame,
            'action_duration': self._parse_frame_ranges,
            'multiple_event_ordering': self._parse_ordered_events,
            'count_frequency': self._parse_count_and_ranges
        }
    
    def _init_evaluators(self):
        """Initialize category-specific evaluators"""
        self.evaluators = {
            'action_onset': self._evaluate_frame_error,
            'action_offset': self._evaluate_frame_error,
            'action_duration': self._evaluate_iou,
            'multiple_event_ordering': self._evaluate_ordered_events,
            'count_frequency': self._evaluate_count_frequency
        }
    
    # =========================================================================
    # Main Evaluation Methods
    # =========================================================================
    
    def run_temporal_benchmark(
        self, 
        annotations: List[Dict],
        ablations: List[str]
    ) -> Dict:
        """
        Run temporal evaluation for all queries and ablations.
        
        Args:
            annotations: List of query dicts from JSON
            ablations: ["multiframe", "multiframe_graph"]
        
        Returns:
            Results dictionary with per-ablation and per-query results
        """
        # Check if frames were loaded successfully
        if self.num_frames == 0:
            print("ERROR: No frames available for temporal evaluation. Cannot proceed.")
            print("Please check that the images directory exists and contains .jpg or .png files.")
            return {
                'ablations': {},
                'per_query_results': [],
                'error': 'No frames loaded'
            }
        
        results = {
            'ablations': {},
            'per_query_results': []
        }
        
        for ablation in ablations:
            print(f"\n{'='*80}")
            print(f"ABLATION: {ablation.upper()}")
            print(f"{'='*80}\n")
            
            ablation_results = []
            
            for query_anno in annotations:
                print(f"[Query {query_anno['query_id']}] {query_anno['question']}")
                
                # Evaluate single query
                result = self.evaluate_query(query_anno, ablation)
                ablation_results.append(result)
                
                # Print result
                self._print_query_result(result)
            
            # Aggregate metrics per ablation
            aggregated = self._aggregate_metrics(ablation_results)
            results['ablations'][ablation] = {
                'metrics': aggregated,
                'results': ablation_results
            }
            
            # Print summary
            print(f"\n{'-'*80}")
            print(f"ABLATION SUMMARY: {ablation}")
            self._print_aggregated_metrics(aggregated)
            print(f"{'-'*80}\n")
        
        return results
    
    def evaluate_query(
        self, 
        query_annotation: Dict, 
        ablation: str
    ) -> Dict:
        """
        Evaluate a single temporal query.
        
        Args:
            query_annotation: Dict with query_id, query_type, question, etc.
            ablation: "multiframe" or "multiframe_graph"
        
        Returns:
            Result dict with metrics
        """
        query_type = query_annotation['query_type']
        # Store for prompt building
        self.current_query_type = query_type
        
        # Stage 1: Query the model
        raw_response = self._query_model(
            question=query_annotation['question'],
            answer_format=query_annotation.get('answer_format', ''),
            ablation=ablation
        )
        
        # Print full model response
        print("\n  Model Response:")
        print(f"  {'-'*76}")
        # Print response with proper indentation, max 500 chars
        # response_preview = raw_response[:500] if len(raw_response) > 500 else raw_response
        response_preview = raw_response
        for line in response_preview.split('\n'):
            print(f"  {line}")
        # if len(raw_response) > 500:
        #     print(f"  ... (truncated, {len(raw_response)} chars total)")
        print(f"  {'-'*76}")
        
        # Stage 2: Parse response
        parser = self.parsers[query_type]
        predicted = parser(raw_response)
        print(f"  Parsed: {predicted}")
        
        # Stage 3: Evaluate
        evaluator = self.evaluators[query_type]
        metrics = evaluator(predicted, query_annotation['ground_truth'], query_type)
        print(f"  Ground Truth: {query_annotation['ground_truth']}")
        return {
            'query_id': query_annotation['query_id'],
            'query_type': query_type,
            'ablation': ablation,
            'question': query_annotation['question'],
            'predicted': predicted,
            'ground_truth': query_annotation['ground_truth'],
            'metrics': metrics,
            'raw_response': raw_response
        }
    
    # =========================================================================
    # Model Querying (Ablation-Specific)
    # =========================================================================
    
    def _query_model(
        self, 
        question: str, 
        answer_format: str, 
        ablation: str
    ) -> str:
        """Query the model based on ablation type"""

        if ablation == "multiframe":
            # Video-only: Downsample frames to align with graph timesteps if available
            # Prefer task-specific template from Hydra
            template = self.temporal_cfg['base_prompts'][getattr(self, 'current_query_type', '')]
            selected_frames = self.video_frames
            systems_cfg = self.temporal_cfg.get('system_prompts', {})
            system_prompt = None
            try:
                mf_cfg = systems_cfg.get('multiframe')
                if isinstance(mf_cfg, dict):
                    system_prompt = mf_cfg.get(getattr(self, 'current_query_type', ''), None)
                else:
                    system_prompt = systems_cfg.get('multiframe', None)
            except Exception:
                system_prompt = None
            try:
                graph_data = self._load_graph_data(self.graph_path)
                if graph_data is not None:
                    num_ts = int(graph_data['adjacency_matrices'].shape[0])
                    if num_ts > 0 and len(self.video_frames) > 0:
                        stride = max(1, round(len(self.video_frames) / num_ts))
                        selected_frames = self.video_frames[::stride]
            except Exception:
                selected_frames = self.video_frames
            # Build prompt using the number of frames actually passed
            prompt = self._build_prompt(
                template=template,
                question=question,
                answer_format=answer_format,
                num_frames_override=len(selected_frames),
                last_frame_override=len(selected_frames) - 1,
            )
            response = self._query_multiframe(selected_frames, prompt, system_prompt=system_prompt)
        
        elif ablation == "multiframe_graph":
            # Video + Graph: Use graph prompt and system prompt; prefer task-specific
            template = self.temporal_cfg['graph_prompts'][getattr(self, 'current_query_type', '')]
            systems_cfg = self.temporal_cfg.get('system_prompts', {})
            # Prefer per-task prompt; fall back to legacy single system_prompt
            try:
                mg_cfg = systems_cfg.get('multiframe_graph')
                if isinstance(mg_cfg, dict):
                    system_prompt = mg_cfg.get(getattr(self, 'current_query_type', ''), None)
                else:
                    system_prompt = systems_cfg.get('multiframe_graph', None)
            except Exception:
                system_prompt = None
            if system_prompt is None:
                system_prompt = self.temporal_cfg.get('system_prompt', None)
            # Determine graph timesteps for accurate prompt variables
            graph_data = self._load_graph_data(self.graph_path)
            if graph_data is not None:
                num_ts = int(graph_data['adjacency_matrices'].shape[0])
            else:
                num_ts = self.num_frames
            prompt = self._build_prompt(
                template=template,
                question=question,
                answer_format=answer_format,
                num_frames_override=num_ts,
                last_frame_override=num_ts - 1,
            )
            response = self._query_with_graph(
                image_paths=self.video_frames,
                graph_path=self.graph_path,
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        elif ablation == "multiframe_descriptors":
            # Descriptors-only (no graph structure): Use descriptors prompt and system prompt; prefer task-specific
            template = self.temporal_cfg['descriptors_prompts'][getattr(self, 'current_query_type', '')]
            systems_cfg = self.temporal_cfg.get('system_prompts', {})
            try:
                md_cfg = systems_cfg.get('multiframe_descriptors')
                if isinstance(md_cfg, dict):
                    system_prompt = md_cfg.get(getattr(self, 'current_query_type', ''), None)
                else:
                    system_prompt = systems_cfg.get('multiframe_descriptors', None)
            except Exception:
                system_prompt = None
            if system_prompt is None:
                system_prompt = self.temporal_cfg.get('system_prompt', None)
            # Determine descriptor timesteps via graph data (node_feats timeline)
            graph_data = self._load_graph_data(self.graph_path)
            if graph_data is not None:
                num_ts = int(graph_data['adjacency_matrices'].shape[0])
            else:
                num_ts = self.num_frames
            prompt = self._build_prompt(
                template=template,
                question=question,
                answer_format=answer_format,
                num_frames_override=num_ts,
                last_frame_override=num_ts - 1,
            )
            response = self._query_with_descriptors(
                image_paths=self.video_frames,
                graph_path=self.graph_path,
                prompt=prompt,
                system_prompt=system_prompt,
            )
        
        else:
            raise ValueError(f"Unknown ablation: {ablation}")
        
        return response
    
    def _build_prompt(
        self, 
        template: str, 
        question: str, 
        answer_format: str,
        *,
        num_frames_override: int | None = None,
        last_frame_override: int | None = None,
    ) -> str:
        """Fill in prompt template with query-specific info.

        Allows overriding num_frames/last_frame to reflect subsampled frames or graph timesteps actually passed to the model.
        """
        nf = num_frames_override if num_frames_override is not None else self.num_frames
        lf = last_frame_override if last_frame_override is not None else (nf - 1)
        return template.format(
            num_frames=nf,
            last_frame=lf,
            question=question,
            answer_format=answer_format,
        )
    
    def _query_multiframe(self, image_paths: List[Path], prompt: str, system_prompt: str | None = None) -> str:
        """Query model with multiple frames (video)"""
        
        content = []
        content.append({"type": "video", "video": [str(p) for p in image_paths]})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def _query_with_graph(
        self, 
        image_paths: List[Path], 
        graph_path: Path, 
        prompt: str, 
        system_prompt: str = None
    ) -> str:
        """Query model with multiple frames AND scene graph"""
        
        # Load graph data
        graph_data = self._load_graph_data(graph_path)
        
        if graph_data is None:
            print("Warning: Could not load graph, falling back to video-only")
            return self._query_multiframe(image_paths, prompt)
        
        # Use dynamic graph prompting across all timesteps
        response = prompt_with_dynamic_graph(
            question=prompt,
            node_feats=graph_data['node_feats'],
            adjacency_matrices=graph_data['adjacency_matrices'],
            node_centers=graph_data['node_centers'],
            node_centroids=graph_data['node_centroids'],
            node_extents=graph_data['node_extents'],
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt if system_prompt is not None else self.temporal_cfg.get('system_prompt', None),
        )
        
        return response

    def _query_with_descriptors(
        self,
        image_paths: List[Path],
        graph_path: Path,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Query model with multiple frames AND descriptor features only (no graph)."""
        # Load graph data to access descriptor npz timeline
        graph_data = self._load_graph_data(graph_path)
        if graph_data is None:
            print("Warning: Could not load descriptor features, falling back to video-only")
            return self._query_multiframe(image_paths, prompt, system_prompt=system_prompt)

        response = prompt_with_dynamic_descriptors(
            question=prompt,
            node_feats=graph_data['node_feats'],
            adjacency_matrices=graph_data['adjacency_matrices'],
            node_centers=graph_data['node_centers'],
            node_centroids=graph_data['node_centroids'],
            node_extents=graph_data['node_extents'],
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt if system_prompt is not None else self.temporal_cfg.get('system_prompt', None),
        )

        return response
    
    def _load_graph_data(self, graph_path: Path) -> Optional[Dict]:
        """Load precomputed 4D graph data"""
        try:
            qwen_feat_file = graph_path / "c_qwen_feats.npz"
            if not qwen_feat_file.exists():
                return None
            
            # Load features
            # Keep NPZ file object so downstream can select timestep appropriately
            qwen_feats_dict = np.load(qwen_feat_file)
            
            # Load spatial properties
            adjacency_matrices = np.load(graph_path / "graph.npy")
            centers = np.load(graph_path / "c_centers.npy")
            centroids = np.load(graph_path / "c_centroids.npy")
            extents = np.load(graph_path / "c_extents.npy")
            
            return {
                'node_feats': qwen_feats_dict,
                'adjacency_matrices': adjacency_matrices,
                'node_centers': centers,
                'node_centroids': centroids,
                'node_extents': extents
            }
        except Exception as e:
            print(f"Warning: Could not load graph data: {e}")
            return None
    
    # =========================================================================
    # Response Parsing (Category-Specific)
    # =========================================================================
    
    def _parse_single_frame(self, response: str) -> Optional[Dict]:
        """Parse response for action_onset or action_offset"""
        try:
            # Expect strict JSON only
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                # Unwrap if model returned {"<query_type>": {...}}
                if 'frame' not in data and hasattr(self, 'current_query_type') and isinstance(data, dict):
                    inner = data.get(self.current_query_type)
                    if isinstance(inner, dict):
                        data = inner

                if 'frame' in data:
                    frame_val = data['frame']
                    if isinstance(frame_val, (int, float)):
                        return {'frame': int(frame_val)}
            return None
        
        except Exception as e:
            print(f"  Error parsing single frame: {e}")
            return None
    
    def _parse_frame_ranges(self, response: str) -> Optional[Dict]:
        """Parse response for action_duration"""
        try:
            # Expect strict JSON only
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                # Unwrap if model returned {"<query_type>": {...}}
                if 'ranges' not in data and hasattr(self, 'current_query_type') and isinstance(data, dict):
                    inner = data.get(self.current_query_type)
                    if isinstance(inner, dict):
                        data = inner
                if 'ranges' in data:
                    ranges = []
                    for r in data['ranges']:
                        if isinstance(r, list) and len(r) == 2:
                            ranges.append([int(r[0]), int(r[1])])
                    if ranges:
                        return {'ranges': ranges}
            return None
        
        except Exception as e:
            print(f"  Error parsing frame ranges: {e}")
            return None
    
    def _parse_ordered_events(self, response: str) -> Optional[Dict]:
        """Parse response for multiple_event_ordering"""
        try:
            # Expect strict JSON only
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                # Unwrap if model returned {"<query_type>": {...}}
                if 'events' not in data and hasattr(self, 'current_query_type') and isinstance(data, dict):
                    inner = data.get(self.current_query_type)
                    if isinstance(inner, dict):
                        data = inner
                if 'events' in data and isinstance(data['events'], list):
                    events = []
                    for event in data['events']:
                        if isinstance(event, dict) and 'order' in event and 'frame_range' in event:
                            events.append({
                                'order': int(event['order']),
                                'description': event.get('description', ''),
                                'frame_range': [int(event['frame_range'][0]), int(event['frame_range'][1])]
                            })
                    if events:
                        events.sort(key=lambda x: x['order'])
                        return {'events': events}
            return None
        
        except Exception as e:
            print(f"  Error parsing ordered events: {e}")
            return None
    
    def _parse_count_and_ranges(self, response: str) -> Optional[Dict]:
        """Parse response for count_frequency"""
        try:
            # Expect strict JSON only
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                # Unwrap if model returned {"<query_type>": {...}}
                if 'count' not in data and hasattr(self, 'current_query_type') and isinstance(data, dict):
                    inner = data.get(self.current_query_type)
                    if isinstance(inner, dict):
                        data = inner
                if 'count' in data:
                    occurrences = []
                    if 'occurrences' in data and isinstance(data['occurrences'], list):
                        for occ in data['occurrences']:
                            if isinstance(occ, list) and len(occ) >= 2:
                                occurrences.append([int(occ[0]), int(occ[1])])
                    return {
                        'count': int(data['count']),
                        'occurrences': occurrences
                    }
            return None
        
        except Exception as e:
            print(f"  Error parsing count and ranges: {e}")
            return None
    
    # =========================================================================
    # Metric Computation (Category-Specific)
    # =========================================================================
    
    def _evaluate_frame_error(
        self, 
        predicted: Optional[Dict], 
        ground_truth: Dict,
        query_type: str
    ) -> Dict:
        """Evaluate action_onset or action_offset using frame error"""
        tolerance = self.temporal_cfg['metrics'][query_type]['tolerance']
        gt_frame = ground_truth['frame']
        
        if predicted is None or 'frame' not in predicted:
            return {
                'frame_error': float('inf'),
                'within_tolerance': False,
                'tolerance_used': tolerance,
                'success': False,
                'note': 'Parsing failed'
            }
        
        pred_frame = predicted['frame']
        error = abs(pred_frame - gt_frame)
        
        return {
            'frame_error': error,
            'within_tolerance': error <= tolerance,
            'tolerance_used': tolerance,
            'success': error <= tolerance,
            'predicted_frame': pred_frame,
            'gt_frame': gt_frame
        }
    
    def _evaluate_iou(
        self, 
        predicted: Optional[Dict], 
        ground_truth: Dict,
        query_type: str
    ) -> Dict:
        """Evaluate action_duration using IoU"""
        threshold = self.temporal_cfg['metrics'][query_type]['threshold']
        gt_ranges = ground_truth['ranges']
        
        if predicted is None or 'ranges' not in predicted:
            return {
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'success': False,
                'threshold': threshold,
                'note': 'Parsing failed'
            }
        
        pred_ranges = predicted['ranges']
        
        # Convert ranges to sets of frame numbers
        gt_frames = set()
        for start, end in gt_ranges:
            gt_frames.update(range(start, end + 1))
        
        pred_frames = set()
        for start, end in pred_ranges:
            pred_frames.update(range(start, end + 1))
        
        # Compute IoU
        intersection = len(gt_frames & pred_frames)
        union = len(gt_frames | pred_frames)
        
        iou = intersection / union if union > 0 else 0.0
        precision = intersection / len(pred_frames) if len(pred_frames) > 0 else 0.0
        recall = intersection / len(gt_frames) if len(gt_frames) > 0 else 0.0
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'success': iou >= threshold,
            'threshold': threshold,
            'gt_frame_count': len(gt_frames),
            'pred_frame_count': len(pred_frames)
        }
    
    def _evaluate_ordered_events(
        self, 
        predicted: Optional[Dict], 
        ground_truth: Dict,
        query_type: str
    ) -> Dict:
        """Evaluate multiple_event_ordering using composite metric"""
        cfg = self.temporal_cfg['metrics'][query_type]
        gt_events = ground_truth['events']
        
        if predicted is None or 'events' not in predicted:
            return {
                'order_correct': False,
                'per_event_iou': [],
                'mean_iou': 0.0,
                'composite_score': 0.0,
                'success': False,
                'note': 'Parsing failed'
            }
        
        pred_events = predicted['events']
        
        # Check order correctness
        order_correct = len(pred_events) == len(gt_events)
        if order_correct:
            for i in range(len(pred_events)):
                if pred_events[i]['order'] != gt_events[i]['order']:
                    order_correct = False
                    break
        
        # Compute per-event IoU
        per_event_iou = []
        for i in range(min(len(pred_events), len(gt_events))):
            pred_range = pred_events[i]['frame_range']
            gt_range = gt_events[i]['frame_range']
            
            pred_frames = set(range(pred_range[0], pred_range[1] + 1))
            gt_frames = set(range(gt_range[0], gt_range[1] + 1))
            
            intersection = len(pred_frames & gt_frames)
            union = len(pred_frames | gt_frames)
            iou = intersection / union if union > 0 else 0.0
            per_event_iou.append(iou)
        
        mean_iou = sum(per_event_iou) / len(per_event_iou) if per_event_iou else 0.0
        
        # Composite score
        order_score = 1.0 if order_correct else 0.0
        composite_score = (cfg['order_weight'] * order_score + 
                          cfg['iou_weight'] * mean_iou)
        
        return {
            'order_correct': order_correct,
            'per_event_iou': per_event_iou,
            'mean_iou': mean_iou,
            'composite_score': composite_score,
            'success': composite_score >= 0.5,
            'num_events_pred': len(pred_events),
            'num_events_gt': len(gt_events)
        }
    
    def _evaluate_count_frequency(
        self, 
        predicted: Optional[Dict], 
        ground_truth: Dict,
        query_type: str
    ) -> Dict:
        """Evaluate count_frequency using composite metric"""
        cfg = self.temporal_cfg['metrics'][query_type]
        gt_count = ground_truth['count']
        gt_occurrences = ground_truth['occurrences']
        
        if predicted is None or 'count' not in predicted:
            return {
                'count_correct': False,
                'count_error': gt_count,
                'per_occurrence_iou': [],
                'mean_iou': 0.0,
                'composite_score': 0.0,
                'success': False,
                'note': 'Parsing failed'
            }
        
        pred_count = predicted['count']
        pred_occurrences = predicted.get('occurrences', [])
        
        # Check count accuracy
        count_correct = (pred_count == gt_count)
        count_error = abs(pred_count - gt_count)
        
        # Compute per-occurrence IoU (match by best overlap)
        per_occurrence_iou = []
        matched_gt = set()
        
        for pred_occ in pred_occurrences[:len(gt_occurrences)]:
            pred_frames = set(range(pred_occ[0], pred_occ[1] + 1))
            
            # Find best matching GT occurrence
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_occ in enumerate(gt_occurrences):
                if j in matched_gt:
                    continue
                gt_frames = set(range(gt_occ[0], gt_occ[1] + 1))
                intersection = len(pred_frames & gt_frames)
                union = len(pred_frames | gt_frames)
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                per_occurrence_iou.append(best_iou)
        
        mean_iou = sum(per_occurrence_iou) / len(per_occurrence_iou) if per_occurrence_iou else 0.0
        
        # Composite score
        count_score = 1.0 if count_correct else 0.0
        composite_score = (cfg['count_weight'] * count_score + 
                          cfg['iou_weight'] * mean_iou)
        
        return {
            'count_correct': count_correct,
            'count_error': count_error,
            'pred_count': pred_count,
            'gt_count': gt_count,
            'per_occurrence_iou': per_occurrence_iou,
            'mean_iou': mean_iou,
            'composite_score': composite_score,
            'success': composite_score >= 0.5
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all queries for one ablation"""
        # Group by query type
        by_type = {}
        for result in results:
            qtype = result['query_type']
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(result)
        
        # Compute per-type aggregates
        aggregated = {}
        for qtype, qresults in by_type.items():
            if qtype in ['action_onset', 'action_offset']:
                # Average frame error
                errors = [r['metrics']['frame_error'] for r in qresults 
                         if r['metrics']['frame_error'] != float('inf')]
                success_rate = sum(r['metrics']['success'] for r in qresults) / len(qresults)
                aggregated[qtype] = {
                    'mean_frame_error': float(np.mean(errors)) if errors else float('inf'),
                    'success_rate': success_rate,
                    'count': len(qresults)
                }
            
            elif qtype == 'action_duration':
                # Average IoU
                ious = [r['metrics']['iou'] for r in qresults]
                success_rate = sum(r['metrics']['success'] for r in qresults) / len(qresults)
                aggregated[qtype] = {
                    'mean_iou': float(np.mean(ious)),
                    'success_rate': success_rate,
                    'count': len(qresults)
                }
            
            elif qtype == 'multiple_event_ordering':
                # Average composite score
                scores = [r['metrics']['composite_score'] for r in qresults]
                order_correct_rate = sum(r['metrics']['order_correct'] for r in qresults) / len(qresults)
                aggregated[qtype] = {
                    'mean_composite_score': float(np.mean(scores)),
                    'order_correct_rate': order_correct_rate,
                    'mean_iou': float(np.mean([r['metrics']['mean_iou'] for r in qresults])),
                    'count': len(qresults)
                }
            
            elif qtype == 'count_frequency':
                # Average composite score
                scores = [r['metrics']['composite_score'] for r in qresults]
                count_correct_rate = sum(r['metrics']['count_correct'] for r in qresults) / len(qresults)
                aggregated[qtype] = {
                    'mean_composite_score': float(np.mean(scores)),
                    'count_correct_rate': count_correct_rate,
                    'mean_iou': float(np.mean([r['metrics']['mean_iou'] for r in qresults])),
                    'count': len(qresults)
                }
        
        return aggregated
    
    def _print_query_result(self, result: Dict):
        """Print human-readable result for a single query"""
        qtype = result['query_type']
        metrics = result['metrics']
        
        if qtype in ['action_onset', 'action_offset']:
            status = "✓" if metrics['success'] else "✗"
            print(f"  {status} Frame error: {metrics['frame_error']} "
                  f"(tolerance: {metrics['tolerance_used']})\n\n")
        
        elif qtype == 'action_duration':
            status = "✓" if metrics['success'] else "✗"
            print(f"  {status} IoU: {metrics['iou']:.3f} "
                  f"(threshold: {metrics['threshold']})\n\n")
        
        elif qtype == 'multiple_event_ordering':
            status = "✓" if metrics['success'] else "✗"
            print(f"  {status} Composite: {metrics['composite_score']:.3f}, "
                  f"Order: {metrics['order_correct']}, "
                  f"Mean IoU: {metrics['mean_iou']:.3f}\n\n")
        
        elif qtype == 'count_frequency':
            status = "✓" if metrics['success'] else "✗"
            print(f"  {status} Composite: {metrics['composite_score']:.3f}, "
                  f"Count: {metrics['pred_count']}/{metrics['gt_count']}, "
                  f"Mean IoU: {metrics['mean_iou']:.3f}\n\n")
    
    def _print_aggregated_metrics(self, aggregated: Dict):
        """Print aggregated metrics per category"""
        for qtype, metrics in aggregated.items():
            print(f"\n{qtype}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {output_path}")

