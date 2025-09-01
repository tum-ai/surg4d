"""
Scene graph loading functionality for 2D, 3D, and 4D scene graphs
"""
import json
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SceneNode:
    """Represents a node in the scene graph"""
    id: str
    label: str
    position: Optional[List[float]] = None  # 3D coordinates
    bbox: Optional[List[float]] = None  # 2D bounding box
    attributes: Dict[str, Any] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class SceneEdge:
    """Represents an edge/relationship in the scene graph"""
    source_id: str
    target_id: str
    relation_type: str
    attributes: Dict[str, Any] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class SceneGraph:
    """Represents a complete scene graph at a single timestep"""
    timestamp: float
    frame_id: str
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_node_by_id(self, node_id: str) -> Optional[SceneNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_for_node(self, node_id: str) -> List[SceneEdge]:
        """Get all edges involving a specific node"""
        edges = []
        for edge in self.edges:
            if edge.source_id == node_id or edge.target_id == node_id:
                edges.append(edge)
        return edges


@dataclass
class TemporalSceneGraph:
    """Represents a sequence of scene graphs over time (4D)"""
    sequence_id: str
    scene_graphs: List[SceneGraph]
    temporal_edges: List[SceneEdge] = None  # Cross-temporal relationships
    
    def __post_init__(self):
        if self.temporal_edges is None:
            self.temporal_edges = []
    
    def get_scene_at_time(self, timestamp: float) -> Optional[SceneGraph]:
        """Get scene graph closest to specified timestamp"""
        if not self.scene_graphs:
            return None
        
        closest_scene = min(self.scene_graphs, 
                          key=lambda sg: abs(sg.timestamp - timestamp))
        return closest_scene
    
    def get_temporal_window(self, center_time: float, window_size: int) -> List[SceneGraph]:
        """Get a temporal window of scene graphs around a center time"""
        center_scene = self.get_scene_at_time(center_time)
        if not center_scene:
            return []
        
        # Find center index
        center_idx = self.scene_graphs.index(center_scene)
        
        # Calculate window bounds
        half_window = window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(self.scene_graphs), center_idx + half_window + 1)
        
        return self.scene_graphs[start_idx:end_idx]


class SceneGraphLoader(ABC):
    """Abstract base class for scene graph loaders"""
    
    @abstractmethod
    def load_scene_graph(self, file_path: str) -> Union[SceneGraph, TemporalSceneGraph]:
        """Load a scene graph from file"""
        pass


class SSGVQASceneGraphLoader(SceneGraphLoader):
    """Loader for original SSG-VQA 2D scene graphs"""
    
    def __init__(self, image_size: tuple = (240, 430)):
        self.image_size = image_size
    
    def load_scene_graph(self, file_path: str) -> SceneGraph:
        """Load SSG-VQA format scene graph"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract frame information
        frame_id = Path(file_path).stem
        timestamp = 0.0  # SSG-VQA doesn't have temporal info
        
        nodes = []
        edges = []
        
        # Convert SSG-VQA format to our format
        if 'scenes' in data and len(data['scenes']) > 0:
            scene = data['scenes'][0]  # Assume single scene
            
            # Create nodes from objects
            for i, obj in enumerate(scene.get('objects', [])):
                node = SceneNode(
                    id=str(i),
                    label=obj.get('component', 'unknown'),
                    bbox=obj.get('bbox', []),
                    attributes={
                        'color': obj.get('color', ''),
                        'location': obj.get('location', ''),
                        'type': obj.get('type', '')
                    }
                )
                nodes.append(node)
            
            # Create edges from relationships
            relationships = scene.get('relationships', {})
            for relation_type, relation_list in relationships.items():
                for source_idx, targets in enumerate(relation_list):
                    if isinstance(targets, list):
                        for target_idx in targets:
                            edge = SceneEdge(
                                source_id=str(source_idx),
                                target_id=str(target_idx),
                                relation_type=relation_type
                            )
                            edges.append(edge)
        
        return SceneGraph(
            timestamp=timestamp,
            frame_id=frame_id,
            nodes=nodes,
            edges=edges,
            metadata={'source': 'ssg_vqa', 'image_size': self.image_size}
        )


class Gaussian4DSceneGraphLoader(SceneGraphLoader):
    """Loader for 4D scene graphs from Gaussian splatting pipeline"""
    
    def __init__(self, file_format: str = "auto", temporal_strategy: str = "single_file"):
        """
        Initialize loader with flexible format support
        
        Args:
            file_format: "json", "pickle", "graphml", or "auto" (detect from extension)
            temporal_strategy: "single_file" or "multi_file" for GraphML loading
        """
        self.file_format = file_format
        self.temporal_strategy = temporal_strategy
    
    def load_scene_graph(self, file_path: str) -> TemporalSceneGraph:
        """Load 4D scene graph from Gaussian clustering results"""
        file_path = Path(file_path)
        
        # Auto-detect format from extension
        if self.file_format == "auto":
            if file_path.suffix.lower() == ".pkl":
                format_to_use = "pickle"
            elif file_path.suffix.lower() == ".json":
                format_to_use = "json"
            elif file_path.suffix.lower() in [".graphml", ".xml"]:
                format_to_use = "graphml"
            else:
                # Try formats in order of preference
                for ext, fmt in [(".graphml", "graphml"), (".pkl", "pickle"), (".json", "json")]:
                    candidate_path = file_path.with_suffix(ext)
                    if candidate_path.exists():
                        file_path = candidate_path
                        format_to_use = fmt
                        break
                else:
                    raise FileNotFoundError(f"No compatible scene graph file found for {file_path}")
        else:
            format_to_use = self.file_format
        
        # Load based on format
        if format_to_use == "pickle":
            return self._load_pickle_format(file_path)
        elif format_to_use == "json":
            return self._load_json_format(file_path)
        elif format_to_use == "graphml":
            return self._load_graphml_format(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_to_use}")
    
    def _load_pickle_format(self, file_path: Path) -> TemporalSceneGraph:
        """Load from pickle format"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert pickle data to our standardized format
        return self._convert_to_temporal_scene_graph(data, source="pickle", file_path=str(file_path))
    
    def _load_graphml_format(self, file_path: Path) -> TemporalSceneGraph:
        """Load from GraphML format"""
        if self.temporal_strategy == "single_file":
            return self._load_graphml_single_file(file_path)
        else:
            return self._load_graphml_multi_file(file_path)
    
    def _load_graphml_single_file(self, file_path: Path) -> TemporalSceneGraph:
        """Load 4D scene graph from single GraphML file with temporal attributes"""
        try:
            G = nx.read_graphml(file_path)
        except Exception as e:
            raise ValueError(f"Error reading GraphML file {file_path}: {e}")
        
        # Group nodes and edges by timestamp
        timestep_data = {}
        
        # Process nodes
        for node_id, attrs in G.nodes(data=True):
            timestamp = float(attrs.get('timestamp', 0.0))
            
            if timestamp not in timestep_data:
                timestep_data[timestamp] = {'nodes': [], 'edges': []}
            
            node = SceneNode(
                id=node_id,
                label=attrs.get('label', 'unknown'),
                position=self._parse_position(attrs.get('position')),
                attributes={k: v for k, v in attrs.items() if k not in ['label', 'position']},
                confidence=float(attrs.get('confidence', 1.0))
            )
            timestep_data[timestamp]['nodes'].append(node)
        
        # Process edges
        for source, target, attrs in G.edges(data=True):
            timestamp = float(attrs.get('timestamp', 0.0))
            relation_type = attrs.get('relation_type', 'unknown')
            
            if timestamp not in timestep_data:
                timestep_data[timestamp] = {'nodes': [], 'edges': []}
            
            # Skip temporal edges (handle separately)
            if relation_type == 'temporal':
                continue
                
            edge = SceneEdge(
                source_id=source,
                target_id=target,
                relation_type=relation_type,
                attributes={k: v for k, v in attrs.items() if k not in ['relation_type']},
                confidence=float(attrs.get('confidence', 1.0))
            )
            timestep_data[timestamp]['edges'].append(edge)
        
        # Create scene graphs for each timestep
        scene_graphs = []
        for timestamp in sorted(timestep_data.keys()):
            data = timestep_data[timestamp]
            scene_graph = SceneGraph(
                timestamp=timestamp,
                frame_id=f"frame_{timestamp}",
                nodes=data['nodes'],
                edges=data['edges'],
                metadata={'source': 'graphml_single_file', 'file_path': str(file_path)}
            )
            scene_graphs.append(scene_graph)
        
        # Process temporal edges
        temporal_edges = []
        for source, target, attrs in G.edges(data=True):
            if attrs.get('relation_type') == 'temporal':
                edge = SceneEdge(
                    source_id=source,
                    target_id=target,
                    relation_type='temporal',
                    attributes=attrs,
                    confidence=float(attrs.get('confidence', 1.0))
                )
                temporal_edges.append(edge)
        
        sequence_id = file_path.stem.replace('_4d', '')
        return TemporalSceneGraph(
            sequence_id=sequence_id,
            scene_graphs=scene_graphs,
            temporal_edges=temporal_edges
        )
    
    def _load_graphml_multi_file(self, file_path: Path) -> TemporalSceneGraph:
        """Load 4D scene graph from multiple GraphML files (one per timestep)"""
        base_path = file_path.parent
        sequence_id = file_path.stem.replace('_4d', '')
        
        # Find all timestep files for this sequence
        pattern = f"{sequence_id}_4d_t*.graphml"
        timestep_files = sorted(base_path.glob(pattern))
        
        if not timestep_files:
            # Fall back to single file if no timestep files found
            return self._load_graphml_single_file(file_path)
        
        scene_graphs = []
        all_temporal_edges = []
        
        for timestep_file in timestep_files:
            # Extract timestamp from filename (e.g., VID01_4d_t001.graphml -> 1.0)
            timestamp_str = timestep_file.stem.split('_t')[-1]
            timestamp = float(timestamp_str)
            
            try:
                G = nx.read_graphml(timestep_file)
            except Exception as e:
                print(f"Warning: Could not load {timestep_file}: {e}")
                continue
            
            # Convert nodes
            nodes = []
            for node_id, attrs in G.nodes(data=True):
                node = SceneNode(
                    id=node_id,
                    label=attrs.get('label', 'unknown'),
                    position=self._parse_position(attrs.get('position')),
                    attributes={k: v for k, v in attrs.items() if k not in ['label', 'position']},
                    confidence=float(attrs.get('confidence', 1.0))
                )
                nodes.append(node)
            
            # Convert spatial edges
            edges = []
            for source, target, attrs in G.edges(data=True):
                relation_type = attrs.get('relation_type', 'unknown')
                
                if relation_type == 'temporal':
                    # Collect temporal edges for later processing
                    temporal_edge = SceneEdge(
                        source_id=source,
                        target_id=target,
                        relation_type='temporal',
                        attributes=attrs,
                        confidence=float(attrs.get('confidence', 1.0))
                    )
                    all_temporal_edges.append(temporal_edge)
                else:
                    edge = SceneEdge(
                        source_id=source,
                        target_id=target,
                        relation_type=relation_type,
                        attributes={k: v for k, v in attrs.items() if k not in ['relation_type']},
                        confidence=float(attrs.get('confidence', 1.0))
                    )
                    edges.append(edge)
            
            scene_graph = SceneGraph(
                timestamp=timestamp,
                frame_id=f"frame_{timestamp:03.0f}",
                nodes=nodes,
                edges=edges,
                metadata={'source': 'graphml_multi_file', 'file_path': str(timestep_file)}
            )
            scene_graphs.append(scene_graph)
        
        return TemporalSceneGraph(
            sequence_id=sequence_id,
            scene_graphs=scene_graphs,
            temporal_edges=all_temporal_edges
        )
    
    def _parse_position(self, position_str):
        """Parse position string to list of floats"""
        if position_str is None:
            return None
        if isinstance(position_str, (list, tuple)):
            return list(position_str)
        if isinstance(position_str, str):
            # Handle comma-separated values like "1.0,2.0,3.0"
            try:
                return [float(x.strip()) for x in position_str.split(',')]
            except:
                return None
        return None
    
    def _load_json_format(self, file_path: Path) -> TemporalSceneGraph:
        """Load from JSON format"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert JSON data to our standardized format
        return self._convert_to_temporal_scene_graph(data, source="json", file_path=str(file_path))
        
        # Convert JSON data to our standardized format
        return self._convert_to_temporal_scene_graph(data, source="json")
    
    def _convert_to_temporal_scene_graph(self, data: Any, source: str, file_path: str = "") -> TemporalSceneGraph:
        """Convert raw data to standardized TemporalSceneGraph format"""
        
        if source == "pickle":
            # Handle pickle data - this depends on what your colleague stores
            # Example structures they might use:
            
            if hasattr(data, 'sequence_id'):
                # If it's already a custom object
                return self._convert_custom_object(data)
            elif isinstance(data, dict) and 'clusters_over_time' in data:
                # If it's a dictionary with time-indexed clusters
                return self._convert_time_indexed_clusters(data)
            elif isinstance(data, list):
                # If it's a list of per-frame data
                return self._convert_frame_list(data)
            else:
                raise ValueError(f"Unknown pickle data structure: {type(data)}")
        
        elif source == "json":
            # Handle JSON data
            sequence_id = data.get('sequence_id', Path(file_path).stem if file_path else 'unknown')
            scene_graphs = []
            
            # Process each timestep
            for timestep_data in data.get('timesteps', []):
                timestamp = timestep_data.get('timestamp', 0.0)
                frame_id = timestep_data.get('frame_id', f"frame_{timestamp}")
                
                nodes = []
                edges = []
                
                # Create nodes from clustered Gaussians
                for cluster in timestep_data.get('clusters', []):
                    node = SceneNode(
                        id=cluster.get('cluster_id', ''),
                        label=cluster.get('label', 'unknown'),
                        position=cluster.get('centroid', [0, 0, 0]),
                        attributes={
                            'feature_vector': cluster.get('feature_vector', []),
                            'gaussian_count': cluster.get('gaussian_count', 0),
                            'confidence': cluster.get('confidence', 1.0)
                        }
                    )
                    nodes.append(node)
                
                # Create spatial edges
                for edge_data in timestep_data.get('spatial_edges', []):
                    edge = SceneEdge(
                        source_id=edge_data.get('source', ''),
                        target_id=edge_data.get('target', ''),
                        relation_type=edge_data.get('relation', 'spatial'),
                        attributes={'distance': edge_data.get('distance', 0.0)}
                    )
                    edges.append(edge)
                
                scene_graph = SceneGraph(
                    timestamp=timestamp,
                    frame_id=frame_id,
                    nodes=nodes,
                    edges=edges,
                    metadata={'source': '4d_gaussian'}
                )
                scene_graphs.append(scene_graph)
            
            # Process temporal edges
            temporal_edges = []
            for edge_data in data.get('temporal_edges', []):
                edge = SceneEdge(
                    source_id=edge_data.get('source', ''),
                    target_id=edge_data.get('target', ''),
                    relation_type='temporal',
                    attributes={
                        'source_time': edge_data.get('source_time', 0.0),
                        'target_time': edge_data.get('target_time', 0.0),
                        'temporal_relation': edge_data.get('temporal_relation', 'follows')
                    }
                )
                temporal_edges.append(edge)
            
            return TemporalSceneGraph(
                sequence_id=sequence_id,
                scene_graphs=scene_graphs,
                temporal_edges=temporal_edges
            )
    
    def _convert_custom_object(self, obj) -> TemporalSceneGraph:
        """Convert custom Python object to TemporalSceneGraph"""
        # This would depend on your colleague's object structure
        # Example implementation:
        
        scene_graphs = []
        for frame_data in getattr(obj, 'frames', []):
            nodes = []
            for cluster in getattr(frame_data, 'clusters', []):
                node = SceneNode(
                    id=str(getattr(cluster, 'id', len(nodes))),
                    label=getattr(cluster, 'predicted_label', 'unknown'),
                    position=getattr(cluster, 'centroid', []).tolist() if hasattr(getattr(cluster, 'centroid', []), 'tolist') else getattr(cluster, 'centroid', []),
                    attributes={
                        'feature_vector': getattr(cluster, 'features', []).tolist() if hasattr(getattr(cluster, 'features', []), 'tolist') else getattr(cluster, 'features', []),
                        'gaussian_count': len(getattr(cluster, 'gaussians', [])),
                        'confidence': getattr(cluster, 'confidence', 1.0)
                    }
                )
                nodes.append(node)
            
            # Convert edges/relationships
            edges = []
            for relation in getattr(frame_data, 'spatial_relations', []):
                edge = SceneEdge(
                    source_id=str(getattr(relation, 'source_id', '')),
                    target_id=str(getattr(relation, 'target_id', '')),
                    relation_type=getattr(relation, 'type', 'unknown'),
                    attributes={'distance': getattr(relation, 'distance', 0.0)}
                )
                edges.append(edge)
            
            scene_graph = SceneGraph(
                timestamp=getattr(frame_data, 'timestamp', 0.0),
                frame_id=getattr(frame_data, 'frame_id', 'unknown'),
                nodes=nodes,
                edges=edges
            )
            scene_graphs.append(scene_graph)
        
        return TemporalSceneGraph(
            sequence_id=getattr(obj, 'sequence_id', 'unknown'),
            scene_graphs=scene_graphs
        )
    
    def _convert_time_indexed_clusters(self, data: Dict) -> TemporalSceneGraph:
        """Convert time-indexed cluster data"""
        scene_graphs = []
        
        for timestamp, clusters in data['clusters_over_time'].items():
            nodes = []
            for cluster_id, cluster_data in clusters.items():
                node = SceneNode(
                    id=str(cluster_id),
                    label=cluster_data.get('label', 'unknown'),
                    position=cluster_data.get('position', [0, 0, 0]),
                    attributes=cluster_data.get('attributes', {})
                )
                nodes.append(node)
            
            scene_graph = SceneGraph(
                timestamp=float(timestamp),
                frame_id=f"frame_{timestamp}",
                nodes=nodes,
                edges=[]  # Add edge conversion if available
            )
            scene_graphs.append(scene_graph)
        
        return TemporalSceneGraph(
            sequence_id=data.get('sequence_id', 'unknown'),
            scene_graphs=scene_graphs
        )
    
    def _convert_frame_list(self, data: List) -> TemporalSceneGraph:
        """Convert list of frame data"""
        scene_graphs = []
        
        for i, frame_data in enumerate(data):
            nodes = []
            if 'clusters' in frame_data:
                for cluster in frame_data['clusters']:
                    node = SceneNode(
                        id=cluster.get('id', str(len(nodes))),
                        label=cluster.get('label', 'unknown'),
                        position=cluster.get('position', [0, 0, 0]),
                        attributes=cluster.get('attributes', {})
                    )
                    nodes.append(node)
            
            scene_graph = SceneGraph(
                timestamp=frame_data.get('timestamp', float(i)),
                frame_id=frame_data.get('frame_id', f"frame_{i}"),
                nodes=nodes,
                edges=[]
            )
            scene_graphs.append(scene_graph)
        
        return TemporalSceneGraph(
            sequence_id='sequence_from_list',
            scene_graphs=scene_graphs
        )


class SceneGraphLoaderFactory:
    """Factory for creating appropriate scene graph loaders"""
    
    @staticmethod
    def create_loader(source_type: str, file_format: str = "auto", **kwargs) -> SceneGraphLoader:
        """Create a scene graph loader based on source type"""
        if source_type == "2d" or source_type == "ssg_vqa":
            return SSGVQASceneGraphLoader(**kwargs)
        elif source_type == "3d":
            return Single3DSceneGraphLoader(file_format=file_format, **kwargs)
        elif source_type == "4d" or source_type == "gaussian":
            return Gaussian4DSceneGraphLoader(file_format=file_format, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}. Supported: 2d, 3d, 4d, ssg_vqa, gaussian")


def load_scene_graph_for_frame(graph_dir: str, sequence_id: str = None, frame_id: str = None, 
                             source_type: str = "2d", file_format: str = "auto", 
                             file_path: str = None) -> Optional[Union[SceneGraph, TemporalSceneGraph]]:
    """
    Convenience function to load scene graph for a specific frame
    
    Args:
        graph_dir: Directory containing scene graphs OR direct file path if file_path is None
        sequence_id: Sequence identifier (required if file_path is None)
        frame_id: Frame identifier (required for 2D graphs if file_path is None)
        source_type: Type of scene graph ("2d", "3d", "4d", "ssg_vqa", "gaussian")  # ✅ Add 3D
        file_format: File format ("auto", "json", "pickle", "graphml")
        file_path: Direct path to scene graph file (overrides path construction)
    
    Returns:
        Loaded scene graph or None if loading fails
    """
    loader = SceneGraphLoaderFactory.create_loader(source_type, file_format=file_format)
    
    # Use direct file path if provided
    if file_path is not None:
        target_file_path = Path(file_path)
    else:
        # Construct file path based on source type and format
        if sequence_id is None:
            raise ValueError("sequence_id is required when file_path is not provided")
        
        graph_path = Path(graph_dir)
        
        if source_type == "2d" or source_type == "ssg_vqa":
            if frame_id is None:
                raise ValueError("frame_id is required for 2D scene graphs when file_path is not provided")
            target_file_path = graph_path / f"{sequence_id}_{frame_id}.json"
        
        elif source_type == "3d":  # ✅ Add 3D path construction
            if file_format == "auto":
                # Try multiple extensions in order of preference for 3D
                for ext in [".graphml", ".json", ".pkl"]:
                    candidate = graph_path / f"{sequence_id}_3d{ext}"
                    if candidate.exists():
                        target_file_path = candidate
                        break
                else:
                    target_file_path = graph_path / f"{sequence_id}_3d.graphml"  # Default to GraphML
            elif file_format == "graphml":
                target_file_path = graph_path / f"{sequence_id}_3d.graphml"
            elif file_format == "pickle":
                target_file_path = graph_path / f"{sequence_id}_3d.pkl"
            else:  # json
                target_file_path = graph_path / f"{sequence_id}_3d.json"
        
        elif source_type == "4d" or source_type == "gaussian":
            if file_format == "auto":
                # Try multiple extensions in order of preference
                for ext in [".graphml", ".pkl", ".json"]:
                    candidate = graph_path / f"{sequence_id}_4d{ext}"
                    if candidate.exists():
                        target_file_path = candidate
                        break
                else:
                    target_file_path = graph_path / f"{sequence_id}_4d.graphml"  # Default to GraphML
            elif file_format == "graphml":
                target_file_path = graph_path / f"{sequence_id}_4d.graphml"
            elif file_format == "pickle":
                target_file_path = graph_path / f"{sequence_id}_4d.pkl"
            else:  # json
                target_file_path = graph_path / f"{sequence_id}_4d.json"
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    if not target_file_path.exists():
        print(f"Warning: Scene graph file not found: {target_file_path}")
        return None
    
    try:
        return loader.load_scene_graph(str(target_file_path))
    except Exception as e:
        print(f"Error loading scene graph from {target_file_path}: {e}")
        return None


def load_3d_scene_graph(graph_dir: str, sequence_id: str = None, 
                       file_format: str = "auto", file_path: str = None) -> Optional[SceneGraph]:
    """
    Convenience function to load a 3D scene graph
    
    Args:
        graph_dir: Directory containing scene graphs (ignored if file_path provided)
        sequence_id: Sequence identifier (required if file_path is None)
        file_format: File format ("auto", "json", "pickle", "graphml")
        file_path: Direct path to scene graph file (overrides path construction)
    
    Returns:
        Loaded 3D scene graph or None if loading fails
    """
    result = load_scene_graph_for_frame(
        graph_dir=graph_dir,
        sequence_id=sequence_id,
        source_type="3d",
        file_format=file_format,
        file_path=file_path
    )
    
    if isinstance(result, SceneGraph):
        return result
    elif result is not None:
        print(f"Warning: Expected SceneGraph but got {type(result)}")
    
    return None


class Single3DSceneGraphLoader(SceneGraphLoader):
    """Loader for single 3D scene graphs (non-temporal)"""
    
    def __init__(self, file_format: str = "auto"):
        """
        Initialize loader for 3D scene graphs
        
        Args:
            file_format: "json", "graphml", or "auto" (detect from extension)
        """
        self.file_format = file_format
    
    def load_scene_graph(self, file_path: str) -> SceneGraph:
        """Load a single 3D scene graph"""
        file_path = Path(file_path)
        
        # Auto-detect format from extension
        if self.file_format == "auto":
            if file_path.suffix.lower() in [".graphml", ".xml"]:
                format_to_use = "graphml"
            elif file_path.suffix.lower() == ".json":
                format_to_use = "json"
            elif file_path.suffix.lower() == ".pkl":
                format_to_use = "pickle"
            else:
                # Try formats in order of preference for 3D
                for ext, fmt in [(".graphml", "graphml"), (".json", "json"), (".pkl", "pickle")]:
                    candidate_path = file_path.with_suffix(ext)
                    if candidate_path.exists():
                        file_path = candidate_path
                        format_to_use = fmt
                        break
                else:
                    raise FileNotFoundError(f"No compatible 3D scene graph file found for {file_path}")
        else:
            format_to_use = self.file_format
        
        # Load based on format
        if format_to_use == "graphml":
            return self._load_graphml_3d(file_path)
        elif format_to_use == "json":
            return self._load_json_3d(file_path)
        elif format_to_use == "pickle":
            return self._load_pickle_3d(file_path)
        else:
            raise ValueError(f"Unsupported format for 3D scene graphs: {format_to_use}")
    
    def _load_graphml_3d(self, file_path: Path) -> SceneGraph:
        """Load single 3D scene from GraphML with proper feature vector reconstruction"""
        try:
            G = nx.read_graphml(file_path)
        except Exception as e:
            raise ValueError(f"Error reading GraphML file {file_path}: {e}")
        
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            # Reconstruct feature vectors from individual attributes
            reconstructed_attrs = self._reconstruct_feature_vectors(attrs)
            
            # Parse position (3D coordinates)
            position = self._parse_position(reconstructed_attrs.get('position'))
            
            # Extract main attributes
            label = reconstructed_attrs.get('label', reconstructed_attrs.get('predicted_label', f'object_{node_id}'))
            confidence = float(reconstructed_attrs.get('confidence', reconstructed_attrs.get('score', 1.0)))
            
            # Collect all other attributes (now including reconstructed feature vectors)
            other_attrs = {}
            for key, value in reconstructed_attrs.items():
                if key not in ['label', 'predicted_label', 'position', 'confidence', 'score']:
                    other_attrs[key] = value
            
            node = SceneNode(
                id=str(node_id),
                label=label,
                position=position,
                attributes=other_attrs,  # Now contains 'clip_features': [0.1, 0.2, ...]
                confidence=confidence
            )
            nodes.append(node)
        
        edges = []
        for source, target, attrs in G.edges(data=True):
            # Extract relation type
            relation_type = attrs.get('relation_type', attrs.get('relation', attrs.get('label', 'spatial')))
            confidence = float(attrs.get('confidence', attrs.get('score', 1.0)))
            
            # Collect other edge attributes
            other_attrs = {}
            for key, value in attrs.items():
                if key not in ['relation_type', 'relation', 'label', 'confidence', 'score']:
                    other_attrs[key] = value
            
            edge = SceneEdge(
                source_id=str(source),
                target_id=str(target),
                relation_type=relation_type,
                attributes=other_attrs,
                confidence=confidence
            )
            edges.append(edge)
        
        # Create scene graph
        frame_id = file_path.stem
        timestamp = 0.0  # Single 3D scene, no temporal info
        
        return SceneGraph(
            timestamp=timestamp,
            frame_id=frame_id,
            nodes=nodes,
            edges=edges,
            metadata={
                'source': 'graphml_3d', 
                'file_path': str(file_path),
                'num_nodes': len(nodes),
                'num_edges': len(edges)
            }
        )
    
    def _reconstruct_feature_vectors(self, attrs: dict) -> dict:
        """Reconstruct feature vectors from individual GraphML attributes"""
        reconstructed = {}
        feature_groups = {}
        
        # Group individual feature attributes by their prefix
        for key, value in attrs.items():
            if '_feat_' in key and key.split('_')[-1].isdigit():
                # Parse feature attribute: 'lang_feat_0' -> ('lang_feat', 0)
                try:
                    parts = key.split('_')
                    # Find the last numeric part
                    index_str = parts[-1]
                    index = int(index_str)
                    prefix = '_'.join(parts[:-1])  # Everything before the last part
                    
                    if prefix not in feature_groups:
                        feature_groups[prefix] = {}
                    feature_groups[prefix][index] = float(value)
                except (ValueError, IndexError):
                    # Not a proper feature index, keep as regular attribute
                    reconstructed[key] = value
            else:
                # Regular attribute, keep as-is
                reconstructed[key] = value
        
        # Reconstruct feature vectors from grouped attributes
        for prefix, indexed_values in feature_groups.items():
            if len(indexed_values) > 1:  # Only reconstruct if we have multiple values
                # Sort by index to maintain order
                sorted_indices = sorted(indexed_values.keys())
                feature_vector = [indexed_values[i] for i in sorted_indices]
                
                # Map to standard feature names
                if 'lang' in prefix.lower():
                    reconstructed['clip_features'] = feature_vector
                elif 'visual' in prefix.lower():
                    reconstructed['visual_features'] = feature_vector  
                elif 'semantic' in prefix.lower():
                    reconstructed['semantic_features'] = feature_vector
                else:
                    # Unknown feature type, use descriptive name
                    reconstructed[f'{prefix}_vector'] = feature_vector
                
                print(f"Reconstructed {prefix} -> {len(feature_vector)}-dim vector")
            else:
                # Single value, keep as individual attribute
                for idx, val in indexed_values.items():
                    reconstructed[f"{prefix}_{idx}"] = val
        
        return reconstructed
    
    def _load_json_3d(self, file_path: Path) -> SceneGraph:
        """Load single 3D scene from JSON format"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        nodes = []
        edges = []
        
        # Handle different JSON structures
        if 'objects' in data and 'relationships' in data:
            # Standard format: {"objects": [...], "relationships": [...]}
            for obj_data in data['objects']:
                node = SceneNode(
                    id=str(obj_data.get('id', len(nodes))),
                    label=obj_data.get('label', obj_data.get('class', 'unknown')),
                    position=obj_data.get('position', obj_data.get('location')),
                    attributes={k: v for k, v in obj_data.items() 
                              if k not in ['id', 'label', 'class', 'position', 'location']},
                    confidence=float(obj_data.get('confidence', obj_data.get('score', 1.0)))
                )
                nodes.append(node)
            
            for rel_data in data['relationships']:
                edge = SceneEdge(
                    source_id=str(rel_data.get('subject', rel_data.get('source', ''))),
                    target_id=str(rel_data.get('object', rel_data.get('target', ''))),
                    relation_type=rel_data.get('relation', rel_data.get('type', 'unknown')),
                    attributes={k: v for k, v in rel_data.items() 
                              if k not in ['subject', 'object', 'source', 'target', 'relation', 'type']},
                    confidence=float(rel_data.get('confidence', 1.0))
                )
                edges.append(edge)
        
        elif 'clusters' in data:
            # Cluster format: {"clusters": [...]}
            for cluster_data in data['clusters']:
                node = SceneNode(
                    id=str(cluster_data.get('cluster_id', cluster_data.get('id', len(nodes)))),
                    label=cluster_data.get('label', cluster_data.get('predicted_class', 'unknown')),
                    position=cluster_data.get('centroid', cluster_data.get('position')),
                    attributes={
                        'feature_vector': cluster_data.get('features', cluster_data.get('clip_features')),
                        'gaussian_count': cluster_data.get('size', cluster_data.get('point_count')),
                        **{k: v for k, v in cluster_data.items() 
                          if k not in ['cluster_id', 'id', 'label', 'predicted_class', 'centroid', 'position']}
                    },
                    confidence=float(cluster_data.get('confidence', cluster_data.get('score', 1.0)))
                )
                nodes.append(node)
            
            # Add spatial relationships if available
            for spatial_edge in data.get('spatial_edges', []):
                edge = SceneEdge(
                    source_id=str(spatial_edge.get('source', '')),
                    target_id=str(spatial_edge.get('target', '')),
                    relation_type=spatial_edge.get('relation', 'spatial'),
                    attributes={'distance': spatial_edge.get('distance', 0.0)},
                    confidence=float(spatial_edge.get('confidence', 1.0))
                )
                edges.append(edge)
        
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
        
        frame_id = data.get('frame_id', file_path.stem)
        timestamp = float(data.get('timestamp', 0.0))
        
        return SceneGraph(
            timestamp=timestamp,
            frame_id=frame_id,
            nodes=nodes,
            edges=edges,
            metadata={
                'source': 'json_3d',
                'file_path': str(file_path),
                'original_data_keys': list(data.keys())
            }
        )
    
    def _load_pickle_3d(self, file_path: Path) -> SceneGraph:
        """Load single 3D scene from pickle format"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different pickle structures
        if hasattr(data, 'nodes') and hasattr(data, 'edges'):
            # Already a scene graph-like object
            return self._convert_pickle_scene_graph(data, file_path)
        elif isinstance(data, dict) and 'clusters' in data:
            # Cluster dictionary
            return self._convert_pickle_clusters(data, file_path)
        elif isinstance(data, list):
            # List of objects/clusters
            return self._convert_pickle_list(data, file_path)
        else:
            raise ValueError(f"Unsupported pickle structure in {file_path}: {type(data)}")
    
    def _convert_pickle_scene_graph(self, data, file_path: Path) -> SceneGraph:
        """Convert pickle scene graph object"""
        nodes = []
        for node_data in data.nodes:
            if hasattr(node_data, 'id'):
                # Custom node object
                node = SceneNode(
                    id=str(node_data.id),
                    label=getattr(node_data, 'label', getattr(node_data, 'predicted_label', 'unknown')),
                    position=getattr(node_data, 'position', getattr(node_data, 'centroid', None)),
                    attributes=getattr(node_data, 'attributes', {}),
                    confidence=getattr(node_data, 'confidence', 1.0)
                )
            else:
                # Dictionary-like node
                node = SceneNode(
                    id=str(node_data.get('id', len(nodes))),
                    label=node_data.get('label', 'unknown'),
                    position=node_data.get('position'),
                    attributes=node_data.get('attributes', {}),
                    confidence=node_data.get('confidence', 1.0)
                )
            nodes.append(node)
        
        edges = []
        for edge_data in getattr(data, 'edges', []):
            if hasattr(edge_data, 'source_id'):
                # Custom edge object
                edge = SceneEdge(
                    source_id=str(edge_data.source_id),
                    target_id=str(edge_data.target_id),
                    relation_type=getattr(edge_data, 'relation_type', 'unknown'),
                    attributes=getattr(edge_data, 'attributes', {}),
                    confidence=getattr(edge_data, 'confidence', 1.0)
                )
            else:
                # Dictionary-like edge
                edge = SceneEdge(
                    source_id=str(edge_data.get('source_id', '')),
                    target_id=str(edge_data.get('target_id', '')),
                    relation_type=edge_data.get('relation_type', 'unknown'),
                    attributes=edge_data.get('attributes', {}),
                    confidence=edge_data.get('confidence', 1.0)
                )
            edges.append(edge)
        
        return SceneGraph(
            timestamp=getattr(data, 'timestamp', 0.0),
            frame_id=getattr(data, 'frame_id', file_path.stem),
            nodes=nodes,
            edges=edges,
            metadata={'source': 'pickle_3d_object', 'file_path': str(file_path)}
        )
    
    def _convert_pickle_clusters(self, data: dict, file_path: Path) -> SceneGraph:
        """Convert pickle cluster dictionary"""
        nodes = []
        for cluster_id, cluster_data in data['clusters'].items():
            node = SceneNode(
                id=str(cluster_id),
                label=cluster_data.get('label', cluster_data.get('predicted_class', 'unknown')),
                position=cluster_data.get('centroid', cluster_data.get('position')),
                attributes={
                    'feature_vector': cluster_data.get('features', cluster_data.get('clip_features')),
                    'gaussian_count': cluster_data.get('size', 0),
                    **{k: v for k, v in cluster_data.items() 
                      if k not in ['label', 'predicted_class', 'centroid', 'position', 'features', 'clip_features']}
                },
                confidence=cluster_data.get('confidence', 1.0)
            )
            nodes.append(node)
        
        # Create basic spatial edges based on proximity (if positions available)
        edges = []
        if 'relationships' in data:
            for rel in data['relationships']:
                edge = SceneEdge(
                    source_id=str(rel.get('source', '')),
                    target_id=str(rel.get('target', '')),
                    relation_type=rel.get('type', 'spatial'),
                    attributes=rel.get('attributes', {}),
                    confidence=rel.get('confidence', 1.0)
                )
                edges.append(edge)
        
        return SceneGraph(
            timestamp=data.get('timestamp', 0.0),
            frame_id=data.get('frame_id', file_path.stem),
            nodes=nodes,
            edges=edges,
            metadata={'source': 'pickle_3d_clusters', 'file_path': str(file_path)}
        )
    
    def _convert_pickle_list(self, data: list, file_path: Path) -> SceneGraph:
        """Convert pickle list of objects"""
        nodes = []
        for i, item in enumerate(data):
            if hasattr(item, 'id'):
                # Object with attributes
                node = SceneNode(
                    id=str(getattr(item, 'id', i)),
                    label=getattr(item, 'label', getattr(item, 'class', f'object_{i}')),
                    position=getattr(item, 'position', getattr(item, 'centroid', None)),
                    attributes={
                        attr: getattr(item, attr) for attr in dir(item)
                        if not attr.startswith('_') and attr not in ['id', 'label', 'class', 'position', 'centroid']
                    },
                    confidence=getattr(item, 'confidence', 1.0)
                )
            else:
                # Dictionary item
                node = SceneNode(
                    id=str(item.get('id', i)),
                    label=item.get('label', item.get('class', f'object_{i}')),
                    position=item.get('position', item.get('centroid')),
                    attributes={k: v for k, v in item.items() 
                              if k not in ['id', 'label', 'class', 'position', 'centroid', 'confidence']},
                    confidence=item.get('confidence', 1.0)
                )
            nodes.append(node)
        
        return SceneGraph(
            timestamp=0.0,
            frame_id=file_path.stem,
            nodes=nodes,
            edges=[],  # No relationships in simple list format
            metadata={'source': 'pickle_3d_list', 'file_path': str(file_path)}
        )
    
    def _parse_position(self, position_str):
        """Parse position string to list of floats (reuse from Gaussian4DSceneGraphLoader)"""
        if position_str is None:
            return None
        if isinstance(position_str, (list, tuple)):
            return list(position_str)
        if isinstance(position_str, str):
            # Handle comma-separated values like "1.0,2.0,3.0"
            try:
                return [float(x.strip()) for x in position_str.split(',')]
            except:
                return None
        return None