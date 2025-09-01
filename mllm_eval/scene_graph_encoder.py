"""
Scene graph encoding functionality for different output formats
"""
import json
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from scene_graph_loader import SceneGraph, TemporalSceneGraph, SceneNode, SceneEdge
from feature_processor import FeatureProcessor

# Set up logger
logger = logging.getLogger(__name__)


class SceneGraphEncoder(ABC):
    """Abstract base class for scene graph encoders"""
    
    @abstractmethod
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Encode scene graph to string format"""
        pass


class TextSceneGraphEncoder(SceneGraphEncoder):
    """Natural language encoder with feature decoding support"""
    
    def __init__(self, 
                 feature_processor: Optional['FeatureProcessor'] = None,
                 enable_feature_decoding: bool = True,
                 clip_model_name: str = "ViT-B-32"):
        
        # Use provided feature processor or create one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        elif enable_feature_decoding:
            try:
                from feature_processor import create_feature_processor
                self.feature_processor = create_feature_processor(
                    clip_model_name=clip_model_name,
                    enable_feature_decoding=True,
                    enable_raw_features=False
                )
            except ImportError as e:
                logger.warning(f"Could not import feature_processor: {e}")
                self.feature_processor = None
        else:
            self.feature_processor = None
    
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Encode with feature decoding"""
        if isinstance(scene_graph, SceneGraph):
            return self._encode_single_graph(scene_graph)
        else:
            return self._encode_temporal_graph(scene_graph)
    
    def _encode_single_graph(self, scene_graph) -> str:
        """Encode single graph with feature decoding"""
        description_parts = ["Scene Description:"]
        
        # Handle both networkx and custom SceneGraph objects
        if hasattr(scene_graph, 'nodes') and callable(scene_graph.nodes):
            node_items = list(scene_graph.nodes(data=True))
        else:
            node_items = [(node.id, self._node_to_dict(node)) for node in scene_graph.nodes]
        
        # Process nodes with feature decoding
        objects = []
        for node_id, attrs in node_items:
            if self.feature_processor:
                try:
                    processed_node = self.feature_processor.process_node_features(str(node_id), attrs)
                    obj_desc = self.feature_processor.create_textual_description(processed_node)
                except Exception as e:
                    logger.warning(f"Feature processing failed for node {node_id}: {e}")
                    obj_desc = self._create_fallback_description(node_id, attrs)
            else:
                obj_desc = self._create_fallback_description(node_id, attrs)
            
            objects.append(obj_desc)
        
        if objects:
            description_parts.append(f"\nObjects ({len(objects)} total):")
            description_parts.extend([f"- {obj}" for obj in objects])
        
        # Process relationships
        if hasattr(scene_graph, 'edges') and callable(scene_graph.edges):
            edge_items = list(scene_graph.edges(data=True))
        else:
            edge_items = [(edge.source_id, edge.target_id, self._edge_to_dict(edge)) for edge in scene_graph.edges]
        
        if edge_items:
            description_parts.append(f"\nRelationships ({len(edge_items)} total):")
            for src, dst, attrs in edge_items:
                rel_desc = f"{src} relates to {dst}"
                if hasattr(attrs, 'get'):
                    rel_type = attrs.get('relation_type', 'relates_to')
                    rel_desc = f"{src} {rel_type} {dst}"
                    if attrs.get('confidence'):
                        rel_desc += f" (conf: {attrs['confidence']:.2f})"
                elif hasattr(attrs, 'items'):
                    attr_str = ", ".join([f"{k}: {v}" for k, v in attrs.items() if v is not None])
                    rel_desc += f" ({attr_str})" if attr_str else ""
                description_parts.append(f"- {rel_desc}")
        
        return "\n".join(description_parts)
    
    def _create_fallback_description(self, node_id: str, attrs: dict) -> str:
        """Create basic description when feature processing is not available"""
        obj_desc = f"Object {node_id}"
        other_attrs = {k: v for k, v in attrs.items() if v is not None}
        if other_attrs:
            attr_str = ", ".join([f"{k}: {v}" for k, v in other_attrs.items()])
            obj_desc += f" - {attr_str}"
        return obj_desc
    
    def _node_to_dict(self, node):
        """Convert SceneNode to dictionary"""
        result = {"label": node.label, "confidence": node.confidence}
        if node.position:
            result["position"] = node.position
        if hasattr(node, 'bbox') and node.bbox:
            result["bbox"] = node.bbox
        if node.attributes:
            result.update(node.attributes)
        return result
    
    def _edge_to_dict(self, edge):
        """Convert SceneEdge to dictionary"""
        result = {"relation_type": edge.relation_type, "confidence": edge.confidence}
        if edge.attributes:
            result.update(edge.attributes)
        return result
    
    def _encode_temporal_graph(self, temporal_graph: TemporalSceneGraph) -> str:
        """Encode temporal graph"""
        if not temporal_graph.scene_graphs:
            return "Empty temporal scene graph"
        
        # Use most recent frame
        current_scene = temporal_graph.scene_graphs[-1]
        return self._encode_single_graph(current_scene)


# Replace the old JSONSceneGraphEncoder with this enhanced version
class JSONSceneGraphEncoder(SceneGraphEncoder):
    """JSON encoder with feature decoding and raw feature support"""
    
    def __init__(self, 
                 feature_processor: Optional[FeatureProcessor] = None,
                 include_raw_features: bool = True,
                 enable_feature_decoding: bool = True,
                 clip_model_name: str = "ViT-B-32",
                 include_positions: bool = True, 
                 include_attributes: bool = True):
        
        # Legacy parameters for backward compatibility
        self.include_positions = include_positions
        self.include_attributes = include_attributes
        self.include_raw_features = include_raw_features
        
        # Use provided feature processor or create one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        elif enable_feature_decoding or include_raw_features:
            try:
                from feature_processor import create_feature_processor
                self.feature_processor = create_feature_processor(
                    clip_model_name=clip_model_name,
                    enable_feature_decoding=enable_feature_decoding,
                    enable_raw_features=include_raw_features
                )
            except ImportError as e:
                logger.warning(f"Could not import feature_processor: {e}")
                self.feature_processor = None
        else:
            self.feature_processor = None
    
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Enhanced JSON encoding with feature processing"""
        if isinstance(scene_graph, SceneGraph):
            return self._encode_single_graph(scene_graph)
        else:
            return self._encode_temporal_graph(scene_graph)
    
    def _encode_single_graph(self, scene_graph) -> str:
        """Encode single graph as enhanced JSON"""
        # Handle both networkx and custom SceneGraph objects
        if hasattr(scene_graph, 'frame_id'):
            # Custom SceneGraph object
            scene_data = {
                "frame_id": scene_graph.frame_id,
                "timestamp": scene_graph.timestamp,
                "objects": [],
                "relationships": [],
                "metadata": {
                    "feature_processing_enabled": self.feature_processor is not None
                }
            }
            node_items = [(node.id, self._node_to_dict(node)) for node in scene_graph.nodes]
            edge_items = [(edge.source_id, edge.target_id, self._edge_to_dict(edge)) for edge in scene_graph.edges]
        
        elif hasattr(scene_graph, 'nodes') and callable(scene_graph.nodes):
            # NetworkX graph
            scene_data = {
                "objects": [],
                "relationships": [],
                "metadata": {
                    "feature_processing_enabled": self.feature_processor is not None
                }
            }
            node_items = list(scene_graph.nodes(data=True))
            edge_items = list(scene_graph.edges(data=True))
        
        else:
            raise ValueError("Unknown scene graph format")
        
        scene_data["metadata"]["num_nodes"] = len(node_items)
        scene_data["metadata"]["num_edges"] = len(edge_items)
        
        # Process nodes
        for node_id, attrs in node_items:
            if self.feature_processor:
                try:
                    processed_node = self.feature_processor.process_node_features(str(node_id), attrs)
                    
                    # Create enhanced node representation
                    node_data = {
                        "id": str(node_id),
                        "attributes": processed_node["original_attributes"]
                    }
                    
                    # Add decoded features if available
                    if processed_node["decoded_features"]:
                        node_data["semantic_descriptions"] = {}
                        for feature_key, decoded in processed_node["decoded_features"].items():
                            node_data["semantic_descriptions"][feature_key] = {
                                "primary": decoded.get("primary_description", "unknown"),
                                "confidence": decoded.get("confidence", 0.0),
                                "alternatives": decoded.get("alternative_descriptions", [])
                            }
                    
                    # Add raw features if requested
                    if self.include_raw_features and processed_node["raw_features"]:
                        node_data["raw_features"] = {}
                        for feature_key, raw_tensor in processed_node["raw_features"].items():
                            # Convert tensor to list for JSON serialization
                            if hasattr(raw_tensor, 'cpu'):
                                node_data["raw_features"][feature_key] = raw_tensor.cpu().tolist()
                            else:
                                node_data["raw_features"][feature_key] = raw_tensor
                
                except Exception as e:
                    logger.warning(f"Feature processing failed for node {node_id}: {e}")
                    node_data = self._create_fallback_node_data(node_id, attrs)
            else:
                node_data = self._create_fallback_node_data(node_id, attrs)
            
            scene_data["objects"].append(node_data)
        
        # Process relationships
        for src, dst, attrs in edge_items:
            rel_data = {
                "subject": str(src),
                "object": str(dst)
            }
            
            # Handle different attribute formats
            if hasattr(attrs, 'get'):
                rel_data["relation"] = attrs.get('relation_type', 'relates_to')
                rel_data["confidence"] = attrs.get('confidence', 0.0)
                if self.include_attributes and attrs:
                    other_attrs = {k: v for k, v in attrs.items() 
                                 if k not in ["relation_type", "confidence"]}
                    if other_attrs:
                        rel_data["attributes"] = other_attrs
            else:
                # Handle direct attribute access or dict
                rel_data["relation"] = getattr(attrs, 'relation_type', 'relates_to')
                rel_data["confidence"] = getattr(attrs, 'confidence', 0.0)
                if self.include_attributes and hasattr(attrs, 'attributes') and attrs.attributes:
                    rel_data["attributes"] = attrs.attributes
            
            scene_data["relationships"].append(rel_data)
        
        return json.dumps(scene_data, indent=2)
    
    def _create_fallback_node_data(self, node_id: str, attrs: dict) -> dict:
        """Create fallback node data when feature processing is not available"""
        node_data = {
            "id": str(node_id),
            "label": attrs.get("label", "unknown"),
            "confidence": attrs.get("confidence", 0.0)
        }
        
        if self.include_positions and attrs.get("position"):
            node_data["position"] = attrs["position"]
        
        if attrs.get("bbox"):
            node_data["bbox"] = attrs["bbox"]
        
        if self.include_attributes:
            # Filter out processed attributes
            other_attrs = {k: v for k, v in attrs.items() 
                         if k not in ["label", "confidence", "position", "bbox"]}
            if other_attrs:
                node_data["attributes"] = other_attrs
        
        return node_data
    
    def _node_to_dict(self, node):
        """Convert SceneNode to dictionary"""
        result = {"label": node.label, "confidence": node.confidence}
        if node.position:
            result["position"] = node.position
        if hasattr(node, 'bbox') and node.bbox:
            result["bbox"] = node.bbox
        if node.attributes:
            result.update(node.attributes)
        return result
    
    def _edge_to_dict(self, edge):
        """Convert SceneEdge to dictionary"""
        result = {"relation_type": edge.relation_type, "confidence": edge.confidence}
        if edge.attributes:
            result.update(edge.attributes)
        return result
    
    def _encode_temporal_graph(self, temporal_graph: TemporalSceneGraph) -> str:
        """Encode temporal graph as enhanced JSON"""
        data = {
            "sequence_id": temporal_graph.sequence_id,
            "temporal_graphs": [],
            "temporal_edges": []
        }
        
        # Encode each timestep
        for sg in temporal_graph.scene_graphs:
            sg_data = json.loads(self._encode_single_graph(sg))
            data["temporal_graphs"].append(sg_data)
        
        # Encode temporal edges
        for edge in temporal_graph.temporal_edges:
            edge_data = {
                "source": edge.source_id,
                "target": edge.target_id,
                "relation": edge.relation_type,
                "confidence": edge.confidence
            }
            
            if self.include_attributes and edge.attributes:
                edge_data["attributes"] = edge.attributes
            
            data["temporal_edges"].append(edge_data)
        
        return json.dumps(data, indent=2)


class XMLSceneGraphEncoder(SceneGraphEncoder):
    """Enhanced XML encoder with feature decoding support"""
    
    def __init__(self, 
                 feature_processor: Optional['FeatureProcessor'] = None,
                 enable_feature_decoding: bool = True,
                 include_raw_features: bool = False,
                 clip_model_name: str = "ViT-B-32"):
        
        self.include_raw_features = include_raw_features
        
        # Use provided feature processor or create one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        elif enable_feature_decoding or include_raw_features:
            try:
                from feature_processor import create_feature_processor
                self.feature_processor = create_feature_processor(
                    clip_model_name=clip_model_name,
                    enable_feature_decoding=enable_feature_decoding,
                    enable_raw_features=include_raw_features
                )
            except ImportError as e:
                logger.warning(f"Could not import feature_processor: {e}")
                self.feature_processor = None
        else:
            self.feature_processor = None
    
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Encode scene graph as XML string"""
        if isinstance(scene_graph, SceneGraph):
            return self._encode_single_graph(scene_graph)
        else:
            return self._encode_temporal_graph(scene_graph)
    
    def _encode_single_graph(self, scene_graph) -> str:
        """Encode a single scene graph as XML with enhanced features"""
        root = ET.Element("scene_graph")
        
        # Handle both networkx and custom SceneGraph objects
        if hasattr(scene_graph, 'frame_id'):
            root.set("frame_id", scene_graph.frame_id)
            root.set("timestamp", str(scene_graph.timestamp))
        
        # Add nodes
        nodes_elem = ET.SubElement(root, "nodes")
        
        # Handle networkx graph
        if hasattr(scene_graph, 'nodes') and callable(scene_graph.nodes):
            node_items = scene_graph.nodes(data=True)
        else:
            # Handle SceneGraph object
            node_items = [(node.id, self._node_to_dict(node)) for node in scene_graph.nodes]
        
        for node_id, attrs in node_items:
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.set("id", str(node_id))
            
            if self.feature_processor:
                try:
                    processed_node = self.feature_processor.process_node_features(str(node_id), attrs)
                    
                    # Add semantic descriptions from decoded features
                    if processed_node["decoded_features"]:
                        semantic_elem = ET.SubElement(node_elem, "semantic_descriptions")
                        for feature_key, decoded in processed_node["decoded_features"].items():
                            desc_elem = ET.SubElement(semantic_elem, "description")
                            desc_elem.set("feature_type", feature_key)
                            desc_elem.set("confidence", str(decoded.get("confidence", 0.0)))
                            desc_elem.text = decoded.get("primary_description", "unknown")
                            
                            # Add alternatives
                            if decoded.get("alternative_descriptions"):
                                alt_elem = ET.SubElement(desc_elem, "alternatives")
                                for i, alt_desc in enumerate(decoded["alternative_descriptions"]):
                                    alt_item = ET.SubElement(alt_elem, "alternative")
                                    alt_item.set("rank", str(i + 2))
                                    alt_item.text = alt_desc
                    
                    # Add regular attributes (non-feature)
                    if processed_node["original_attributes"]:
                        attrs_elem = ET.SubElement(node_elem, "attributes")
                        for key, value in processed_node["original_attributes"].items():
                            attr_elem = ET.SubElement(attrs_elem, "attribute")
                            attr_elem.set("name", key)
                            attr_elem.text = str(value)
                    
                    # Add raw features if requested (as base64 for compactness)
                    if self.include_raw_features and processed_node["raw_features"]:
                        raw_elem = ET.SubElement(node_elem, "raw_features")
                        for feature_key, raw_tensor in processed_node["raw_features"].items():
                            feature_elem = ET.SubElement(raw_elem, "feature")
                            feature_elem.set("type", feature_key)
                            
                            if hasattr(raw_tensor, 'shape'):
                                feature_elem.set("shape", str(list(raw_tensor.shape)))
                                # Convert to base64 for XML storage
                                import base64
                                tensor_bytes = raw_tensor.cpu().numpy().tobytes()
                                feature_elem.text = base64.b64encode(tensor_bytes).decode('utf-8')
                            else:
                                feature_elem.text = str(raw_tensor)
                
                except Exception as e:
                    logger.warning(f"Feature processing failed for node {node_id}: {e}")
                    self._add_fallback_node_attributes(node_elem, attrs)
            else:
                self._add_fallback_node_attributes(node_elem, attrs)
        
        # Add edges
        edges_elem = ET.SubElement(root, "edges")
        
        # Handle networkx vs custom edges
        if hasattr(scene_graph, 'edges') and callable(scene_graph.edges):
            edge_items = scene_graph.edges(data=True)
        else:
            edge_items = [(edge.source_id, edge.target_id, self._edge_to_dict(edge)) for edge in scene_graph.edges]
        
        for src, dst, attrs in edge_items:
            edge_elem = ET.SubElement(edges_elem, "edge")
            edge_elem.set("source", str(src))
            edge_elem.set("target", str(dst))
            
            if hasattr(attrs, 'items'):
                for key, value in attrs.items():
                    if key in ['relation_type', 'confidence']:
                        edge_elem.set(key.replace('_', ''), str(value))
                    else:
                        attr_elem = ET.SubElement(edge_elem, key)
                        attr_elem.text = str(value)
        
        return ET.tostring(root, encoding='unicode')
    
    def _add_fallback_node_attributes(self, node_elem: ET.Element, attrs: dict):
        """Add basic node attributes when feature processing is not available"""
        if hasattr(attrs, 'items'):
            for key, value in attrs.items():
                if key in ['label', 'confidence']:
                    node_elem.set(key, str(value))
                else:
                    attr_elem = ET.SubElement(node_elem, key)
                    attr_elem.text = str(value)
    
    def _node_to_dict(self, node):
        """Convert SceneNode to dictionary"""
        result = {"label": node.label, "confidence": node.confidence}
        if node.position:
            result["position"] = node.position
        if hasattr(node, 'bbox') and node.bbox:
            result["bbox"] = node.bbox
        if node.attributes:
            result.update(node.attributes)
        return result
    
    def _edge_to_dict(self, edge):
        """Convert SceneEdge to dictionary"""
        result = {"relation_type": edge.relation_type, "confidence": edge.confidence}
        if edge.attributes:
            result.update(edge.attributes)
        return result
    
    def _encode_temporal_graph(self, temporal_graph: TemporalSceneGraph) -> str:
        """Encode a temporal scene graph as XML"""
        root = ET.Element("temporal_scene_graph")
        root.set("sequence_id", temporal_graph.sequence_id)
        
        # Add temporal graphs
        for sg in temporal_graph.scene_graphs:
            sg_xml = self._encode_single_graph(sg)
            sg_elem = ET.fromstring(sg_xml)
            root.append(sg_elem)
        
        return ET.tostring(root, encoding='unicode')


class SoMSceneGraphEncoder(SceneGraphEncoder):
    """Enhanced Set-of-Mark encoder with feature decoding support"""
    
    def __init__(self, 
                 feature_processor: Optional['FeatureProcessor'] = None,
                 use_numbers: bool = True,
                 enable_feature_decoding: bool = True,
                 include_confidence: bool = True,
                 clip_model_name: str = "ViT-B-32"):
        
        self.use_numbers = use_numbers
        self.include_confidence = include_confidence
        
        # Use provided feature processor or create one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        elif enable_feature_decoding:
            try:
                from feature_processor import create_feature_processor
                self.feature_processor = create_feature_processor(
                    clip_model_name=clip_model_name,
                    enable_feature_decoding=True,
                    enable_raw_features=False
                )
            except ImportError as e:
                logger.warning(f"Could not import feature_processor: {e}")
                self.feature_processor = None
        else:
            self.feature_processor = None
    
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Encode scene graph with numbered/marked objects for SoM prompting"""
        if isinstance(scene_graph, SceneGraph):
            return self._encode_single_graph(scene_graph)
        else:
            return self._encode_temporal_graph(scene_graph)
    
    def _encode_single_graph(self, scene_graph) -> str:
        """Encode a single scene graph with enhanced SoM style marking"""
        text_parts = []
        text_parts.append("Scene objects with reference marks:")
        
        # Handle both networkx and custom SceneGraph objects
        if hasattr(scene_graph, 'nodes') and callable(scene_graph.nodes):
            node_items = list(scene_graph.nodes(data=True))
        else:
            node_items = [(node.id, self._node_to_dict(node)) for node in scene_graph.nodes]
        
        # Create enhanced numbered list of objects
        node_to_mark = {}
        for i, (node_id, attrs) in enumerate(node_items, 1):
            mark = f"<{i}>" if self.use_numbers else f"<{node_id}>"
            node_to_mark[str(node_id)] = mark
            
            if self.feature_processor:
                try:
                    processed_node = self.feature_processor.process_node_features(str(node_id), attrs)
                    
                    # Use decoded description as primary label
                    primary_desc = "unknown_object"
                    confidence_score = 0.0
                    
                    if processed_node["decoded_features"]:
                        for feature_key, decoded in processed_node["decoded_features"].items():
                            primary_desc = decoded.get("primary_description", "unknown_object")
                            confidence_score = decoded.get("confidence", 0.0)
                            break  # Use first decoded feature
                    
                    # Build description
                    if self.include_confidence and confidence_score > 0:
                        description = f"{mark} {primary_desc} (conf: {confidence_score:.2f})"
                    else:
                        description = f"{mark} {primary_desc}"
                    
                    # Add additional attributes
                    if processed_node["original_attributes"]:
                        extra_attrs = []
                        for key, value in processed_node["original_attributes"].items():
                            if key in ['color', 'location', 'type', 'size', 'position'] and value:
                                extra_attrs.append(str(value))
                        if extra_attrs:
                            description += f" [{', '.join(extra_attrs)}]"
                
                except Exception as e:
                    logger.warning(f"Feature processing failed for node {node_id}: {e}")
                    description = self._create_fallback_som_description(mark, node_id, attrs)
            else:
                description = self._create_fallback_som_description(mark, node_id, attrs)
            
            text_parts.append(description)
        
        # Add relationships with marks
        if hasattr(scene_graph, 'edges') and callable(scene_graph.edges):
            edge_items = list(scene_graph.edges(data=True))
        else:
            edge_items = [(edge.source_id, edge.target_id, self._edge_to_dict(edge)) for edge in scene_graph.edges]
        
        if edge_items:
            text_parts.append("\nSpatial relationships:")
            for src, dst, attrs in edge_items:
                source_mark = node_to_mark.get(str(src), str(src))
                target_mark = node_to_mark.get(str(dst), str(dst))
                
                relation_type = attrs.get('relation_type', 'relates_to') if hasattr(attrs, 'get') else 'relates_to'
                relation_text = f"- {source_mark} is {relation_type} {target_mark}"
                
                # Add confidence if available
                if hasattr(attrs, 'get') and attrs.get('confidence'):
                    relation_text += f" (conf: {attrs['confidence']:.2f})"
                
                text_parts.append(relation_text)
        
        return "\n".join(text_parts)
    
    def _create_fallback_som_description(self, mark: str, node_id: str, attrs: dict) -> str:
        """Create fallback SoM description when feature processing is not available"""
        label = attrs.get('label', f'object_{node_id}')
        description = f"{mark} {label}"
        
        if hasattr(attrs, 'items'):
            extra_attrs = []
            for key, value in attrs.items():
                if key in ['color', 'location', 'type'] and value:
                    extra_attrs.append(str(value))
            if extra_attrs:
                description += f" ({', '.join(extra_attrs)})"
        
        return description
    
    def _node_to_dict(self, node):
        """Convert SceneNode to dictionary"""
        result = {"label": node.label, "confidence": node.confidence}
        if node.position:
            result["position"] = node.position
        if hasattr(node, 'bbox') and node.bbox:
            result["bbox"] = node.bbox
        if node.attributes:
            result.update(node.attributes)
        return result
    
    def _edge_to_dict(self, edge):
        """Convert SceneEdge to dictionary"""
        result = {"relation_type": edge.relation_type, "confidence": edge.confidence}
        if edge.attributes:
            result.update(edge.attributes)
        return result
    
    def _encode_temporal_graph(self, temporal_graph: TemporalSceneGraph) -> str:
        """Encode a temporal scene graph with SoM style marking"""
        if temporal_graph.scene_graphs:
            current_scene = temporal_graph.scene_graphs[-1]
            return self._encode_single_graph(current_scene)
        return "No scene graphs available"


class GraphTextEncoder(SceneGraphEncoder):
    """Enhanced GRAPHTEXT encoder with feature decoding support"""
    
    def __init__(self,
                 feature_processor: Optional['FeatureProcessor'] = None,
                 enable_feature_decoding: bool = True,
                 include_confidence: bool = True,
                 clip_model_name: str = "ViT-B-32"):
        
        self.include_confidence = include_confidence
        
        # Use provided feature processor or create one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        elif enable_feature_decoding:
            try:
                from feature_processor import create_feature_processor
                self.feature_processor = create_feature_processor(
                    clip_model_name=clip_model_name,
                    enable_feature_decoding=True,
                    enable_raw_features=False
                )
            except ImportError as e:
                logger.warning(f"Could not import feature_processor: {e}")
                self.feature_processor = None
        else:
            self.feature_processor = None
    
    def encode(self, scene_graph: Union[SceneGraph, TemporalSceneGraph]) -> str:
        """Encode scene graph using enhanced tree traversal for GRAPHTEXT approach"""
        if isinstance(scene_graph, SceneGraph):
            return self._encode_single_graph(scene_graph)
        else:
            return self._encode_temporal_graph(scene_graph)
    
    def _encode_single_graph(self, scene_graph) -> str:
        """Encode using depth-first traversal with semantic descriptions"""
        
        # Handle both networkx and custom SceneGraph objects
        if hasattr(scene_graph, 'nodes') and callable(scene_graph.nodes):
            node_items = list(scene_graph.nodes(data=True))
            edge_items = list(scene_graph.edges(data=True))
        else:
            node_items = [(node.id, self._node_to_dict(node)) for node in scene_graph.nodes]
            edge_items = [(edge.source_id, edge.target_id, self._edge_to_dict(edge)) for edge in scene_graph.edges]
        
        if not node_items:
            return "Empty scene graph"
        
        # Build adjacency list for traversal
        adjacency = {}
        edge_info = {}
        
        for node_id, attrs in node_items:
            adjacency[str(node_id)] = []
        
        for src, dst, attrs in edge_items:
            src_str, dst_str = str(src), str(dst)
            if src_str in adjacency:
                adjacency[src_str].append(dst_str)
            edge_info[(src_str, dst_str)] = attrs
        
        visited = set()
        traversal_parts = []
        
        # Start traversal from first node
        start_node_id = str(node_items[0][0])
        self._dfs_traversal_enhanced(start_node_id, dict(node_items), adjacency, edge_info, 
                                   visited, traversal_parts, depth=0)
        
        # Add any unvisited nodes
        for node_id, attrs in node_items:
            node_id_str = str(node_id)
            if node_id_str not in visited:
                self._dfs_traversal_enhanced(node_id_str, dict(node_items), adjacency, edge_info,
                                           visited, traversal_parts, depth=0)
        
        return " -> ".join(traversal_parts)
    
    def _dfs_traversal_enhanced(self, node_id: str, node_data: dict, adjacency: dict, 
                              edge_info: dict, visited: set, traversal_parts: list, depth: int):
        """Perform enhanced depth-first traversal with semantic descriptions"""
        if node_id in visited:
            return
        
        visited.add(node_id)
        attrs = node_data.get(node_id, {})
        
        # Create enhanced node representation
        if self.feature_processor:
            try:
                processed_node = self.feature_processor.process_node_features(node_id, attrs)
                
                # Use decoded description
                primary_desc = f"object_{node_id}"
                confidence_score = 0.0
                
                if processed_node["decoded_features"]:
                    for feature_key, decoded in processed_node["decoded_features"].items():
                        primary_desc = decoded.get("primary_description", primary_desc)
                        confidence_score = decoded.get("confidence", 0.0)
                        break  # Use first decoded feature
                
                # Build node representation
                if self.include_confidence and confidence_score > 0:
                    node_repr = f"{primary_desc}({node_id})[conf:{confidence_score:.2f}]"
                else:
                    node_repr = f"{primary_desc}({node_id})"
                
                # Add additional attributes
                if processed_node["original_attributes"]:
                    extra_attrs = []
                    for key, value in processed_node["original_attributes"].items():
                        if key in ['position', 'color', 'size'] and value:
                            extra_attrs.append(f"{key}:{value}")
                    if extra_attrs:
                        node_repr += f"[{','.join(extra_attrs)}]"
            
            except Exception as e:
                logger.warning(f"Feature processing failed for node {node_id}: {e}")
                node_repr = self._create_fallback_graphtext_node(node_id, attrs)
        else:
            node_repr = self._create_fallback_graphtext_node(node_id, attrs)
        
        # Add indentation for depth
        indent = "  " * depth
        traversal_parts.append(f"{indent}{node_repr}")
        
        # Visit connected nodes
        for target_id in adjacency.get(node_id, []):
            if target_id not in visited:
                # Add edge information
                edge_attrs = edge_info.get((node_id, target_id), {})
                relation_type = edge_attrs.get('relation_type', 'connects_to')
                
                edge_repr = f"{indent}--{relation_type}-->"
                if self.include_confidence and edge_attrs.get('confidence'):
                    edge_repr += f"[conf:{edge_attrs['confidence']:.2f}]"
                
                traversal_parts.append(edge_repr)
                
                # Recursively visit target
                self._dfs_traversal_enhanced(target_id, node_data, adjacency, edge_info,
                                           visited, traversal_parts, depth + 1)
    
    def _create_fallback_graphtext_node(self, node_id: str, attrs: dict) -> str:
        """Create fallback GraphText node representation"""
        label = attrs.get('label', f'object_{node_id}')
        return f"{label}({node_id})"
    
    def _node_to_dict(self, node):
        """Convert SceneNode to dictionary"""
        result = {"label": node.label, "confidence": node.confidence}
        if node.position:
            result["position"] = node.position
        if hasattr(node, 'bbox') and node.bbox:
            result["bbox"] = node.bbox  
        if node.attributes:
            result.update(node.attributes)
        return result
    
    def _edge_to_dict(self, edge):
        """Convert SceneEdge to dictionary"""
        result = {"relation_type": edge.relation_type, "confidence": edge.confidence}
        if edge.attributes:
            result.update(edge.attributes)
        return result
    
    def _encode_temporal_graph(self, temporal_graph: TemporalSceneGraph) -> str:
        """Encode temporal graph with temporal markers"""
        if not temporal_graph.scene_graphs:
            return "Empty temporal scene graph"
        
        parts = []
        for i, sg in enumerate(temporal_graph.scene_graphs):
            parts.append(f"T{i}: {self._encode_single_graph(sg)}")
        
        return " | ".join(parts)


class SceneGraphEncoderFactory:
    """Factory for creating scene graph encoders"""
    
    @staticmethod
    def create_encoder(encoding_format: str, feature_processor: Optional['FeatureProcessor'] = None, **kwargs) -> SceneGraphEncoder:
        """Create appropriate encoder based on format"""
        
        # Create encoders with feature processor
        if encoding_format.lower() == "json":
            return JSONSceneGraphEncoder(feature_processor=feature_processor, **kwargs)
        elif encoding_format.lower() == "xml":
            return XMLSceneGraphEncoder(feature_processor=feature_processor, **kwargs)
        elif encoding_format.lower() in ["text", "natural_language", "nl"]:
            return TextSceneGraphEncoder(feature_processor=feature_processor, **kwargs)
        elif encoding_format.lower() == "som":
            return SoMSceneGraphEncoder(feature_processor=feature_processor, **kwargs)
        elif encoding_format.lower() == "graphtext":
            return GraphTextEncoder(feature_processor=feature_processor, **kwargs)
        else:
            raise ValueError(f"Unknown encoding format: {encoding_format}")
    
    @staticmethod
    def list_available_encoders() -> List[str]:
        """List all available encoder formats"""
        return ["json", "xml", "text", "natural_language", "som", "graphtext"]


# Rest of the classes remain the same...
class TemporalGraphProcessor:
    """Process temporal aspects of 4D scene graphs"""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
    
    def extract_temporal_context(self, temporal_graph: TemporalSceneGraph, 
                                target_time: float) -> str:
        """Extract temporal context around a target time"""
        window_graphs = temporal_graph.get_temporal_window(target_time, self.window_size)
        
        if not window_graphs:
            return "No temporal context available"
        
        context_parts = []
        context_parts.append(f"Temporal context (window size: {len(window_graphs)}):")
        
        for i, sg in enumerate(window_graphs):
            time_label = "PAST" if sg.timestamp < target_time else "CURRENT" if sg.timestamp == target_time else "FUTURE"
            context_parts.append(f"{time_label} (t={sg.timestamp}): {len(sg.nodes)} objects, {len(sg.edges)} relationships")
        
        return "\n".join(context_parts)
    
    def get_object_trajectory(self, temporal_graph: TemporalSceneGraph, 
                            object_label: str) -> str:
        """Get trajectory description for a specific object"""
        trajectory_parts = []
        trajectory_parts.append(f"Trajectory for {object_label}:")
        
        for sg in temporal_graph.scene_graphs:
            # Find object in this timestep
            object_nodes = [node for node in sg.nodes if node.label == object_label]
            
            if object_nodes:
                node = object_nodes[0]  # Take first match
                if node.position:
                    pos_str = f"({node.position[0]:.2f}, {node.position[1]:.2f}, {node.position[2]:.2f})"
                    trajectory_parts.append(f"  t={sg.timestamp}: {pos_str}")
                else:
                    trajectory_parts.append(f"  t={sg.timestamp}: present")
            else:
                trajectory_parts.append(f"  t={sg.timestamp}: not visible")
        
        return "\n".join(trajectory_parts)


if __name__ == "__main__":
    # Example usage and testing
    from scene_graph_loader import SceneNode, SceneEdge, SceneGraph
    
    # Create example scene graph
    nodes = [
        SceneNode("1", "grasper", position=[1.0, 2.0, 3.0], attributes={"color": "silver"}),
        SceneNode("2", "gallbladder", position=[2.0, 3.0, 4.0], attributes={"color": "yellow"})
    ]
    edges = [
        SceneEdge("1", "2", "grasping")
    ]
    
    scene_graph = SceneGraph(
        timestamp=0.0,
        frame_id="test_frame",
        nodes=nodes,
        edges=edges
    )
    
    # Test different encoders
    encoders = {
        "JSON": SceneGraphEncoderFactory.create_encoder("json"),
        "XML": SceneGraphEncoderFactory.create_encoder("xml"),
        "Text": SceneGraphEncoderFactory.create_encoder("text"),
        "SoM": SceneGraphEncoderFactory.create_encoder("som"),
        "GraphText": SceneGraphEncoderFactory.create_encoder("graphtext")
    }
    
    for name, encoder in encoders.items():
        print(f"\n{name} Encoding:")
        try:
            result = encoder.encode(scene_graph)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)