"""
Feature processing utilities for handling CLIP and other visual/textual features
in scene graphs for MLLM consumption.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureProcessorConfig:
    """Configuration for feature processing with sensible defaults"""
    
    # Preset configuration name (overrides individual settings if specified)
    preset: Optional[str] = None  # "default", "surgical_optimized", "fast_inference", "research_mode"
    
    # CLIP Model Configuration
    clip_model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: Optional[str] = None
    
    # Feature Processing Options
    enable_feature_decoding: bool = True
    enable_raw_features: bool = True
    
    # Decoding Configuration
    top_k_matches: int = 5
    confidence_threshold: float = 0.1
    
    # Vocabulary Configuration
    surgical_vocabulary_file: Optional[str] = None
    use_extended_vocabulary: bool = True
    custom_vocabulary_terms: Optional[List[str]] = None
    
    # Output Configuration
    include_alternative_descriptions: bool = True
    max_description_length: int = 100
    include_confidence_scores: bool = True
    
    # Fallback Configuration
    fallback_to_basic_description: bool = True
    broken_feature_reconstruction: bool = True
    
    def __post_init__(self):
        """Apply preset configurations if specified"""
        if self.preset:
            self._apply_preset(self.preset)
    
    def _apply_preset(self, preset_name: str):
        """Apply predefined configuration presets"""
        presets = {
            "default": {
                # Keep current values - this is the baseline
            },
            "surgical_optimized": {
                "clip_model_name": "ViT-L-14",
                "pretrained": "laion400m_e32",
                "top_k_matches": 3,
                "confidence_threshold": 0.2,
                "use_extended_vocabulary": True,
                "include_alternative_descriptions": True
            },
            "fast_inference": {
                "clip_model_name": "ViT-B-16",
                "pretrained": "laion2b_s34b_b88k",
                "top_k_matches": 3,
                "enable_raw_features": False,
                "include_alternative_descriptions": False,
                "use_extended_vocabulary": False
            },
            "research_mode": {
                "clip_model_name": "ViT-H-14",
                "pretrained": "laion2b_s34b_b79k",
                "top_k_matches": 10,
                "confidence_threshold": 0.05,
                "enable_raw_features": True,
                "include_alternative_descriptions": True,
                "use_extended_vocabulary": True
            }
        }
        
        if preset_name in presets:
            preset_config = presets[preset_name]
            for key, value in preset_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            logger.warning(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")


class FeatureDecoder(ABC):
    """Abstract base class for feature decoders"""
    
    @abstractmethod
    def decode_features(self, features: Union[torch.Tensor, np.ndarray, List]) -> Dict[str, Any]:
        """Decode features to human-readable format"""
        pass


class CLIPFeatureDecoder(FeatureDecoder):
    """Decode CLIP features using nearest neighbor search against surgical vocabulary"""
    
    def __init__(self, config: FeatureProcessorConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Open-CLIP model
        if open_clip is None:
            raise ImportError("open_clip not installed. Install with: pip install open-clip-torch")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                config.clip_model_name, 
                pretrained=config.pretrained,
                device=self.device,
                precision="fp16",
            )
            self.tokenizer = open_clip.get_tokenizer(config.clip_model_name)
            
            logger.info(f"Loaded Open-CLIP model: {config.clip_model_name} with {config.pretrained} weights")
        except Exception as e:
            logger.error(f"Failed to load Open-CLIP model {config.clip_model_name}: {e}")
            # Fallback to a common model
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained='openai',
                    device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                logger.info("Loaded fallback Open-CLIP model: ViT-B-32")
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model: {fallback_e}")
                raise
        
        # Initialize surgical vocabulary using config
        self.surgical_vocabulary = self._load_surgical_vocabulary()
        self.text_features = self._encode_vocabulary()
        
        logger.info(f"CLIPFeatureDecoder initialized with {len(self.surgical_vocabulary)} terms")
    
    def _load_surgical_vocabulary(self, vocab_file: Optional[str] = None) -> List[str]:
        """Load surgical vocabulary from file or use default"""
        
        if vocab_file:
            try:
                with open(vocab_file, 'r') as f:
                    vocabulary = [line.strip() for line in f if line.strip()]
                return vocabulary
            except FileNotFoundError:
                logger.warning(f"Vocabulary file {vocab_file} not found, using default")
        
        # Default surgical vocabulary
        return [
            # Surgical instruments
            "grasper", "forceps", "surgical forceps", "laparoscopic grasper",
            "scissors", "surgical scissors", "laparoscopic scissors", "curved scissors",
            "scalpel", "surgical knife", "electrocautery", "cautery device",
            "clipper", "clip applier", "titanium clips", "hemoclips",
            "irrigator", "irrigation device", "suction device", "aspirator",
            "retractor", "surgical retractor", "liver retractor",
            "trocar", "laparoscopic trocar", "port", "surgical port",
            "camera", "laparoscope", "endoscope", "surgical camera",
            "hook", "surgical hook", "electrocautery hook", "L-hook",
            "bipolar", "bipolar forceps", "bipolar device",
            
            # Anatomical structures
            "gallbladder", "cholecyst", "gallbladder wall", "inflated gallbladder",
            "liver", "hepatic tissue", "liver surface", "liver edge",
            "cystic artery", "hepatic artery", "blood vessel", "artery",
            "cystic duct", "common bile duct", "hepatocystic duct", "bile duct",
            "peritoneum", "peritoneal surface", "abdominal wall", "peritoneal cavity",
            "adhesion", "scar tissue", "fibrous tissue", "inflammatory tissue",
            "fat", "adipose tissue", "omental fat", "peritoneal fat",
            "connective tissue", "fascia", "muscle", "abdominal muscle",
            "bowel", "intestine", "colon", "duodenum",
            "diaphragm", "hepatic flexure", "stomach", "gastric surface",
            
            # Surgical materials and objects
            "suture", "surgical thread", "stitches", "suture material",
            "mesh", "surgical mesh", "hernia mesh",
            "sponge", "surgical sponge", "gauze", "surgical gauze",
            "drain", "surgical drain", "jackson-pratt drain",
            "clip", "surgical clip", "metal clip", "titanium clip",
            
            # Tissue states and conditions
            "inflamed tissue", "healthy tissue", "diseased tissue",
            "bleeding", "hemorrhage", "blood", "coagulated blood",
            "edematous tissue", "swollen tissue", "congested tissue",
            "necrotic tissue", "ischemic tissue", "viable tissue",
            
            # Colors and textures (surgical context)
            "red tissue", "pink tissue", "pale tissue", "dark tissue",
            "smooth surface", "rough surface", "textured surface",
            "shiny surface", "matte surface", "wet surface", "dry surface",
            
            # Generic objects
            "metallic object", "surgical tool", "medical device", "instrument",
            "anatomical structure", "organ", "tissue", "biological structure"
        ]
    
    def _encode_vocabulary(self) -> torch.Tensor:
        """Pre-encode the surgical vocabulary using Open-CLIP text encoder"""
        # Tokenize text using Open-CLIP tokenizer
        text_tokens = self.tokenizer(self.surgical_vocabulary).to(self.device)
        
        with torch.no_grad():
            # Use Open-CLIP's encode_text method
            text_features = self.model.encode_text(text_tokens)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def decode_features(self, features: Union[torch.Tensor, np.ndarray, List]) -> Dict[str, Any]:
        """Decode CLIP features using nearest neighbor search"""
        
        # Convert input to tensor
        if isinstance(features, (list, str)):
            if isinstance(features, str):
                # Handle string representations like "[0.1, 0.2, ...]"
                try:
                    features = eval(features) if features.startswith('[') else None
                except:
                    logger.warning("Failed to parse feature string")
                    return {"error": "Invalid feature format"}
            
            if features is None:
                return {"error": "Could not parse features"}
                
            features = torch.tensor(features, dtype=torch.float32)
        
        elif isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        features = features.to(self.device)
        
        # Ensure proper shape and normalization
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with vocabulary
        with torch.no_grad():
            # Open-CLIP uses the same cosine similarity approach
            similarities = torch.cosine_similarity(features, self.text_features.unsqueeze(0), dim=-1)
            similarities = similarities.squeeze()
            
            # Get top matches
            top_k = min(5, len(self.surgical_vocabulary))
            top_scores, top_indices = similarities.topk(top_k)
            
            top_terms = [self.surgical_vocabulary[idx.item()] for idx in top_indices]
            top_scores = top_scores.cpu().tolist()
        
        return {
            "primary_description": top_terms[0],
            "confidence": top_scores[0],
            "alternative_descriptions": top_terms[1:],
            "alternative_confidences": top_scores[1:],
            "all_matches": list(zip(top_terms, top_scores))
        }
    
    def get_raw_features(self, features: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """Get raw features in proper tensor format for direct MLLM consumption"""
        if isinstance(features, (list, str)):
            if isinstance(features, str):
                try:
                    features = eval(features) if features.startswith('[') else None
                except:
                    return None
            features = torch.tensor(features, dtype=torch.float32) if features else None
        elif isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        return features.to(self.device) if features is not None else None


class FeatureProcessor:
    """Main processor for handling different types of features in scene graphs"""
    
    def __init__(self, config: Optional[FeatureProcessorConfig] = None, **kwargs):
        # Handle both new config-based and legacy parameter-based initialization
        if config is None:
            # Legacy support: create config from individual parameters
            config = FeatureProcessorConfig(
                clip_model_name=kwargs.get('clip_model_name', 'ViT-B-32'),
                pretrained=kwargs.get('pretrained', 'openai'),
                device=kwargs.get('device'),
                enable_feature_decoding=kwargs.get('enable_feature_decoding', True),
                enable_raw_features=kwargs.get('enable_raw_features', True)
            )
        
        self.config = config
        
        # Initialize decoders with Open-CLIP
        if self.config.enable_feature_decoding:
            try:
                self.clip_decoder = CLIPFeatureDecoder(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize Open-CLIP decoder: {e}")
                self.clip_decoder = None
        else:
            self.clip_decoder = None
    
    def _reconstruct_broken_features(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct feature vectors from broken GraphML individual attributes as fallback"""
        reconstructed = {}
        feature_groups = {}
        
        # Group individual feature attributes
        for key, value in attrs.items():
            if '_feat_' in key and key.split('_')[-1].isdigit():
                try:
                    parts = key.split('_')
                    index_str = parts[-1]
                    index = int(index_str)
                    prefix = '_'.join(parts[:-1])
                    
                    if prefix not in feature_groups:
                        feature_groups[prefix] = {}
                    feature_groups[prefix][index] = float(value)
                except (ValueError, IndexError):
                    reconstructed[key] = value
            else:
                reconstructed[key] = value
        
        # Reconstruct vectors
        for prefix, indexed_values in feature_groups.items():
            if len(indexed_values) > 1:
                sorted_indices = sorted(indexed_values.keys())
                feature_vector = [indexed_values[i] for i in sorted_indices]
                
                if 'lang' in prefix.lower():
                    reconstructed['clip_features'] = feature_vector
                elif 'visual' in prefix.lower():
                    reconstructed['visual_features'] = feature_vector
                else:
                    reconstructed[f'{prefix}_vector'] = feature_vector
                
                logger.info(f"Reconstructed broken {prefix} -> {len(feature_vector)}-dim vector")
        
        return reconstructed

    def process_node_features(self, node_id: str, node_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Process all features in a node's attributes with broken feature reconstruction"""
        processed = {
            "id": node_id,
            "original_attributes": {},
            "decoded_features": {},
            "raw_features": {}
        }
        
        # First, try to reconstruct any broken feature vectors as fallback
        reconstructed_attrs = self._reconstruct_broken_features(node_attrs)
        
        for attr_key, attr_value in reconstructed_attrs.items():
            if self._is_feature_attribute(attr_key):
                # This is a feature attribute
                if self.config.enable_feature_decoding and self.clip_decoder:
                    try:
                        decoded = self.clip_decoder.decode_features(attr_value)
                        processed["decoded_features"][attr_key] = decoded
                    except Exception as e:
                        logger.warning(f"Failed to decode {attr_key}: {e}")
                
                if self.config.enable_raw_features:
                    raw_features = self.clip_decoder.get_raw_features(attr_value) if self.clip_decoder else None
                    if raw_features is not None:
                        processed["raw_features"][attr_key] = raw_features
            else:
                # Regular attribute
                processed["original_attributes"][attr_key] = attr_value
        
        return processed

    def _is_feature_attribute(self, attr_key: str) -> bool:
        """Determine if an attribute contains features that need processing"""
        feature_indicators = [
            "clip", "feature", "embedding", "vector", "representation"
        ]
        
        # Standard feature attribute names
        if any(indicator in attr_key.lower() for indicator in feature_indicators):
            return True
        
        # Detect broken GraphML individual feature attributes (fallback)
        if '_feat_' in attr_key and attr_key.split('_')[-1].isdigit():
            return True
        
        return False
    
    def create_textual_description(self, processed_node: Dict[str, Any]) -> str:
        """Create a textual description from processed node data with config limits"""
        node_id = processed_node["id"]
        description_parts = [f"Object {node_id}"]
        
        # Add decoded feature descriptions
        if processed_node["decoded_features"]:
            for feature_key, decoded in processed_node["decoded_features"].items():
                if "primary_description" in decoded:
                    desc_part = f"({decoded['primary_description']}"
                    
                    if self.config.include_confidence_scores:
                        conf = decoded.get("confidence", 0)
                        desc_part += f", confidence: {conf:.3f}"
                    
                    # Add alternatives if configured
                    if (self.config.include_alternative_descriptions and 
                        decoded.get("alternative_descriptions")):
                        alts = decoded["alternative_descriptions"][:2]  # Limit to 2 alternatives
                        desc_part += f", alternatives: {', '.join(alts)}"
                    
                    desc_part += ")"
                    description_parts.append(desc_part)
        
        # Add original attributes
        if processed_node["original_attributes"]:
            attr_strings = [
                f"{k}: {v}" for k, v in processed_node["original_attributes"].items()
                if v is not None
            ]
            if attr_strings:
                description_parts.append(f"[{', '.join(attr_strings)}]")
        
        full_description = " ".join(description_parts)
        
        # Truncate if too long
        if len(full_description) > self.config.max_description_length:
            full_description = full_description[:self.config.max_description_length-3] + "..."
        
        return full_description


# Let's patch the CLIPFeatureDecoder to use general vocabulary
class GeneralCLIPFeatureDecoder(CLIPFeatureDecoder):
    """CLIP decoder that uses general object categories instead of surgical vocabulary"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Open-CLIP model (same as parent)
        if open_clip is None:
            raise ImportError("open_clip not installed. Install with: pip install open-clip-torch")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                config.clip_model_name, 
                pretrained=config.pretrained,
                device=self.device,
                precision="fp16",
            )
            self.tokenizer = open_clip.get_tokenizer(config.clip_model_name)
            print(f"Loaded Open-CLIP model: {config.clip_model_name} with {config.pretrained} weights")
        except Exception as e:
            print(f"Failed to load Open-CLIP model: {e}")
            raise
        
        # Use general vocabulary instead of surgical
        self.general_vocabulary = self._get_general_vocabulary()
        self.text_features = self._encode_general_vocabulary()
        
        print(f"GeneralCLIPFeatureDecoder initialized with {len(self.general_vocabulary)} general terms")
    
    def _get_general_vocabulary(self):
        """General object vocabulary for everyday scenes"""
        return [
            # People and body parts
            "person", "people", "man", "woman", "child", "face", "hand", "body", "head",
            
            # Animals
            "animal", "dog", "cat", "bird", "horse", "chicken", "cow", "fish", "pet",
            
            # Common objects
            "object", "thing", "item", "tool", "container", "box", "bag", "bottle",
            
            # Vehicles
            "car", "truck", "bicycle", "motorcycle", "bus", "vehicle", "wheel",
            
            # Furniture
            "chair", "table", "bed", "sofa", "desk", "shelf", "cabinet", "furniture",
            
            # Electronics
            "computer", "phone", "screen", "television", "camera", "device", "machine",
            
            # Food and kitchen
            "food", "plate", "cup", "bowl", "spoon", "fork", "knife", "kitchen", "egg", "board", "shell", "chick", "chicken",
            
            # Buildings and structures
            "building", "house", "wall", "door", "window", "roof", "room", "structure",
            
            # Nature
            "tree", "flower", "plant", "grass", "rock", "water", "sky", "cloud",
            
            # Colors
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", 
            "black", "white", "gray", "grey", "colorful",
            
            # Shapes and sizes
            "round", "square", "rectangular", "circular", "triangular",
            "large", "small", "big", "little", "tiny", "huge", "long", "short",
            
            # Materials
            "wooden", "metal", "plastic", "glass", "fabric", "paper", "leather",
            "stone", "concrete", "ceramic",
            
            # Textures and properties
            "smooth", "rough", "soft", "hard", "shiny", "matte", "transparent",
            "opaque", "bright", "dark", "light", "heavy",
            
            # Locations and positions
            "indoor", "outdoor", "inside", "outside", "center", "corner", "edge",
            "top", "bottom", "left", "right", "front", "back",
            
            # Generic descriptors
            "visible", "hidden", "clear", "blurry", "detailed", "simple", "complex",
            "new", "old", "clean", "dirty", "broken", "whole", "empty", "full"
        ]
    
    def _encode_general_vocabulary(self):
        """Encode the general vocabulary using Open-CLIP"""
        text_tokens = self.tokenizer(self.general_vocabulary).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def decode_features(self, features):
        """Decode features using general vocabulary"""
        # Convert input to tensor (same logic as parent)
        if isinstance(features, (list, str)):
            if isinstance(features, str):
                try:
                    features = eval(features) if features.startswith('[') else None
                except:
                    return {"error": "Invalid feature format"}
            
            if features is None:
                return {"error": "Could not parse features"}
                
            features = torch.tensor(features, dtype=torch.float32)
        
        elif isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        features = features.to(self.device)
        
        # Ensure proper shape and normalization
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with general vocabulary
        with torch.no_grad():
            similarities = torch.cosine_similarity(features, self.text_features.unsqueeze(0), dim=-1)
            similarities = similarities.squeeze()
            
            # Get top matches
            top_k = min(self.config.top_k_matches, len(self.general_vocabulary))
            top_scores, top_indices = similarities.topk(top_k)
            
            top_terms = [self.general_vocabulary[idx.item()] for idx in top_indices]
            top_scores = top_scores.cpu().tolist()
        
        return {
            "primary_description": top_terms[0],
            "confidence": top_scores[0],
            "alternative_descriptions": top_terms[1:] if len(top_terms) > 1 else [],
            "alternative_confidences": top_scores[1:] if len(top_scores) > 1 else [],
            "all_matches": list(zip(top_terms, top_scores))
        }

# Create a general feature processor using the patched decoder
class GeneralFeatureProcessor(FeatureProcessor):
    """Feature processor that uses general vocabulary instead of surgical"""
    
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = FeatureProcessorConfig(
                clip_model_name=kwargs.get('clip_model_name', 'ViT-B-32'),
                pretrained=kwargs.get('pretrained', 'openai'),
                device=kwargs.get('device'),
                enable_feature_decoding=kwargs.get('enable_feature_decoding', True),
                enable_raw_features=kwargs.get('enable_raw_features', True),
                top_k_matches=kwargs.get('top_k_matches', 5),
                confidence_threshold=kwargs.get('confidence_threshold', 0.1)
            )
        
        self.config = config
        
        # Use the general decoder instead of surgical one
        if self.config.enable_feature_decoding:
            try:
                self.clip_decoder = GeneralCLIPFeatureDecoder(self.config)
            except Exception as e:
                print(f"Failed to initialize general CLIP decoder: {e}")
                self.clip_decoder = None
        else:
            self.clip_decoder = None


# Factory function with Open-CLIP defaults
def create_feature_processor(**kwargs) -> FeatureProcessor:
    """Factory function to create a feature processor with Open-CLIP configuration"""
    # Set Open-CLIP compatible defaults
    defaults = {
        'clip_model_name': 'ViT-B-32',
        'pretrained': 'openai'
    }
    defaults.update(kwargs)
    return FeatureProcessor(**defaults)

# Updated factory functions with preset support
def create_feature_processor(config: Optional[FeatureProcessorConfig] = None, **kwargs) -> FeatureProcessor:
    """Factory function to create a feature processor with configuration or legacy parameters"""
    if config is None:
        # Create config from kwargs (supports both new preset approach and legacy parameters)
        config = FeatureProcessorConfig(**kwargs)
    return FeatureProcessor(config)

def create_feature_processor_from_config(config: FeatureProcessorConfig) -> FeatureProcessor:
    """Create feature processor from configuration object"""
    return FeatureProcessor(config)

def create_feature_processor_from_preset(preset_name: str, **overrides) -> FeatureProcessor:
    """Create feature processor from preset name with optional overrides"""
    config = FeatureProcessorConfig(preset=preset_name)
    
    # Apply any override parameters
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return FeatureProcessor(config)

# Convenience factory functions for common presets
def create_surgical_optimized_processor(**overrides) -> FeatureProcessor:
    """Create processor optimized for surgical scene understanding"""
    return create_feature_processor_from_preset("surgical_optimized", **overrides)

def create_fast_processor(**overrides) -> FeatureProcessor:
    """Create processor optimized for fast inference"""
    return create_feature_processor_from_preset("fast_inference", **overrides)

def create_research_processor(**overrides) -> FeatureProcessor:
    """Create processor for research with maximum information"""
    return create_feature_processor_from_preset("research_mode", **overrides)

# Legacy factory function (backward compatibility)
def create_feature_processor_legacy(**kwargs) -> FeatureProcessor:
    """Legacy factory function - use create_feature_processor instead"""
    logger.warning("create_feature_processor_legacy is deprecated. Use create_feature_processor instead.")
    # Set Open-CLIP compatible defaults for backward compatibility
    defaults = {
        'clip_model_name': 'ViT-B-32',
        'pretrained': 'openai'
    }
    defaults.update(kwargs)
    return FeatureProcessor(None, **defaults)


if __name__ == "__main__":
    # Test the feature processor with Open-CLIP and presets
    print("=== Testing Open-CLIP Feature Processor with Presets ===")
    
    try:
        # Test 1: Default processor
        print("\n1. Testing default processor...")
        processor_default = create_feature_processor()
        print("✅ Default processor initialized successfully")
        
        # Test 2: Preset-based processors
        print("\n2. Testing preset-based processors...")
        
        processor_surgical = create_surgical_optimized_processor()
        print("✅ Surgical optimized processor initialized")
        print(f"   Model: {processor_surgical.config.clip_model_name}")
        print(f"   Pretrained: {processor_surgical.config.pretrained}")
        print(f"   Top-k matches: {processor_surgical.config.top_k_matches}")
        
        processor_fast = create_fast_processor()
        print("✅ Fast processor initialized")
        print(f"   Model: {processor_fast.config.clip_model_name}")
        print(f"   Enable raw features: {processor_fast.config.enable_raw_features}")
        
        processor_research = create_research_processor()
        print("✅ Research processor initialized")
        print(f"   Model: {processor_research.config.clip_model_name}")
        print(f"   Top-k matches: {processor_research.config.top_k_matches}")
        
        # Test 3: Preset with overrides
        print("\n3. Testing preset with overrides...")
        processor_custom = create_surgical_optimized_processor(
            top_k_matches=2,
            confidence_threshold=0.3,
            custom_vocabulary_terms=["my_custom_tool"]
        )
        print("✅ Custom surgical processor initialized")
        print(f"   Top-k matches (overridden): {processor_custom.config.top_k_matches}")
        print(f"   Confidence threshold (overridden): {processor_custom.config.confidence_threshold}")
        print(f"   Has custom vocabulary: {processor_custom.config.custom_vocabulary_terms is not None}")
        
        # Test 4: Direct config creation
        print("\n4. Testing direct config creation...")
        config = FeatureProcessorConfig(
            preset="fast_inference",
            clip_model_name="ViT-B-16",  # Override the preset
            custom_vocabulary_terms=["special_instrument"]
        )
        processor_config = create_feature_processor(config)
        print("✅ Config-based processor initialized")
        print(f"   Model (overridden): {processor_config.config.clip_model_name}")
        print(f"   Enable raw features (from preset): {processor_config.config.enable_raw_features}")
        
        # Test 5: Legacy support
        print("\n5. Testing legacy parameter support...")
        processor_legacy = create_feature_processor(
            clip_model_name="ViT-L-14",
            enable_feature_decoding=True,
            top_k_matches=3
        )
        print("✅ Legacy parameter processor initialized")
        print(f"   Model: {processor_legacy.config.clip_model_name}")
        print(f"   Top-k: {processor_legacy.config.top_k_matches}")
        
        # Test actual feature processing
        print("\n6. Testing actual feature processing...")
        
        # Test with dummy CLIP features (512-dim for ViT-B-32)
        dummy_features = torch.randn(512).tolist()
        
        test_node = {
            "clip_features": dummy_features,
            "position": [100, 200],
            "size": "large"
        }
        
        processed = processor_surgical.process_node_features("test_node", test_node)
        description = processor_surgical.create_textual_description(processed)
        
        print("✅ Feature processing successful")
        print("Processed node keys:", list(processed.keys()))
        print("Description:", description)
        
        # Test broken feature reconstruction
        print("\n7. Testing broken feature reconstruction...")
        broken_node = {
            "lang_feat_0": 0.123,
            "lang_feat_1": 0.456,
            "lang_feat_2": 0.789,
            "label": "test_object"
        }
        
        processed_broken = processor_surgical.process_node_features("broken_test", broken_node)
        description_broken = processor_surgical.create_textual_description(processed_broken)
        
        print("✅ Broken feature reconstruction successful")
        print("Broken features processed:", list(processed_broken['decoded_features'].keys()))
        print("Broken description:", description_broken)
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()