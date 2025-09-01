"""
Unified interface for Multimodal Large Language Models (MLLMs)
Supports both local HuggingFace models and API-based models
"""
import torch
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
import openai
from config_manager import ModelConfig


class MLLMInterface(ABC):
    """Abstract base class for MLLM interfaces"""
    
    @abstractmethod
    def query(self, image: Image.Image, question: str, 
              scene_graph: Optional[str] = None, 
              additional_context: Optional[str] = None) -> str:
        """Query the model with image, question and optional scene graph"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class LocalMLLMInterface(MLLMInterface):
    """Interface for local HuggingFace models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor based on model type
        self._load_model()
    
    def _load_model(self):
        """Load the specified model and processor"""
        model_path = self.config.model_path
        
        try:
            # Check for Qwen2.5-VL models
            if "qwen2.5-vl" in model_path.lower() or "qwen2_5_vl" in model_path.lower():
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.model_type = "qwen2_5_vl"
                print(f"Loaded Qwen2.5-VL model: {model_path}")
            
            else:
                # Generic multimodal model loading
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                self.model_type = "generic"
                print(f"Loaded generic model: {model_path}")
            
            print(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise
    
    def query(self, image: Image.Image, question: str, 
              scene_graph: Optional[str] = None, 
              additional_context: Optional[str] = None) -> str:
        """Query the local model"""
        
        try:
            if self.model_type == "qwen2_5_vl":
                return self._query_qwen2_5_vl(image, question, scene_graph, additional_context)
            else:
                return self._query_generic(image, question, scene_graph, additional_context)
                
        except Exception as e:
            print(f"Error during model inference: {e}")
            return f"Error: {str(e)}"
    
    def _construct_prompt(self, question: str, scene_graph: Optional[str], 
                         additional_context: Optional[str]) -> str:
        """Construct the complete prompt with all context"""
        prompt_parts = []
        
        # Add scene graph context if available
        if scene_graph:
            prompt_parts.append("Scene Graph Information:")
            prompt_parts.append(scene_graph)
            prompt_parts.append("")
        
        # Add additional context if available
        if additional_context:
            prompt_parts.append("Additional Context:")
            prompt_parts.append(additional_context)
            prompt_parts.append("")
        
        # Add the main question
        prompt_parts.append("Question:")
        prompt_parts.append(question)
        
        return "\n".join(prompt_parts)
    
    def _query_qwen2_5_vl(self, image: Image.Image, question: str, 
                          scene_graph: Optional[str] = None, 
                          additional_context: Optional[str] = None) -> str:
        """Query Qwen2.5-VL model"""
        
        # Construct the full prompt with context
        prompt_text = self._construct_prompt(question, scene_graph, additional_context)
        
        # Construct messages in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Preparation for inference
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
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                **self.config.additional_params
            )
        
        # Decode response (extract only the generated part)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Return the first (and likely only) response
        return output_text[0].strip() if output_text else ""
    
    def _query_generic(self, image: Image.Image, question: str,
                      scene_graph: Optional[str] = None, 
                      additional_context: Optional[str] = None) -> str:
        """Query generic multimodal model"""
        prompt = self._construct_prompt(question, scene_graph, additional_context)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **self.config.additional_params
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _query_qwen2_5_vl_with_features(self, image: Image.Image, question: str, 
                                       scene_graph: Optional[str] = None,
                                       raw_features: Optional[Dict[str, torch.Tensor]] = None,
                                       additional_context: Optional[str] = None) -> str:
        """Query Qwen2.5-VL with optional raw feature injection"""
        
        # Construct the full prompt with context
        prompt_text = self._construct_prompt(question, scene_graph, additional_context)
        
        # Construct messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Standard processing
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
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        # Inject raw features if available (experimental)
        if raw_features and hasattr(self.model, 'inject_features'):
            # This would require custom model modifications
            self.model.inject_features(raw_features)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                **self.config.additional_params
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip() if output_text else ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.name,
            "type": "local",
            "model_path": self.config.model_path,
            "device": str(self.device),
            "model_type": self.model_type
        }


class APIMLLMInterface(MLLMInterface):
    """Interface for API-based models (OpenRouter, OpenAI, etc.)"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Set up OpenAI client for OpenRouter or direct OpenAI API
        openai.api_key = config.api_key
        if config.api_base_url:
            openai.api_base = config.api_base_url
        
        self.client = openai
    
    def query(self, image: Image.Image, question: str, 
              scene_graph: Optional[str] = None, 
              additional_context: Optional[str] = None) -> str:
        """Query the API model"""
        
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Construct messages
            messages = self._construct_messages(question, scene_graph, 
                                              additional_context, image_b64)
            
            # Make API call
            response = self.client.ChatCompletion.create(
                model=self.config.name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **self.config.additional_params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error during API call: {e}")
            return f"API Error: {str(e)}"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        return image_b64
    
    def _construct_messages(self, question: str, scene_graph: Optional[str], 
                          additional_context: Optional[str], image_b64: str) -> List[Dict]:
        """Construct messages for API call"""
        
        # Build system message with context
        system_parts = [
            "You are a helpful assistant that analyzes surgical videos and answers questions about them.",
            "Pay careful attention to the provided image and any additional context."
        ]
        
        # Build user message content
        content = []
        
        # Add scene graph context
        if scene_graph:
            content.append({
                "type": "text",
                "text": f"Scene Graph Information:\n{scene_graph}\n"
            })
        
        # Add additional context
        if additional_context:
            content.append({
                "type": "text", 
                "text": f"Additional Context:\n{additional_context}\n"
            })
        
        # Add image
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })
        
        # Add question
        content.append({
            "type": "text",
            "text": f"Question: {question}\n\nPlease provide a clear and concise answer based on the image and any provided context."
        })
        
        messages = [
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": content}
        ]
        
        return messages
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.name,
            "type": "api",
            "api_base_url": self.config.api_base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }


class MLLMFactory:
    """Factory for creating MLLM interfaces"""
    
    @staticmethod
    def create_interface(config: ModelConfig) -> MLLMInterface:
        """Create an MLLM interface based on configuration"""
        
        if config.type.lower() == "local":
            return LocalMLLMInterface(config)
        elif config.type.lower() == "api":
            return APIMLLMInterface(config)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")


class ResponseProcessor:
    """Process and normalize responses from different MLLMs"""
    
    def __init__(self):
        # Map common response formats to SSG-VQA label format
        self.label_mapping = {
            "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
            "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
            "false": "False", "true": "True", "yes": "True", "no": "False",
            "gallbladder": "gallbladder", "liver": "liver", "grasper": "grasper",
            "scissors": "scissors", "hook": "hook", "clipper": "clipper",
            "irrigator": "irrigator", "bipolar": "bipolar",
            "red": "red", "blue": "blue", "yellow": "yellow", "white": "white",
            "silver": "silver", "brown": "brown",
            "grasp": "grasp", "cut": "cut", "dissect": "dissect", "coagulate": "coagulate",
            "irrigate": "irrigate", "retract": "retract", "clip": "clip", "aspirate": "aspirate"
        }
    
    def process_response(self, response: str, expected_type: str = "classification") -> str:
        """Process response and map to expected format"""
        response = response.strip().lower()
        
        if expected_type == "classification":
            # Try to find a mappable label in the response
            for key, value in self.label_mapping.items():
                if key in response:
                    return value
            
            # If no exact match, try to extract numbers or boolean values
            if any(word in response for word in ["yes", "true", "correct"]):
                return "True"
            elif any(word in response for word in ["no", "false", "incorrect"]):
                return "False"
            
            # Try to extract numbers
            import re
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                return numbers[0]
        
        # Return original response if no processing needed
        return response
    
    def extract_confidence(self, response: str) -> float:
        """Extract confidence score from response if available"""
        # Look for confidence indicators
        import re
        
        confidence_patterns = [
            r'confidence[:\s]*([0-9]*\.?[0-9]+)',
            r'certainty[:\s]*([0-9]*\.?[0-9]+)',
            r'probability[:\s]*([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Default confidence based on response characteristics
        if any(word in response.lower() for word in ["certain", "sure", "definitely"]):
            return 0.9
        elif any(word in response.lower() for word in ["likely", "probably"]):
            return 0.7
        elif any(word in response.lower() for word in ["maybe", "possibly", "might"]):
            return 0.5
        else:
            return 0.8  # Default moderate confidence


if __name__ == "__main__":
    # Example usage
    from config_manager import ModelConfig
    
    # Test Qwen2.5-VL model config
    qwen_config = ModelConfig(
        name="qwen2.5-vl-7b",
        type="local",
        model_path="/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct",
        max_tokens=500,
        temperature=0.1
    )
    
    # Test API model config
    api_config = ModelConfig(
        name="gpt-4-vision-preview",
        type="api",
        api_key="your-api-key",
        api_base_url="https://openrouter.ai/api/v1",
        max_tokens=500
    )
    
    print("MLLM Interface configurations created successfully")
    
    # Example of creating interfaces (commented out to avoid actual model loading)
    # qwen_interface = MLLMFactory.create_interface(qwen_config)
    # api_interface = MLLMFactory.create_interface(api_config)
