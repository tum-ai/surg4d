"""Utilities for JSON serialization of benchmark results."""
from typing import Any, Dict, List, Optional
import json


def sanitize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove non-JSON-serializable objects (tensors) from tool call results.
    
    Args:
        tool_calls: List of tool call dicts potentially containing tensors
        
    Returns:
        List of sanitized tool call dicts safe for JSON serialization
    """
    sanitized = []
    for tc in tool_calls:
        clean_tc = {
            "tool_name": tc.get("tool_name"),
            "arguments": tc.get("arguments"),
        }
        # Only keep the text part of results, drop vision_features (tensors)
        result = tc.get("result", {})
        if isinstance(result, dict):
            clean_tc["result"] = {"text": result.get("text", "")}
        else:
            clean_tc["result"] = str(result)
        sanitized.append(clean_tc)
    return sanitized

def parse_json(response: str) -> Optional[Dict[str, Any]]:
    """Extract last valid JSON object from response.
    
    Tries to find the last valid JSON object by trying different starting positions.
    This handles cases where the response contains multiple { } pairs or invalid JSON.
    """
    last_brace = response.rfind('}')
    if last_brace < 0:
        return None
    
    # Try to find valid JSON by working backwards from the last }
    # Find all opening braces and try parsing from each one to the last }
    opening_braces = []
    for i, char in enumerate(response[:last_brace + 1]):
        if char == '{':
            opening_braces.append(i)
    
    # Try parsing from each opening brace (starting from the last one)
    for start in reversed(opening_braces):
        json_str = response[start:last_brace + 1]
        try:
            data = json.loads(json_str)
            return data
        except (json.JSONDecodeError, ValueError):
            continue
    
    return None