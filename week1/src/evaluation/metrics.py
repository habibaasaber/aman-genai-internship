import json
import tiktoken
from typing import Dict, Any, Tuple

def validate_json_output(content: str) -> bool:
    """Checks if the LLM output is valid JSON. Useful for Task 1."""
    try:
        # Strip potential markdown formatting
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
        if content.endswith("```"):
            content = content[::-1].replace("```", "", 1)[::-1]
            
        json.loads(content.strip())
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def count_tokens_fallback(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Fallback token counter using tiktoken if the API doesn't return usage.
    Used mainly as an estimation for non-OpenAI models if their token count is missing.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Default to cl100k_base which is standard for current GPT models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def calculate_cost(
    prompt_tokens: int, 
    completion_tokens: int, 
    cost_per_1m_prompt: float, 
    cost_per_1m_completion: float
) -> float:
    """Calculates the estimated cost based on token usage."""
    prompt_cost = (prompt_tokens / 1_000_000) * cost_per_1m_prompt
    completion_cost = (completion_tokens / 1_000_000) * cost_per_1m_completion
    return prompt_cost + completion_cost
