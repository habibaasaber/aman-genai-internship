import os
from google import genai
from google.genai import types
from typing import Dict, Any
from src.models.base_client import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient(BaseLLMClient):
    """Client wrapper for Google Gemini APIs using the new google.genai SDK."""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        super().__init__(model_name, temperature, max_tokens)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            content = response.text

            # Extract token usage from usage_metadata if available
            prompt_tokens = None
            completion_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return {
                "content": f"Error: {str(e)}",
                "prompt_tokens": None,
                "completion_tokens": None
            }
