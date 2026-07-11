import os
import openai
from typing import Dict, Any
from src.models.base_client import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIClient(BaseLLMClient):
    """Client wrapper for OpenAI APIs."""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        super().__init__(model_name, temperature, max_tokens)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        # using the new OpenAI client instantiated class
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            return {
                "content": content,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None
            }
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            return {
                "content": f"Error: {str(e)}",
                "prompt_tokens": None,
                "completion_tokens": None
            }
