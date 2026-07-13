import os
from google import genai
from google.genai import types
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_gemini(model_name: str, temperature: float, max_tokens: int, prompt: str) -> Dict[str, Any]:
    """Generates text using Gemini and returns response content and token usage."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        content = response.text
        prompt_tokens = None
        completion_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {
            "content": f"Error: {str(e)}",
            "prompt_tokens": None,
            "completion_tokens": None,
        }
