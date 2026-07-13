import os
import openai
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_openai(model_name: str, temperature: float, max_tokens: int, prompt: str) -> Dict[str, Any]:
    """Generates text using OpenAI and returns response content and token usage."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables.")

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        usage = response.usage

        return {
            "content": content,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return {
            "content": f"Error: {str(e)}",
            "prompt_tokens": None,
            "completion_tokens": None,
        }
