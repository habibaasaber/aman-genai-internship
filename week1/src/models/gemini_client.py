import os
import time
from google import genai
from google.genai import types
from typing import Dict, Any
from src.utils.logger import get_logger
from src.utils.langfuse_client import get_langfuse_client

logger = get_logger(__name__)


def generate_gemini(model_name: str, temperature: float, max_tokens: int, prompt: str) -> Dict[str, Any]:
    """Generates text using Gemini and returns response content and token usage."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)

    langfuse = get_langfuse_client()

    try:
        start_time = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency_ms = (time.time() - start_time) * 1000

        content = response.text
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

        # --- Langfuse tracing (SDK v4) ---
        if langfuse is not None:
            try:
                # start_observation with as_type="generation" is the v4 API.
                # .trace() no longer exists in Langfuse SDK v4.
                usage_details = {}
                if prompt_tokens is not None:
                    usage_details["input"] = prompt_tokens
                if completion_tokens is not None:
                    usage_details["output"] = completion_tokens
                if total_tokens is not None:
                    usage_details["total"] = total_tokens

                gen = langfuse.start_observation(
                    name="gemini-generate-content",
                    as_type="generation",
                    model=model_name,
                    input=prompt,
                    metadata={
                        "provider": "gemini",
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                gen.update(
                    output=content,
                    usage_details=usage_details if usage_details else None,
                )
                gen.end()
                langfuse.flush()
            except Exception as trace_exc:
                logger.warning(f"Langfuse tracing failed (Gemini): {trace_exc}")

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
