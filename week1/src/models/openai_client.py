import os
import time
import openai
from typing import Dict, Any
from src.utils.logger import get_logger
from src.utils.langfuse_client import get_langfuse_client

logger = get_logger(__name__)


def generate_openai(model_name: str, temperature: float, max_tokens: int, prompt: str) -> Dict[str, Any]:
    """Generates text using OpenAI and returns response content and token usage."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables.")

    client = openai.OpenAI(api_key=api_key)

    langfuse = get_langfuse_client()

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        # --- Langfuse tracing ---
        if langfuse is not None:
            try:
                trace = langfuse.trace(
                    name="openai-generation",
                    metadata={"provider": "openai"},
                )
                trace.generation(
                    name="openai-chat-completion",
                    model=model_name,
                    input=prompt,
                    output=content,
                    metadata={
                        "provider": "openai",
                        "latency_ms": round(latency_ms, 2),
                    },
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": total_tokens,
                    },
                )
                langfuse.flush()
            except Exception as trace_exc:
                logger.warning(f"Langfuse tracing failed (OpenAI): {trace_exc}")

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return {
            "content": f"Error: {str(e)}",
            "prompt_tokens": None,
            "completion_tokens": None,
        }
