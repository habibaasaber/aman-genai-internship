"""
inference_service.py
--------------------
Thin service layer that:
  1. Resolves the correct prompt template file.
  2. Loads it and injects {input_text}.
  3. Reads model config from config.yaml.
  4. Delegates to the existing OpenAI / Gemini client functions.
  5. Returns the plain response string.

No new business logic – reuses openai_client.generate_openai
and gemini_client.generate_gemini exactly as the evaluator does.
"""

import time
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv

from src.models.openai_client import generate_openai
from src.models.gemini_client import generate_gemini
from src.evaluation.metrics import calculate_cost, count_tokens_fallback, validate_json_output
from src.evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Canonical key maps (UI label → internal key)
# ---------------------------------------------------------------------------

TASK_MAP: dict[str, str] = {
    "task1": "task1",
    "task2": "task2",
    "task3": "task3",
}

STRATEGY_MAP: dict[str, str] = {
    "zero_shot": "zero_shot",
    "few_shot": "few_shot",
    "chain_of_thought": "chain_of_thought",
}

MODEL_MAP: dict[str, str] = {
    "gpt_4o_mini": "gpt_4o_mini",
    "gemini_2_0_flash": "gemini_2_0_flash",
}

# ---------------------------------------------------------------------------
# Project root (week1/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompt(task: str, strategy: str, input_text: str) -> str:
    """Read the prompt template and substitute {input_text}."""
    template_path = _PROJECT_ROOT / "prompts" / task / f"{strategy}.txt"
    template = template_path.read_text(encoding="utf-8")
    return template.replace("{input_text}", input_text)


def run_inference(
    model_key: str,
    task: str,
    strategy: str,
    input_text: str,
) -> dict:
    """
    Parameters
    ----------
    model_key  : one of 'gpt_4o_mini' | 'gemini_2_0_flash' (or other config keys)
    task       : one of 'task1' | 'task2' | 'task3'
    strategy   : one of 'zero_shot' | 'few_shot' | 'chain_of_thought'
    input_text : raw user input to inject into the prompt template

    Returns
    -------
    A dict containing response text and execution metrics.
    """
    # Ensure API keys are loaded from .env
    load_dotenv(_PROJECT_ROOT / ".env")

    config = _load_config()
    
    # Handle possible mismatch between UI selection and config.yaml key
    if model_key not in config["models"]:
        if model_key == "gemini_3.5_flash" and "gemini_2_0_flash" in config["models"]:
            model_key = "gemini_2_0_flash"
            
    model_cfg = config["models"][model_key]

    prompt = _load_prompt(task, strategy, input_text)

    provider = model_cfg["provider"]
    model_name = model_cfg["model_name"]
    temperature = model_cfg["temperature"]
    max_tokens = model_cfg["max_tokens"]

    start_time = time.perf_counter()
    if provider == "openai":
        result = generate_openai(model_name, temperature, max_tokens, prompt)
    elif provider == "gemini":
        result = generate_gemini(model_name, temperature, max_tokens, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    latency_sec = time.perf_counter() - start_time

    content = result.get("content", "")
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")

    if prompt_tokens is None:
        prompt_tokens = count_tokens_fallback(prompt)
    if completion_tokens is None:
        completion_tokens = count_tokens_fallback(content)

    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_cost(
        prompt_tokens,
        completion_tokens,
        model_cfg["cost_per_1m_prompt_tokens"],
        model_cfg["cost_per_1m_completion_tokens"],
    )

    # Re-use evaluator to score accuracy and check JSON validity
    evaluator = Evaluator()
    accuracy_score = evaluator._score_output(task, content)
    is_valid_json = validate_json_output(content) if task == "task1" else None

    return {
        "response": content,
        "latency_sec": round(latency_sec, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
        "accuracy_score": round(accuracy_score, 3),
        "valid_json": is_valid_json,
    }
