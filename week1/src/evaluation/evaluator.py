import json
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.evaluation.metrics import calculate_cost, count_tokens_fallback, validate_json_output
from src.models.gemini_client import generate_gemini
from src.models.openai_client import generate_openai
from src.utils.cache import ResponseCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Orchestrates the evaluation of multiple models against multiple prompts and inputs."""

    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[2] / config_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_root = config_path.parent.parent
        self.cache_dir = self.project_root / self.config.get("execution", {}).get("cache_dir", ".cache")
        self.cache = ResponseCache(str(self.cache_dir))

    def load_prompt_template(self, task: str, strategy: str) -> str:
        """Loads a prompt template from disk."""
        path = self.project_root / "prompts" / task / f"{strategy}.txt"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {path}")
            return "{input_text}"

    def _score_output(self, task: str, content: str) -> float:
        """Assigns a simple heuristic accuracy score for each task."""
        if task == "task1":
            return self._score_task1(content)
        if task == "task2":
            return self._score_task2(content)
        if task == "task3":
            return self._score_task3(content)
        return 0.0

    def _score_task1(self, content: str) -> float:
        if not validate_json_output(content):
            return 0.0

        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return 0.0

        keys = set(payload.keys()) if isinstance(payload, dict) else set()
        expected = {"error_type", "root_cause", "affected_component", "severity"}
        if not expected.issubset(keys):
            return 0.6

        severity = str(payload.get("severity", "")).upper()
        if severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            return 1.0
        return 0.8

    def _score_task2(self, content: str) -> float:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return 0.0

        if not isinstance(payload, list) or not payload:
            return 0.0

        if not all(isinstance(item, dict) and {"entity", "type", "speaker"}.issubset(item.keys()) for item in payload):
            return 0.4

        score = 0.8
        entities = [str(item.get("entity", "")) for item in payload]
        if any(re.search(r"\d{11}", entity) for entity in entities):
            score += 0.05
        if any(re.search(r"\d{14}", entity) for entity in entities):
            score += 0.05
        if any(re.search(r"[أ-ي]{2,}", entity) for entity in entities):
            score += 0.05
        return round(min(score, 1.0), 3)

    def _score_task3(self, content: str) -> float:
        text = (content or "").strip()
        if not text:
            return 0.0

        lower_text = text.lower()
        apology = any(token in lower_text for token in ["sorry", "apologize", "apology", "أسف", "آسف", "نأسف"])
        mentions_issue = any(token in lower_text for token in ["billing", "bill", "internet", "sim", "delay", "charged", "issue", "problem", "فاتورة", "إنترنت", "شريحة", "تأخير", "مشكلة"])
        next_step = any(token in lower_text for token in ["next step", "will", "we will", "contact", "follow up", "resolve", "sincerely", "سوف", "سن", "نتابع", "نواصل", "نرسل"])
        sentences = [segment for segment in re.split(r"[.!?]+", text) if segment.strip()]
        sentence_count = len(sentences)

        if apology and mentions_issue and next_step and 3 <= sentence_count <= 4:
            return 1.0
        if apology and mentions_issue and next_step:
            return 0.7
        return 0.4

    def run_evaluation(self, tasks: List[str], strategies: List[str], inputs: Dict[str, List[str]]) -> pd.DataFrame:
        """Runs the entire stress-test matrix."""
        results = []

        for task in tasks:
            logger.info(f"--- Starting evaluation for {task} ---")
            task_inputs = inputs.get(task, [])

            for strategy in strategies:
                prompt_template = self.load_prompt_template(task, strategy)

                for input_idx, input_text in enumerate(task_inputs):
                    formatted_prompt = prompt_template.replace("{input_text}", input_text)

                    for model_key, model_config in self.config["models"].items():
                        provider = model_config["provider"]
                        model_name = model_config["model_name"]
                        temp = model_config["temperature"]
                        max_tokens = model_config["max_tokens"]

                        logger.info(f"Running: {task} | {strategy} | Input {input_idx + 1} | {model_key}")

                        cache_kwargs = {
                            "model_name": model_name,
                            "temperature": temp,
                            "prompt": formatted_prompt,
                        }

                        cached_response = self.cache.get(**cache_kwargs)

                        if cached_response:
                            response_data = cached_response["response_data"]
                            latency = cached_response["latency"]
                        else:
                            start_time = time.perf_counter()
                            if provider == "openai":
                                response_data = generate_openai(model_name, temp, max_tokens, formatted_prompt)
                            elif provider == "gemini":
                                response_data = generate_gemini(model_name, temp, max_tokens, formatted_prompt)
                            else:
                                logger.warning(f"Unknown provider {provider} for model {model_key}")
                                response_data = {
                                    "content": f"Error: Unknown provider {provider}",
                                    "prompt_tokens": None,
                                    "completion_tokens": None,
                                }
                            latency = time.perf_counter() - start_time

                            self.cache.set({
                                "response_data": response_data,
                                "latency": latency,
                            }, **cache_kwargs)

                        content = response_data.get("content", "")
                        p_tokens = response_data.get("prompt_tokens")
                        c_tokens = response_data.get("completion_tokens")

                        if p_tokens is None:
                            p_tokens = count_tokens_fallback(formatted_prompt)
                        if c_tokens is None:
                            c_tokens = count_tokens_fallback(content)

                        total_tokens = p_tokens + c_tokens
                        cost = calculate_cost(
                            p_tokens,
                            c_tokens,
                            model_config["cost_per_1m_prompt_tokens"],
                            model_config["cost_per_1m_completion_tokens"],
                        )

                        is_valid_json = validate_json_output(content) if task == "task1" else None
                        accuracy_score = self._score_output(task, content)

                        results.append({
                            "Task": task,
                            "Strategy": strategy,
                            "Input_Index": input_idx + 1,
                            "Model": model_key,
                            "Provider": model_config["provider"],
                            "Latency_sec": round(latency, 4),
                            "Prompt_Tokens": p_tokens,
                            "Completion_Tokens": c_tokens,
                            "Total_Tokens": total_tokens,
                            "Estimated_Cost_USD": round(cost, 6),
                            "Output_Length": len(content),
                            "Valid_JSON": is_valid_json,
                            "Accuracy_Score": round(accuracy_score, 3),
                            "Response": content,
                        })

        return pd.DataFrame(results)
