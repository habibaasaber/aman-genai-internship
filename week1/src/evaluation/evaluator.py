import os
import time
import json
import yaml
import pandas as pd
from typing import List, Dict, Any

from src.models.openai_client import OpenAIClient
from src.models.gemini_client import GeminiClient
from src.utils.logger import get_logger
from src.utils.cache import cache
from src.evaluation.metrics import count_tokens_fallback, calculate_cost, validate_json_output

logger = get_logger(__name__)

class Evaluator:
    """
    Orchestrates the evaluation of multiple models against multiple prompts and inputs.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.clients = self._initialize_clients()
        
    def _initialize_clients(self) -> Dict[str, Any]:
        """Initializes all LLM clients specified in config."""
        clients = {}
        for model_key, model_config in self.config["models"].items():
            provider = model_config["provider"]
            model_name = model_config["model_name"]
            temp = model_config["temperature"]
            max_tokens = model_config["max_tokens"]
            
            if provider == "openai":
                clients[model_key] = OpenAIClient(model_name, temp, max_tokens)
            elif provider == "gemini":
                clients[model_key] = GeminiClient(model_name, temp, max_tokens)
            else:
                logger.warning(f"Unknown provider {provider} for model {model_key}")
                
        return clients

    def load_prompt_template(self, task: str, strategy: str) -> str:
        """Loads a prompt template from disk."""
        path = os.path.join("prompts", task, f"{strategy}.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {path}")
            return "{input_text}"

    def run_evaluation(self, tasks: List[str], strategies: List[str], inputs: Dict[str, List[str]]) -> pd.DataFrame:
        """Runs the entire stress-test matrix."""
        results = []
        
        for task in tasks:
            logger.info(f"--- Starting evaluation for {task} ---")
            task_inputs = inputs.get(task, [])
            
            for strategy in strategies:
                prompt_template = self.load_prompt_template(task, strategy)
                
                for input_idx, input_text in enumerate(task_inputs):
                    formatted_prompt = prompt_template.format(input_text=input_text)
                    
                    for model_key, client in self.clients.items():
                        model_config = self.config["models"][model_key]
                        
                        logger.info(f"Running: {task} | {strategy} | Input {input_idx+1} | {model_key}")
                        
                        # Cache key generation
                        cache_kwargs = {
                            "model_name": client.model_name,
                            "temperature": client.temperature,
                            "prompt": formatted_prompt
                        }
                        
                        cached_response = cache.get(**cache_kwargs)
                        
                        if cached_response:
                            response_data = cached_response["response_data"]
                            latency = cached_response["latency"]
                        else:
                            start_time = time.perf_counter()
                            response_data = client.generate(formatted_prompt)
                            latency = time.perf_counter() - start_time
                            
                            cache.set({
                                "response_data": response_data, 
                                "latency": latency
                            }, **cache_kwargs)
                        
                        # Metric calculations
                        content = response_data.get("content", "")
                        p_tokens = response_data.get("prompt_tokens")
                        c_tokens = response_data.get("completion_tokens")
                        
                        if p_tokens is None:
                            p_tokens = count_tokens_fallback(formatted_prompt)
                        if c_tokens is None:
                            c_tokens = count_tokens_fallback(content)
                            
                        total_tokens = p_tokens + c_tokens
                        cost = calculate_cost(
                            p_tokens, c_tokens,
                            model_config["cost_per_1m_prompt_tokens"],
                            model_config["cost_per_1m_completion_tokens"]
                        )
                        
                        # JSON validation is highly relevant for task1
                        is_valid_json = validate_json_output(content) if task == "task1" else None
                        
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
                            "Response": content
                        })
                        
        return pd.DataFrame(results)
