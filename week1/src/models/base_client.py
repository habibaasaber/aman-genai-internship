from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    Ensures a consistent interface across different API providers.
    """
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Given a prompt, generates a response from the LLM.
        
        Returns a dictionary containing:
        - "content": (str) The text response
        - "prompt_tokens": (int) Number of tokens in the prompt (optional if not provided by API)
        - "completion_tokens": (int) Number of tokens in the response (optional if not provided by API)
        """
        pass
