import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        try:
            from ollama import Client
            self.client = Client(host=base_url)
            logger.info(f"LLM client initialized successfully with model: {model}")
        except ImportError:
            logger.error("ollama not installed; LLM functionality will be disabled. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM client: {e}")


    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        if not self.client:
            logger.error("Ollama LLM client not available; cannot generate response")
            raise RuntimeError("Ollama LLM client not available")

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": temp,
                    "num_predict": tokens,
                },
            )
            return response['response']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM generation failed: {e}")