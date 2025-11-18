import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
    ):
        self.client = None

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            model_config = config.get("llm", {})
            self.base_url = model_config.get("base_url")
            self.model = model_config.get("model_name")
            self.temperature = model_config.get("temperature")
            self.max_tokens = model_config.get("max_tokens")

            from ollama import Client
            self.client = Client(host=self.base_url)
            logger.info(f"LLM client initialized successfully with model: {self.model}")
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