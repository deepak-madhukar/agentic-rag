import logging
from typing import Optional, List
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
    ):
        self.client = None

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            embedding_config = config.get("embedding", {})
            self.base_url = embedding_config.get("base_url")
            self.model_name = embedding_config.get("model_name")

            from ollama import Client

            self.client = Client(host=self.base_url)
            logger.info(
                f"Embedding client initialized successfully with model: {self.model_name}"
            )
        except ImportError:
            logger.error(
                "ollama not installed; embedding functionality will be disabled. Install with: pip install ollama"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding client: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        if not self.client:
            logger.error("Ollama embedding client not available; cannot get embedding")
            raise RuntimeError("Ollama embedding client not available")

        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text,
            )
            embedding = np.array(response["embedding"], dtype="float32")
            logger.debug(f"Generated embedding with dimension: {embedding.shape[0]}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {e}")