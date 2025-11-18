import logging
import os
from pathlib import Path
from typing import Optional
import json
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class Dependencies:
    """Central dependency container for RAG system."""

    MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
    ACL_RULES_PATH = Path("configs/acl_rules.yaml")
    DEFAULT_INDEX_DIR = "indexes/store"
    KNOWLEDGE_GRAPH_FILE = "kg_data.json"

    def __init__(self):
        self.index_builder: Optional["IndexBuilder"] = None
        self.rag_pipeline: Optional["RAGPipeline"] = None
        self.acl_checker: Optional["ACLChecker"] = None
        self.llm_client: Optional["LLMClient"] = None
        self.embedding_client: Optional["EmbeddingClient"] = None
        
        self.latest_trace: Optional[dict] = None

        self._initialize_all_components()

    def _initialize_all_components(self):
        """Initialize all system components in correct order."""
        model_config = self._load_model_config(self.MODEL_CONFIG_PATH)
        acl_rules = self._load_acl_rules(self.ACL_RULES_PATH)

        index_dir = Path(os.getenv("INDEX_STORE_DIR", self.DEFAULT_INDEX_DIR))

        self.llm_client = self._initialize_llm_client(model_config)
        self.embedding_client = self._initialize_embedding_client(model_config)

        self.index_builder = self._initialize_index_builder(index_dir)
        self.acl_checker = self._initialize_acl_checker(self.ACL_RULES_PATH)

        if self.index_builder:
            self.rag_pipeline = self._initialize_rag_pipeline(
                index_dir, model_config, acl_rules
            )

    def _initialize_index_builder(self, index_dir: Path) -> Optional["IndexBuilder"]:
        """Initialize index builder if index directory exists."""
        from ingestion_pipeline.index_builder import IndexBuilder

        if not index_dir.exists():
            logger.warning(f"Index directory not found: {index_dir}")
            return None

        logger.debug(f"Index builder initialized with directory: {index_dir}")
        return IndexBuilder(index_dir, embedding_client=self.embedding_client)

    def _initialize_acl_checker(self, acl_file: Path) -> "ACLChecker":
        """Initialize ACL checker."""
        from app.service.acl import ACLChecker

        return ACLChecker(acl_file)

    def _initialize_llm_client(
        self, model_config: dict
    ) -> Optional["LLMClient"]:
        """Initialize LLM client with Ollama configuration."""
        from utils.llm_client import LLMClient

        return LLMClient(config_path="configs/model_config.yaml")

    def _initialize_embedding_client(
        self, model_config: dict
    ) -> Optional["EmbeddingClient"]:
        """Initialize Embedding client with Ollama configuration."""
        from utils.embedding_client import EmbeddingClient

        return EmbeddingClient(config_path="configs/model_config.yaml")

    def _initialize_rag_pipeline(
        self, index_dir: Path, model_config: dict, acl_rules: dict
    ) -> "RAGPipeline":
        """Initialize RAG pipeline with all dependencies."""
        from rag_pipeline.pipeline import RAGPipeline

        # Load knowledge graph if available
        kg_data = self._load_knowledge_graph(index_dir)

        # Create and return pipeline
        pipeline = RAGPipeline(
            llm_client=self.llm_client,
            embedding_client=self.embedding_client,
            acl_rules=acl_rules,
            index_builder=self.index_builder,
            kg_data=kg_data,
        )
        logger.debug("RAG pipeline initialized")
        return pipeline

    def _load_knowledge_graph(self, index_dir: Path) -> Optional[dict]:
        """Load knowledge graph data if available."""
        kg_file = index_dir / self.KNOWLEDGE_GRAPH_FILE

        if not kg_file.exists():
            return None

        try:
            with open(kg_file, "r") as f:
                kg_data = json.load(f)
            logger.debug(f"Knowledge graph loaded: {kg_file}")
            return kg_data
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
            return None

    def _load_model_config(self, config_file: Path) -> dict:
        """Load model configuration from YAML file."""
        if not config_file.exists():
            logger.warning(f"Model config file not found: {config_file}")
            return {}

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            logger.debug(f"Model config loaded from: {config_file}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to parse model config {config_file}: {e}", exc_info=True)
            return {}

    def _load_acl_rules(self, acl_file: Path) -> dict:
        """Load ACL rules from YAML file."""
        if not acl_file.exists():
            logger.warning(f"ACL rules file not found: {acl_file}")
            return {}

        try:
            with open(acl_file, "r") as f:
                rules = yaml.safe_load(f)
            logger.debug(f"ACL rules loaded from: {acl_file}")
            return rules or {}
        except Exception as e:
            logger.error(f"Failed to parse ACL rules {acl_file}: {e}", exc_info=True)
            return {}

    def is_ready(self) -> bool:
        """Check if system has loaded all required components."""
        logger.info(f"RAG Pipeline ready: {self.rag_pipeline is not None}")
        logger.info(f"LLM Client ready: {self.llm_client is not None}")
        logger.info(f"Embedding Client ready: {self.embedding_client is not None}")
        return self.rag_pipeline is not None and self.llm_client is not None and self.embedding_client is not None

    def store_trace(self, trace: dict) -> None:
        """Store query trace for debugging and monitoring."""
        self.latest_trace = trace

    def get_latest_trace(self) -> Optional[dict]:
        """Retrieve latest query trace."""
        return self.latest_trace


_dependencies: Optional[Dependencies] = None


def get_dependencies() -> Dependencies:
    global _dependencies
    if _dependencies is None:
        _dependencies = Dependencies()
    return _dependencies