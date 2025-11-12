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
    """
    Central dependency container for RAG system.
    Initializes and manages: index builder, RAG pipeline, ACL checker, and LLM client.
    """

    # Configuration file paths
    MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
    ACL_RULES_PATH = Path("configs/acl_rules.yaml")
    DEFAULT_INDEX_DIR = "indexes/store"
    KNOWLEDGE_GRAPH_FILE = "kg_data.json"

    def __init__(self):
        # Core components
        self.index_builder: Optional["IndexBuilder"] = None
        self.rag_pipeline: Optional["RAGPipeline"] = None
        self.acl_checker: Optional["ACLChecker"] = None
        self.llm_client: Optional["LLMClient"] = None
        
        # Trace tracking for debugging
        self.latest_trace: Optional[dict] = None

        self._initialize_all_components()

    def _initialize_all_components(self):
        """Initialize all system components in correct order."""
        # Step 1: Load configuration files
        model_config = self._load_model_config(self.MODEL_CONFIG_PATH)
        acl_rules = self._load_acl_rules(self.ACL_RULES_PATH)

        # Step 2: Get environment variables
        index_dir = Path(os.getenv("INDEX_STORE_DIR", self.DEFAULT_INDEX_DIR))

        # Step 3: Initialize storage layer
        self.index_builder = self._initialize_index_builder(index_dir)
        self.acl_checker = self._initialize_acl_checker(self.ACL_RULES_PATH)

        # Step 4: Initialize LLM layer
        self.llm_client = self._initialize_llm_client(model_config)

        # Step 5: Initialize RAG pipeline (uses all above)
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
        return IndexBuilder(index_dir)

    def _initialize_acl_checker(self, acl_file: Path) -> "ACLChecker":
        """Initialize ACL checker."""
        from app.acl import ACLChecker

        return ACLChecker(acl_file)

    def _initialize_llm_client(
        self, model_config: dict
    ) -> Optional["LLMClient"]:
        """Initialize LLM client with Ollama configuration."""
        from app.llm_client import LLMClient

        try:
            llm_config = model_config.get("model", {})
            model_name = llm_config.get("model_name", "qwen2.5:1.5b")
            base_url = llm_config.get("base_url", "http://localhost:11434")
            
            llm_client = LLMClient(
                base_url=base_url,
                model=model_name,
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 2048),
            )
            logger.info(f"LLM client initialized: {model_name}")
            return llm_client
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            return None

    def _initialize_rag_pipeline(
        self, index_dir: Path, model_config: dict, acl_rules: dict
    ) -> "RAGPipeline":
        """Initialize RAG pipeline with all dependencies."""
        from rag_pipeline.pipeline import RAGPipeline

        # Load knowledge graph if available
        kg_data = self._load_knowledge_graph(index_dir)

        # Extract embedding configuration
        embedding_config = model_config.get("embedding", {})
        embedding_model_name = embedding_config.get(
            "model_name", "nomic-embed-text"
        )
        embedding_base_url = embedding_config.get(
            "base_url", "http://localhost:11434"
        )

        # Create and return pipeline
        pipeline = RAGPipeline(
            index_builder=self.index_builder,
            kg_data=kg_data,
            llm_client=self.llm_client,
            acl_rules=acl_rules,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
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
        return (
            self.index_builder is not None
            and self.index_builder.load_index_exists()
            and self.rag_pipeline is not None
        )

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
