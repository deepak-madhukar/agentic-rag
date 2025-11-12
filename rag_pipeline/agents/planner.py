import logging
from typing import Optional
from dataclasses import dataclass
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A decomposed sub-query for retrieval."""

    sub_query_id: str
    text: str
    strategy: str  # "vector" or "graph"
    filters: dict
    priority: float  # 1.0 = highest priority


@dataclass
class QueryPlan:
    """Planner's decision on how to handle a query."""

    strategy: str  # "vector", "graph", or "hybrid"
    filters: dict  # Metadata filters to apply during retrieval
    expected_sources: str  # Human-readable description of sources
    sub_queries: Optional[list[SubQuery]] = None  # Decomposed sub-queries


class QueryPlanner:
    """
    Query planning agent.
    Analyzes queries to determine: retrieval strategy, metadata filters,
    and decompose complex queries into sub-queries.
    
    Capabilities:
    - Analyzes query complexity
    - Breaks complex queries into sub-queries
    - Determines optimal retrieval strategy
    - Extracts metadata filters
    """

    # Strategy selection keywords
    HYBRID_KEYWORDS = ["what", "describe", "explain", "summarize", "compare", "analyze"]
    GRAPH_KEYWORDS = ["who", "which team", "assigned", "related", "connected", "author"]
    VECTOR_KEYWORDS = ["when", "date", "recent", "latest", "current"]
    COMPLEX_KEYWORDS = ["and", "or", "both", "also", "additionally"]

    # Strategy to description mapping
    STRATEGY_DESCRIPTIONS = {
        "vector": "Vector Index (FAISS)",
        "graph": "Knowledge Graph",
        "hybrid": "Vector Index + Knowledge Graph (RRF)",
    }

    # Default document types if not configured
    DEFAULT_DOCUMENT_TYPES = ["INTERNAL", "PUBLIC", "CONFIDENTIAL", "TEAM"]

    def __init__(self, config_path: Optional[str] = None):
        """Initialize planner with optional config."""
        self.config = {}
        self.products = []
        self.valid_document_types = self.DEFAULT_DOCUMENT_TYPES

        if config_path:
            self._load_config(config_path)
        else:
            logger.debug("Planner using default configuration")

    def _load_config(self, config_path: str) -> None:
        """Load optional configuration from file."""
        config_file = Path(config_path)

        if not config_file.exists():
            logger.debug(f"Config file not found: {config_path}")
            return

        try:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f) or {}
            logger.debug(f"Planner config loaded from: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load planner config: {e}")

    def plan_query(
        self, query: str, user_role: str = "Admin"
    ) -> QueryPlan:
        """
        Analyze query and produce a retrieval plan.
        
        If query is complex (multiple topics), decompose into sub-queries.
        Otherwise, create a single retrieval plan.

        Returns:
            QueryPlan with strategy, filters, and optional sub-queries
        """
        strategy = self._determine_retrieval_strategy(query)
        filters = self._extract_metadata_filters(query)
        expected_sources = self._describe_sources(strategy)

        # Check if query needs decomposition
        is_complex = self._is_complex_query(query)
        sub_queries = None

        if is_complex:
            logger.debug(f"Complex query detected; decomposing into sub-queries")
            sub_queries = self._decompose_query(query, strategy, filters)

        logger.debug(
            f"Query plan: strategy={strategy}, "
            f"filters={filters}, complex={is_complex}, "
            f"sub_queries={len(sub_queries) if sub_queries else 0}"
        )

        return QueryPlan(
            strategy=strategy,
            filters=filters,
            expected_sources=expected_sources,
            sub_queries=sub_queries,
        )

    def _is_complex_query(self, query: str) -> bool:
        """Detect if query is complex (multiple topics/operations)."""
        query_lower = query.lower()
        
        # Check for multiple topics
        complex_indicators = [
            query_lower.count(" and ") > 1,
            " or " in query_lower and len(query_lower) > 80,
            query_lower.count(",") > 2,
            any(kw in query_lower for kw in self.COMPLEX_KEYWORDS) and len(query_lower) > 100,
        ]
        
        is_complex = sum(complex_indicators) >= 2
        logger.debug(f"Query complexity check: {is_complex} (indicators: {sum(complex_indicators)})")
        return is_complex

    def _decompose_query(
        self, 
        query: str, 
        parent_strategy: str,
        parent_filters: dict
    ) -> list[SubQuery]:
        """
        Decompose complex query into sub-queries.
        
        Example:
        Input: "What features does ProductA have and which team manages it?"
        Output: [
            SubQuery("1", "What features does ProductA have?", "vector", {...}),
            SubQuery("2", "Which team manages ProductA?", "graph", {...})
        ]
        """
        sub_queries = []
        
        # Split by "and" or "or"
        parts = query.lower().replace(" and ", "|").replace(" or ", "|").split("|")
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) <= 1:
            # No actual decomposition needed
            return sub_queries
        
        logger.debug(f"Decomposing query into {len(parts)} sub-queries")
        
        for idx, part in enumerate(parts, 1):
            sub_strategy = self._determine_retrieval_strategy(part)
            sub_filters = self._extract_metadata_filters(part)
            
            # Merge with parent filters (parent filters take precedence)
            merged_filters = {**sub_filters, **parent_filters}
            
            sub_query = SubQuery(
                sub_query_id=f"sub_{idx}",
                text=part,
                strategy=sub_strategy,
                filters=merged_filters,
                priority=1.0 / idx,  # First sub-query has highest priority
            )
            sub_queries.append(sub_query)
            
            logger.debug(
                f"Sub-query {idx}: '{part}' -> strategy={sub_strategy}"
            )
        
        return sub_queries

    def _determine_retrieval_strategy(self, query: str) -> str:
        """Determine retrieval strategy based on query keywords."""
        query_lower = query.lower()

        # Check for hybrid strategy indicators
        if any(keyword in query_lower for keyword in self.HYBRID_KEYWORDS):
            return "hybrid"

        # Check for graph strategy indicators
        if any(keyword in query_lower for keyword in self.GRAPH_KEYWORDS):
            return "graph"

        # Check for vector strategy indicators
        if any(keyword in query_lower for keyword in self.VECTOR_KEYWORDS):
            return "vector"

        # Default to hybrid
        return "hybrid"

    def _extract_metadata_filters(self, query: str) -> dict:
        """Extract metadata filters from query."""
        filters = {}
        query_lower = query.lower()

        # Extract product filter if configured
        if self.products:
            for product in self.products:
                if product.lower() in query_lower:
                    filters["product"] = product
                    break

        # Extract document type filter
        for doc_type in self.valid_document_types:
            if doc_type.lower() in query_lower:
                filters["document_type"] = doc_type
                break

        if filters:
            logger.debug(f"Extracted filters: {filters}")

        return filters

    def _describe_sources(self, strategy: str) -> str:
        """Get human-readable description of retrieval sources."""
        return self.STRATEGY_DESCRIPTIONS.get(strategy, "Unknown")
