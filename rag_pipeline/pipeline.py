import logging
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field

from rag_pipeline.agents.planner import QueryPlanner, SubQuery
from rag_pipeline.agents.retriever import HybridRetriever
from rag_pipeline.agents.synthesizer import SynthesizerAgent

logger = logging.getLogger(__name__)


@dataclass
class QueryPlanDecision:
    """Planner's decision on how to handle a query."""

    strategy: str  # "vector", "graph", or "hybrid"
    filters: dict  # Filters to apply during retrieval
    expected_sources: str  # Description of where results will come from
    sub_queries: list = field(default_factory=list)  # Sub-queries if decomposed


@dataclass
class AgentInteraction:
    """Record of interaction between two agents."""

    from_agent: str  # "planner", "retriever", or "synthesizer"
    to_agent: str
    interaction_type: str  # "plan_decision", "retrieval_request", "chunk_delivery", "synthesis_request"
    details: dict
    timestamp_ms: float


@dataclass
class AgentTrace:
    """Complete execution trace for a query."""

    request_id: str
    query: str
    planner_decision: dict
    retriever_stats: dict
    synthesizer_stats: dict
    total_time_ms: int
    agent_interactions: list[AgentInteraction] = field(default_factory=list)
    sub_query_results: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.
    Coordinates: planning, retrieval, and synthesis for query processing.
    
    AGENT INTERACTION FLOW:
    
    1. PLANNER → PIPELINE
       Output: QueryPlan (strategy, filters, sub_queries)
    
    2. RETRIEVER ← PIPELINE (receives plan from planner)
       Input: Query + Plan (strategy, filters, sub_queries)
       Output: Retrieved chunks
    
    3. SYNTHESIZER ← PIPELINE (receives chunks from retriever)
       Input: Query + Retrieved chunks
       Output: Answer with inline citations + hallucination score
    
    4. PIPELINE
       Coordinates all agent outputs
       Builds complete trace
    """

    # Default configuration
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

    def __init__(
        self,
        index_builder: "IndexBuilder",
        kg_data: Optional[dict] = None,
        llm_client: Optional["LLMClient"] = None,
        acl_rules: Optional[dict] = None,
        embedding_model_name: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        """Initialize pipeline with all required components."""
        # Agent initialization
        self.planner = QueryPlanner()
        self.retriever = HybridRetriever(index_builder, kg_data, acl_rules)
        self.synthesizer = SynthesizerAgent(llm_client)

        # Store reference to index builder
        self.index_builder = index_builder
        self.llm_client = llm_client

        # Configure retriever with embedding model name and base URL
        embedding_model = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL
        embedding_url = embedding_base_url or "http://localhost:11434"
        self.retriever.set_embedding_model_name(embedding_model)
        self.retriever.set_embedding_base_url(embedding_url)

        logger.debug(
            "RAG pipeline initialized with "
            f"planner, retriever, synthesizer (agentic architecture)"
        )

    def process_query(
        self,
        request_id: str,
        query: str,
        user_role: str = "Admin",
        retrieval_depth: int = 10,
    ) -> tuple["SynthesisResult", AgentTrace]:
        """
        Process a single query end-to-end using agentic RAG flow.
        
        AGENT INTERACTION:
        
        Step 1: PLANNER AGENT
          - Analyzes query complexity
          - Breaks complex queries into sub-queries
          - Determines retrieval strategy (vector/graph/hybrid)
          - Extracts metadata filters
          → Output: QueryPlan
        
        Step 2: RETRIEVER AGENT
          - Receives plan from planner
          - Executes retrieval based on strategy
          - Handles both main query and sub-queries
          - Applies filters and ACL
          → Output: Retrieved chunks
        
        Step 3: SYNTHESIZER AGENT
          - Receives retrieved chunks from retriever
          - Generates answer with INLINE CITATIONS
          - Validates every claim against chunks
          - Computes hallucination score
          → Output: Cited answer with metrics
        
        Step 4: ORCHESTRATION
          - Build complete trace with all agent interactions
          - Track metrics and timings

        Returns:
            Tuple of (synthesis_result, execution_trace)
        """
        start_time = time.time()
        logger.info(f"[{request_id}] Starting agentic RAG: {query[:100]}")

        interactions = []

        # ═════════════════════════════════════════════════════════════
        # STEP 1: PLANNER AGENT - Analyze query and create plan
        # ═════════════════════════════════════════════════════════════
        planner_start = time.time()
        plan = self.planner.plan_query(query, user_role)
        planner_time = time.time() - planner_start

        interactions.append(
            AgentInteraction(
                from_agent="planner",
                to_agent="pipeline",
                interaction_type="plan_decision",
                details={
                    "strategy": plan.strategy,
                    "filters": plan.filters,
                    "sub_queries_count": len(plan.sub_queries) if plan.sub_queries else 0,
                },
                timestamp_ms=planner_time * 1000,
            )
        )

        logger.info(
            f"[{request_id}] AGENT: Planner decided "
            f"strategy={plan.strategy}, filters={plan.filters}"
        )

        if plan.sub_queries:
            logger.info(
                f"[{request_id}] AGENT: Planner decomposed into "
                f"{len(plan.sub_queries)} sub-queries"
            )

        # ═════════════════════════════════════════════════════════════
        # STEP 2: RETRIEVER AGENT - Execute retrieval plan
        # ═════════════════════════════════════════════════════════════
        retriever_start = time.time()

        # Process main query + any sub-queries
        all_chunks = []
        sub_query_results = {}

        # Main query retrieval
        main_chunks = self._retrieve_for_query(
            query=query,
            plan=plan,
            k=retrieval_depth,
            acl_role=user_role,
            request_id=request_id,
        )
        all_chunks.extend(main_chunks)

        logger.info(
            f"[{request_id}] AGENT: Retriever returned "
            f"{len(main_chunks)} chunks for main query"
        )

        # Sub-query retrievals (if decomposed)
        if plan.sub_queries:
            for sub_query in plan.sub_queries:
                sub_chunks = self._retrieve_for_query(
                    query=sub_query.text,
                    plan=plan,
                    k=max(5, retrieval_depth // len(plan.sub_queries)),
                    acl_role=user_role,
                    request_id=request_id,
                    sub_query_id=sub_query.sub_query_id,
                )
                all_chunks.extend(sub_chunks)
                sub_query_results[sub_query.sub_query_id] = {
                    "query": sub_query.text,
                    "chunks_count": len(sub_chunks),
                    "strategy": sub_query.strategy,
                }

                logger.info(
                    f"[{request_id}] AGENT: Retriever returned "
                    f"{len(sub_chunks)} chunks for {sub_query.sub_query_id}"
                )

        # Deduplicate chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)
        retriever_time = time.time() - retriever_start

        interactions.append(
            AgentInteraction(
                from_agent="retriever",
                to_agent="synthesizer",
                interaction_type="chunk_delivery",
                details={
                    "total_chunks": len(unique_chunks),
                    "unique_chunks": len(unique_chunks),
                    "sources": list(set(c.source for c in unique_chunks)),
                    "sub_queries_processed": len(plan.sub_queries) if plan.sub_queries else 0,
                },
                timestamp_ms=retriever_time * 1000,
            )
        )

        # ═════════════════════════════════════════════════════════════
        # STEP 3: SYNTHESIZER AGENT - Generate answer with citations
        # ═════════════════════════════════════════════════════════════
        synthesizer_start = time.time()

        synthesis_result = self.synthesizer.synthesize(
            query=query,
            retrieved_chunks=unique_chunks,
            synthesis_time_ms=int((time.time() - synthesizer_start) * 1000),
        )

        synthesizer_time = time.time() - synthesizer_start

        interactions.append(
            AgentInteraction(
                from_agent="synthesizer",
                to_agent="pipeline",
                interaction_type="answer_generation",
                details={
                    "answer_length": len(synthesis_result.answer),
                    "citations_count": len(synthesis_result.citations),
                    "hallucination_score": synthesis_result.hallucination_score,
                    "justified_sentences": synthesis_result.justified_sentences,
                    "total_sentences": synthesis_result.total_sentences,
                },
                timestamp_ms=synthesizer_time * 1000,
            )
        )

        logger.info(
            f"[{request_id}] AGENT: Synthesizer generated answer "
            f"with {len(synthesis_result.citations)} citations, "
            f"hallucination_score={synthesis_result.hallucination_score:.2f}"
        )

        # ═════════════════════════════════════════════════════════════
        # STEP 4: BUILD EXECUTION TRACE
        # ═════════════════════════════════════════════════════════════
        total_time = time.time() - start_time
        trace = self._create_trace(
            request_id=request_id,
            query=query,
            plan=plan,
            retrieved_chunks=unique_chunks,
            synthesis_result=synthesis_result,
            timing={
                "planner_ms": int(planner_time * 1000),
                "retriever_ms": int(retriever_time * 1000),
                "synthesizer_ms": int(synthesizer_time * 1000),
                "total_ms": int(total_time * 1000),
            },
            interactions=interactions,
            sub_query_results=sub_query_results,
        )

        logger.info(
            f"[{request_id}] Agentic RAG pipeline completed in {total_time:.2f}s"
        )

        return synthesis_result, trace

    def _retrieve_for_query(
        self,
        query: str,
        plan,
        k: int,
        acl_role: str,
        request_id: str,
        sub_query_id: Optional[str] = None,
    ) -> list:
        """Execute retrieval for a query (main or sub-query)."""
        query_label = f"{sub_query_id}" if sub_query_id else "main"

        logger.debug(
            f"[{request_id}] AGENT: Retriever executing {query_label} query: {query[:80]}"
        )

        chunks = self.retriever.hybrid_retrieve(
            query=query,
            filters=plan.filters,
            k=k,
            acl_role=acl_role,
        )

        return chunks

    def _deduplicate_chunks(self, chunks: list) -> list:
        """Remove duplicate chunks by chunk_id."""
        seen = set()
        unique = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)
        return unique

    def _create_trace(
        self,
        request_id: str,
        query: str,
        plan,
        retrieved_chunks: list,
        synthesis_result: "SynthesisResult",
        timing: dict,
        interactions: list,
        sub_query_results: dict,
    ) -> AgentTrace:
        """Create detailed execution trace with all agent interactions."""
        return AgentTrace(
            request_id=request_id,
            query=query,
            planner_decision={
                "strategy": plan.strategy,
                "filters": plan.filters,
                "expected_sources": plan.expected_sources,
                "sub_queries": len(plan.sub_queries) if plan.sub_queries else 0,
            },
            retriever_stats={
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_time_ms": timing["retriever_ms"],
                "sources": list(set(c.source for c in retrieved_chunks)),
            },
            synthesizer_stats={
                "citations_count": len(synthesis_result.citations),
                "hallucination_score": synthesis_result.hallucination_score,
                "justified_sentences": synthesis_result.justified_sentences,
                "total_sentences": synthesis_result.total_sentences,
                "synthesis_time_ms": timing["synthesizer_ms"],
            },
            total_time_ms=timing["total_ms"],
            agent_interactions=interactions,
            sub_query_results=sub_query_results,
        )
