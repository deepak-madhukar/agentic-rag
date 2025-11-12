import json
import logging
import os
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedResult:
    chunk_id: str
    content: str
    score: float
    metadata: dict
    source: str


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and knowledge graph traversal.
    Fuses results using Reciprocal Rank Fusion (RRF).
    """

    # Fusion algorithm parameters
    RRF_CONSTANT = 60  # k parameter for RRF formula
    HYBRID_ALPHA = 0.6  # Weight for vector results in fusion

    def __init__(
        self,
        index_builder: "IndexBuilder",
        kg_data: Optional[dict] = None,
        acl_rules: Optional[dict] = None,
    ):
        """Initialize hybrid retriever."""
        self.index_builder = index_builder
        self.kg_data = kg_data or {}
        self.acl_rules = acl_rules or {}

        # Embedding configuration (set by pipeline)
        self.embedding_model_name: Optional[str] = None
        self.embedding_base_url: Optional[str] = None

    def set_embedding_model_name(self, model_name: str) -> None:
        """Configure the embedding model to use for vector retrieval."""
        self.embedding_model_name = model_name
        logger.debug(f"Embedding model configured: {model_name}")

    def set_embedding_base_url(self, base_url: str) -> None:
        """Configure the base URL for Ollama embeddings."""
        self.embedding_base_url = base_url
        logger.debug(f"Embedding base URL configured: {base_url}")

    def hybrid_retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        k: int = 10,
        acl_role: Optional[str] = None,
    ) -> list[RetrievedResult]:
        """
        Retrieve results using hybrid approach.

        Steps:
        1. Vector search
        2. Knowledge graph search
        3. RRF fusion
        4. Filter by metadata
        5. Filter by ACL rules
        """
        logger.debug(
            f"Hybrid retrieval: k={k}, filters={filters}, role={acl_role}"
        )

        # Step 1: Vector retrieval
        vector_results = self._retrieve_by_vector(query, k * 2)

        # Step 2: Graph retrieval
        graph_results = self._retrieve_by_graph(query, k * 2)

        # Step 3: Fuse results using RRF
        fused_results = self._fuse_results_with_rrf(vector_results, graph_results)

        # Step 4: Apply metadata filters
        if filters:
            fused_results = self._apply_metadata_filters(fused_results, filters)

        # Step 5: Apply access control filters
        if acl_role:
            fused_results = self._apply_acl_filters(fused_results, acl_role)

        # Return top k results
        final_results = fused_results[:k]
        logger.debug(f"Hybrid retrieval returning {len(final_results)} results")

        return final_results

    def _retrieve_by_vector(self, query: str, k: int) -> list[RetrievedResult]:
        """Retrieve documents using dense vector similarity."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            logger.error("FAISS not available; cannot perform vector retrieval")
            raise RuntimeError("FAISS required for vector retrieval not available")

        index_file = self.index_builder.faiss_index_file
        if not index_file.exists():
            logger.error(f"Vector index file not found: {index_file}")
            raise FileNotFoundError(f"Vector index file not found: {index_file}")

        index = faiss.read_index(str(index_file))
        embedding_dim = index.d

        try:
            from ollama import Client
        except ImportError:
            logger.error("ollama not available; cannot perform vector retrieval")
            raise ImportError("Ollama required but not installed")

        try:
            # Initialize Ollama client
            base_url = self.embedding_base_url or "http://localhost:11434"
            client = Client(host=base_url)
            
            # Generate query embedding
            response = client.embeddings(
                model=self.embedding_model_name or "nomic-embed-text",
                prompt=query,
            )
            query_embedding = np.array(response["embedding"]).astype("float32")

            # Validate dimension match
            if query_embedding.shape[0] != embedding_dim:
                logger.error(
                    f"Embedding dimension mismatch: "
                    f"query={query_embedding.shape[0]}, "
                    f"index={embedding_dim}. "
                    f"Please rebuild the index with the correct model."
                )
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"{query_embedding.shape[0]} != {embedding_dim}"
                )
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            raise RuntimeError(f"Query embedding generation failed: {e}")

        # Search in FAISS index
        distances, indices = index.search(query_embedding.reshape(1, -1), k)

        # Convert FAISS results to RetrievedResult objects
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_metadata = self.index_builder.get_chunk_by_id(f"chunk_{idx}")
            if chunk_metadata:
                score = 1.0 / (1.0 + distance)
                results.append(
                    RetrievedResult(
                        chunk_id=chunk_metadata["chunk_id"],
                        content=chunk_metadata["content"],
                        score=score,
                        metadata={
                            "document_id": chunk_metadata["document_id"],
                            "document_type": chunk_metadata["document_type"],
                            "product": chunk_metadata["product"],
                            "section": chunk_metadata["section"],
                        },
                        source="vector",
                    )
                )

        logger.debug(f"Vector retrieval returned {len(results)} results")
        return results

    def _retrieve_by_graph(self, query: str, k: int) -> list[RetrievedResult]:
        """Retrieve documents using knowledge graph traversal."""
        if not self.kg_data or "nodes" not in self.kg_data:
            return []

        results = []
        query_terms = query.lower().split()

        for node in self.kg_data.get("nodes", []):
            if node["type"] != "Chunk":
                continue

            label = node["label"].lower()
            matching_terms = sum(1 for term in query_terms if term in label)

            if matching_terms > 0:
                chunk_metadata = self.index_builder.get_chunk_by_id(node["label"])
                if chunk_metadata:
                    score = min(matching_terms / len(query_terms), 1.0)
                    results.append(
                        RetrievedResult(
                            chunk_id=node["label"],
                            content=chunk_metadata["content"],
                            score=score,
                            metadata={
                                "document_id": chunk_metadata["document_id"],
                                "document_type": chunk_metadata["document_type"],
                                "product": chunk_metadata["product"],
                                "section": chunk_metadata["section"],
                            },
                            source="graph",
                        )
                    )

        logger.debug(f"Graph retrieval returned {len(results)} results")
        return results

    def _fuse_results_with_rrf(
        self,
        vector_results: list[RetrievedResult],
        graph_results: list[RetrievedResult],
    ) -> list[RetrievedResult]:
        """Fuse vector and graph results using Reciprocal Rank Fusion."""
        combined = {}

        # Add vector results with RRF scoring
        for rank, result in enumerate(vector_results):
            rrf_score = 1.0 / (self.RRF_CONSTANT + rank + 1)
            if result.chunk_id not in combined:
                combined[result.chunk_id] = {
                    "result": result,
                    "vector_rrf": rrf_score,
                    "graph_rrf": 0.0,
                }
            else:
                combined[result.chunk_id]["vector_rrf"] = rrf_score

        # Add graph results with RRF scoring
        for rank, result in enumerate(graph_results):
            rrf_score = 1.0 / (self.RRF_CONSTANT + rank + 1)
            if result.chunk_id not in combined:
                combined[result.chunk_id] = {
                    "result": result,
                    "vector_rrf": 0.0,
                    "graph_rrf": rrf_score,
                }
            else:
                combined[result.chunk_id]["graph_rrf"] = rrf_score

        # Compute fused score with alpha weighting
        fused_list = []
        for chunk_id, data in combined.items():
            vector_rrf = data["vector_rrf"]
            graph_rrf = data["graph_rrf"]

            fused_score = (
                self.HYBRID_ALPHA * vector_rrf
                + (1 - self.HYBRID_ALPHA) * graph_rrf
            )

            result = data["result"]
            result.score = fused_score
            fused_list.append(result)

        # Sort by fused score descending
        fused_list.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"RRF fusion combined {len(vector_results)} vector + "
                     f"{len(graph_results)} graph results")
        return fused_list

    def _apply_metadata_filters(
        self,
        results: list[RetrievedResult],
        filters: dict,
    ) -> list[RetrievedResult]:
        """Filter results by metadata (product, document type, etc)."""
        filtered = results

        if "product" in filters:
            product = filters["product"]
            filtered = [
                r for r in filtered
                if r.metadata.get("product") == product
            ]
            logger.debug(f"Product filter '{product}': {len(results)} -> "
                        f"{len(filtered)} results")

        if "document_type" in filters:
            doc_type = filters["document_type"]
            filtered = [
                r for r in filtered
                if r.metadata.get("document_type") == doc_type
            ]
            logger.debug(f"Document type filter '{doc_type}': "
                        f"{len(filtered)} results")

        return filtered

    def _apply_acl_filters(
        self,
        results: list[RetrievedResult],
        user_role: str,
    ) -> list[RetrievedResult]:
        """Filter results based on user's ACL permissions."""
        if not self.acl_rules:
            logger.warning("No ACL rules configured; allowing all access")
            return results

        document_type_access = self.acl_rules.get("document_type_access", {})

        # Find document types this role can access
        accessible_types = []
        for doc_type, allowed_roles in document_type_access.items():
            if user_role in allowed_roles:
                accessible_types.append(doc_type)

        if not accessible_types:
            logger.warning(
                f"Role '{user_role}' has no document type access in ACL rules"
            )
            return []

        # Filter results to only accessible types
        filtered = [
            r for r in results
            if r.metadata.get("document_type") in accessible_types
        ]

        logger.debug(
            f"ACL filter for role '{user_role}': "
            f"{len(results)} -> {len(filtered)} results"
        )
        return filtered
