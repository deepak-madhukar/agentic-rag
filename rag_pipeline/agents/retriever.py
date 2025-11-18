import json
import logging
from typing import Optional
from dataclasses import dataclass
from ingestion_pipeline.index_builder import IndexBuilder
from utils.embedding_client import EmbeddingClient

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
        embedding_client: "EmbeddingClient" = None,
    ):
        """Initialize hybrid retriever."""
        if embedding_client is None:
            raise ValueError("embedding_client is required and cannot be None")
        self.index_builder = index_builder
        self.kg_data = kg_data or {}
        self.acl_rules = acl_rules or {}
        self.embedding_client = embedding_client

    def hybrid_retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        k: int = 10,
        acl_role: Optional[str] = None,
    ) -> list[RetrievedResult]:
        """Retrieve results using hybrid vector and graph search with RRF fusion."""
        logger.debug(f"Hybrid retrieval: k={k}, filters={filters}, role={acl_role}")

        vector_results = self._retrieve_by_vector(query, k * 2, self.embedding_client)

        graph_results = self._retrieve_by_graph(query, k * 2)

        fused_results = self._fuse_results_with_rrf(vector_results, graph_results)

        if filters:
            fused_results = self._apply_metadata_filters(fused_results, filters)

        if acl_role:
            fused_results = self._apply_acl_filters(fused_results, acl_role)

        final_results = fused_results[:k]
        logger.debug(f"Hybrid retrieval returning {len(final_results)} results")

        return final_results

    def _retrieve_by_vector(self, query: str, k: int, embedding_client: "EmbeddingClient") -> list[RetrievedResult]:
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
            query_embedding = np.array(
                embedding_client.get_embedding(query)
            ).astype("float32")

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

        distances, indices = index.search(query_embedding.reshape(1, -1), k)

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
