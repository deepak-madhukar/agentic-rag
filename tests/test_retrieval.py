import pytest
from pathlib import Path
from ingestion_pipeline.index_builder import IndexBuilder
from ingestion_pipeline.chunker import Chunk
from rag_pipeline.agents.retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="doc1:chunk_0",
            content="ProductA has excellent performance metrics",
            document_id="doc1",
            document_type="INTERNAL",
            product="ProductA",
            date="2024-01-01",
            owner="engineering",
            category="performance",
            section="Features",
            subsection="General",
            start_idx=0,
            end_idx=50,
        ),
        Chunk(
            chunk_id="doc2:chunk_0",
            content="The deployment process is straightforward",
            document_id="doc2",
            document_type="PUBLIC",
            product="ProductB",
            date="2024-01-02",
            owner="devops",
            category="operations",
            section="Deployment",
            subsection="General",
            start_idx=0,
            end_idx=45,
        ),
    ]


def test_hybrid_retriever_initialization(tmp_path, sample_chunks):
    index_builder = IndexBuilder(tmp_path)
    retriever = HybridRetriever(index_builder)

    assert retriever.rrf_k == 60
    assert retriever.hybrid_alpha == 0.6


def test_rrf_fusion(tmp_path, sample_chunks):
    index_builder = IndexBuilder(tmp_path)
    retriever = HybridRetriever(index_builder)

    from rag_pipeline.agents.retriever import RetrievedResult

    vector_results = [
        RetrievedResult(
            chunk_id=sample_chunks[0].chunk_id,
            content=sample_chunks[0].content,
            score=0.9,
            metadata={},
            source="vector",
        )
    ]

    kg_results = [
        RetrievedResult(
            chunk_id=sample_chunks[1].chunk_id,
            content=sample_chunks[1].content,
            score=0.7,
            metadata={},
            source="graph",
        )
    ]

    fused = retriever._rrf_fusion(vector_results, kg_results)
    assert len(fused) == 2
    assert fused[0].score > 0
    assert fused[1].score > 0


def test_acl_filter(tmp_path, sample_chunks):
    index_builder = IndexBuilder(tmp_path)
    retriever = HybridRetriever(index_builder)

    from rag_pipeline.agents.retriever import RetrievedResult

    results = [
        RetrievedResult(
            chunk_id=sample_chunks[0].chunk_id,
            content=sample_chunks[0].content,
            score=0.9,
            metadata={"document_type": "INTERNAL"},
            source="vector",
        ),
        RetrievedResult(
            chunk_id=sample_chunks[1].chunk_id,
            content=sample_chunks[1].content,
            score=0.8,
            metadata={"document_type": "PUBLIC"},
            source="vector",
        ),
    ]

    filtered = retriever._apply_acl_filter(results, "Analyst")
    assert len(filtered) == 2

    filtered = retriever._apply_acl_filter(results, "Contractor")
    assert len(filtered) == 1
    assert filtered[0].metadata["document_type"] == "PUBLIC"
