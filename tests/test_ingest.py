import pytest
import asyncio
from pathlib import Path
from ingestion_pipeline.loader import DocumentLoader
from ingestion_pipeline.chunker import SemanticChunker
from ingestion_pipeline.index_builder import IndexBuilder


@pytest.fixture
def sample_documents():
    from ingestion_pipeline.loader import Document, DocumentMetadata
    from datetime import datetime

    return [
        Document(
            metadata=DocumentMetadata(
                document_id="test_doc_1",
                document_type="INTERNAL",
                title="Test Document 1",
                product="ProductA",
                date=datetime.now().isoformat(),
                owner="test_owner",
                category="test",
                source_path="/test/path",
            ),
            content="This is a test document. It contains information about ProductA. The system is well designed.",
            doc_type="test",
        ),
    ]


@pytest.mark.asyncio
async def test_loader_pdfs(tmp_path):
    from utils.embedding_client import EmbeddingClient
    
    embedding_client = EmbeddingClient(config_path="configs/model_config.yaml")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    loader = DocumentLoader(
        pdf_dir=pdf_dir,
        html_dir=tmp_path / "html",
        json_dir=tmp_path / "json",
        email_dir=tmp_path / "emails",
        embedding_client=embedding_client,
    )

    documents = await loader.load_all()
    assert isinstance(documents, list)


@pytest.mark.asyncio
async def test_loader_html(tmp_path):
    from utils.embedding_client import EmbeddingClient
    
    embedding_client = EmbeddingClient(config_path="configs/model_config.yaml")
    html_dir = tmp_path / "html"
    html_dir.mkdir()

    html_file = html_dir / "test.html"
    html_file.write_text("<html><body>Test content</body></html>")

    loader = DocumentLoader(
        pdf_dir=tmp_path / "pdfs",
        html_dir=html_dir,
        json_dir=tmp_path / "json",
        email_dir=tmp_path / "emails",
        embedding_client=embedding_client,
    )

    documents = await loader.load_all()
    assert len(documents) == 1
    assert documents[0].doc_type == "html"


def test_chunker(sample_documents):
    chunker = SemanticChunker(chunk_size=128, chunk_overlap=32)
    chunks = chunker.chunk_documents(sample_documents)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert chunk.document_id == "test_doc_1"


def test_index_builder(tmp_path, sample_documents):
    from utils.embedding_client import EmbeddingClient
    
    embedding_client = EmbeddingClient(config_path="configs/model_config.yaml")
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(sample_documents)

    index_builder = IndexBuilder(tmp_path, embedding_client=embedding_client)
    index_builder.build_vector_index(chunks)
    index_builder.build_knowledge_graph(sample_documents, chunks)

    assert index_builder.kg_file.exists()


def test_chunk_retrieval(tmp_path, sample_documents):
    from utils.embedding_client import EmbeddingClient
    
    embedding_client = EmbeddingClient(config_path="configs/model_config.yaml")
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(sample_documents)

    index_builder = IndexBuilder(tmp_path, embedding_client=embedding_client)
    index_builder._store_chunk_metadata(chunks)

    chunk = index_builder.get_chunk_by_id(chunks[0].chunk_id)
    assert chunk is not None
    assert chunk["document_id"] == "test_doc_1"
