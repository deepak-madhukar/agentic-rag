import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ingestion_pipeline.loader import DocumentLoader
from ingestion_pipeline.chunker import SemanticChunker
from ingestion_pipeline.index_builder import IndexBuilder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    data_dir = Path("tests/test_data")
    index_dir = Path("indexes/store")
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ingestion pipeline...")

    from utils.embedding_client import EmbeddingClient
    
    embedding_client = EmbeddingClient(config_path="configs/model_config.yaml")

    loader = DocumentLoader(
        pdf_dir=data_dir / "pdfs",
        html_dir=data_dir / "html",
        json_dir=data_dir / "json",
        email_dir=data_dir / "emails",
        embedding_client=embedding_client,
    )

    documents = await loader.load_all()
    logger.info(f"Loaded {len(documents)} documents")

    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    builder = IndexBuilder(
        index_dir=index_dir,
        embedding_client=embedding_client,
    )
    builder.build_vector_index(chunks)
    builder.build_knowledge_graph(documents, chunks)

    logger.info("Ingestion pipeline completed successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
