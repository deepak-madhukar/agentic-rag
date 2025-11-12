import os
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
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    data_dir = Path("tests/test_data")
    index_dir = Path("indexes/store")
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ingestion pipeline...")

    loader = DocumentLoader(
        pdf_dir=data_dir / "pdfs",
        html_dir=data_dir / "html",
        json_dir=data_dir / "json",
        email_dir=data_dir / "emails",
    )

    documents = await loader.load_all()
    logger.info(f"Loaded {len(documents)} documents")

    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    model_config_file = Path("configs/model_config.yaml")
    if not model_config_file.exists():
        logger.error(f"Model config file not found: {model_config_file}")
        raise FileNotFoundError(f"Model config file required: {model_config_file}")
    
    with open(model_config_file, "r") as f:
        config = yaml.safe_load(f)
    
    if not config:
        logger.error("Model config is empty")
        raise ValueError("Model config is empty")
    
    embedding_config = config.get("embedding", {})
    if not embedding_config:
        logger.error("Embedding config not found in model_config.yaml")
        raise ValueError("Embedding config not found in model_config.yaml")
    
    embedding_model_name = embedding_config.get("model_name", "nomic-embed-text")
    embedding_base_url = embedding_config.get("base_url", "http://localhost:11434")

    logger.info(f"Using embedding model: {embedding_model_name}")

    builder = IndexBuilder(
        index_dir=index_dir,
        model_name=embedding_model_name,
        base_url=embedding_base_url,
    )
    builder.build_vector_index(chunks)
    builder.build_knowledge_graph(documents, chunks)

    logger.info("Ingestion pipeline completed successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
