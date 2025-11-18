import json
import logging
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from utils.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, index_dir: Path, embedding_client: "EmbeddingClient" = None):
        if embedding_client is None:
            raise ValueError("embedding_client is required and cannot be None")
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db = self.index_dir / "metadata.db"
        self.kg_file = self.index_dir / "kg_data.json"
        self.faiss_index_file = self.index_dir / "vector_index.bin"
        self.embedding_client = embedding_client

    def build_vector_index(self, chunks: list) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError:
            logger.error("FAISS not available; cannot build vector index")
            raise RuntimeError("FAISS required but not installed")

        if not chunks:
            logger.error("No chunks provided to build vector index")
            raise ValueError("Cannot build vector index with empty chunks list")

        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = [self.embedding_client.get_embedding(text) for text in texts]
            embeddings = np.array(embeddings).astype("float32")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings: {e}")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.faiss_index_file))

        self._store_chunk_metadata(chunks)
        logger.info(f"Built FAISS vector index with {len(chunks)} vectors of dimension {dimension}")

    def _store_chunk_metadata(self, chunks: list) -> None:
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                content TEXT,
                document_type TEXT,
                product TEXT,
                date TEXT,
                owner TEXT,
                category TEXT,
                section TEXT,
                subsection TEXT
            )
        """
        )

        for chunk in chunks:
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, content, document_type, product, date, owner, category, section, subsection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.document_type,
                    chunk.product,
                    chunk.date,
                    chunk.owner,
                    chunk.category,
                    chunk.section,
                    chunk.subsection,
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"Stored metadata for {len(chunks)} chunks in SQLite")

    def build_knowledge_graph(self, documents: list, chunks: list) -> None:
        kg_data = {
            "nodes": [],
            "edges": [],
            "document_map": {},
        }

        doc_id_counter = 0
        for doc in documents:
            doc_node_id = f"doc_{doc_id_counter}"
            kg_data["nodes"].append(
                {
                    "id": doc_node_id,
                    "type": "Document",
                    "label": doc.metadata.title,
                    "metadata": {
                        "document_id": doc.metadata.document_id,
                        "document_type": doc.metadata.document_type,
                        "product": doc.metadata.product,
                        "date": doc.metadata.date,
                        "owner": doc.metadata.owner,
                    },
                }
            )
            kg_data["document_map"][doc.metadata.document_id] = doc_node_id
            doc_id_counter += 1

        chunk_id_counter = 0
        for chunk in chunks:
            chunk_node_id = f"chunk_{chunk_id_counter}"
            kg_data["nodes"].append(
                {
                    "id": chunk_node_id,
                    "type": "Chunk",
                    "label": chunk.chunk_id,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "section": chunk.section,
                        "subsection": chunk.subsection,
                        "category": chunk.category,
                    },
                }
            )

            doc_node_id = kg_data["document_map"].get(chunk.document_id)
            if doc_node_id:
                kg_data["edges"].append(
                    {
                        "source": doc_node_id,
                        "target": chunk_node_id,
                        "relation": "contains",
                    }
                )

            chunk_id_counter += 1

        section_map = {}
        for chunk in chunks:
            section_key = f"{chunk.document_id}::{chunk.section}"
            if section_key not in section_map:
                section_node_id = f"section_{len(section_map)}"
                section_map[section_key] = section_node_id
                kg_data["nodes"].append(
                    {
                        "id": section_node_id,
                        "type": "Section",
                        "label": chunk.section,
                        "metadata": {"document_id": chunk.document_id},
                    }
                )

                doc_node_id = kg_data["document_map"].get(chunk.document_id)
                if doc_node_id:
                    kg_data["edges"].append(
                        {
                            "source": doc_node_id,
                            "target": section_node_id,
                            "relation": "has_section",
                        }
                    )

        with open(self.kg_file, "w") as f:
            json.dump(kg_data, f, indent=2)

        logger.info(f"Built knowledge graph with {len(kg_data['nodes'])} nodes and {len(kg_data['edges'])} edges")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.metadata_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def load_index_exists(self) -> bool:
        return self.faiss_index_file.exists() and self.metadata_db.exists()
