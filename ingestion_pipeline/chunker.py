import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    content: str
    document_id: str
    document_type: str
    product: str
    date: str
    owner: str
    category: str
    section: str
    subsection: str
    start_idx: int
    end_idx: int
    embedding: Optional[list] = None


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_pattern = re.compile(r"[.!?]\s+")

    def chunk_documents(self, documents: list) -> list[Chunk]:
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_single_document(self, document) -> list[Chunk]:
        chunks = []
        content = document.content
        sections = self._extract_sections(content)

        chunk_idx = 0
        for section_title, section_content in sections:
            subsections = self._extract_subsections(section_content)
            for subsection_title, subsection_content in subsections:
                text_chunks = self._semantic_chunk_text(
                    subsection_content,
                    document.metadata.document_id,
                    section_title,
                    subsection_title,
                    document.metadata,
                )
                for text_chunk in text_chunks:
                    chunks.append(text_chunk)
                    chunk_idx += 1

        return chunks

    def _extract_sections(self, content: str) -> list[tuple[str, str]]:
        section_pattern = re.compile(r"^(#{1,2})\s+(.+?)$", re.MULTILINE)
        sections = []
        last_end = 0
        section_title = "Introduction"

        for match in section_pattern.finditer(content):
            if sections or last_end > 0:
                sections.append((section_title, content[last_end : match.start()]))
            section_title = match.group(2).strip()
            last_end = match.end()

        if last_end < len(content):
            sections.append((section_title, content[last_end:]))

        if not sections:
            sections = [("Content", content)]

        return sections

    def _extract_subsections(self, content: str) -> list[tuple[str, str]]:
        subsection_pattern = re.compile(r"^(###)\s+(.+?)$", re.MULTILINE)
        subsections = []
        last_end = 0
        subsection_title = "General"

        for match in subsection_pattern.finditer(content):
            if subsections or last_end > 0:
                subsections.append((subsection_title, content[last_end : match.start()]))
            subsection_title = match.group(2).strip()
            last_end = match.end()

        if last_end < len(content):
            subsections.append((subsection_title, content[last_end:]))

        if not subsections:
            subsections = [("General", content)]

        return subsections

    def _semantic_chunk_text(
        self,
        text: str,
        document_id: str,
        section: str,
        subsection: str,
        metadata,
    ) -> list[Chunk]:
        chunks = []
        sentences = self.sentence_pattern.split(text)

        current_chunk = ""
        current_start = 0
        chunk_count = 0

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunk_id = f"{document_id}:chunk_{chunk_count}"
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        content=current_chunk.strip(),
                        document_id=document_id,
                        document_type=metadata.document_type,
                        product=metadata.product,
                        date=metadata.date,
                        owner=metadata.owner,
                        category=metadata.category,
                        section=section,
                        subsection=subsection,
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                    )
                    chunks.append(chunk)
                    chunk_count += 1

                current_chunk = sentence + " "
                current_start = idx

        if current_chunk.strip():
            chunk_id = f"{document_id}:chunk_{chunk_count}"
            chunk = Chunk(
                chunk_id=chunk_id,
                content=current_chunk.strip(),
                document_id=document_id,
                document_type=metadata.document_type,
                product=metadata.product,
                date=metadata.date,
                owner=metadata.owner,
                category=metadata.category,
                section=section,
                subsection=subsection,
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
            )
            chunks.append(chunk)

        return chunks
