import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from utils.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    document_id: str
    document_type: str
    title: str
    product: str
    date: str
    owner: str
    category: str
    source_path: str


@dataclass
class Document:
    metadata: DocumentMetadata
    content: str
    doc_type: str


class DocumentLoader:
    def __init__(
        self,
        pdf_dir: Path,
        html_dir: Path,
        json_dir: Path,
        email_dir: Path,
        embedding_client: "EmbeddingClient" = None,
    ):
        if embedding_client is None:
            raise ValueError("embedding_client is required and cannot be None")
        self.pdf_dir = Path(pdf_dir)
        self.html_dir = Path(html_dir)
        self.json_dir = Path(json_dir)
        self.email_dir = Path(email_dir)
        self.embedding_client = embedding_client

    async def load_all(self) -> list[Document]:
        documents = []
        documents.extend(await self._load_pdfs())
        documents.extend(await self._load_html())
        documents.extend(await self._load_json())
        documents.extend(await self._load_emails())
        return documents

    async def _load_pdfs(self) -> list[Document]:
        documents = []
        if not self.pdf_dir.exists():
            return documents

        try:
            import PyPDF2

            for pdf_file in self.pdf_dir.glob("*.pdf"):
                try:
                    with open(pdf_file, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()

                    metadata = DocumentMetadata(
                        document_id=pdf_file.stem,
                        document_type="INTERNAL",
                        title=pdf_file.stem.replace("_", " ").title(),
                        product="ProductA",
                        date=datetime.now().isoformat(),
                        owner="engineering",
                        category="technical",
                        source_path=str(pdf_file),
                    )
                    documents.append(
                        Document(
                            metadata=metadata,
                            content=text,
                            doc_type="pdf",
                        )
                    )
                    logger.info(f"Loaded PDF: {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file}: {e}")
        except ImportError:
            logger.warning("PyPDF2 not installed, skipping PDFs")

        return documents

    async def _load_html(self) -> list[Document]:
        documents = []
        if not self.html_dir.exists():
            return documents

        try:
            from bs4 import BeautifulSoup

            for html_file in self.html_dir.glob("*.html"):
                try:
                    with open(html_file, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "html.parser")
                        text = soup.get_text(separator="\n", strip=True)

                    metadata = DocumentMetadata(
                        document_id=html_file.stem,
                        document_type="PUBLIC",
                        title=html_file.stem.replace("_", " ").title(),
                        product="ProductB",
                        date=datetime.now().isoformat(),
                        owner="marketing",
                        category="documentation",
                        source_path=str(html_file),
                    )
                    documents.append(
                        Document(
                            metadata=metadata,
                            content=text,
                            doc_type="html",
                        )
                    )
                    logger.info(f"Loaded HTML: {html_file.name}")
                except Exception as e:
                    logger.error(f"Error loading HTML {html_file}: {e}")
        except ImportError:
            logger.warning("BeautifulSoup not installed, skipping HTML")

        return documents

    async def _load_json(self) -> list[Document]:
        documents = []
        if not self.json_dir.exists():
            return documents

        for json_file in self.json_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        content = json.dumps(item, indent=2)
                        doc_id = f"{json_file.stem}_{idx}"
                else:
                    content = json.dumps(data, indent=2)
                    doc_id = json_file.stem

                metadata = DocumentMetadata(
                    document_id=doc_id,
                    document_type="TEAM",
                    title=json_file.stem.replace("_", " ").title(),
                    product="ProductA",
                    date=datetime.now().isoformat(),
                    owner="devops",
                    category="tickets",
                    source_path=str(json_file),
                )
                documents.append(
                    Document(
                        metadata=metadata,
                        content=content,
                        doc_type="json",
                    )
                )
                logger.info(f"Loaded JSON: {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading JSON {json_file}: {e}")

        return documents

    async def _load_emails(self) -> list[Document]:
        documents = []
        if not self.email_dir.exists():
            return documents

        for eml_file in self.email_dir.glob("*.eml"):
            try:
                with open(eml_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                lines = content.split("\n")
                subject = next(
                    (line.split(": ", 1)[1] for line in lines if line.startswith("Subject:")),
                    eml_file.stem,
                )

                metadata = DocumentMetadata(
                    document_id=eml_file.stem,
                    document_type="INTERNAL",
                    title=subject,
                    product="ProductC",
                    date=datetime.now().isoformat(),
                    owner="support",
                    category="communication",
                    source_path=str(eml_file),
                )
                documents.append(
                    Document(
                        metadata=metadata,
                        content=content,
                        doc_type="email",
                    )
                )
                logger.info(f"Loaded Email: {eml_file.name}")
            except Exception as e:
                logger.error(f"Error loading email {eml_file}: {e}")

        return documents
