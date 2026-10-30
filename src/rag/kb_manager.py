"""
TicketAI - Knowledge Base Ingestion & Management System
Supports PDF, TXT, Markdown, FAQ, and verified troubleshooting guides.
Implements PII sanitization, duplicate hash detection, chunking, and embedding sync.
"""

import os
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.database.repositories import KnowledgeRepository
from src.rag.text_chunker import TextChunker
from src.rag.embedding_service import EmbeddingService

logger = logging.getLogger("ticketai.kb_manager")


class PIIFilter:
    """Sanitizes sensitive customer identifiers before knowledge base indexing."""

    # Common PII regex patterns
    PATTERNS = [
        (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[REDACTED_CARD]"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
        (re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S+"), "[REDACTED_SECRET]")
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


class KnowledgeBaseManager:
    """Central interface for ingesting, indexing, and updating knowledge base documents."""

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
        kb_directory: str = "knowledge_base"
    ):
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.kb_directory = kb_directory
        os.makedirs(self.kb_directory, exist_ok=True)

    def extract_text_from_file(self, file_path: str) -> str:
        """Extracts plain text from PDF, TXT, or Markdown files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".md", ".markdown"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                pages = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(f"<!-- Page {idx + 1} -->\n{page_text}")
                return "\n\n".join(pages)
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                raise ValueError(f"Failed to extract text from PDF: {e}")

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .txt, .md")

    def ingest_document(
        self,
        title: str,
        content: str,
        category: str,
        source: str = "manual_entry",
        version: str = "1.0",
        allow_duplicate: bool = False
    ) -> Dict[str, Any]:
        """
        Sanitizes, checks duplicates, stores in NeonDB, chunks, and creates pgvector embeddings.
        """
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Document title is required.")

        # 1. PII Sanitization
        sanitized_content = PIIFilter.sanitize(content.strip())
        if not sanitized_content:
            raise ValueError("Document content is empty after sanitization.")

        # 2. Hash calculation for duplicate detection
        file_hash = hashlib.sha256(sanitized_content.encode("utf-8")).hexdigest()

        existing = KnowledgeRepository.find_by_hash(file_hash)
        if existing and not allow_duplicate:
            logger.info(f"Duplicate document detected (id={existing['id']}, title='{existing['title']}'). Reusing.")
            return {
                "document_id": existing["id"],
                "status": "already_exists",
                "title": existing["title"],
                "chunks_indexed": KnowledgeRepository.get_document_chunk_count(existing["id"])
            }

        # 3. Insert or update document in NeonDB
        doc_id = KnowledgeRepository.create_document(
            title=clean_title,
            content=sanitized_content,
            category=category,
            source=source,
            version=version,
            file_hash=file_hash
        )

        # 4. Chunk document
        metadata = {
            "title": clean_title,
            "category": category,
            "source": source,
            "version": version
        }
        chunks = self.chunker.chunk_text(
            text=sanitized_content,
            metadata=metadata,
            document_id=doc_id
        )

        # 5. Generate embeddings
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        for i, c in enumerate(chunks):
            c["embedding"] = embeddings[i]

        # 6. Save chunks into pgvector
        saved_count = KnowledgeRepository.save_chunks(doc_id, chunks)

        logger.info(f"Successfully ingested document '{clean_title}' (id={doc_id}) with {saved_count} chunks.")
        return {
            "document_id": doc_id,
            "status": "created",
            "title": clean_title,
            "chunks_indexed": saved_count
        }

    def ingest_faq(
        self,
        question: str,
        answer: str,
        category: str,
        source: str = "faq_system"
    ) -> Dict[str, Any]:
        """Ingests a verified FAQ item as an authoritative knowledge document."""
        title = f"FAQ: {question[:100]}"
        content = f"### Frequently Asked Question\n\n**Question:** {question}\n\n**Official Answer:** {answer}"
        return self.ingest_document(
            title=title,
            content=content,
            category=category,
            source=source,
            version="1.0"
        )

    def reindex_document(self, doc_id: int) -> bool:
        """Re-generates chunks and embeddings for an updated document."""
        doc = KnowledgeRepository.get_document_by_id(doc_id)
        if not doc:
            return False

        content = doc["content"]
        metadata = {
            "title": doc["title"],
            "category": doc["category"],
            "source": doc["source"],
            "version": doc["version"]
        }
        chunks = self.chunker.chunk_text(content, metadata=metadata, document_id=doc_id)
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        for i, c in enumerate(chunks):
            c["embedding"] = embeddings[i]

        KnowledgeRepository.save_chunks(doc_id, chunks)
        logger.info(f"Reindexed document id={doc_id} with {len(chunks)} chunks.")
        return True

