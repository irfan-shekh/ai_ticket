"""
TicketAI - Intelligent Text Chunker
Splits support documents while preserving paragraph structure, headings,
list items, and document metadata. Configurable token-aware windowing.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ticketai.chunker")


class TextChunker:
    """
    Intelligent document chunker that maintains markdown headings,
    bulleted procedures, and paragraph coherence.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _estimate_tokens(self, text: str) -> int:
        """Lightweight token estimator (~4 chars per token, or word-split)."""
        return max(1, len(text.split()))

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks preserving semantic paragraph boundaries.
        Returns a list of dicts with:
        - 'chunk_index': int
        - 'chunk_text': str
        - 'metadata': dict
        - 'document_id': int (if provided)
        """
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on double newlines (paragraphs/headings) or Markdown headers
        paragraphs = re.split(r"\n{2,}", normalized)
        
        chunks = []
        current_chunk_paragraphs = []
        current_token_count = 0
        current_heading = metadata.get("title", "")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph is a Markdown heading
            heading_match = re.match(r"^(#{1,6}\s+.*)", para)
            if heading_match:
                current_heading = heading_match.group(1).lstrip("#").strip()

            para_tokens = self._estimate_tokens(para)

            # If a single paragraph exceeds chunk_size, split by sentences
            if para_tokens > self.chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_tokens = self._estimate_tokens(sent)
                    if current_token_count + sent_tokens > self.chunk_size and current_chunk_paragraphs:
                        # Finalize chunk
                        chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                        chunks.append(self._create_chunk_dict(
                            len(chunks), chunk_str, current_heading, metadata, document_id
                        ))
                        # Keep overlap
                        overlap_paras = []
                        overlap_tokens = 0
                        for p in reversed(current_chunk_paragraphs):
                            t = self._estimate_tokens(p)
                            if overlap_tokens + t <= self.chunk_overlap:
                                overlap_paras.insert(0, p)
                                overlap_tokens += t
                            else:
                                break
                        current_chunk_paragraphs = overlap_paras
                        current_token_count = overlap_tokens

                    current_chunk_paragraphs.append(sent)
                    current_token_count += sent_tokens
            else:
                if current_token_count + para_tokens > self.chunk_size and current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                    chunks.append(self._create_chunk_dict(
                        len(chunks), chunk_str, current_heading, metadata, document_id
                    ))
                    # Retain overlap
                    overlap_paras = []
                    overlap_tokens = 0
                    for p in reversed(current_chunk_paragraphs):
                        t = self._estimate_tokens(p)
                        if overlap_tokens + t <= self.chunk_overlap:
                            overlap_paras.insert(0, p)
                            overlap_tokens += t
                        else:
                            break
                    current_chunk_paragraphs = overlap_paras
                    current_token_count = overlap_tokens

                current_chunk_paragraphs.append(para)
                current_token_count += para_tokens

        # Final trailing chunk
        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
            if chunk_str:
                chunks.append(self._create_chunk_dict(
                    len(chunks), chunk_str, current_heading, metadata, document_id
                ))

        logger.info(f"Chunked document (id={document_id}) into {len(chunks)} chunks.")
        return chunks

    def _create_chunk_dict(
        self,
        index: int,
        chunk_text: str,
        heading: str,
        parent_meta: Dict[str, Any],
        document_id: Optional[int]
    ) -> Dict[str, Any]:
        meta = dict(parent_meta)
        if heading:
            meta["section_heading"] = heading
        meta["estimated_tokens"] = self._estimate_tokens(chunk_text)

        result = {
            "chunk_index": index,
            "chunk_text": chunk_text,
            "metadata": meta
        }
        if document_id is not None:
            result["document_id"] = document_id
        return result

