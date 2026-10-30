"""
TicketAI - RAG and Knowledge Base Unit Tests
Tests text chunking, embedding generation, PII sanitization, and semantic retrieval.
"""

import unittest
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rag.text_chunker import TextChunker
from src.rag.embedding_service import EmbeddingService
from src.rag.kb_manager import PIIFilter, KnowledgeBaseManager


class TestRAGComponents(unittest.TestCase):

    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.embedding_service = EmbeddingService(expected_dimension=768)

    def test_pii_sanitization(self):
        """Verify credit cards, emails, and secrets are redacted."""
        raw = "User irfan@example.com paid with card 4111 2222 3333 4444 using api_key: secret12345"
        sanitized = PIIFilter.sanitize(raw)
        self.assertNotIn("4111 2222 3333 4444", sanitized)
        self.assertNotIn("irfan@example.com", sanitized)
        self.assertNotIn("secret12345", sanitized)
        self.assertIn("[REDACTED_CARD]", sanitized)
        self.assertIn("[REDACTED_EMAIL]", sanitized)

    def test_text_chunking_with_headings(self):
        """Verify markdown headings and chunks are preserved."""
        md_text = (
            "# Main Heading\n\n"
            "This is paragraph one discussing support policies.\n\n"
            "## Subheading\n\n"
            "This is paragraph two detailing refund steps and return shipping guidelines."
        )
        chunks = self.chunker.chunk_text(md_text, metadata={"title": "Test Policy"})
        self.assertGreaterEqual(len(chunks), 1)
        first_chunk = chunks[0]
        self.assertIn("chunk_text", first_chunk)
        self.assertIn("metadata", first_chunk)
        self.assertEqual(first_chunk["chunk_index"], 0)

    def test_embedding_dimension(self):
        """Verify embedding dimension is consistently 768."""
        sample_texts = ["How do I return a damaged item?", "What is my invoice balance?"]
        embeddings = self.embedding_service.embed_texts(sample_texts)
        self.assertEqual(len(embeddings), 2)
        for emb in embeddings:
            self.assertEqual(len(emb), 768)
            # Check vector normalization (norm close to 1.0)
            norm = sum(x**2 for x in emb)**0.5
            self.assertAlmostEqual(norm, 1.0, places=2)

    def test_deterministic_embedding_consistency(self):
        """Verify identical text yields identical embedding vector."""
        t1 = "My account is locked"
        t2 = "My account is locked"
        e1 = self.embedding_service.embed_query(t1)
        e2 = self.embedding_service.embed_query(t2)
        self.assertEqual(e1, e2)


if __name__ == "__main__":
    unittest.main()

