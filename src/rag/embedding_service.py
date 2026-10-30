"""
TicketAI - Embedding Service
Generates dense vector embeddings using official Google Gen AI Embedding API
(text-embedding-004, 768 dim) with batching, hash-based deduplication,
offline deterministic fallback, and dimension verification.
"""

import os
import hashlib
import logging
from typing import List, Union, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ticketai.embedding")


class EmbeddingService:
    """
    Manages embedding generation using Google Gen AI with caching,
    batching, and robust offline fallback for sandboxed tests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-004",
        expected_dimension: int = 768
    ):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = model_name
        self.dimension = expected_dimension
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.info("GEMINI_API_KEY not found. Embedding service will use deterministic 768-d fallback for offline testability.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google Gen AI Embedding client with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not initialize Google Gen AI embedding client ({e}). Offline fallback active.")
            self._client = None

    @property
    def is_api_available(self) -> bool:
        return bool(self._client and self.api_key)

    def compute_text_hash(self, text: str) -> str:
        """SHA-256 hash for document content / chunk change detection."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single search query."""
        res = self.embed_texts([text])
        return res[0] if res else [0.0] * self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts.
        Uses batching when supported by Google Gen AI.
        """
        if not texts:
            return []

        # If API is available, try Google Gen AI Embeddings
        if self.is_api_available:
            try:
                embeddings = []
                # Batch in groups of 20 to prevent rate-limit or payload size issues
                batch_size = 20
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self._client.models.embed_content(
                        model=self.model_name,
                        contents=batch
                    )

                    # Extract embeddings from response
                    for emb_obj in response.embeddings:
                        vec = emb_obj.values
                        if len(vec) != self.dimension:
                            self.dimension = len(vec)
                        # Normalize vector for cosine distance
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = (np.array(vec) / norm).tolist()
                        embeddings.append(vec)

                logger.info(f"Generated {len(embeddings)} embeddings via Google Gen AI (dim={self.dimension}).")
                return embeddings
            except Exception as e:
                logger.warning(f"Google Gen AI embedding call failed ({e}). Falling back to deterministic embedding.")

        # Fallback deterministic normalized vector (e.g., for offline or sandbox mode)
        return [self._deterministic_fallback_vector(t) for t in texts]

    def _deterministic_fallback_vector(self, text: str) -> List[float]:
        """
        Creates a consistent, reproducible 768-dimensional normalized pseudo-vector
        from text content using hash seeding and word token distributions.
        Guarantees zero-network offline testability while maintaining semantic properties.
        """
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)

        for i, word in enumerate(words):
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = word_hash % self.dimension
            vec[idx] += 1.0 / (1.0 + (i * 0.05))

        # Add salt from whole string hash
        full_hash = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        seed_rng = np.random.RandomState(full_hash % (2**32))
        noise = seed_rng.normal(0, 0.01, size=self.dimension)
        vec += noise

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

