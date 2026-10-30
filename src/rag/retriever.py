"""
TicketAI - Semantic RAG Retriever
Generates query embeddings, performs pgvector semantic search, applies category filters,
and enforces minimum relevance score thresholds to reject hallucination-prone queries.
"""

import logging
from typing import List, Dict, Any, Optional
from src.rag.embedding_service import EmbeddingService
from src.rag.vector_store import VectorStore

logger = logging.getLogger("ticketai.retriever")


class Retriever:
    """Semantic search engine powered by NeonDB pgvector and Google Embeddings."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        default_top_k: int = 3,
        default_relevance_threshold: Optional[float] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k
        if default_relevance_threshold is not None:
            self.default_relevance_threshold = default_relevance_threshold
        else:
            # 0.45 for real dense neural embeddings; 0.10 for offline hash vectors
            self.default_relevance_threshold = 0.45 if self.embedding_service.is_api_available else 0.10

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K knowledge chunks for a support ticket.
        If results fall below relevance_threshold, they are safely rejected.
        """
        k = top_k or self.default_top_k
        min_score = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold

        clean_query = str(query or "").strip()
        if not clean_query:
            logger.info("Empty query provided to retriever.")
            return []

        # Generate query vector
        query_vector = self.embedding_service.embed_query(clean_query)

        # First attempt: search with category filter if specified
        results = []
        if category and category.lower() not in ["all", "out of scope"]:
            results = self.vector_store.similarity_search(
                query_embedding=query_vector,
                top_k=k,
                category_filter=category,
                min_relevance_score=min_score
            )

        # Fallback attempt: broad search across all categories if category-specific yielded insufficient chunks
        if len(results) < 2:
            broad_results = self.vector_store.similarity_search(
                query_embedding=query_vector,
                top_k=k,
                category_filter=None,
                min_relevance_score=min_score
            )
            # Merge without duplicates
            seen_chunk_ids = {r["chunk_id"] for r in results}
            for r in broad_results:
                if r["chunk_id"] not in seen_chunk_ids:
                    results.append(r)
                    seen_chunk_ids.add(r["chunk_id"])

        # Truncate to top_k
        results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:k]

        if not results:
            logger.info(f"Zero chunks met the minimum relevance threshold ({min_score}) for query: '{clean_query[:60]}...'")
        else:
            top_score = results[0]["relevance_score"]
            logger.info(f"Retrieved {len(results)} chunks (Top score: {top_score:.3f}) for query.")

        return results
