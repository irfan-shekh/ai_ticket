"""
TicketAI - RAG Pipeline
End-to-end retrieval coordinator that handles query preparation,
relevance gating, and context structuring.
"""

import logging
from typing import Dict, Any, List, Optional
from src.rag.retriever import Retriever

logger = logging.getLogger("ticketai.rag_pipeline")


class RAGPipeline:
    """Coordinates retrieval and preparation of grounded context for Gemini."""

    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()

    def retrieve_context_for_ticket(
        self,
        ticket_text: str,
        predicted_category: str,
        top_k: int = 5,
        relevance_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieves context for a customer support ticket.
        Returns:
        - 'documents': list of chunk dicts with source metadata
        - 'has_sufficient_context': bool
        - 'top_relevance_score': float
        - 'rejection_reason': str if insufficient
        """
        min_threshold = relevance_threshold if relevance_threshold is not None else self.retriever.default_relevance_threshold
        chunks = self.retriever.retrieve(
            query=ticket_text,
            category=predicted_category,
            top_k=top_k,
            relevance_threshold=min_threshold
        )

        if not chunks:
            return {
                "documents": [],
                "has_sufficient_context": False,
                "top_relevance_score": 0.0,
                "rejection_reason": "No authoritative knowledge base articles matched the required relevance threshold."
            }

        top_score = chunks[0]["relevance_score"]
        # If top score is weak, mark as insufficient context to prevent hallucinations
        has_sufficient = top_score >= min_threshold

        return {
            "documents": chunks,
            "has_sufficient_context": has_sufficient,
            "top_relevance_score": top_score,
            "rejection_reason": "" if has_sufficient else "Retrieved knowledge chunks had low similarity score."
        }
