"""
TicketAI - NeonDB pgvector Vector Store
Handles high-performance similarity search, category filtering,
and embedding persistence in Neon PostgreSQL using pgvector.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from src.database.connection import get_db

logger = logging.getLogger("ticketai.vector_store")


class VectorStore:
    """NeonDB pgvector vector storage and semantic search operations."""

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        category_filter: Optional[str] = None,
        min_relevance_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Executes a pgvector cosine similarity search:
        relevance_score = 1 - (embedding <=> query_vec)
        Only considers chunks belonging to active knowledge documents.
        """
        if not query_embedding:
            return []

        emb_str = f"[{','.join(str(float(x)) for x in query_embedding)}]"

        query = """
            SELECT 
                kc.id as chunk_id,
                kc.document_id,
                kc.chunk_index,
                kc.chunk_text,
                kc.metadata as chunk_metadata,
                kd.title as doc_title,
                kd.category as doc_category,
                kd.source as doc_source,
                kd.version as doc_version,
                1 - (kc.embedding <=> %s::vector) AS relevance_score
            FROM knowledge_chunks kc
            JOIN knowledge_documents kd ON kc.document_id = kd.id
            WHERE kd.is_active = TRUE
        """
        params = [emb_str]

        if category_filter and category_filter.lower() != "all":
            query += " AND kd.category = %s"
            params.append(category_filter)

        query += " ORDER BY kc.embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([emb_str, top_k])

        results = []
        try:
            with get_db() as db:
                rows = db.execute(query, tuple(params)).fetchall()
                for row in rows:
                    score = float(row.get("relevance_score", 0.0))
                    if score >= min_relevance_score:
                        results.append({
                            "chunk_id": row["chunk_id"],
                            "document_id": row["document_id"],
                            "chunk_index": row["chunk_index"],
                            "chunk_text": row["chunk_text"],
                            "metadata": row.get("chunk_metadata") or {},
                            "title": row["doc_title"],
                            "category": row["doc_category"],
                            "source": row["doc_source"],
                            "version": row["doc_version"],
                            "relevance_score": round(score, 4)
                        })
            logger.info(f"Retrieved {len(results)} relevant chunks (top_k={top_k}, category={category_filter}).")
            return results
        except Exception as e:
            logger.error(f"Error querying pgvector: {e}")
            return []

    def get_chunk_count(self) -> int:
        """Returns total active chunks stored in NeonDB."""
        try:
            with get_db() as db:
                row = db.execute(
                    """
                    SELECT COUNT(*) as count 
                    FROM knowledge_chunks kc
                    JOIN knowledge_documents kd ON kc.document_id = kd.id
                    WHERE kd.is_active = TRUE;
                    """
                ).fetchone()
                return row["count"] if row else 0
        except Exception as e:
            logger.warning(f"Could not count chunks: {e}")
            return 0

