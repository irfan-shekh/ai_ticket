"""
TicketAI - Database Repositories
Clean data access layer for documents, chunks, tickets, resolutions, feedback, and audit events.
Uses parameterized queries and proper transaction handling.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from src.database.connection import get_db

logger = logging.getLogger("ticketai.repositories")


class KnowledgeRepository:
    """Manages knowledge documents and their vectorized chunks in NeonDB."""

    @staticmethod
    def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
        with get_db() as db:
            row = db.execute("SELECT * FROM knowledge_documents WHERE id = %s", (doc_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def find_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
        if not file_hash:
            return None
        with get_db() as db:
            row = db.execute("SELECT * FROM knowledge_documents WHERE file_hash = %s", (file_hash,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_documents(active_only: bool = True, category: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM knowledge_documents WHERE 1=1"
        params = []
        if active_only:
            query += " AND is_active = TRUE"
        if category and category.lower() != "all":
            query += " AND category = %s"
            params.append(category)
        query += " ORDER BY created_at DESC"

        with get_db() as db:
            rows = db.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows] if rows else []

    @staticmethod
    def create_document(
        title: str,
        content: str,
        category: str,
        source: str,
        version: str = "1.0",
        file_hash: str = ""
    ) -> int:
        with get_db() as db:
            cur = db.execute(
                """
                INSERT INTO knowledge_documents (title, content, category, source, version, file_hash, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                RETURNING id;
                """,
                (title, content, category, source, version, file_hash)
            )
            doc_id = cur.lastrowid
            db.commit()
            return doc_id

    @staticmethod
    def update_document(
        doc_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        version: Optional[str] = None,
        is_active: Optional[bool] = None,
        file_hash: Optional[str] = None
    ) -> bool:
        updates = []
        params = []
        if title is not None:
            updates.append("title = %s")
            params.append(title)
        if content is not None:
            updates.append("content = %s")
            params.append(content)
        if category is not None:
            updates.append("category = %s")
            params.append(category)
        if version is not None:
            updates.append("version = %s")
            params.append(version)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(is_active)
        if file_hash is not None:
            updates.append("file_hash = %s")
            params.append(file_hash)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(doc_id)
        sql = f"UPDATE knowledge_documents SET {', '.join(updates)} WHERE id = %s;"

        with get_db() as db:
            db.execute(sql, tuple(params))
            db.commit()
            return True

    @staticmethod
    def set_document_active(doc_id: int, is_active: bool) -> bool:
        with get_db() as db:
            db.execute("UPDATE knowledge_documents SET is_active = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s", (is_active, doc_id))
            db.commit()
            return True

    @staticmethod
    def delete_document(doc_id: int) -> bool:
        with get_db() as db:
            db.execute("DELETE FROM knowledge_documents WHERE id = %s", (doc_id,))
            db.commit()
            return True

    @staticmethod
    def save_chunks(document_id: int, chunks: List[Dict[str, Any]]) -> int:
        """
        Saves chunks with embeddings for a document.
        First clears old chunks for this document to ensure clean reindexing.
        """
        with get_db() as db:
            # Delete old chunks
            db.execute("DELETE FROM knowledge_chunks WHERE document_id = %s;", (document_id,))
            count = 0
            for idx, chunk in enumerate(chunks):
                text = chunk.get("chunk_text") or chunk.get("text", "")
                emb = chunk.get("embedding")
                meta = chunk.get("metadata", {})
                meta_json = json.dumps(meta)

                # Format embedding as string '[x1,x2,...]' for pgvector
                if isinstance(emb, list):
                    emb_str = f"[{','.join(str(float(x)) for x in emb)}]"
                else:
                    emb_str = None

                db.execute(
                    """
                    INSERT INTO knowledge_chunks (document_id, chunk_index, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb);
                    """,
                    (document_id, idx, text, emb_str, meta_json)
                )
                count += 1
            db.commit()
            return count

    @staticmethod
    def get_document_chunk_count(document_id: int) -> int:
        with get_db() as db:
            row = db.execute("SELECT COUNT(*) as cnt FROM knowledge_chunks WHERE document_id = %s", (document_id,)).fetchone()
            return row["cnt"] if row else 0


class TicketRepository:
    """Manages tickets, resolutions, feedback, and audit events."""

    @staticmethod
    def get_ticket(ticket_id: int) -> Optional[Dict[str, Any]]:
        with get_db() as db:
            row = db.execute(
                """
                SELECT t.*, u.username, u.email as user_email
                FROM tickets t
                LEFT JOIN users u ON t.user_id = u.id
                WHERE t.id = %s;
                """,
                (ticket_id,)
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def update_status(ticket_id: int, new_status: str, description: str = "") -> bool:
        with get_db() as db:
            # Get current status
            curr = db.execute("SELECT status FROM tickets WHERE id = %s;", (ticket_id,)).fetchone()
            old_status = curr["status"] if curr else ""

            db.execute("UPDATE tickets SET status = %s WHERE id = %s;", (new_status, ticket_id))
            # Log event
            db.execute(
                """
                INSERT INTO ticket_events (ticket_id, previous_status, new_status, event_description)
                VALUES (%s, %s, %s, %s);
                """,
                (ticket_id, old_status, new_status, description or f"Status changed from {old_status} to {new_status}")
            )
            db.commit()
            return True

    @staticmethod
    def save_resolution(
        ticket_id: int,
        proposed_solution: str,
        model_name: str,
        problem_summary: str = "",
        identified_issue: str = "",
        resolution_status: str = "needs_review",
        requires_human_review: bool = True,
        grounding_status: str = "unverified",
        escalation_reason: str = "",
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        with get_db() as db:
            cur = db.execute(
                """
                INSERT INTO ticket_resolutions (
                    ticket_id, problem_summary, identified_issue, proposed_solution,
                    model_name, resolution_status, requires_human_review,
                    grounding_status, escalation_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    ticket_id, problem_summary, identified_issue, proposed_solution,
                    model_name, resolution_status, requires_human_review,
                    grounding_status, escalation_reason
                )
            )
            res_id = cur.lastrowid

            if sources:
                for src in sources:
                    chunk_id = src.get("chunk_id")
                    doc_id = src.get("doc_id") or src.get("document_id")
                    score = float(src.get("relevance_score", 0.0))
                    db.execute(
                        """
                        INSERT INTO ticket_resolution_sources (resolution_id, knowledge_chunk_id, document_id, relevance_score)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (res_id, chunk_id, doc_id, score)
                    )

            db.commit()
            return res_id

    @staticmethod
    def get_latest_resolution(ticket_id: int) -> Optional[Dict[str, Any]]:
        with get_db() as db:
            row = db.execute(
                """
                SELECT * FROM ticket_resolutions
                WHERE ticket_id = %s
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (ticket_id,)
            ).fetchone()
            if not row:
                return None

            res_dict = dict(row)
            # Fetch attached sources
            sources = db.execute(
                """
                SELECT s.*, kd.title as doc_title, kd.category as doc_category, kc.chunk_text
                FROM ticket_resolution_sources s
                LEFT JOIN knowledge_documents kd ON s.document_id = kd.id
                LEFT JOIN knowledge_chunks kc ON s.knowledge_chunk_id = kc.id
                WHERE s.resolution_id = %s
                ORDER BY s.relevance_score DESC;
                """,
                (res_dict["id"],)
            ).fetchall()
            res_dict["sources"] = [dict(s) for s in sources] if sources else []
            return res_dict

    @staticmethod
    def save_feedback(
        ticket_id: int,
        resolution_id: Optional[int],
        rating: int,
        comment: str = "",
        was_resolved: bool = False
    ) -> int:
        with get_db() as db:
            cur = db.execute(
                """
                INSERT INTO ticket_feedback (ticket_id, resolution_id, feedback_rating, feedback_comment, was_resolved)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (ticket_id, resolution_id, rating, comment, was_resolved)
            )
            fb_id = cur.lastrowid

            # Update ticket status if verified resolved
            if was_resolved:
                db.execute(
                    """
                    UPDATE tickets
                    SET status = 'Resolved', resolved_at = CURRENT_TIMESTAMP, resolution_notes = %s
                    WHERE id = %s;
                    """,
                    (f"Verified resolution via customer feedback (Rating: {rating}/5): {comment}".strip(), ticket_id)
                )
                db.execute(
                    """
                    INSERT INTO ticket_events (ticket_id, previous_status, new_status, event_description)
                    VALUES (%s, 'awaiting_customer', 'Resolved', %s);
                    """,
                    (ticket_id, f"Customer confirmed resolution (Rating: {rating}/5)")
                )
            else:
                db.execute(
                    """
                    UPDATE tickets
                    SET status = 'needs_review', needs_human_review = 1,
                        review_reason = %s
                    WHERE id = %s;
                    """,
                    (f"Customer reported issue not resolved (Rating: {rating}/5): {comment}".strip(), ticket_id)
                )
                db.execute(
                    """
                    INSERT INTO ticket_events (ticket_id, previous_status, new_status, event_description)
                    VALUES (%s, 'awaiting_customer', 'needs_review', %s);
                    """,
                    (ticket_id, f"Customer flagged unsatisfied resolution (Rating: {rating}/5). Escalated to human review.")
                )

            db.commit()
            return fb_id

    @staticmethod
    def get_events(ticket_id: int) -> List[Dict[str, Any]]:
        with get_db() as db:
            rows = db.execute("SELECT * FROM ticket_events WHERE ticket_id = %s ORDER BY created_at ASC;", (ticket_id,)).fetchall()
            return [dict(r) for r in rows] if rows else []

