"""
TicketAI - Ticket Orchestrator
Master coordinator executing the end-to-end support ticket pipeline:
Input Validation -> ML Classifier -> NeonDB Insertion -> RAG Retrieval ->
Gemini LLM -> Response Validation -> Status Escalation -> Resolution Storage.
"""

import re
import logging
from typing import Dict, Any, Optional

from src.classifier.ticket_classifier import TicketClassifier
from src.resolution.resolution_engine import ResolutionEngine
from src.database.connection import get_db

logger = logging.getLogger("ticketai.orchestrator")


class TicketOrchestrator:
    """Master workflow orchestrator for TicketAI."""

    def __init__(
        self,
        classifier: Optional[TicketClassifier] = None,
        resolution_engine: Optional[ResolutionEngine] = None
    ):
        self.classifier = classifier or TicketClassifier.get_instance()
        self.resolution_engine = resolution_engine or ResolutionEngine()

    def process_ticket_submission(
        self,
        user_id: int,
        ticket_text: str,
        notification_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a full customer ticket submission through the unified pipeline.
        """
        raw_text = str(ticket_text or "").strip()
        if not raw_text:
            raise ValueError("Ticket text cannot be empty.")

        # Email validation if provided
        clean_email = (notification_email or "").strip()
        if clean_email and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", clean_email):
            raise ValueError("Invalid notification email format.")

        logger.info(f"Initiating TicketAI pipeline for user #{user_id}: '{raw_text[:60]}...'")

        # 1. Existing ML Classification
        ml_decision = self.classifier.predict(raw_text)
        predicted_cat = ml_decision["predicted_category"]
        confidence = float(ml_decision["confidence"])
        is_oos = bool(ml_decision.get("is_out_of_scope", False))
        needs_review = bool(ml_decision.get("needs_human_review", False) or confidence < 60.0)
        review_reason = ml_decision.get("review_reason", "")
        top_predictions = ml_decision.get("top_predictions", [])

        # Display category override for out of scope
        display_category = "Out of Scope" if is_oos else predicted_cat

        # 2. Persist Initial Ticket into NeonDB
        with get_db() as db:
            cur = db.execute(
                """
                INSERT INTO tickets (
                    user_id, ticket_text, predicted_category, confidence,
                    status, notification_email, is_out_of_scope,
                    needs_human_review, review_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    user_id,
                    raw_text,
                    display_category,
                    confidence,
                    "Open",
                    clean_email,
                    int(is_oos),
                    int(needs_review),
                    review_reason
                )
            )
            ticket_id = cur.lastrowid
            db.commit()

        logger.info(f"Persisted Ticket #{ticket_id} in NeonDB with category '{display_category}'.")

        # 3. RAG Retrieval + LLM Generation + Automated Resolution Engine
        clean_ml_review_reason = review_reason if (review_reason and not review_reason.lower().startswith("none")) else ""
        resolution_result = self.resolution_engine.process_ticket(
            ticket_id=ticket_id,
            customer_message=raw_text,
            predicted_category=predicted_cat,
            confidence=confidence,
            is_out_of_scope=is_oos,
            needs_human_review_flag=needs_review,
            review_reason=clean_ml_review_reason
        )

        return {
            "ticket_id": ticket_id,
            "category": display_category,
            "raw_category": predicted_cat,
            "confidence": round(confidence, 2),
            "confidence_level": ml_decision.get("confidence_level", "Medium"),
            "is_out_of_scope": is_oos,
            "ml_needs_review": needs_review,
            "ml_review_reason": review_reason,
            "needs_human_review": resolution_result["requires_human_review"],
            "review_reason": resolution_result["escalation_reason"],
            "status": resolution_result["status"],
            "top_predictions": top_predictions,
            "resolution": resolution_result["resolution_data"],
            "sources": resolution_result["retrieved_documents"],
            "latency_ms": resolution_result.get("latency_ms", 0.0),
            "success": True
        }

