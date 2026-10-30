"""
TicketAI - Automated Resolution Decision Engine
Applies multi-stage decision gating:
Classification -> Risk Check -> RAG Context -> Gemini Generation -> Validation -> NeonDB Storage.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.database.repositories import TicketRepository
from src.llm.resolution_generator import ResolutionGenerator
from src.resolution.response_validator import ResponseValidator
from src.resolution.escalation import EscalationEngine
from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger("ticketai.resolution_engine")


class ResolutionEngine:
    """End-to-end resolution decision maker and workflow coordinator."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        resolution_generator: Optional[ResolutionGenerator] = None
    ):
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.resolution_generator = resolution_generator or ResolutionGenerator()

    def process_ticket(
        self,
        ticket_id: int,
        customer_message: str,
        predicted_category: str,
        confidence: float = 0.0,
        is_out_of_scope: bool = False,
        needs_human_review_flag: bool = False,
        review_reason: str = ""
    ) -> Dict[str, Any]:
        """
        Executes resolution pipeline for an incoming or existing ticket.
        Returns complete decision packet and persists resolution into NeonDB.
        """
        logger.info(f"Processing resolution for Ticket #{ticket_id} [{predicted_category}, {confidence:.1f}%]...")

        # 1. Operational Risk Assessment
        risk = EscalationEngine.assess_risk(
            ticket_text=customer_message,
            predicted_category=predicted_category,
            confidence=confidence,
            is_out_of_scope=is_out_of_scope,
            needs_review=needs_human_review_flag
        )

        # 2. RAG Semantic Retrieval
        retrieval_result = self.rag_pipeline.retrieve_context_for_ticket(
            ticket_text=customer_message,
            predicted_category=predicted_category,
            top_k=5,
            relevance_threshold=None
        )
        retrieved_docs = retrieval_result["documents"]
        has_sufficient_context = retrieval_result["has_sufficient_context"]

        # 3. LLM Structured Generation
        gen_result = self.resolution_generator.generate_resolution(
            ticket_id=ticket_id,
            customer_message=customer_message,
            predicted_category=predicted_category,
            classification_confidence=confidence,
            retrieved_documents=retrieved_docs,
            risk_assessment=risk["risk_reason"]
        )

        resolution_data = gen_result.get("data", {})

        # 4. Response Validation
        is_valid, validation_issues = ResponseValidator.validate_resolution(
            resolution_data=resolution_data,
            retrieved_documents=retrieved_docs,
            customer_message=customer_message
        )

        # 5. Final Decision Logic
        # A ticket can only be 'awaiting_customer' IF:
        # - Not out-of-scope
        # - Low risk
        # - Sufficient grounded retrieval
        # - Response passed validation
        # - Confidence >= 60.0%
        # Otherwise: 'needs_review'
        can_auto_respond = (
            not is_out_of_scope
            and not risk["requires_human"]
            and has_sufficient_context
            and is_valid
            and confidence >= 60.0
            and resolution_data.get("grounding_status") in ["grounded", "partially_grounded"]
        )

        if can_auto_respond:
            final_status = "awaiting_customer"
            requires_human = False
            escalation_reason = ""
        else:
            final_status = "needs_review"
            requires_human = True
            reasons = []
            if review_reason and not review_reason.lower().startswith("none"):
                reasons.append(review_reason)
            if risk["requires_human"]:
                reasons.append(risk["risk_reason"])
            if not has_sufficient_context:
                reasons.append(retrieval_result.get("rejection_reason", "Insufficient knowledge retrieved"))
            if not is_valid:
                reasons.extend(validation_issues)
            if resolution_data.get("grounding_status") == "ungrounded" and not reasons:
                reasons.append("Ungrounded AI response requires agent verification")
            escalation_reason = " | ".join(filter(None, reasons)) or "Requires human specialist review."

        # Update resolution_data with final decision
        resolution_data["requires_human_review"] = requires_human
        resolution_data["escalation_reason"] = escalation_reason
        resolution_data["suggested_ticket_status"] = final_status

        # 6. NeonDB Persistence
        # Format proposed solution as clean readable text
        proposed_steps = resolution_data.get("proposed_solution", [])
        if isinstance(proposed_steps, list):
            formatted_solution = "\n".join(f"{i+1}. {s}" for i, s in enumerate(proposed_steps))
        else:
            formatted_solution = str(proposed_steps)

        # Prepare source mappings
        source_records = []
        for doc in retrieved_docs:
            source_records.append({
                "chunk_id": doc.get("chunk_id"),
                "doc_id": doc.get("document_id"),
                "relevance_score": doc.get("relevance_score", 0.0)
            })

        try:
            res_id = TicketRepository.save_resolution(
                ticket_id=ticket_id,
                proposed_solution=formatted_solution,
                model_name=gen_result.get("model") or "gemini-2.5-flash",
                problem_summary=resolution_data.get("problem_summary", ""),
                identified_issue=resolution_data.get("identified_issue", ""),
                resolution_status=final_status,
                requires_human_review=requires_human,
                grounding_status=resolution_data.get("grounding_status", "ungrounded"),
                escalation_reason=escalation_reason,
                sources=source_records
            )
            logger.info(f"Saved ticket resolution #{res_id} in NeonDB for Ticket #{ticket_id} (Status: {final_status}).")
        except Exception as e:
            logger.error(f"Failed to persist ticket resolution in NeonDB: {e}")
            res_id = None

        # Update ticket status in tickets table and audit events
        try:
            TicketRepository.update_status(
                ticket_id=ticket_id,
                new_status=final_status,
                description=f"Automated resolution generated. Status: {final_status}. Reason: {escalation_reason or 'AI auto-response dispatched to customer.'}"
            )
        except Exception as e:
            logger.warning(f"Failed to update ticket status in database: {e}")

        return {
            "ticket_id": ticket_id,
            "resolution_id": res_id,
            "status": final_status,
            "requires_human_review": requires_human,
            "escalation_reason": escalation_reason,
            "resolution_data": resolution_data,
            "retrieved_documents": retrieved_docs,
            "latency_ms": gen_result.get("latency_ms", 0.0),
            "validation_passed": is_valid,
            "validation_issues": validation_issues
        }
