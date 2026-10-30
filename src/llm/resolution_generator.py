"""
TicketAI - Structured Resolution Generator
Coordinates Gemini LLM invocation with strict Pydantic schema validation,
grounding checks, and fallback resolution templates.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from src.llm.gemini_client import get_gemini_client, GeminiClient
from src.llm.prompt_templates import SYSTEM_RESOLUTION_INSTRUCTION, RESOLUTION_PROMPT_TEMPLATE

logger = logging.getLogger("ticketai.resolution_generator")


class TicketResolutionSchema(BaseModel):
    """Pydantic schema for structured AI ticket resolutions."""
    problem_summary: str = Field(description="Brief 1-2 sentence overview of the issue.")
    predicted_category: str = Field(description="Customer support category.")
    identified_issue: str = Field(description="Identified root cause or inquiry type.")
    proposed_solution: List[str] = Field(description="Step-by-step actionable resolution steps.")
    retrieved_sources: List[int] = Field(default_factory=list, description="List of supporting document IDs.")
    grounding_status: Literal["grounded", "partially_grounded", "ungrounded"] = Field(
        default="ungrounded",
        description="Factual grounding confidence relative to retrieved context."
    )
    requires_human_review: bool = Field(default=True, description="Whether an agent must inspect this before dispatch.")
    escalation_reason: str = Field(default="", description="Reason for escalation if human review required.")
    suggested_ticket_status: Literal["awaiting_customer", "needs_review", "pending_review"] = Field(
        default="needs_review",
        description="Initial workflow status for the ticket."
    )


class ResolutionGenerator:
    """Generates structured resolutions using Gemini and retrieved knowledge."""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client or get_gemini_client()

    def generate_resolution(
        self,
        ticket_id: int,
        customer_message: str,
        predicted_category: str,
        classification_confidence: float = 0.0,
        retrieved_documents: Optional[List[Dict[str, Any]]] = None,
        risk_assessment: str = "Standard low-risk inquiry",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a validated resolution. If Gemini is unavailable or errors,
        produces a deterministic, safe human-escalation payload.
        """
        retrieved_documents = retrieved_documents or []
        
        # Format retrieved context
        if retrieved_documents:
            context_blocks = []
            for doc in retrieved_documents:
                doc_id = doc.get("document_id") or doc.get("id") or "N/A"
                title = doc.get("title", "Untitled Document")
                category = doc.get("category", "General")
                text = doc.get("chunk_text") or doc.get("content", "")
                relevance = doc.get("relevance_score", 0.0)
                context_blocks.append(
                    f"--- Document ID: {doc_id} | Title: {title} | Category: {category} | Score: {relevance:.3f} ---\n{text}"
                )
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No authoritative knowledge base articles were retrieved for this query."

        prompt = RESOLUTION_PROMPT_TEMPLATE.format(
            ticket_id=ticket_id,
            customer_message=customer_message,
            predicted_category=predicted_category,
            classification_confidence=classification_confidence,
            risk_assessment=risk_assessment,
            retrieved_context=context_str
        )

        # If LLM client is unavailable or no API key, return safe fallback
        if not self.gemini_client or not self.gemini_client.is_available:
            logger.info("Gemini client unavailable; applying deterministic review fallback.")
            fallback_data = self._build_offline_fallback(
                ticket_id=ticket_id,
                customer_message=customer_message,
                predicted_category=predicted_category,
                retrieved_documents=retrieved_documents,
                reason="AI generation engine operating offline or credentials not configured."
            )
            return {
                "success": True,
                "data": fallback_data,
                "latency_ms": 0.0,
                "model": "offline-fallback",
                "usage": {}
            }

        try:
            call_result = self.gemini_client.generate_content(
                prompt=prompt,
                system_instruction=SYSTEM_RESOLUTION_INSTRUCTION,
                response_schema=TicketResolutionSchema,
                temperature=0.2
            )

            parsed: Optional[TicketResolutionSchema] = call_result.get("parsed")
            if parsed:
                res_dict = parsed.model_dump()
            else:
                # Attempt manual parse if parsed object not populated
                raw_text = call_result.get("text", "")
                res_dict = self._parse_json_fallback(raw_text, predicted_category)

            # Ensure retrieved_sources are mapped
            if not res_dict.get("retrieved_sources") and retrieved_documents:
                res_dict["retrieved_sources"] = [
                    int(d.get("document_id") or d.get("id"))
                    for d in retrieved_documents if (d.get("document_id") or d.get("id"))
                ]

            return {
                "success": True,
                "data": res_dict,
                "latency_ms": call_result.get("latency_ms", 0.0),
                "model": call_result.get("model", ""),
                "usage": call_result.get("usage", {})
            }

        except Exception as e:
            logger.error(f"Error during Gemini resolution generation: {e}")
            return {
                "success": False,
                "data": self._build_offline_fallback(
                    ticket_id=ticket_id,
                    customer_message=customer_message,
                    predicted_category=predicted_category,
                    retrieved_documents=retrieved_documents,
                    reason=f"LLM API generation error: {str(e)[:100]}"
                ),
                "error": str(e)
            }

    def _parse_json_fallback(self, raw_text: str, default_category: str) -> Dict[str, Any]:
        """Safely extracts JSON or constructs standard structured dict."""
        try:
            clean = raw_text.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            data = json.loads(clean.strip())
            return TicketResolutionSchema(**data).model_dump()
        except Exception:
            return {
                "problem_summary": "Customer inquiry received.",
                "predicted_category": default_category,
                "identified_issue": "General customer request",
                "proposed_solution": [
                    "We have received your ticket and routed it to our support specialists.",
                    "An agent will review the request and follow up shortly."
                ],
                "retrieved_sources": [],
                "grounding_status": "ungrounded",
                "requires_human_review": True,
                "escalation_reason": "Response formatting required human validation.",
                "suggested_ticket_status": "needs_review"
            }

    def _build_offline_fallback(
        self,
        ticket_id: int,
        customer_message: str,
        predicted_category: str,
        retrieved_documents: List[Dict[str, Any]],
        reason: str
    ) -> Dict[str, Any]:
        """Builds a deterministic, safe response for escalation when LLM is unavailable."""
        doc_ids = [int(d.get("document_id") or d.get("id")) for d in retrieved_documents if (d.get("document_id") or d.get("id"))]
        return {
            "problem_summary": f"Inquiry regarding {predicted_category}",
            "predicted_category": predicted_category,
            "identified_issue": f"{predicted_category} issue pending support evaluation",
            "proposed_solution": [
                f"Your request has been logged under #{ticket_id}.",
                "A support representative has been assigned to evaluate the details.",
                "You will receive an update once the ticket has been reviewed."
            ],
            "retrieved_sources": doc_ids,
            "grounding_status": "partially_grounded" if doc_ids else "ungrounded",
            "requires_human_review": True,
            "escalation_reason": reason,
            "suggested_ticket_status": "needs_review"
        }
