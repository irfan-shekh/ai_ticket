"""
TicketAI - Human Escalation Routing Engine
Evaluates risk profiles, classification uncertainty, policy boundaries,
and triggers human handoffs with structured escalation reasons.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("ticketai.escalation")


class EscalationEngine:
    """Calculates escalation necessity and builds human agent triage summaries."""

    @staticmethod
    def assess_risk(
        ticket_text: str,
        predicted_category: str,
        confidence: float,
        is_out_of_scope: bool,
        needs_review: bool
    ) -> Dict[str, Any]:
        """
        Assesses operational risk of ticket based on text signals,
        confidence levels, and triage decision outputs.
        """
        text_lower = ticket_text.lower()
        high_risk_keywords = [
            "lawyer", "attorney", "sue", "legal action", "chargeback",
            "fraud", "stolen", "unauthorized charge", "police", "databreach", "compromised"
        ]

        is_legal_or_fraud = any(kw in text_lower for kw in high_risk_keywords)
        is_low_confidence = confidence < 60.0

        if is_legal_or_fraud:
            risk_level = "HIGH"
            reason = "High-risk terms detected (potential legal, fraud, or chargeback dispute)."
            requires_human = True
        elif is_out_of_scope:
            risk_level = "HIGH"
            reason = "Input classified as out-of-scope or non-standard customer request."
            requires_human = True
        elif is_low_confidence:
            risk_level = "MEDIUM"
            reason = f"Classification confidence ({confidence:.1f}%) is below automated threshold (60.0%)."
            requires_human = True
        elif needs_review:
            risk_level = "MEDIUM"
            reason = "Classification flagged for human verification by triage policy."
            requires_human = True
        elif predicted_category in ["Billing", "Complaint"]:
            risk_level = "MEDIUM"
            reason = f"Sensitive category ({predicted_category}) requires agent review for financial/satisfaction safety."
            # Billing and Complaint can still suggest AI response, but will flag for review if actions needed
            requires_human = False
        else:
            risk_level = "LOW"
            reason = "Standard informational or technical inquiry."
            requires_human = False

        return {
            "risk_level": risk_level,
            "requires_human": requires_human,
            "risk_reason": reason
        }

    @staticmethod
    def build_escalation_payload(
        ticket_id: int,
        primary_reason: str,
        validation_issues: List[str]
    ) -> Dict[str, Any]:
        """Formats structured human escalation record."""
        reasons = [primary_reason] if primary_reason else []
        reasons.extend(validation_issues)
        full_reason = " | ".join(reasons) if reasons else "Escalated for standard human agent review."

        logger.info(f"Ticket #{ticket_id} escalated: {full_reason}")
        return {
            "ticket_id": ticket_id,
            "requires_human_review": True,
            "suggested_ticket_status": "needs_review",
            "escalation_reason": full_reason
        }

