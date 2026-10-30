"""
TicketAI - Response Validator & Guardrails
Validates LLM-generated resolutions for safety, grounding compliance,
PII leakage, unauthorized action claims, and prompt injection attempts.
"""

import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("ticketai.validator")


class ResponseValidator:
    """Rigorous multi-layer validation engine for AI ticket resolutions."""

    # Prohibited claims where the LLM promises actions that require backend authorization
    PROHIBITED_ACTION_PATTERNS = [
        re.compile(r"(?i)\b(?:i have|we have|i've|we've)\s+(?:refunded|credited|charged|processed your payment|modified your account|reset your password|deleted your data)\b"),
        re.compile(r"(?i)\b(?:your refund of\s+[\$\€\£]?\d+|has been issued|has been refunded to your card)\b"),
        re.compile(r"(?i)\b(?:please send (?:us|me) your password|your credit card number|your pin)\b")
    ]

    # Prompt injection flags
    INJECTION_INDICATORS = [
        "ignore all previous",
        "system prompt",
        "developer mode",
        "unrestricted ai",
        "jailbreak",
        "dan mode"
    ]

    @classmethod
    def validate_resolution(
        cls,
        resolution_data: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
        customer_message: str
    ) -> Tuple[bool, List[str]]:
        """
        Validates resolution structure and semantic safety.
        Returns (is_valid: bool, issues: List[str]).
        """
        issues = []

        # 1. Structural checks
        if not resolution_data.get("problem_summary"):
            issues.append("Missing problem summary.")

        proposed_solution = resolution_data.get("proposed_solution", [])
        if not proposed_solution or not isinstance(proposed_solution, list):
            issues.append("Proposed solution must be a non-empty list of steps.")

        full_text = " ".join([
            resolution_data.get("problem_summary", ""),
            resolution_data.get("identified_issue", ""),
            " ".join(str(s) for s in proposed_solution)
        ])

        # 2. Check for unauthorized financial / administrative action claims
        for pattern in cls.PROHIBITED_ACTION_PATTERNS:
            if pattern.search(full_text):
                issues.append(
                    "AI response erroneously claims external action was completed (e.g. issued refund or modified account). Requires human agent review."
                )
                break

        # 3. Grounding vs Source check
        grounding = resolution_data.get("grounding_status", "ungrounded")
        sources = resolution_data.get("retrieved_sources", [])

        if grounding == "grounded" and not retrieved_documents:
            issues.append("Response claimed 'grounded' status but zero knowledge base documents were retrieved.")

        # 4. Check for prompt injection attempts in customer message or output
        msg_lower = (customer_message + " " + full_text).lower()
        for indicator in cls.INJECTION_INDICATORS:
            if indicator in msg_lower:
                issues.append(f"Potential prompt injection pattern detected: '{indicator}'. Escalating to human.")
                break

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(f"Resolution validation failed with issues: {issues}")
        return is_valid, issues

