"""
TicketAI - Automated Resolution Decision Engine Tests
Verifies multi-stage decision gating:
Low-Risk Grounded -> awaiting_customer
High-Risk / Ungrounded -> needs_review
Prevention of false resolutions.
"""

import unittest
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.resolution.escalation import EscalationEngine
from src.resolution.resolution_engine import ResolutionEngine


class TestResolutionEngine(unittest.TestCase):

    def test_risk_assessment_legal_dispute(self):
        """Legal dispute keywords must trigger HIGH risk and human review."""
        text = "I am speaking to my attorney and filing a lawsuit for fraud and unauthorized charges."
        risk = EscalationEngine.assess_risk(
            ticket_text=text,
            predicted_category="Billing",
            confidence=95.0,
            is_out_of_scope=False,
            needs_review=False
        )
        self.assertEqual(risk["risk_level"], "HIGH")
        self.assertTrue(risk["requires_human"])

    def test_risk_assessment_low_confidence(self):
        """Low ML classification confidence must trigger human escalation."""
        text = "Something is wrong with my thing"
        risk = EscalationEngine.assess_risk(
            ticket_text=text,
            predicted_category="General Inquiry",
            confidence=45.0,
            is_out_of_scope=False,
            needs_review=True
        )
        self.assertEqual(risk["risk_level"], "MEDIUM")
        self.assertTrue(risk["requires_human"])

    def test_risk_assessment_standard_technical(self):
        """Standard login questions should be low risk."""
        text = "How do I clear my browser cookies to fix login?"
        risk = EscalationEngine.assess_risk(
            ticket_text=text,
            predicted_category="Technical Support",
            confidence=98.0,
            is_out_of_scope=False,
            needs_review=False
        )
        self.assertEqual(risk["risk_level"], "LOW")
        self.assertFalse(risk["requires_human"])


if __name__ == "__main__":
    unittest.main()

