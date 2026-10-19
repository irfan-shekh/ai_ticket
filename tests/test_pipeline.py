"""
TicketAI Pipeline & Triage Unit Tests
Verifies primary classification pipeline, out-of-scope rejection,
short/informal valid tickets, boundary handling, and triage decision contracts.
"""

import unittest
import os
import sys
import numpy as np
import joblib

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.triage_decision_layer import TriageDecisionLayer, get_default_triage_layer, check_gibberish
from app import classify_ticket, preprocess_text, production_model


class TestTicketClassificationPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.triage = get_default_triage_layer()
        cls.model = production_model

    def test_production_model_loaded(self):
        """Ensure primary LinearSVC model is loaded and has 5 classes."""
        self.assertIsNotNone(self.model, "Production model should be loaded successfully.")
        expected_classes = {'Billing', 'Complaint', 'General Inquiry', 'Returns', 'Technical Support'}
        self.assertEqual(set(self.model.classes_), expected_classes)

    def test_valid_technical_support_ticket(self):
        text = "My account login is not working, getting a 500 internal server error when accessing the dashboard."
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'Technical Support')
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreaterEqual(res['confidence'], 60.0)

    def test_valid_billing_ticket(self):
        text = "I was double charged on my credit card invoice for this month's subscription payment."
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'Billing')
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreaterEqual(res['confidence'], 60.0)

    def test_valid_returns_ticket(self):
        text = "The package arrived with a broken and damaged screen, I need to return it for a full refund."
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'Returns')
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreaterEqual(res['confidence'], 60.0)

    def test_valid_general_inquiry_ticket(self):
        text = "What are your business operating hours and do you ship internationally to Canada?"
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'General Inquiry')
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreaterEqual(res['confidence'], 50.0)

    def test_business_hours_inquiry(self):
        """Ensure concise informational questions like business hours route to General Inquiry."""
        text = "What are your business hours?"
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'General Inquiry')
        self.assertFalse(res['is_out_of_scope'])
        self.assertFalse(res['needs_human_review'])
        self.assertGreaterEqual(res['confidence'], 70.0)

    def test_valid_complaint_ticket(self):
        text = "Your customer service representative was extremely rude and unhelpful on the phone."
        res = classify_ticket(text, return_structured=True)
        self.assertEqual(res['predicted_category'], 'Complaint')
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreaterEqual(res['confidence'], 60.0)

    def test_short_valid_ticket_not_rejected(self):
        """Short tickets with clear customer intent must NOT be rejected as out of scope."""
        short_tickets = [
            ("Where is my refund?", ["Returns"]),
            ("Cancel subscription please", ["Billing", "Complaint"]),
            ("Password reset failed", ["Technical Support"])
        ]
        for text, expected_cats in short_tickets:
            res = classify_ticket(text, return_structured=True)
            self.assertFalse(
                res['is_out_of_scope'],
                f"Valid short ticket '{text}' should not be marked out-of-scope"
            )
            self.assertIn(
                res['predicted_category'], expected_cats,
                f"Ticket '{text}' predicted as {res['predicted_category']}, expected one of {expected_cats}"
            )

    def test_non_technical_complaint_grounding(self):
        """Food, service, and dining complaints must route to Complaint, never Technical Support."""
        samples = [
            "the food is not good",
            "the waiter was extremely rude and the meal was cold"
        ]
        for s in samples:
            res = classify_ticket(s, return_structured=True)
            self.assertEqual(res['predicted_category'], 'Complaint', f"'{s}' should be Complaint")

    def test_out_of_scope_rejection(self):
        """Unrelated topics must be flagged as out-of-scope and routed for human review."""
        unrelated_prompts = [
            "good morning",
            "how to cook rice",
            "What is the capital city of Australia?",
            "Can you give me a recipe for chocolate chip cookies?",
            "Write a python function to sort an array using quicksort",
            "Who won the 2022 FIFA World Cup in Qatar?",
            "What is your favorite color and are you an AI?",
            "The hotel room had a dirty bathroom and the flight was delayed"
        ]
        for prompt in unrelated_prompts:
            res = classify_ticket(prompt, return_structured=True)
            self.assertTrue(
                res['is_out_of_scope'],
                f"Prompt '{prompt}' should be recognized as out-of-scope"
            )
            self.assertTrue(res['needs_human_review'])
            self.assertEqual(res['predicted_category'], 'Out of Scope')
            self.assertLess(res['confidence'], 60.0)

    def test_gibberish_and_noise_detection(self):
        """Pure noise and keyboard mashing must be trapped as out-of-scope."""
        gibberish_inputs = [
            "asdfghjklzxcvbnmqwertyuiop12345",
            "12345678901234567890",
            "!@#$%^&*()_+{}[]:;<>",
            "zzzzzzzzzzzzzzzzzzzzzz"
        ]
        for gib in gibberish_inputs:
            self.assertTrue(check_gibberish(gib), f"'{gib}' should be detected as gibberish")
            res = classify_ticket(gib, return_structured=True)
            self.assertTrue(res['is_out_of_scope'])
            self.assertTrue(res['needs_human_review'])

    def test_empty_and_whitespace_input(self):
        """Empty, None, or pure whitespace must be safely handled without exceptions."""
        for val in ["", "   ", "\n\t", None]:
            res = classify_ticket(val, return_structured=True)
            self.assertTrue(res['is_out_of_scope'])
            self.assertTrue(res['needs_human_review'])
            self.assertIn('review_reason', res)

    def test_very_long_input_safety(self):
        """Inputs longer than 5,000 characters must truncate gracefully without crashing."""
        long_text = "I need help with my account password reset. " * 300
        res = classify_ticket(long_text, return_structured=True)
        self.assertIsNotNone(res['predicted_category'])
        self.assertFalse(res['is_out_of_scope'])

    def test_api_contract_structure(self):
        """Verify presence and types of all required contract fields."""
        text = "I want to exchange my shirt for a larger size."
        res = classify_ticket(text, return_structured=True)
        required_keys = [
            'predicted_category', 'confidence', 'confidence_level',
            'needs_human_review', 'is_out_of_scope', 'review_reason',
            'margin', 'top_predictions', 'domain_signals'
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Response missing required contract key '{key}'")
            
        self.assertIsInstance(res['confidence'], float)
        self.assertIsInstance(res['needs_human_review'], bool)
        self.assertIsInstance(res['is_out_of_scope'], bool)
        self.assertIsInstance(res['top_predictions'], list)
        self.assertGreaterEqual(len(res['top_predictions']), 1)
    def test_ambiguity_flagging(self):
        """Ambiguous or borderline inputs must trigger needs_human_review = True."""
        ambiguous_text = "I am having an issue with something"
        res = classify_ticket(ambiguous_text, return_structured=True)
        self.assertTrue(res['needs_human_review'], "Ambiguous text should be routed to human review")

    def test_flask_classify_endpoint(self):
        """Verify the Flask /classify HTTP route with an authenticated session."""
        from app import app
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['username'] = 'test_user'
                sess['is_admin'] = False

            resp = client.post('/classify', json={
                'ticket_text': 'I need a refund for damaged item'
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertIn('category', data)
            self.assertIn('is_out_of_scope', data)
            self.assertIn('needs_human_review', data)
            self.assertIn('top_categories', data)

    def test_flask_batch_classify_endpoint(self):
        """Verify the Flask /batch_classify HTTP route."""
        from app import app
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['username'] = 'test_user'
                sess['is_admin'] = False

            batch_payload = {
                'tickets': [
                    {'text': 'My screen is black and laptop crashes'},
                    {'text': 'Double charge on my credit card invoice'},
                    {'text': 'Where is the moon located in space?'}
                ]
            }
            resp = client.post('/batch_classify', json=batch_payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['total'], 3)
            self.assertEqual(len(data['results']), 3)
            # The third ticket should be flagged out of scope
            self.assertTrue(data['results'][2]['is_out_of_scope'])


if __name__ == '__main__':
    unittest.main()

