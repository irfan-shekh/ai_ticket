"""
TicketAI - Machine Learning Classifier Tests
Verifies that existing TF-IDF + LinearSVC model and TriageDecisionLayer
correctly classify all 5 categories, preserve preprocessing, and reject out-of-scope inputs.
"""

import unittest
import os
import sys

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.classifier.ticket_classifier import TicketClassifier, default_preprocess_text, SUPPORTED_CATEGORIES


class TestTicketClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = TicketClassifier.get_instance()

    def test_all_five_categories_supported(self):
        """Ensure all 5 categories are strictly preserved."""
        expected = {"Billing", "Complaint", "General Inquiry", "Returns", "Technical Support"}
        self.assertEqual(set(SUPPORTED_CATEGORIES), expected)
        model_classes = getattr(self.classifier.model, "classes_", None)
        if model_classes is not None:
            self.assertEqual(set(model_classes), expected)

    def test_preprocessing_pipeline(self):
        """Verify contractions, negations, URLs, and lowercasing."""
        raw = "I won't accept this! Check https://example.com/receipt or email support@test.com"
        processed = default_preprocess_text(raw)
        self.assertIn("not", processed, "Negations like won't -> will not must be preserved")
        self.assertIn("url", processed, "URLs should be normalized to token 'url'")
        self.assertIn("email", processed, "Emails should be normalized to token 'email'")
        self.assertNotIn("https", processed)

    def test_technical_support_classification(self):
        text = "I cannot log in to my account. Receiving a 500 internal server error on the dashboard."
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "Technical Support")
        self.assertGreaterEqual(res["confidence"], 60.0)
        self.assertFalse(res["is_out_of_scope"])

    def test_billing_classification(self):
        text = "I was double charged on my monthly subscription credit card statement."
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "Billing")
        self.assertGreaterEqual(res["confidence"], 60.0)
        self.assertFalse(res["is_out_of_scope"])

    def test_returns_classification(self):
        text = "The product arrived damaged with a broken screen. I want to return it for a refund."
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "Returns")
        self.assertGreaterEqual(res["confidence"], 60.0)
        self.assertFalse(res["is_out_of_scope"])

    def test_general_inquiry_classification(self):
        text = "What are your business operating hours and do you ship internationally to Canada?"
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "General Inquiry")
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertFalse(res["is_out_of_scope"])

    def test_complaint_classification(self):
        text = "Your customer service agent was extremely rude and unhelpful on the phone."
        res = self.classifier.predict(text)
        self.assertEqual(res["predicted_category"], "Complaint")
        self.assertGreaterEqual(res["confidence"], 60.0)
        self.assertFalse(res["is_out_of_scope"])

    def test_out_of_scope_rejection(self):
        """Irrelevant or gibberish messages must be flagged as out-of-scope."""
        gibberish = "asdfjkllqwer uioppzxcv"
        res = self.classifier.predict(gibberish)
        self.assertTrue(res["is_out_of_scope"] or res["needs_human_review"])


if __name__ == "__main__":
    unittest.main()
