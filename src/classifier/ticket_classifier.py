"""
TicketAI - Machine Learning Classifier Integration
Preserves the trained Calibrated LinearSVC + FeatureUnion (TF-IDF word & char_wb) model
and integrates the Triage Decision Layer for robust confidence and guardrails.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import joblib

logger = logging.getLogger("ticketai.classifier")

# 5 Authoritative Categories (Preserved strictly)
SUPPORTED_CATEGORIES = [
    "Billing",
    "Complaint",
    "General Inquiry",
    "Returns",
    "Technical Support"
]

CATEGORY_DESCRIPTIONS = {
    "Technical Support": "Issues related to login, access, software, or technical problems",
    "Returns": "Product returns, refunds, exchanges, and damaged items",
    "Billing": "Payment issues, invoices, billing questions, and subscription renewals",
    "General Inquiry": "General questions about products, services, hours, or company information",
    "Complaint": "Customer complaints regarding service quality, staff, or delayed delivery"
}


def default_preprocess_text(text: str) -> str:
    """Production preprocessing pipeline preserving contractions and negations."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)

    # Contractions
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "cannot", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # Filter special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()

    # Negation words to retain
    negation_words = {"not", "no", "never", "neither", "nor", "none", "cannot", "down", "without", "off"}
    common_stops = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are"}

    cleaned = [t for t in tokens if (t in negation_words or t not in common_stops) and len(t) > 1]
    return " ".join(cleaned)


class TicketClassifier:
    """Wrapper around enhanced_ticket_classifier.joblib and triage decision layer."""

    _instance: Optional["TicketClassifier"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._find_model_path()
        self.model = None
        self.triage_layer = None
        self.metadata = {}
        self._load_model()

    @classmethod
    def get_instance(cls) -> "TicketClassifier":
        if cls._instance is None:
            cls._instance = TicketClassifier()
        return cls._instance

    def _find_model_path(self) -> str:
        candidates = [
            os.path.join(os.getcwd(), "models", "enhanced_ticket_classifier.joblib"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "enhanced_ticket_classifier.joblib"),
            "models/enhanced_ticket_classifier.joblib"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded existing ML model from: {self.model_path}")
            else:
                logger.warning(f"ML Model file not found at {self.model_path}")

            # Load triage layer
            try:
                from models.triage_decision_layer import get_default_triage_layer
                self.triage_layer = get_default_triage_layer(min_confidence=60.0, min_margin=0.15)
                logger.info("Initialized Triage Decision Layer.")
            except Exception as e:
                logger.warning(f"Could not load custom triage layer: {e}")

            # Load metadata if exists
            meta_path = os.path.join(os.path.dirname(self.model_path), "enhanced_model_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies ticket text with calibrated confidence, out-of-scope check,
        and margin evaluation using the preserved LinearSVC model.
        """
        raw_text = str(text or "")[:5000].strip()

        if self.triage_layer and self.model:
            decision = self.triage_layer.decide(
                raw_text,
                primary_model=self.model,
                preprocessor_fn=default_preprocess_text
            )
            return decision

        # Direct model inference if triage layer unavailable
        if self.model:
            processed = default_preprocess_text(raw_text)
            pred_class = self.model.predict([processed])[0]
            confidence = 80.0
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([processed])[0]
                classes = list(self.model.classes_)
                confidence = float(probs[classes.index(pred_class)] * 100.0)

            return {
                "predicted_category": pred_class,
                "confidence": round(confidence, 2),
                "confidence_level": "High" if confidence >= 75 else "Medium",
                "needs_human_review": confidence < 60.0,
                "is_out_of_scope": False,
                "review_reason": "" if confidence >= 60.0 else "Low ML confidence",
                "margin": 0.20,
                "top_predictions": [{"category": pred_class, "confidence": round(confidence, 2)}]
            }

        # Fallback if no model file found
        return {
            "predicted_category": "General Inquiry",
            "confidence": 50.0,
            "confidence_level": "Low (Review Suggested)",
            "needs_human_review": True,
            "is_out_of_scope": False,
            "review_reason": "ML Model unavailable",
            "margin": 0.0,
            "top_predictions": [{"category": "General Inquiry", "confidence": 50.0}]
        }

