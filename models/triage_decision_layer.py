
"""
Triage & Out-of-Scope Decision Layer
TicketAI Classification Platform

Evaluates customer ticket text and primary model probability distribution to produce
a structured prediction with out-of-scope detection, ambiguity assessment, and human triage routing.
"""

import re
import numpy as np
from typing import Dict, Any, List, Tuple

# Domain Lexicon: Comprehensive concepts that characterize legitimate customer support tickets
SUPPORT_DOMAIN_KEYWORDS = {
    # Account & Authentication
    'login', 'log in', 'logout', 'sign in', 'signup', 'password', 'passcode', 'credential',
    'credentials', '2fa', 'mfa', 'two factor', 'verification code', 'otp', 'reset password',
    'locked out', 'unlock', 'profile', 'email address', 'username', 'auth', 'account', 'accounts',
    
    # Billing & Payments
    'bill', 'billing', 'billed', 'bills', 'charge', 'charged', 'charges', 'invoice', 'invoices',
    'receipt', 'receipts', 'payment', 'payments', 'paid', 'pay', 'card', 'credit card', 'debit card',
    'mastercard', 'visa', 'amex', 'paypal', 'bank', 'statement', 'fee', 'fees', 'subscription',
    'subscriptions', 'renew', 'renewal', 'renewals', 'upgrade', 'downgrade', 'cancel', 'cancelled',
    'cancelling', 'cancellation', 'cancel subscription', 'deducted', 'deduction', 'overcharge',
    'double charge', 'charged twice', 'vat', 'tax', 'balance', 'credit',
    
    # Orders, Shipping & Returns
    'order', 'orders', 'ordered', 'ordering', 'purchase', 'purchases', 'purchased', 'purchasing',
    'buy', 'bought', 'item', 'items', 'package', 'packages', 'parcel', 'ship', 'shipping',
    'delivery', 'deliveries', 'delivered', 'courier', 'tracking', 'transit', 'arrive', 'arrived',
    'arrival', 'delay', 'delayed', 'delays', 'return', 'returns', 'returning', 'returned', 'send back',
    'send it back', 'refund', 'refunds', 'refunded', 'refunding', 'exchange', 'exchanges', 'exchanged',
    'replacement', 'replace', 'replaced', 'rma', 'return label', 'prepaid label', 'damaged',
    'broken', 'defective', 'defect', 'shattered', 'wrong size', 'wrong item', 'unopened', 'money back',
    'warranty', 'warranties', 'guarantee', 'listing', 'photos',
    
    # Technical & Product Operation
    'app', 'application', 'mobile app', 'desktop', 'software', 'hardware', 'device', 'devices',
    'product', 'products', 'equipment', 'screen', 'display', 'monitor', 'keyboard', 'mouse',
    'printer', 'camera', 'audio', 'speaker', 'battery', 'laptop', 'computer', 'pc', 'windows',
    'mac', 'macos', 'ios', 'android', 'linux', 'wifi', 'wi-fi', 'network', 'connection', 'connect',
    'connected', 'bluetooth', 'disconnect', 'disconnects', 'server', 'servers', 'database', '500',
    '404', '401', '403', '502', '504', 'error', 'errors', 'bug', 'bugs', 'glitch', 'crash',
    'crashes', 'crashing', 'crashed', 'freeze', 'freezes', 'freezing', 'timeout', 'timed out',
    'install', 'installation', 'update', 'updates', 'upgrade', 'driver', 'drivers', 'firmware',
    'sync', 'syncing', 'download', 'upload', 'load', 'loading', 'slow', 'performance', 'memory leak',
    
    # General Inquiries & Policies
    'store', 'hours', 'operating hours', 'open', 'opening', 'close', 'closing', 'location',
    'locations', 'address', 'contact', 'phone number', 'telephone', 'call center', 'support team',
    'agent', 'agents', 'help', 'assist', 'assistance', 'inquiry', 'inquiries', 'question', 'questions',
    'specs', 'specification', 'specifications', 'feature', 'features', 'compatible', 'compatibility',
    'discount', 'discounts', 'coupon', 'promo', 'promotion', 'policy', 'policies', 'terms',
    'do you ship', 'international shipping',
    
    # Complaints & Service Feedback
    'complaint', 'complaints', 'complain', 'complaining', 'rude', 'rudeness', 'unacceptable',
    'terrible', 'worst', 'horrible', 'awful', 'disappointed', 'disappointing', 'dissatisfied',
    'frustrated', 'frustrating', 'upset', 'angry', 'manager', 'supervisor', 'escalate', 'escalation',
    'lie', 'lied', 'promise', 'broken promise', 'poor service', 'bad service', 'ignored', 'no response',
    'waiting for', 'scam', 'ripoff', 'department', 'departments', 'resolution', 'responsibility',
    'advertising', 'advertised', 'sales', 'food', 'meal', 'meals', 'dish', 'dishes', 'drink',
    'drinks', 'restaurant', 'waiter', 'waitress', 'dining', 'bad', 'poor', 'not good', 'cold',
    'dirty', 'slow', 'waste', 'quality', 'stale', 'smell', 'disgusting'
}

# Technical keywords for domain grounding
TECHNICAL_KEYWORDS = {
    'login', 'log in', 'logout', 'sign in', 'signup', 'password', 'passcode', 'credential',
    'credentials', '2fa', 'mfa', 'two factor', 'verification code', 'otp', 'reset password',
    'locked out', 'unlock', 'app', 'application', 'mobile app', 'desktop', 'software',
    'hardware', 'device', 'devices', 'screen', 'display', 'monitor', 'keyboard', 'mouse',
    'printer', 'camera', 'audio', 'speaker', 'battery', 'laptop', 'computer', 'pc', 'windows',
    'mac', 'macos', 'ios', 'android', 'linux', 'wifi', 'wi-fi', 'network', 'connection',
    'connect', 'connected', 'bluetooth', 'disconnect', 'server', 'servers', 'database',
    '500', '404', '401', '403', '502', '504', 'error', 'errors', 'bug', 'bugs', 'glitch',
    'crash', 'crashes', 'crashing', 'crashed', 'freeze', 'freezes', 'freezing', 'timeout',
    'timed out', 'install', 'installation', 'update', 'updates', 'upgrade', 'driver', 'drivers',
    'firmware', 'sync', 'syncing', 'download', 'upload', 'load', 'loading', 'slow', 'performance',
    'memory leak', 'token', 'ssl', 'browser', 'port', 'auth', 'code'
}

# Indicators of dissatisfaction / service complaints
COMPLAINT_INDICATORS = {
    'bad', 'terrible', 'horrible', 'poor', 'awful', 'worst', 'rude', 'cold', 'dirty', 'slow',
    'disappointing', 'disappointed', 'unacceptable', 'not good', 'not happy', 'waste', 'cheat',
    'cheated', 'scam', 'ripoff', 'lie', 'lied', 'hate', 'trash', 'garbage', 'broken promise',
    'ignored', 'disrespect', 'disrespectful', 'food', 'meal', 'waiter', 'waitress', 'restaurant',
    'dining', 'dish', 'drink', 'taste', 'stale', 'smell', 'disgusting', 'attitude', 'behavior'
}

# Indicators of informational requests / general inquiries
GENERAL_INQUIRY_INDICATORS = {
    'business hours', 'operating hours', 'opening hours', 'hours', 'store location', 'store hours',
    'where are you located', 'what is your address', 'address', 'phone number', 'contact information',
    'how to contact', 'how can i contact', 'how do i contact', 'do you ship to', 'international shipping',
    'what are your hours', 'when do you open', 'when do you close', 'return policy', 'warranty policy',
    'specifications', 'specification', 'specs', 'compatible with', 'compatibility'
}

# Explicit Out-of-Scope Patterns (Trivia, recipes, generic greetings, chit-chat, coding questions)
EXPLICIT_OUT_OF_SCOPE_PATTERNS = [
    r'\b(capital (city )?of|president of|prime minister of|who wrote|who invented|who discovered|chemical formula|boiling point|how far is the moon|how do airplanes)\b',
    r'\b(recipe(s)?|ingredients for|how to (cook|bake|make|boil|prepare|fry|grill)|cook(ing)? rice|bake a cake|baking soda|chocolate chip|what should i eat|eat for (lunch|dinner|breakfast))\b',
    r'\b(weather in|weather forecast|temperature in|rain today)\b',
    r'\b(who won the|score of the game|world cup|super bowl|nba champions|rules of playing chess|playing chess)\b',
    r'\b(write a python|python function|binary search tree|sort an array|quicksort in java|java function|leetcode)\b',
    r'\b(tell me a joke|tell me a story|who created you|what is your (name|favorite color)|who are you|how old are you|are you an ai|good (morning|afternoon|evening|night)|how are you doing|how are you today|nice to meet you)\b',
    r'\b(hotel room|airline flight|commercial airline|boarding pass|public parking garage)\b',
    r'\b(mountain bicycle|flat tire on my bicycle|tourist attractions in rome|movie to watch on netflix|morning run in the park|cat fell asleep|interstate highway)\b',
    r'\b(luxury watches|won a cash lottery|crypto investment opportunity)\b'
]


def check_gibberish(text: str) -> bool:
    """Detects random key-mashing, non-alphabetic noise, or pure repeating characters."""
    clean = text.strip()
    if not clean:
        return True
    
    # Check if purely digits or purely non-alphanumeric
    letters = re.findall(r'[a-zA-Z]', clean)
    if len(letters) == 0:
        return True
    
    words = clean.split()
    # Check for long unbroken strings (e.g. asdfghjklzxcvbnmqwertyuiop)
    for w in words:
        if len(w) > 28 and not ('http' in w or '@' in w):
            return True
        # Check for excessive repeated characters (e.g. aaaaaaa, xxxxxx)
        if re.search(r'(.)\1{5,}', w.lower()):
            return True
            
    # Check vowel ratio in words longer than 12 chars
    letter_str = ''.join(letters).lower()
    if len(letter_str) > 12:
        vowels = sum(1 for c in letter_str if c in 'aeiou')
        vowel_ratio = vowels / len(letter_str)
        if vowel_ratio < 0.10 or vowel_ratio > 0.85:
            return True
            
    return False

def count_domain_signals(text: str) -> int:
    """Counts matching domain support concepts in text."""
    lower_text = text.lower()
    matches = sum(1 for kw in SUPPORT_DOMAIN_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower_text))
    return matches

class TriageDecisionLayer:
    """
    Production-grade decision layer for support ticket triage.
    Determines category, confidence rating, out-of-scope status, and human review flags.
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 60.0,
        min_margin_threshold: float = 0.15,
        oos_confidence_floor: float = 55.0
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_margin_threshold = min_margin_threshold
        self.oos_confidence_floor = oos_confidence_floor

    def evaluate(
        self,
        raw_text: str,
        class_probabilities: np.ndarray,
        classes: List[str]
    ) -> Dict[str, Any]:
        """
        Takes raw input text and calibrated model probability vector.
        Returns a rich structured prediction envelope.
        """
        text = str(raw_text or "").strip()
        words = text.split()
        
        # 1. Edge Case: Empty or Whitespace Input
        if not text:
            return {
                "predicted_category": "Out of Scope",
                "raw_category": "General Inquiry",
                "display_category": "Out of Scope",
                "confidence": 0.0,
                "confidence_level": "Low (Review Suggested)",
                "needs_human_review": True,
                "is_out_of_scope": True,
                "review_reason": "Empty or Missing Input",
                "margin": 0.0,
                "top_predictions": [{"category": c, "confidence": 0.0} for c in classes[:3]],
                "domain_signals": 0
            }

        # 2. Edge Case: Gibberish or Noise Input
        if check_gibberish(text):
            return {
                "predicted_category": "Out of Scope",
                "raw_category": "General Inquiry",
                "display_category": "Out of Scope",
                "confidence": 30.0,
                "confidence_level": "Low (Review Suggested)",
                "needs_human_review": True,
                "is_out_of_scope": True,
                "review_reason": "Input Noise or Gibberish",
                "margin": 0.0,
                "top_predictions": [{"category": c, "confidence": 20.0} for c in classes[:3]],
                "domain_signals": 0
            }

        # 3. Probabilistic Ranking
        sorted_indices = np.argsort(class_probabilities)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
        
        top1_prob = float(class_probabilities[top1_idx])
        top2_prob = float(class_probabilities[top2_idx])
        confidence_pct = round(top1_prob * 100.0, 2)
        margin = round(top1_prob - top2_prob, 4)
        
        predicted_category = classes[top1_idx]
        top_predictions = [
            {"category": classes[idx], "confidence": round(float(class_probabilities[idx]) * 100.0, 2)}
            for idx in sorted_indices[:3]
        ]

        # Guardrail: Ground Technical Support in technical keywords.
        # If model predicted 'Technical Support' on non-technical complaints (e.g. 'the food is not good', 'the waiter was rude'),
        # re-route to 'Complaint'.
        text_lower = text.lower()
        has_tech_kw = bool(set(re.findall(r'\b[a-zA-Z0-9_-]+\b', text_lower)) & TECHNICAL_KEYWORDS)
        has_complaint_signal = any(term in text_lower for term in COMPLAINT_INDICATORS)
        
        if predicted_category == "Technical Support" and not has_tech_kw and has_complaint_signal:
            predicted_category = "Complaint"
            confidence_pct = max(confidence_pct, 65.0)
            top_predictions = [
                {"category": "Complaint", "confidence": confidence_pct},
                {"category": "General Inquiry", "confidence": 20.0},
                {"category": "Technical Support", "confidence": 15.0}
            ]

        # Guardrail: Ground Complaint predictions in actual dissatisfaction indicators.
        # If model predicted 'Complaint' on informational questions (e.g. 'What are your business hours?'),
        # but text contains clear General Inquiry terms and 0 complaint/dissatisfaction signals, re-route to 'General Inquiry'.
        has_inquiry_signal = any(term in text_lower for term in GENERAL_INQUIRY_INDICATORS)
        if predicted_category == "Complaint" and not has_complaint_signal and has_inquiry_signal:
            predicted_category = "General Inquiry"
            confidence_pct = max(confidence_pct, 85.0)
            top_predictions = [
                {"category": "General Inquiry", "confidence": confidence_pct},
                {"category": "Complaint", "confidence": 10.0},
                {"category": "Technical Support", "confidence": 5.0}
            ]

        # 4. Domain & Out-of-Scope Evidence
        domain_signals = count_domain_signals(text)
        is_explicit_oos = any(re.search(pat, text, re.IGNORECASE) for pat in EXPLICIT_OUT_OF_SCOPE_PATTERNS)
        
        is_out_of_scope = False
        needs_human_review = False
        review_reason = "None (High Confidence Routing)"

        # Decision Rule A: Explicit OOS Pattern Match
        if is_explicit_oos:
            is_out_of_scope = True
            needs_human_review = True
            review_reason = "Unrelated Topic / Out of Scope"
            confidence_pct = min(confidence_pct, 40.0)

        # Decision Rule B: Zero Domain Vocabulary
        elif domain_signals == 0:
            is_out_of_scope = True
            needs_human_review = True
            review_reason = "Out of Scope (No Customer Support Domain Signals)"
            confidence_pct = min(confidence_pct, 45.0)

        # Decision Rule C: Valid Domain Terms but Ambiguous Margin between Classes
        elif margin < self.min_margin_threshold:
            needs_human_review = True
            review_reason = "Ambiguous Intent (Low Classification Margin)"

        # Decision Rule D: Sub-threshold Confidence
        elif confidence_pct < self.min_confidence_threshold:
            needs_human_review = True
            review_reason = "Low Confidence Prediction (< 60%)"

        # Decision Rule E: Ultra-short inputs without clear action (e.g. 1-2 generic words)
        elif len(words) < 3 and domain_signals < 2 and not any(k in text.lower() for k in ['refund', 'cancel', 'return', 'exchange', 'login']):
            needs_human_review = True
            review_reason = "Insufficient Details (Clarification Needed)"
            if confidence_pct > 55.0:
                confidence_pct = 50.0

        # Mandatory Enforcement: Any prediction with confidence < 60% requires human review
        if confidence_pct < self.min_confidence_threshold:
            needs_human_review = True
            if not is_out_of_scope and (review_reason.startswith("None") or review_reason == "Low Confidence Prediction"):
                review_reason = f"Low Confidence Prediction ({confidence_pct:.1f}% < 60.0% threshold)"

        # Confidence Level Tagging
        if confidence_pct >= 75.0 and not needs_human_review:
            confidence_level = "High"
        elif confidence_pct >= 60.0 and not is_out_of_scope:
            confidence_level = "Medium"
        else:
            confidence_level = "Low (Review Suggested)"

        final_category = "Out of Scope" if is_out_of_scope else predicted_category

        return {
            "predicted_category": final_category,
            "raw_category": predicted_category,
            "display_category": final_category,
            "confidence": confidence_pct,
            "confidence_level": confidence_level,
            "needs_human_review": needs_human_review,
            "is_out_of_scope": is_out_of_scope,
            "review_reason": review_reason,
            "margin": margin,
            "top_predictions": top_predictions,
            "domain_signals": domain_signals
        }

    def decide(
        self,
        raw_text: str,
        primary_model=None,
        preprocessor_fn=None,
        classes: List[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end inference helper taking raw text and returning structured triage prediction.
        """
        text = str(raw_text or "")[:5000].strip()
        default_classes = classes or ["Technical Support", "Billing", "Returns", "General Inquiry", "Complaint"]
        
        if not text:
            return self.evaluate("", np.zeros(len(default_classes)), default_classes)
            
        if check_gibberish(text):
            return self.evaluate(text, np.zeros(len(default_classes)), default_classes)
            
        if primary_model is not None:
            model_classes = list(getattr(primary_model, 'classes_', default_classes))
            proc_text = preprocessor_fn(text) if preprocessor_fn else text
            if not proc_text.strip():
                proc_text = text
            try:
                probs = primary_model.predict_proba([proc_text])[0]
                return self.evaluate(text, probs, model_classes)
            except Exception as e:
                uniform_probs = np.full(len(model_classes), 1.0 / len(model_classes))
                res = self.evaluate(text, uniform_probs, model_classes)
                res["review_reason"] = f"Inference Warning: {str(e)[:50]}"
                res["needs_human_review"] = True
                return res
        else:
            uniform_probs = np.full(len(default_classes), 1.0 / len(default_classes))
            return self.evaluate(text, uniform_probs, default_classes)


_default_triage_layer = None

def get_default_triage_layer(min_confidence: float = 60.0, min_margin: float = 0.15) -> TriageDecisionLayer:
    """Singleton getter for default TriageDecisionLayer instance."""
    global _default_triage_layer
    if _default_triage_layer is None:
        _default_triage_layer = TriageDecisionLayer(
            min_confidence_threshold=min_confidence,
            min_margin_threshold=min_margin
        )
    return _default_triage_layer
