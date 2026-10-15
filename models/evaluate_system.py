"""
Comprehensive Reproducible Model & Triage Evaluation Benchmark
TicketAI Customer Support Classification Platform

Compares:
  A. Primary Classifier Alone (TF-IDF + Calibrated LinearSVC)
  B. Primary Classifier with DistilBERT Fallback (Legacy app.py logic)
  C. Primary Classifier with Improved Triage Decision Layer (Production Candidate)

Reports:
  - Accuracy and Macro-F1 for valid tickets
  - Per-class precision, recall, F1
  - Confusion matrix
  - Out-of-scope rejection precision, recall, F1
  - False rejection rate (valid tickets incorrectly rejected)
  - False acceptance rate (unrelated messages admitted as valid tickets)
  - Human review rate
  - Average, p50, p95, p99 inference latency
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.triage_decision_layer import TriageDecisionLayer

# -----------------------------------------------------------------------------
# 1. Standardized Untouched Benchmark Suite (80 Samples)
# Provenance: 50 Human-phrased in-scope tickets + 30 diverse out-of-scope messages
# -----------------------------------------------------------------------------
BENCHMARK_DATA = [
    # --- TECHNICAL SUPPORT (10) ---
    {"text": "My database server is throwing 500 internal server error and connection dropped", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "The mobile app keeps crashing immediately upon launch on iOS 18", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Cannot connect to the VPN server, receiving authentication failed error 691", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Bluetooth mouse keeps disconnecting every few minutes from my Windows 11 laptop", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Two-factor authentication 2FA SMS code is never delivered to my registered phone", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Database query timeout occurs whenever generating large monthly analytical reports", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Camera hardware is not detected by video conferencing software after driver update", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Printer driver incompatibility error prevents document printing on MacOS Sequoia", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "Application memory leak causes total CPU utilization to spike to 100 percent", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "SSL certificate expired error warning appears on the secure checkout portal", "ground_truth": "Technical Support", "is_oos": False},

    # --- BILLING (10) ---
    {"text": "I was charged twice on my credit card for invoice INV-4820", "ground_truth": "Billing", "is_oos": False},
    {"text": "Why was there an unexpected 25 dollar charge on my monthly subscription bill?", "ground_truth": "Billing", "is_oos": False},
    {"text": "Need the official VAT receipt and tax invoice for our corporate accounting records", "ground_truth": "Billing", "is_oos": False},
    {"text": "Payment failed at checkout but funds were deducted from my checking account", "ground_truth": "Billing", "is_oos": False},
    {"text": "My debit card on file expired and subscription cancelled, need to update billing info", "ground_truth": "Billing", "is_oos": False},
    {"text": "I noticed an unauthorized charge of 120 dollars from your company on my statement", "ground_truth": "Billing", "is_oos": False},
    {"text": "Can you explain why my recurring subscription renewal fee is higher this billing cycle?", "ground_truth": "Billing", "is_oos": False},
    {"text": "I requested cancellation 10 days ago but my account was still billed for this month", "ground_truth": "Billing", "is_oos": False},
    {"text": "Need clarification regarding the foreign transaction fee listed on our latest bill", "ground_truth": "Billing", "is_oos": False},
    {"text": "Double debit on my bank statement for the exact same order number", "ground_truth": "Billing", "is_oos": False},

    # --- RETURNS (10) ---
    {"text": "Item arrived with broken glass and shattered display, how do I return it for a refund?", "ground_truth": "Returns", "is_oos": False},
    {"text": "Received the wrong shoe size in the mail, need to exchange size 9 for size 10", "ground_truth": "Returns", "is_oos": False},
    {"text": "How do I print the prepaid UPS return shipping label for my RMA request?", "ground_truth": "Returns", "is_oos": False},
    {"text": "I would like to return an unopened product within the 30-day money-back guarantee window", "ground_truth": "Returns", "is_oos": False},
    {"text": "Returned my defective item two weeks ago with tracking 1Z999 but refund has not credited yet", "ground_truth": "Returns", "is_oos": False},
    {"text": "Defective electronics right out of the box, requesting immediate replacement or full money back", "ground_truth": "Returns", "is_oos": False},
    {"text": "The jacket color looks completely different from the website photos, want to send it back", "ground_truth": "Returns", "is_oos": False},
    {"text": "Need to initiate an exchange for a defective power adapter that arrived non-functional", "ground_truth": "Returns", "is_oos": False},
    {"text": "Can I return items purchased during the seasonal promotion for store credit?", "ground_truth": "Returns", "is_oos": False},
    {"text": "Received a refurbished unit instead of the brand new retail box I paid for, need a return", "ground_truth": "Returns", "is_oos": False},

    # --- GENERAL INQUIRY (10) ---
    {"text": "What time does your retail flagship store open on Saturdays and what is your customer phone number?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Do you offer international shipping to Australia and what are the carrier delivery rates?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Is this external monitor compatible with MacBook Pro M3 or only Windows laptops?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Can you provide more information regarding product technical specifications and battery lifespan?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Do you offer academic or military service discounts on new equipment purchases?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Where on your website can I track package shipping milestones and carrier updates?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "What are your customer support call center operating hours during federal holidays?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Does purchasing this hardware include the standard manufacturer 2-year warranty?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "Can multiple user profiles be connected simultaneously to the same smart speaker device?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "What are the dimensions and physical weight of the packaged delivery box?", "ground_truth": "General Inquiry", "is_oos": False},

    # --- COMPLAINT (10) ---
    {"text": "Your representative was terribly rude and hung up the telephone on me, I demand a manager", "ground_truth": "Complaint", "is_oos": False},
    {"text": "Paid for 2-day express shipping and waited 18 business days with zero status updates, unacceptable", "ground_truth": "Complaint", "is_oos": False},
    {"text": "I have been transferred between 6 different departments without resolution, nobody takes responsibility", "ground_truth": "Complaint", "is_oos": False},
    {"text": "Worst customer support experience of my life, submitted 4 tickets and received only canned copy-paste bots", "ground_truth": "Complaint", "is_oos": False},
    {"text": "Delivery carrier literally threw the fragile electronic parcel onto my concrete porch steps", "ground_truth": "Complaint", "is_oos": False},
    {"text": "Your software update completely broke my production system and your support team does not seem to care", "ground_truth": "Complaint", "is_oos": False},
    {"text": "I was promised a management callback within 24 hours and have been waiting over a week, disgusting treatment", "ground_truth": "Complaint", "is_oos": False},
    {"text": "False product advertising: the features prominently advertised on your sales page do not exist in the product", "ground_truth": "Complaint", "is_oos": False},
    {"text": "I have spent over 15 hours troubleshooting an issue caused entirely by your engineering bug", "ground_truth": "Complaint", "is_oos": False},
    {"text": "Your support supervisor was completely dismissive, condescending, and refused to provide their employee name", "ground_truth": "Complaint", "is_oos": False},

    # --- OUT-OF-SCOPE / UNRELATED MESSAGES (30) ---
    {"text": "hi how are you doing today?", "ground_truth": "OOS", "is_oos": True},
    {"text": "good morning have a wonderful and productive day ahead", "ground_truth": "OOS", "is_oos": True},
    {"text": "tell me a funny joke about computers and programmers", "ground_truth": "OOS", "is_oos": True},
    {"text": "who created you and what is your favorite color?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what should I eat for lunch today?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the capital city of Australia?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how far is the moon from planet earth in miles?", "ground_truth": "OOS", "is_oos": True},
    {"text": "who was the 16th president of the united states?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the chemical formula and boiling point of water?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do airplanes generate aerodynamic lift to fly?", "ground_truth": "OOS", "is_oos": True},
    {"text": "can you give me a simple recipe for homemade chocolate chip cookies?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do I change a flat tire on my mountain bicycle?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what are the top tourist attractions to visit in Rome Italy?", "ground_truth": "OOS", "is_oos": True},
    {"text": "explain the basic rules and strategy of playing chess for beginners", "ground_truth": "OOS", "is_oos": True},
    {"text": "recommend an exciting science fiction movie to watch on Netflix tonight", "ground_truth": "OOS", "is_oos": True},
    {"text": "write a python function to find the maximum element in a binary search tree", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do I sort an array of integers using quicksort in java?", "ground_truth": "OOS", "is_oos": True},
    {"text": "the food was completely cold and the restaurant waiter was extremely rude at dinner", "ground_truth": "OOS", "is_oos": True},
    {"text": "the hotel room was dirty and the bed sheets had noticeable stains", "ground_truth": "OOS", "is_oos": True},
    {"text": "my commercial airline flight was delayed 4 hours at the airport terminal", "ground_truth": "OOS", "is_oos": True},
    {"text": "where is the closest public parking garage near the downtown train station?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the weather forecast for Chicago this upcoming weekend?", "ground_truth": "OOS", "is_oos": True},
    {"text": "asdfghjkl qwertyuiop zxcvbnm", "ground_truth": "OOS", "is_oos": True},
    {"text": "1234567890987654321", "ground_truth": "OOS", "is_oos": True},
    {"text": "visit our discount store to buy cheap designer luxury watches free worldwide shipping", "ground_truth": "OOS", "is_oos": True},
    {"text": "congratulations you have won a cash lottery prize of one million dollars click here", "ground_truth": "OOS", "is_oos": True},
    {"text": "crypto investment opportunity guaranteed 500 percent return in seven days", "ground_truth": "OOS", "is_oos": True},
    {"text": "I went for a five mile morning run in the park today", "ground_truth": "OOS", "is_oos": True},
    {"text": "my cat fell asleep on top of the warm radiator again", "ground_truth": "OOS", "is_oos": True},
    {"text": "traffic on the interstate highway was unusually heavy during rush hour", "ground_truth": "OOS", "is_oos": True}
]

def run_evaluation():
    print("=" * 80)
    print("TICKETAI: RIGOROUS THREE-SYSTEM BENCHMARK EVALUATION")
    print("=" * 80)
    
    # Load primary model
    model_path = os.path.join(os.path.dirname(__file__), 'enhanced_ticket_classifier.joblib')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    primary_model = joblib.load(model_path)
    classes = list(primary_model.classes_)
    print(f"Primary Model Loaded: {primary_model.named_steps['classifier'].__class__.__name__}")
    print(f"Operational Categories ({len(classes)}): {classes}")

    # Initialize DistilBERT optionally
    distilbert_pipeline = None
    try:
        from transformers import pipeline
        print("Loading DistilBERT zero-shot pipeline (for System B evaluation)...")
        distilbert_pipeline = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
        print("DistilBERT loaded successfully.")
    except Exception as e:
        print(f"Notice: DistilBERT could not be initialized: {e}")

    triage_layer = TriageDecisionLayer(min_confidence_threshold=60.0, min_margin_threshold=0.15, oos_confidence_floor=55.0)

    # -------------------------------------------------------------------------
    # System Functions
    # -------------------------------------------------------------------------
    def predict_system_a(text: str):
        """System A: Primary Classifier Alone (baseline triage rule: conf < 60 or words < 3)"""
        probs = primary_model.predict_proba([text])[0]
        best_idx = np.argmax(probs)
        pred_cat = classes[best_idx]
        conf = float(probs[best_idx] * 100.0)
        is_rejected = bool(conf < 60.0 or len(text.split()) < 3)
        needs_review = is_rejected
        is_oos = is_rejected
        return pred_cat, conf, is_rejected, needs_review

    tech_kw = {
        'login', 'log in', 'password', 'crash', 'crashed', 'crashing', 'error', 'bug', 'software',
        'app', 'hardware', 'screen', 'server', 'wifi', 'network', 'connection', 'database',
        '500', '404', 'code', 'device', 'laptop', 'phone', 'browser', 'install', 'update',
        'port', 'driver', 'sync', 'reset', 'auth', '2fa', 'token', 'ssl', 'timeout', 'bluetooth',
        'mouse', 'keyboard', 'printer', 'pc', 'windows', 'ios', 'android', 'mac'
    }
    
    def predict_system_b(text: str):
        """System B: Primary Classifier with DistilBERT Fallback (legacy app.py)"""
        probs = primary_model.predict_proba([text])[0]
        best_idx = np.argmax(probs)
        pred_cat = classes[best_idx]
        conf = float(probs[best_idx] * 100.0)
        
        tokens = set(text.lower().split())
        has_tech = bool(tokens & tech_kw)
        needs_bert = ((pred_cat == "Technical Support" and not has_tech) or (conf < 72.0))
        
        if distilbert_pipeline is not None and needs_bert:
            try:
                bert_res = distilbert_pipeline(text, classes)
                bert_top = bert_res['labels'][0]
                bert_conf = float(bert_res['scores'][0] * 100.0)
                min_threshold = 35.0 if (pred_cat == "Technical Support" and not has_tech) else 50.0
                if bert_top != pred_cat and bert_conf >= min_threshold:
                    pred_cat = bert_top
                    conf = bert_conf
            except Exception:
                pass
                
        is_rejected = bool(conf < 60.0 or len(text.split()) < 3)
        return pred_cat, conf, is_rejected, is_rejected

    def predict_system_c(text: str):
        """System C: Primary Classifier with Improved Triage Decision Layer"""
        probs = primary_model.predict_proba([text])[0]
        triage_res = triage_layer.evaluate(text, probs, classes)
        return (
            triage_res["predicted_category"],
            triage_res["confidence"],
            triage_res["is_out_of_scope"],
            triage_res["needs_human_review"]
        )

    systems = [
        ("System A: Primary Classifier Alone (LinearSVC)", predict_system_a),
        ("System B: Primary + Legacy DistilBERT Fallback", predict_system_b),
        ("System C: Primary + Improved Triage Decision Layer", predict_system_c)
    ]

    results_table = []

    for sys_name, predict_fn in systems:
        print(f"\nEvaluating: {sys_name} ...")
        
        latencies = []
        valid_y_true = []
        valid_y_pred = []
        valid_rejected_count = 0  # False rejection of valid tickets (FP)
        
        oos_total = 0
        oos_rejected_count = 0    # True rejection of OOS messages (TP)
        oos_accepted_count = 0    # False acceptance of OOS messages (FN)
        total_human_reviews = 0

        for item in BENCHMARK_DATA:
            t0 = time.perf_counter()
            pred_cat, conf, is_oos_flag, needs_review = predict_fn(item["text"])
            lat = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(lat)

            if needs_review:
                total_human_reviews += 1

            if not item["is_oos"]:
                valid_y_true.append(item["ground_truth"])
                valid_y_pred.append(pred_cat)
                if is_oos_flag:
                    valid_rejected_count += 1
            else:
                oos_total += 1
                if is_oos_flag:
                    oos_rejected_count += 1
                else:
                    oos_accepted_count += 1

        # Metrics computation
        valid_total = len(valid_y_true)
        valid_acc = accuracy_score(valid_y_true, valid_y_pred) * 100.0
        valid_macro_f1 = f1_score(valid_y_true, valid_y_pred, average="macro") * 100.0
        
        # OOS Rejection metrics: TP = oos_rejected, FP = valid_rejected, FN = oos_accepted
        tp = oos_rejected_count
        fp = valid_rejected_count
        fn = oos_accepted_count
        
        oos_prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        oos_rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        oos_f1 = (2 * oos_prec * oos_rec / (oos_prec + oos_rec)) if (oos_prec + oos_rec) > 0 else 0.0

        false_rejection_rate = (valid_rejected_count / valid_total) * 100.0
        false_acceptance_rate = (oos_accepted_count / oos_total) * 100.0
        human_review_rate = (total_human_reviews / len(BENCHMARK_DATA)) * 100.0

        lat_mean = float(np.mean(latencies))
        lat_p50 = float(np.percentile(latencies, 50))
        lat_p95 = float(np.percentile(latencies, 95))
        lat_p99 = float(np.percentile(latencies, 99))

        print(f"  • Valid Ticket Acc:            {valid_acc:.2f}% (Macro F1: {valid_macro_f1:.2f}%)")
        print(f"  • Valid Tickets False Reject:  {valid_rejected_count}/{valid_total} ({false_rejection_rate:.2f}%)")
        print(f"  • OOS Rejection Recall:        {oos_rejected_count}/{oos_total} ({oos_rec:.2f}%)")
        print(f"  • OOS Rejection Precision:     {oos_prec:.2f}% (F1: {oos_f1:.2f}%)")
        print(f"  • OOS False Acceptance Rate:   {false_acceptance_rate:.2f}%")
        print(f"  • Human Review Routing Rate:   {human_review_rate:.2f}%")
        print(f"  • Latency (Mean / P50 / P95):  {lat_mean:.2f}ms / {lat_p50:.2f}ms / {lat_p95:.2f}ms")

        # Per-class report for System C
        if sys_name.startswith("System C"):
            print("\n  Detailed Classification Report (System C on Valid Tickets):")
            print("  " + "\n  ".join(classification_report(valid_y_true, valid_y_pred, digits=4).splitlines()))
            cm = confusion_matrix(valid_y_true, valid_y_pred, labels=classes)
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            print("\n  Confusion Matrix on Valid Tickets:")
            print(cm_df.to_string())

        results_table.append({
            "System": sys_name,
            "Valid Acc": f"{valid_acc:.1f}%",
            "Macro F1": f"{valid_macro_f1:.1f}%",
            "False Rejection Rate": f"{false_rejection_rate:.1f}%",
            "OOS Rejection Prec": f"{oos_prec:.1f}%",
            "OOS Rejection Rec": f"{oos_rec:.1f}%",
            "OOS Rejection F1": f"{oos_f1:.1f}%",
            "False Acceptance Rate": f"{false_acceptance_rate:.1f}%",
            "Human Review Rate": f"{human_review_rate:.1f}%",
            "Latency Mean (ms)": f"{lat_mean:.2f}",
            "Latency P95 (ms)": f"{lat_p95:.2f}"
        })

    print("\n" + "=" * 90)
    print("FINAL STANDARDIZED BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))
    print("=" * 90)

    # Save to JSON
    json_path = os.path.join(os.path.dirname(__file__), 'system_evaluation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved evaluation metrics to: {json_path}")

if __name__ == '__main__':
    run_evaluation()
