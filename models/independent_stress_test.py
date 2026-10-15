"""
Independent QA Stress-Test & Generalization Benchmark
TicketAI Customer Support Classification Platform

Audits the primary classifier, DistilBERT fallback, and Triage Decision Layer
against a brand new, unseen 80-sample stress-test suite:
- 40 Unseen Out-of-Scope (OOS) messages (creative writing, philosophy, science, medicine,
  adversarial domain word collisions like 'order', 'help', 'open', 'close', chit-chat)
- 40 Unseen In-Scope Valid customer support tickets (8 per category) with informal phrasing,
  colloquialisms, real-world typos, and compound expressions.

Includes:
- Warm-up and multi-iteration latency measurement (strictly excluding model load time)
- Transparent metric calculation
- Per-class breakdown and confusion matrices
- Analysis of 100% OOS rejection claim and generalization limits
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
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

from models.triage_decision_layer import TriageDecisionLayer, get_default_triage_layer
from app import preprocess_text

# -----------------------------------------------------------------------------
# INDEPENDENT STRESS-TEST BENCHMARK (80 SAMPLES: 40 VALID + 40 OOS)
# Note: These samples were NOT used to design any regex, vocabulary, or thresholds.
# -----------------------------------------------------------------------------
STRESS_TEST_DATA = [
    # =========================================================================
    # PART 1: 40 UNSEEN VALID IN-SCOPE CUSTOMER SUPPORT TICKETS (8 per category)
    # =========================================================================
    
    # --- TECHNICAL SUPPORT (8) ---
    {"text": "getting blank white screen after typing my credentials on chrome browser", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "error code ERR_SSL_PROTOCOL_ERROR when trying to reach account management portal", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "my wireless keyboard lags badly and misses keystrokes while typing fast", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "the sync client keeps throwing socket timeout exception 10054 on windows 10", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "cant reset my password, the verification link in the email says token expired", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "android application crashes whenever i tap on the export pdf button", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "device status shows offline in the dashboard even though the green wifi led is on", "ground_truth": "Technical Support", "is_oos": False},
    {"text": "audio crackles and cuts out completely during audio calls on the desktop app", "ground_truth": "Technical Support", "is_oos": False},

    # --- BILLING (8) ---
    {"text": "was charged 49 dollars on my mastercard today but i downgraded to the free tier last week", "ground_truth": "Billing", "is_oos": False},
    {"text": "my corporate credit card was charged three times for transaction ref 99482", "ground_truth": "Billing", "is_oos": False},
    {"text": "where can i download the paid vat invoices for the last quarter for taxes?", "ground_truth": "Billing", "is_oos": False},
    {"text": "my payment was declined by stripe even though my bank confirms funds are available", "ground_truth": "Billing", "is_oos": False},
    {"text": "please cancel my monthly recurring subscription before the next renewal on the 25th", "ground_truth": "Billing", "is_oos": False},
    {"text": "there is a 15 dollar foreign currency processing surcharge on my statement that was never disclosed", "ground_truth": "Billing", "is_oos": False},
    {"text": "i need an updated receipt showing our company tax identification number", "ground_truth": "Billing", "is_oos": False},
    {"text": "charged twice for the annual plan renewal on my amex card", "ground_truth": "Billing", "is_oos": False},

    # --- RETURNS (8) ---
    {"text": "the ceramic mug arrived with the handle completely snapped off inside the packaging", "ground_truth": "Returns", "is_oos": False},
    {"text": "sent back the wrong size sneakers last monday with usps tracking, when will my refund show up?", "ground_truth": "Returns", "is_oos": False},
    {"text": "i received a red medium hoodie instead of the black large hoodie i purchased, need an exchange", "ground_truth": "Returns", "is_oos": False},
    {"text": "need a prepaid return mailing label to ship this defective monitor back to your warehouse", "ground_truth": "Returns", "is_oos": False},
    {"text": "box was delivered unsealed and missing the charging cable, want to return for a full replacement", "ground_truth": "Returns", "is_oos": False},
    {"text": "how many days do i have to return an unopened item under your 30 day return guarantee?", "ground_truth": "Returns", "is_oos": False},
    {"text": "product stopped turning on after two days, requesting an rma return authorization number", "ground_truth": "Returns", "is_oos": False},
    {"text": "i want to return these unopened filters for store credit", "ground_truth": "Returns", "is_oos": False},

    # --- GENERAL INQUIRY (8) ---
    {"text": "what are the customer service telephone support hours during memorial day weekend?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "does this mechanical keyboard support macos key remapping out of the box?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "can i purchase an extended 3 year warranty coverage after the initial purchase date?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "do you offer bulk educational pricing discounts for university student clubs?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "is your retail store in downtown chicago open for curbside order pickup on sundays?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "what is the maximum load weight capacity for this standing desk model?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "do your products comply with the european union rohs environmental standards?", "ground_truth": "General Inquiry", "is_oos": False},
    {"text": "how long does standard ground shipping usually take to arrive in seattle?", "ground_truth": "General Inquiry", "is_oos": False},

    # --- COMPLAINT (8) ---
    {"text": "your telephone agent called me stupid and terminated the call while i was calmly explaining my issue", "ground_truth": "Complaint", "is_oos": False},
    {"text": "waited on hold for over ninety minutes only to be disconnected when transferred to a manager", "ground_truth": "Complaint", "is_oos": False},
    {"text": "delivery driver tossed my fragile laptop box over an eight foot gate directly onto concrete", "ground_truth": "Complaint", "is_oos": False},
    {"text": "your software update deleted our quarterly database backup and your tier 1 staff was utterly indifferent", "ground_truth": "Complaint", "is_oos": False},
    {"text": "blatant false advertising: the features shown in your keynote presentation do not exist in the live release", "ground_truth": "Complaint", "is_oos": False},
    {"text": "three weeks of empty promises from customer care and still nobody has solved my ticket", "ground_truth": "Complaint", "is_oos": False},
    {"text": "i will be filing a formal complaint with the consumer financial protection bureau regarding your billing deceit", "ground_truth": "Complaint", "is_oos": False},
    {"text": "appalling service quality, treating loyal ten year enterprise customers with complete disdain", "ground_truth": "Complaint", "is_oos": False},


    # =========================================================================
    # PART 2: 40 UNSEEN OUT-OF-SCOPE (OOS) MESSAGES
    # Diverse domains, paraphrases, creative text, and domain-colliding vocabulary
    # =========================================================================

    # Creative, philosophical, and chit-chat (8)
    {"text": "write a short haiku about cherry blossoms in the spring breeze", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the meaning of life according to existentialist philosophy?", "ground_truth": "OOS", "is_oos": True},
    {"text": "hey there, just feeling bored and wondering if you wanted to chat", "ground_truth": "OOS", "is_oos": True},
    {"text": "compose a bedtime poem for a six year old child who loves dragons", "ground_truth": "OOS", "is_oos": True},
    {"text": "do you believe artificial consciousness will surpass human creativity?", "ground_truth": "OOS", "is_oos": True},
    {"text": "knock knock who is there orange orange who orange you glad i said hello", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is your opinion on modern abstract impressionist painting?", "ground_truth": "OOS", "is_oos": True},
    {"text": "hope you are having a pleasant afternoon my digital companion", "ground_truth": "OOS", "is_oos": True},

    # Science, geography, nature, and history (8)
    {"text": "why does the sky appear deep blue during midday but crimson red at dusk?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what caused the bronze age collapse in the eastern mediterranean?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how many moons does jupiter currently have documented by astronomers?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the approximate distance between the sun and alpha centauri in parsecs?", "ground_truth": "OOS", "is_oos": True},
    {"text": "explain how tectonic plate subduction causes deep oceanic trenches and volcanic arcs", "ground_truth": "OOS", "is_oos": True},
    {"text": "who was the architect behind the design of the eiffel tower in paris?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the biological life cycle and migration route of monarch butterflies?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do hydrothermal vents support deep sea ecosystems without sunlight?", "ground_truth": "OOS", "is_oos": True},

    # Medical, culinary, sports, and daily life (8)
    {"text": "how long should i slow cook beef short ribs in red wine broth in a dutch oven?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what are the early symptoms of acute appendicitis versus ordinary stomach cramps?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the proper stance and grip when hitting a topspin forehand in tennis?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do you make a traditional french bechamel sauce from butter and flour?", "ground_truth": "OOS", "is_oos": True},
    {"text": "can a residential landlord enter a tenant apartment without twenty four hours notice in california?", "ground_truth": "OOS", "is_oos": True},
    {"text": "my tomato plants in the backyard garden have yellow spots on their lower leaves", "ground_truth": "OOS", "is_oos": True},
    {"text": "what are the best exercises for strengthening the lower back after a lumbar sprain?", "ground_truth": "OOS", "is_oos": True},
    {"text": "who is favored to win the tour de france cycling race this summer?", "ground_truth": "OOS", "is_oos": True},

    # Mathematics, programming, and logic puzzles (8)
    {"text": "if two trains depart stations two hundred miles apart at differing speeds when do they meet?", "ground_truth": "OOS", "is_oos": True},
    {"text": "solve the quadratic equation 3x squared minus 5x plus 2 equals zero using the quadratic formula", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do i implement a depth first search algorithm on an adjacency list in rust?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the time complexity difference between mergesort and heapsort in the worst case?", "ground_truth": "OOS", "is_oos": True},
    {"text": "explain the mathematical proof that the square root of two is an irrational number", "ground_truth": "OOS", "is_oos": True},
    {"text": "how does the rsa public key encryption algorithm utilize prime factorization?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do i write a sql query to find the second highest employee salary across departments?", "ground_truth": "OOS", "is_oos": True},
    {"text": "what is the difference between a mutex lock and a semaphore in concurrent multithreading?", "ground_truth": "OOS", "is_oos": True},

    # ADVERSARIAL OOS (Contains common English words that overlap with domain words like 'order', 'help', 'open', 'close', 'question') (8)
    {"text": "can you help me understand the concept of quantum wave function collapse?", "ground_truth": "OOS", "is_oos": True},
    {"text": "in what chronological order should a new student read the plays of william shakespeare?", "ground_truth": "OOS", "is_oos": True},
    {"text": "how do you open a raw oyster shell safely using an oyster knife without cutting yourself?", "ground_truth": "OOS", "is_oos": True},
    {"text": "i have a philosophical question about free will versus genetic determinism", "ground_truth": "OOS", "is_oos": True},
    {"text": "the government decided to close all public schools on tuesday due to heavy snowfall", "ground_truth": "OOS", "is_oos": True},
    {"text": "what order of magnitude is the mass of an electron compared to a proton?", "ground_truth": "OOS", "is_oos": True},
    {"text": "help me choose a good gift for my grandmother eightieth birthday celebration", "ground_truth": "OOS", "is_oos": True},
    {"text": "can you answer a trivia question regarding the origin of the ancient olympic games?", "ground_truth": "OOS", "is_oos": True}
]


def run_stress_test():
    print("=" * 85)
    print("TICKETAI: INDEPENDENT QUALITY ASSURANCE & STRESS-TEST BENCHMARK")
    print("=" * 85)
    print(f"Total Evaluation Samples: {len(STRESS_TEST_DATA)}")
    print(f"  • Unseen Valid Customer Tickets: {sum(1 for d in STRESS_TEST_DATA if not d['is_oos'])} (8 per category)")
    print(f"  • Unseen Out-of-Scope (OOS) Texts: {sum(1 for d in STRESS_TEST_DATA if d['is_oos'])} (Diverse & Adversarial)")
    print("-" * 85)

    # 1. Load Primary Production Model
    model_path = os.path.join(os.path.dirname(__file__), 'enhanced_ticket_classifier.joblib')
    if not os.path.exists(model_path):
        print(f"Error: Model artifact not found at {model_path}")
        return
    
    primary_model = joblib.load(model_path)
    classes = list(primary_model.classes_)
    print(f"Primary Production Model: {primary_model.named_steps['classifier'].__class__.__name__}")
    print(f"Target Classes ({len(classes)}): {classes}")

    # 2. DistilBERT Pipeline (for System B comparison)
    distilbert_pipeline = None
    try:
        from transformers import pipeline
        distilbert_pipeline = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1
        )
        print("DistilBERT zero-shot fallback loaded for System B comparison.")
    except Exception as e:
        print(f"Notice: DistilBERT unavailable ({e}); System B will match System A.")

    # 3. Triage Decision Layer (System C)
    triage_layer = get_default_triage_layer()

    # Pre-defined system prediction wrappers
    def predict_system_a(text: str):
        """System A: LinearSVC with naive confidence threshold (<60% or <3 words rejected)."""
        proc = preprocess_text(text)
        if not proc.strip():
            proc = text
        probs = primary_model.predict_proba([proc])[0]
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx] * 100.0)
        is_rejected = bool(conf < 60.0 or len(text.split()) < 3)
        return classes[top_idx], conf, is_rejected, is_rejected

    def predict_system_b(text: str):
        """System B: Primary Classifier + Legacy DistilBERT Fallback."""
        proc = preprocess_text(text)
        if not proc.strip():
            proc = text
        probs = primary_model.predict_proba([proc])[0]
        top_idx = int(np.argmax(probs))
        pred_cat = classes[top_idx]
        conf = float(probs[top_idx] * 100.0)
        
        tech_kw = {'login', 'password', 'crash', 'error', 'bug', 'software', 'app', 'hardware', 'screen', 'server', 'wifi', 'network'}
        tokens = set(proc.split())
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
        """System C: Primary Classifier with Triage Decision Layer."""
        res = triage_layer.decide(text, primary_model=primary_model, preprocessor_fn=preprocess_text)
        return (
            res["predicted_category"],
            res["confidence"],
            res["is_out_of_scope"],
            res["needs_human_review"]
        )

    systems = [
        ("System A: Primary Classifier Alone (LinearSVC)", predict_system_a),
        ("System B: Primary + Legacy DistilBERT Fallback", predict_system_b),
        ("System C: Primary + Triage Decision Layer (Production)", predict_system_c)
    ]

    # =========================================================================
    # 4. RIGOROUS LATENCY BENCHMARK (Warm-up + Multi-iteration timing)
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 1: RIGOROUS INFERENCE LATENCY BENCHMARKING")
    print("Conditions: Identical machine, 10 pre-warm iterations, 5 timed passes per sample")
    print("Note: Model loading time is strictly EXCLUDED. Only per-request pipeline is timed.")
    print("=" * 85)

    latency_summary = {}

    for sys_name, predict_fn in systems:
        # Pre-warm (10 runs)
        for _ in range(10):
            predict_fn("Warmup request test ticket for system calibration")

        # Timed evaluations (5 passes per sample across 80 samples = 400 timed runs)
        run_latencies = []
        for _ in range(5):
            for item in STRESS_TEST_DATA:
                t0 = time.perf_counter()
                _ = predict_fn(item["text"])
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                run_latencies.append(elapsed_ms)

        mean_lat = float(np.mean(run_latencies))
        median_lat = float(np.median(run_latencies))
        p95_lat = float(np.percentile(run_latencies, 95))
        p99_lat = float(np.percentile(run_latencies, 99))
        min_lat = float(np.min(run_latencies))
        max_lat = float(np.max(run_latencies))

        latency_summary[sys_name] = {
            "mean": mean_lat,
            "median": median_lat,
            "p95": p95_lat,
            "p99": p99_lat,
            "min": min_lat,
            "max": max_lat,
            "total_runs": len(run_latencies)
        }
        print(f"[{sys_name}]")
        print(f"  Runs: {len(run_latencies)} | Mean: {mean_lat:.2f}ms | Median: {median_lat:.2f}ms | P95: {p95_lat:.2f}ms | P99: {p99_lat:.2f}ms")

    # =========================================================================
    # 5. PREDICTION ACCURACY & OOS CLASSIFICATION EVALUATION
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 2: PREDICTION ACCURACY, OOS REJECTION & HUMAN REVIEW METRICS")
    print("=" * 85)

    comparison_results = []
    system_c_reports = {}

    for sys_name, predict_fn in systems:
        valid_y_true = []
        valid_y_pred = []
        valid_rejected_count = 0  # Valid tickets rejected as OOS (False Positive rejection)
        
        oos_total = 0
        oos_rejected_count = 0    # OOS messages correctly rejected (True Positive)
        oos_accepted_count = 0    # OOS messages admitted as tickets (False Negative)
        total_human_reviews = 0

        # Detailed tracking of per-sample decisions
        predictions_log = []

        for item in STRESS_TEST_DATA:
            pred_cat, conf, is_oos_flag, needs_review = predict_fn(item["text"])
            
            predictions_log.append({
                "text": item["text"],
                "ground_truth": item["ground_truth"],
                "is_oos_gt": item["is_oos"],
                "pred_cat": pred_cat,
                "confidence": conf,
                "is_oos_pred": is_oos_flag,
                "needs_review": needs_review
            })

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

        valid_total = len(valid_y_true)
        valid_acc = accuracy_score(valid_y_true, valid_y_pred) * 100.0
        valid_macro_f1 = f1_score(valid_y_true, valid_y_pred, average="macro") * 100.0

        tp = oos_rejected_count
        fp = valid_rejected_count
        fn = oos_accepted_count

        oos_prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        oos_rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        oos_f1 = (2 * oos_prec * oos_rec / (oos_prec + oos_rec)) if (oos_prec + oos_rec) > 0 else 0.0

        false_rejection_rate = (valid_rejected_count / valid_total) * 100.0
        false_acceptance_rate = (oos_accepted_count / oos_total) * 100.0
        human_review_rate = (total_human_reviews / len(STRESS_TEST_DATA)) * 100.0

        lat_stats = latency_summary[sys_name]

        comparison_results.append({
            "System": sys_name,
            "Valid Acc": f"{valid_acc:.1f}%",
            "Macro F1": f"{valid_macro_f1:.1f}%",
            "Valid False Reject Rate": f"{false_rejection_rate:.1f}% ({valid_rejected_count}/{valid_total})",
            "OOS Rejection Recall": f"{oos_rec:.1f}% ({oos_rejected_count}/{oos_total})",
            "OOS Rejection Precision": f"{oos_prec:.1f}%",
            "OOS Rejection F1": f"{oos_f1:.1f}%",
            "OOS False Accept Rate": f"{false_acceptance_rate:.1f}% ({oos_accepted_count}/{oos_total})",
            "Human Review Rate": f"{human_review_rate:.1f}% ({total_human_reviews}/{len(STRESS_TEST_DATA)})",
            "Latency Mean (ms)": f"{lat_stats['mean']:.2f}",
            "Latency P95 (ms)": f"{lat_stats['p95']:.2f}"
        })

        if sys_name.startswith("System C"):
            report_str = classification_report(valid_y_true, valid_y_pred, digits=4)
            cm = confusion_matrix(valid_y_true, valid_y_pred, labels=classes)
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            system_c_reports = {
                "classification_report": report_str,
                "confusion_matrix": cm_df.to_dict(),
                "confusion_matrix_df": cm_df,
                "valid_acc": valid_acc,
                "valid_macro_f1": valid_macro_f1,
                "oos_rec": oos_rec,
                "oos_prec": oos_prec,
                "oos_f1": oos_f1,
                "false_rejection_rate": false_rejection_rate,
                "false_acceptance_rate": false_acceptance_rate,
                "human_review_rate": human_review_rate,
                "predictions_log": predictions_log
            }

    # Print summary table
    df_comp = pd.DataFrame(comparison_results)
    print("\n" + "=" * 90)
    print("STRESS-TEST BENCHMARK RESULTS TABLE (80 UNSEEN SAMPLES)")
    print("=" * 90)
    print(df_comp.to_string(index=False))
    print("=" * 90)

    # Print Detailed Per-Class breakdown for System C
    print("\nDETAILED PER-CLASS REPORT (System C on 40 Unseen Valid Tickets):")
    print(system_c_reports["classification_report"])
    print("\nCONFUSION MATRIX (System C on 40 Unseen Valid Tickets):")
    print(system_c_reports["confusion_matrix_df"].to_string())

    # Deep-dive: Check any OOS false acceptances or valid ticket rejections
    print("\n" + "=" * 90)
    print("INVESTIGATION: ANALYSIS OF 100% OOS REJECTION CLAIM ON UNSEEN DATA")
    print("=" * 90)
    oos_leaks = [p for p in system_c_reports["predictions_log"] if p["is_oos_gt"] and not p["is_oos_pred"]]
    valid_rejections = [p for p in system_c_reports["predictions_log"] if not p["is_oos_gt"] and p["is_oos_pred"]]

    print(f"Total Unseen OOS False Acceptances (Leaks): {len(oos_leaks)} / 40")
    if oos_leaks:
        print("Detailed Leaked OOS Queries:")
        for idx, leak in enumerate(oos_leaks, 1):
            print(f"  {idx}. \"{leak['text']}\" -> Predicted: {leak['pred_cat']} ({leak['confidence']:.1f}%) [Needs Review: {leak['needs_review']}]")
    else:
        print("✓ All 40 unseen out-of-scope queries were successfully trapped!")

    print(f"\nTotal Valid Customer Tickets Falsely Rejected: {len(valid_rejections)} / 40")
    if valid_rejections:
        for idx, rej in enumerate(valid_rejections, 1):
            print(f"  {idx}. \"{rej['text']}\" -> Category: {rej['pred_cat']}")
    else:
        print("✓ Zero valid customer tickets were falsely rejected!")

    # Save stress test report to JSON
    export_data = {
        "benchmark_metadata": {
            "total_samples": len(STRESS_TEST_DATA),
            "valid_samples": 40,
            "oos_samples": 40,
            "description": "Independent stress-test set untouched during rule and threshold development"
        },
        "latency_summary": latency_summary,
        "comparison_table": comparison_results,
        "system_c_per_class": system_c_reports["classification_report"],
        "system_c_confusion_matrix": system_c_reports["confusion_matrix"],
        "oos_leaks": oos_leaks,
        "valid_rejections": valid_rejections
    }

    report_path = os.path.join(os.path.dirname(__file__), 'independent_stress_test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    print(f"\nExported complete stress test report to: {report_path}")

    return export_data

if __name__ == '__main__':
    run_stress_test()
