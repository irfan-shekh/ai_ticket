# TicketAI - Intelligent Customer Support Ticket Classification & Automation Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange?style=for-the-badge&logo=scikit-learn)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-yellow?style=for-the-badge)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)

**TicketAI** is an end-to-end AI-powered customer support ticketing and automated classification system. It combines modern Natural Language Processing (NLP), a soft **Voting Ensemble** of calibrated linear models, and pre-trained **DistilBERT Transformers** with a full-stack Flask web application. The platform automatically classifies, prioritizes, and routes customer support tickets, providing real-time category predictions, calibrated confidence scores, and automated email notifications.

---

## 🌟 Key Features

### 🧠 Machine Learning & NLP Classification Engine
- **Automated Categorization**: Real-time classification into 5 core operational categories:
  - 🛠️ **Technical Support**: Login issues, 500/404 server errors, software bugs, app crashes, connectivity.
  - 💳 **Billing**: Invoices, duplicate charges, payment failures, subscription renewals.
  - 📦 **Returns**: Return requests, damaged goods, incorrect shipments, refunds.
  - ℹ️ **General Inquiry**: Store hours, contact information, product specifications, shipping estimates.
  - ⚠️ **Complaint**: Service dissatisfaction, delivery delays, rude staff, executive escalations.
- **Dual-Granularity Feature Extraction**:
  - **Word-Level TF-IDF (1–2 n-grams, 12,000 features)** with sublinear frequency scaling to capture contextual phrases.
  - **Character-Level TF-IDF (3–5 n-grams, 20,000 features)** (`char_wb`) to robustly handle typos, misspellings, and morphological variants.
- **Ensemble Model Architecture**:
  - Soft **Voting Ensemble** combining **Calibrated LinearSVC**, **Multinomial/Complement Naive Bayes**, and **L2-Regularized Logistic Regression**.
  - Produces well-calibrated posterior probabilities with categorized confidence ratings (*High, Medium, Low (Review Suggested)*).
  - **Top-3 Ranked Predictions**: Transparent multi-category suggestions with probability breakdown.
- **DistilBERT Out-of-Distribution (OOD) Semantic Engine**:
  - Leverages pre-trained **DistilBERT** (`typeform/distilbert-base-uncased-mnli`) transformer to understand deep semantic intent and sentiment for out-of-dataset or domain-shifted customer inquiries (e.g., *"the food is not good"*, *"waiter was rude"*), preventing out-of-domain inputs from defaulting to vocabulary-sparse classes.
- **Production Guardrails & Anti-Leakage Hardening**:
  - **Zero-Leakage Dataset Split**: Verified 100% disjoint train/test partitions (5,156 training samples, 1,290 strictly held-out test samples; 0 overlapping records).
  - **Ambiguity Detection**: Automatically flags ultra-brief or vague inputs (< 3 words) and caps confidence to 50% for manual triage.
  - **Review Suggested Flag**: Tickets with confidence under 60% are flagged (`needs_human_review = True`) to prevent erroneous automation.
- **High Accuracy & Real-World Generalization**:
  - Achieves **99.46% test accuracy**, **0.995 weighted F1-score**, and **0.995 macro F1-score** on the clean held-out test set.

### 💻 Full-Stack Web Application
- **Customer Portal**:
  - Intuitive ticket submission interface with live AI classification preview.
  - Ticket history tracking, status badges, and resolution notes.
- **Administrative Dashboard**:
  - Role-Based Access Control (RBAC) separating regular users and support agents.
  - Real-time ticket statistics (Total, Open, Resolved, Breakdown by Category).
  - Filter tickets by status (*Open, In Progress, Resolved*) and category.
  - Resolution workflow: Add resolution notes, resolve tickets, and trigger automated resolution emails.
- **Adaptive Light & Dark Theme UI**:
  - Complete dark and light mode system powered by CSS variables (`theme.css`) with persistent browser preference storage and optimal readability.
- **Automated Transactional Emails**:
  - SMTP/MIME integration with responsive HTML email templates.
  - Instant confirmation upon ticket creation with AI classification badge and confidence score.
  - Resolution notification email delivering the resolution summary to the customer.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([Customer / User]) -->|1. Submit Ticket| WebUI[Flask Web App]
    WebUI -->|2. Preprocess Text| Preprocessor[NLTK Preprocessor<br/>Lemmatization & Negation Filtering]
    Preprocessor -->|3. Feature Extraction| Features[Dual TF-IDF Union<br/>Word 1-2 & Char-wb 3-5 n-grams]
    Features -->|4. Primary Inference| Ensemble[Voting Ensemble<br/>Calibrated SVC + LR + Complement NB]
    Ensemble -->|Low Confidence / OOD Check| Arbiter{Semantic Arbiter<br/>DistilBERT Transformer}
    Arbiter -->|Calibrated Prediction| WebUI
    WebUI -->|5. Store Ticket| DB[(SQLite Database<br/>tickets.db)]
    WebUI -->|6. Send Notification Email| SMTP[SMTP Server / Gmail API]
    SMTP -->|7. Email Notification| CustomerInbox([Customer Inbox])
    Admin([Support Agent / Admin]) -->|Manage & Resolve| AdminUI[Admin Dashboard]
    AdminUI -->|Update Status & Notes| DB
```

---

## 📁 Project Structure

```text
ai_ticket/
├── app.py                                        # Main Flask application (routes, auth, APIs, DistilBERT arbiter, email)
├── requirements.txt                              # Python dependencies
├── tickets.db                                    # SQLite database (users, tickets, settings)
├── download.py                                   # NLTK resource downloader (stopwords, wordnet)
├── static/                                       # Static assets
│   └── css/
│       └── theme.css                             # Unified Light & Dark mode styling and CSS variables
├── templates/                                    # Jinja2 HTML templates
│   ├── landing.html                              # Public landing page
│   ├── index.html                                # Customer ticket creation & live AI preview
│   ├── dashboard.html                            # User ticket history & dashboard
│   ├── admin_dashboard.html                      # Support agent admin management portal
│   ├── login.html                                # Authentication login page
│   └── register.html                             # User registration page
├── models/                                       # Machine learning models & training pipeline
│   ├── customer_support_tickets_train_split.csv  # Verified training split (5,156 records, 0 leakage)
│   ├── customer_support_tickets_test_split.csv   # Strictly held-out test split (1,290 records)
│   ├── train_model.py                            # Complete training, benchmarking & serialization script
│   ├── ticket_classification_updated.ipynb       # Interactive Jupyter notebook
│   ├── enhanced_ticket_classifier.joblib         # Production serialized ensemble pipeline
│   ├── enhanced_customer_support_model_90plus.pkl# Serialized model artifact backup
│   └── enhanced_model_metadata.json              # Model metrics, per-class stats & metadata
└── documentation/                                # Academic papers, presentations & research docs
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/irfan-shekh/ai_ticket.git
cd ai_ticket
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
```bash
python download.py
```

### 5. Configure Email Credentials (Optional)
To enable real email dispatch, configure your SMTP settings in [app.py](app.py) or set environment variables:
```python
app.config['SMTP_SERVER'] = 'smtp.gmail.com'
app.config['SMTP_PORT'] = 587
app.config['EMAIL_USER'] = 'your-email@gmail.com'
app.config['EMAIL_PASSWORD'] = 'your-gmail-app-password'
```

---

## 🚀 Running the Application

Start the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Default Credentials
| Role | Username | Password |
| :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` |
| **Demo User** | `demo` | Register a new account on `/register` |

---

## 🧪 Model Training & Retraining

To train or retrain the enhanced classification pipeline:

```bash
python models/train_model.py
```

This script will:
1. Ingest `models/customer_support_tickets_train_split.csv` (5,156 verified training records) and `models/customer_support_tickets_test_split.csv` (1,290 strictly held-out records).
2. Augment training with curated domain patterns and colloquial conversational variations (**5,676 train / 1,290 test samples**; 0 overlap / 100% disjoint).
3. Apply dual-granularity word (1–2 ngrams, 12,000 max features) and character (3–5 ngrams, 20,000 max features) TF-IDF feature extraction.
4. Train and benchmark candidate algorithms: **Calibrated LinearSVC**, **Logistic Regression**, **Complement Naive Bayes**, and a soft **Voting Ensemble**.
5. Output detailed classification reports, precision, recall, and confusion matrices.
6. Export production models to `models/enhanced_ticket_classifier.joblib` and `models/enhanced_customer_support_model_90plus.pkl`.
7. Update `models/enhanced_model_metadata.json` with current performance figures.

You can also run [models/ticket_classification_updated.ipynb](models/ticket_classification_updated.ipynb) interactively in Jupyter or Google Colab.

---

## 📡 REST API Reference

### `POST /classify`
Submits a ticket description for automated AI classification and storage.

* **Headers**: `Content-Type: application/json`
* **Authentication**: Requires active user session

#### Request Body
```json
{
  "ticket_text": "I was charged twice on my credit card for invoice #INV-4820",
  "notification_email": "customer@example.com"
}
```

#### Response (`200 OK`)
```json
{
  "category": "Billing",
  "confidence": 84.4,
  "confidence_level": "High",
  "needs_human_review": false,
  "top_predictions": [
    {"category": "Billing", "confidence": 84.4},
    {"category": "Technical Support", "confidence": 8.1},
    {"category": "General Inquiry", "confidence": 4.5}
  ],
  "description": "Payment issues, invoices, billing questions, and subscription renewals",
  "ticket_id": 14,
  "email_sent": true,
  "success": true
}
```

> **Note**: For ambiguous queries (< 3 words or < 12 characters, e.g. *"help please"*), confidence is automatically capped at 50% and `"needs_human_review": true` is returned to prompt manual agent triage.

---

## 📊 Benchmark Results (Verified Disjoint Test Set)

Evaluated on the strictly held-out test partition (`customer_support_tickets_test_split.csv`, 1,290 samples, 0 train-test leakage):

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Billing** | **1.000** | **0.996** | **0.998** | 250 |
| **Technical Support** | **0.996** | **1.000** | **0.998** | 268 |
| **General Inquiry** | **0.996** | **0.992** | **0.994** | 249 |
| **Returns** | **0.989** | **0.996** | **0.992** | 264 |
| **Complaint** | **0.992** | **0.988** | **0.990** | 259 |
| **Overall Accuracy** | — | — | **99.46%** | **1,290 test samples** |
| **Weighted F1-Score**| — | — | **0.995** | — |
| **Macro F1-Score**   | — | — | **0.995** | — |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
