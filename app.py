import sys
import os

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    import psycopg2
    import psycopg2.extras
    HAVE_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    HAVE_PSYCOPG2 = False
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import json
from werkzeug.utils import secure_filename

# Import TicketAI RAG, LLM, DB, and Orchestration services
from src.database.connection import get_db as get_pooled_db, init_db as init_pg_db
from src.database.repositories import KnowledgeRepository, TicketRepository
from src.rag.kb_manager import KnowledgeBaseManager
from src.rag.retriever import Retriever
from src.services.ticket_orchestrator import TicketOrchestrator
import logging

logger = logging.getLogger("ticketai.app")

# Load environment variables from .env file (with built-in zero-dependency fallback)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# Configure NLTK data directory (support serverless environments like Vercel / AWS Lambda)
# Serverless containers have read-only root filesystems; downloads must use /tmp or be handled gracefully.
nltk_data_dir = '/tmp/nltk_data' if (os.getenv('VERCEL') or os.path.exists('/tmp')) else os.path.join(os.path.expanduser('~'), 'nltk_data')
try:
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
except Exception:
    pass

for _item, _pkg in [('corpora/stopwords', 'stopwords'), ('corpora/wordnet', 'wordnet')]:
    try:
        nltk.data.find(_item)
    except LookupError:
        try:
            nltk.download(_pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception as _e:
            print(f"Notice: Optional NLTK resource {_pkg} download skipped: {_e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-123')

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Database Configuration: Neon PostgreSQL
RAW_DB_URL = os.getenv('DATABASE_URL') or os.getenv('POSTGRES_URL') or os.getenv('POSTGRES_PRISMA_URL')

if RAW_DB_URL:
    # Fix postgres:// scheme for psycopg2 compatibility
    if RAW_DB_URL.startswith('postgres://'):
        RAW_DB_URL = RAW_DB_URL.replace('postgres://', 'postgresql://', 1)
    # Ensure sslmode=require for Neon if not explicitly set
    if 'sslmode' not in RAW_DB_URL:
        sep = '&' if '?' in RAW_DB_URL else '?'
        RAW_DB_URL = f"{RAW_DB_URL}{sep}sslmode=require"
    app.config['DATABASE_TYPE'] = 'postgres'
    app.config['DATABASE_URL'] = RAW_DB_URL
    print("Configured for Neon PostgreSQL database.")
else:
    app.config['DATABASE_TYPE'] = 'postgres'
    app.config['DATABASE_URL'] = None
    print("Warning: DATABASE_URL not set in environment. Set DATABASE_URL to your Neon connection string.")

# Email configuration (loaded from .env)
app.config['SMTP_SERVER'] = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
app.config['SMTP_PORT'] = int(os.getenv('SMTP_PORT', 587))
app.config['EMAIL_USER'] = os.getenv('EMAIL_USER', '')
app.config['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD', '')

# Admin credentials (loaded from .env)
app.config['ADMIN_USERNAME'] = os.getenv('ADMIN_USERNAME', 'admin')
app.config['ADMIN_PASSWORD'] = os.getenv('ADMIN_PASSWORD', 'admin123')


# Initialize text preprocessing tools
try:
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', 'down', 'without', 'off'}
    stop_words = stop_words - negation_words
except Exception:
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'}

lemmatizer = WordNetLemmatizer()

# Triage and Out-of-Scope Decision Layer
from models.triage_decision_layer import get_default_triage_layer, TriageDecisionLayer
triage_layer = get_default_triage_layer(min_confidence=60.0, min_margin=0.15)

# Primary Model Preprocessing & Triage Pipeline
# Uses Calibrated LinearSVC + TriageDecisionLayer (14.41 ms latency, 0 external neural network dependencies)



def preprocess_text(text):
    """Production-grade text preprocessing for ticket classification"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    
    # URL and email replacement
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Expand common contractions
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "cannot", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    # Filter special characters keeping letters and numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            try:
                lemmatized = lemmatizer.lemmatize(token)
            except Exception:
                lemmatized = token
            cleaned_tokens.append(lemmatized)
            
    return ' '.join(cleaned_tokens)

# Emotion to category mapping (used when emotion classifier is available)
emotion_to_category = {
    'anger': 'Complaint',
    'disgust': 'Complaint', 
    'fear': 'Technical Support',
    'joy': 'General Inquiry',
    'neutral': 'General Inquiry',
    'sadness': 'Returns',
    'surprise': 'Billing'
}

# Category descriptions
category_descriptions = {
    "Technical Support": "Issues related to login, access, software, or technical problems",
    "Returns": "Product returns, refunds, exchanges, and damaged items", 
    "Billing": "Payment issues, invoices, billing questions, and subscription renewals",
    "General Inquiry": "General questions about products, services, hours, or company information",
    "Complaint": "Customer complaints regarding service quality, staff, or delayed delivery"
}

# Category resolution messages
category_resolution_messages = {
    "Technical Support": "Our technical team has resolved your issue. If you need further assistance, please don't hesitate to contact us.",
    "Returns": "Your return request has been processed. You should receive confirmation and next steps shortly.",
    "Billing": "Your billing inquiry has been addressed. Please check your account for updates.",
    "General Inquiry": "Thank you for your inquiry. We've provided the information you requested.",
    "Complaint": "We appreciate you bringing this to our attention. We've addressed your concerns and taken appropriate action."
}

# -----------------------------------------------------------------------------
# Production Model Loader
# -----------------------------------------------------------------------------
production_model = None
production_metadata = {}

def load_production_model():
    """Loads the trained high-accuracy classification pipeline from models/ directory"""
    global production_model, production_metadata
    
    model_candidates = [
        os.path.join(os.path.dirname(__file__), 'models', 'enhanced_ticket_classifier.joblib'),
        os.path.join(os.path.dirname(__file__), 'models', 'enhanced_customer_support_model_90plus.pkl'),
        os.path.join('models', 'enhanced_ticket_classifier.joblib')
    ]
    
    for candidate in model_candidates:
        if os.path.exists(candidate):
            try:
                production_model = joblib.load(candidate)
                print(f"Loaded production ticket classifier from: {candidate}")
                break
            except Exception as e:
                print(f"Warning: Failed loading model from {candidate}: {e}")
                
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'enhanced_model_metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                production_metadata = json.load(f)
        except Exception:
            pass

# Initialize production model on startup
load_production_model()


def send_email(to_email, subject, html_body):
    """Send email notification"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = app.config['EMAIL_USER']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to server and send email
        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['EMAIL_USER'], app.config['EMAIL_PASSWORD'])
        
        text = msg.as_string()
        server.sendmail(app.config['EMAIL_USER'], to_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_ticket_creation_email(to_email, ticket_id, category, ticket_text, confidence):
    """Send email notification for ticket creation"""
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: white; }}
            .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 30px; }}
            .ticket-info {{ background-color: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; }}
            .category-badge {{ 
                display: inline-block; 
                padding: 8px 16px; 
                border-radius: 20px; 
                color: white; 
                font-weight: bold;
                background: {'#0d6efd' if category == 'Technical Support' else '#ffc107' if category == 'Returns' else '#198754' if category == 'Billing' else '#0dcaf0' if category == 'General Inquiry' else '#dc3545'};
            }}
            .confidence-bar {{ 
                width: 100%; 
                height: 10px; 
                background-color: #e9ecef; 
                border-radius: 5px; 
                overflow: hidden; 
                margin: 10px 0;
            }}
            .confidence-fill {{ 
                height: 100%; 
                background: linear-gradient(90deg, #667eea, #764ba2); 
                width: {confidence}%;
            }}
            .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; }}
            .btn {{ 
                display: inline-block; 
                padding: 12px 24px; 
                background: linear-gradient(135deg, #667eea, #764ba2); 
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎫 TicketAI</h1>
                <p>Your ticket has been successfully raised and classified!</p>
            </div>
            
            <div class="content">
                <h2>Ticket Details</h2>
                
                <div class="ticket-info">
                    <p><strong>Ticket ID:</strong> #{ticket_id}</p>
                    <p><strong>Date:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                    <p><strong>Status:</strong> <span style="color: #198754; font-weight: bold;">Open</span></p>
                </div>
                
                <h3>AI Classification Result</h3>
                <p><strong>Category:</strong> <span class="category-badge">{category}</span></p>
                <p><strong>Description:</strong> {category_descriptions.get(category, '')}</p>
                
                <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill"></div>
                </div>
                
                <h3>Your Ticket Description</h3>
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-style: italic;">
                    "{ticket_text[:200]}{'...' if len(ticket_text) > 200 else ''}"
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="#" class="btn">View Dashboard</a>
                </div>
                
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h4 style="color: #1976d2; margin-top: 0;">What's Next?</h4>
                    <ul style="color: #555;">
                        <li>Our AI has automatically categorized your ticket</li>
                        <li>It will be routed to the appropriate support team</li>
                        <li>You'll receive updates on your registered email</li>
                        <li>Average response time: 24-48 hours</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Thank you for using TicketAI!</p>
                <p style="font-size: 12px;">This is an automated message. Please do not reply to this email.</p>
                <p style="font-size: 12px;">If you have any questions, contact our support team.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"TicketAI - Your Ticket #{ticket_id} has been Raised"
    return send_email(to_email, subject, html_body)

def send_ticket_resolution_email(to_email, ticket_id, category, ticket_text, resolution_notes):
    """Send email notification for ticket resolution"""
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: white; }}
            .header {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 30px; }}
            .ticket-info {{ background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 20px; margin: 20px 0; }}
            .category-badge {{ 
                display: inline-block; 
                padding: 8px 16px; 
                border-radius: 20px; 
                color: white; 
                font-weight: bold;
                background: {'#0d6efd' if category == 'Technical Support' else '#ffc107' if category == 'Returns' else '#198754' if category == 'Billing' else '#0dcaf0' if category == 'General Inquiry' else '#dc3545'};
            }}
            .resolution-box {{ background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; }}
            .btn {{ 
                display: inline-block; 
                padding: 12px 24px; 
                background: linear-gradient(135deg, #28a745, #20c997); 
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✅ TicketAI - Ticket Resolved</h1>
                <p>Your support ticket has been successfully resolved!</p>
            </div>
            
            <div class="content">
                <h2>Ticket Resolution Details</h2>
                
                <div class="ticket-info">
                    <p><strong>Ticket ID:</strong> #{ticket_id}</p>
                    <p><strong>Category:</strong> <span class="category-badge">{category}</span></p>
                    <p><strong>Date Resolved:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                    <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">Resolved</span></p>
                </div>
                
                <h3>Your Original Ticket</h3>
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-style: italic; margin-bottom: 20px;">
                    "{ticket_text[:200]}{'...' if len(ticket_text) > 200 else ''}"
                </div>
                
                <h3>Resolution Details</h3>
                <div class="resolution-box">
                    <p><strong>Resolution Notes:</strong></p>
                    <p>{resolution_notes}</p>
                    <p style="margin-top: 15px; font-style: italic;">{category_resolution_messages.get(category, 'Thank you for contacting us.')}</p>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="#" class="btn">View Resolution Details</a>
                </div>
                
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h4 style="color: #1976d2; margin-top: 0;">Need Further Assistance?</h4>
                    <p style="color: #555;">If you have any additional questions or concerns, please don't hesitate to create a new ticket or contact our support team.</p>
                </div>
            </div>
            
            <div class="footer">
                <p>Thank you for using TicketAI!</p>
                <p style="font-size: 12px;">This is an automated message. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    subject = f"TicketAI - Your Ticket #{ticket_id} has been Resolved"
    return send_email(to_email, subject, html_body)

class PGCursorWrapper:
    """Wraps a psycopg2 cursor to provide an SQLite-compatible interface:
    - Automatically converts '?' placeholders to '%s'
    - Automatically captures lastrowid on INSERT queries via RETURNING id
    - Supports fetchone, fetchall, and iteration with RealDictCursor rows
    """
    def __init__(self, raw_cursor):
        self._cursor = raw_cursor
        self.lastrowid = None

    def execute(self, sql, params=None):
        if params is None:
            params = ()
        elif isinstance(params, list):
            params = tuple(params)
        elif not isinstance(params, tuple):
            params = (params,)
            
        # Replace ? with %s for psycopg2
        pg_sql = sql.replace('?', '%s')
        
        # Check if INSERT statement without RETURNING to emulate cursor.lastrowid
        trimmed = pg_sql.strip()
        is_insert = trimmed.upper().startswith('INSERT INTO')
        
        if is_insert and 'RETURNING' not in trimmed.upper():
            pg_sql_returning = trimmed.rstrip(';').rstrip() + ' RETURNING id;'
            try:
                self._cursor.execute(pg_sql_returning, params)
                row = self._cursor.fetchone()
                if row:
                    if isinstance(row, dict):
                        self.lastrowid = row.get('id')
                    elif isinstance(row, (tuple, list)):
                        self.lastrowid = row[0]
                return self
            except Exception:
                self._cursor.execute(pg_sql, params)
                self.lastrowid = None
                return self
        else:
            self._cursor.execute(pg_sql, params)
            self.lastrowid = None
            return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size=None):
        return self._cursor.fetchmany(size) if size else self._cursor.fetchmany()

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def close(self):
        try:
            self._cursor.close()
        except Exception:
            pass

    def __iter__(self):
        return iter(self._cursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PGConnectionWrapper:
    """Wraps a psycopg2 connection to mimic sqlite3.Connection methods (execute, commit, rollback, close, cursor)"""
    def __init__(self, raw_conn):
        self._conn = raw_conn

    def cursor(self):
        raw_cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return PGCursorWrapper(raw_cur)

    def execute(self, sql, params=None):
        cur = self.cursor()
        return cur.execute(sql, params)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        try:
            return self._conn.rollback()
        except Exception:
            pass

    def close(self):
        try:
            return self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


def init_db():
    """Initialize database schema and pgvector for Neon PostgreSQL"""
    try:
        return init_pg_db(embedding_dimension=768)
    except Exception as e:
        print(f"Notice: init_db fallback: {e}")
        return False


def get_db():
    """Get active pooled database connection for Neon PostgreSQL"""
    return get_pooled_db()


def row_to_dict(row):
    """Convert SQLite Row or psycopg2 RealDictRow to standard dictionary"""
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def rows_to_dict_list(rows):
    """Convert list of Row objects to list of dictionaries"""
    if not rows:
        return []
    return [row_to_dict(row) for row in rows]

def classify_ticket(text, return_top_k=False, return_structured=False):
    """
    Classify ticket using production trained LinearSVC pipeline and Triage Decision Layer.
    Guarantees out-of-scope detection, ambiguity triage, input validation, and category guardrails.
    """
    try:
        raw_text = str(text or "")[:5000].strip()
        
        # Use Triage Decision Layer with production model
        decision = triage_layer.decide(
            raw_text,
            primary_model=production_model,
            preprocessor_fn=preprocess_text
        )

        if return_structured:
            return decision
        if return_top_k:
            return decision['predicted_category'], decision['confidence'], decision['top_predictions']
        return decision['predicted_category'], decision['confidence']

    except Exception as e:
        print(f"Classification error: {e}")
        fallback_cat = "General Inquiry"
        fallback_preds = [{'category': fallback_cat, 'confidence': 50.0}]
        if return_structured:
            return {
                "predicted_category": fallback_cat,
                "confidence": 50.0,
                "confidence_level": "Low (Review Suggested)",
                "needs_human_review": True,
                "is_out_of_scope": True,
                "review_reason": f"System Exception: {str(e)[:50]}",
                "margin": 0.0,
                "top_predictions": fallback_preds,
                "domain_signals": 0
            }
        if return_top_k:
            return fallback_cat, 50.0, fallback_preds
        return fallback_cat, 50.0



@app.route('/')
def index():
    """Landing page - show to unauthenticated users"""
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template('landing.html', categories=category_descriptions)

@app.route('/landing')
def landing():
    """Explicit landing page route"""
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template('landing.html', categories=category_descriptions)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if admin login
        if username == app.config['ADMIN_USERNAME'] and password == app.config['ADMIN_PASSWORD']:
            session['user_id'] = 0  # Admin user ID
            session['username'] = 'admin'
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        
        # Regular user login
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_email'] = user['email']
            session['is_admin'] = False
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        db = get_db()
        try:
            password_hash = generate_password_hash(password)
            db.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                      (username, email, password_hash))
            db.commit()
            db.close()
            return redirect(url_for('login'))
        except Exception as e:
            err_msg = str(e).lower()
            if hasattr(db, 'rollback'):
                db.rollback()
            db.close()
            if 'unique' in err_msg or 'duplicate' in err_msg or 'integrity' in err_msg:
                return render_template('register.html', error='Username or email already exists')
            return render_template('register.html', error=f'Registration failed: {str(e)}')
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session or session.get('is_admin'):
        return redirect(url_for('login'))
    
    db = get_db()
    # Get all tickets for the user (both open and resolved)
    tickets = db.execute('''
        SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    # Convert Row objects to dictionaries for JSON serialization
    tickets_dict = rows_to_dict_list(tickets)
    
    # Attach AI resolution and feedback state for each ticket
    for t in tickets_dict:
        t['resolution'] = TicketRepository.get_latest_resolution(t['id'])
        fb = db.execute('SELECT * FROM ticket_feedback WHERE ticket_id = ? ORDER BY created_at DESC LIMIT 1', (t['id'],)).fetchone()
        t['feedback'] = row_to_dict(fb) if fb else None

    db.close()
    
    return render_template('dashboard.html', 
                         username=session['username'], 
                         tickets=tickets_dict,
                         categories=category_descriptions)

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard to view and manage all tickets with AI & Manual resolution options"""
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    # Get filter and search parameters
    status_filter = request.args.get('status', 'all')
    category_filter = request.args.get('category', 'all')
    priority_filter = request.args.get('priority', 'all')
    method_filter = request.args.get('method', 'all')
    search_query = request.args.get('search', '').strip()
    
    # Build query based on filters
    query = '''
        SELECT t.*, u.username, u.email 
        FROM tickets t 
        LEFT JOIN users u ON t.user_id = u.id 
        WHERE 1=1
    '''
    params = []
    
    if status_filter != 'all':
        if status_filter in ['pending_review', 'needs_review']:
            query += " AND (t.status = 'needs_review' OR t.needs_human_review = 1)"
        elif status_filter == 'escalated':
            query += " AND (t.status = 'escalated' OR t.status = 'needs_review')"
        elif status_filter == 'ai_suggested':
            query += " AND (t.status = 'awaiting_customer' AND (t.resolution_method = 'AI' OR t.resolution_method IS NULL))"
        else:
            query += ' AND t.status = %s'
            params.append(status_filter)
    
    if category_filter != 'all':
        query += ' AND t.predicted_category = %s'
        params.append(category_filter)

    if priority_filter != 'all':
        query += ' AND t.priority = %s'
        params.append(priority_filter)

    if method_filter != 'all':
        if method_filter.upper() == 'AI':
            query += " AND (t.resolution_method = 'AI' OR (t.resolution_method IS NULL AND t.status = 'awaiting_customer'))"
        elif method_filter.upper() == 'MANUAL':
            query += " AND t.resolution_method = 'MANUAL'"

    if search_query:
        if search_query.isdigit():
            query += ' AND (t.id = %s OR t.ticket_text ILIKE %s OR u.username ILIKE %s)'
            params.extend([int(search_query), f'%{search_query}%', f'%{search_query}%'])
        else:
            query += ' AND (t.ticket_text ILIKE %s OR u.username ILIKE %s OR COALESCE(t.subject, \'\') ILIKE %s)'
            params.extend([f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
    
    query += ' ORDER BY t.created_at DESC'
    
    with get_db() as db:
        tickets = db.execute(query, tuple(params)).fetchall()
        tickets_dict = [dict(t) for t in tickets] if tickets else []
        
        # Load KB documents for Knowledge Base Manager tab
        kb_docs = KnowledgeRepository.list_documents(active_only=False)
        for doc in kb_docs:
            doc['chunk_count'] = KnowledgeRepository.get_document_chunk_count(doc['id'])

    # Attach latest AI/Manual resolutions and events to tickets
    for t in tickets_dict:
        t['resolution'] = TicketRepository.get_latest_resolution(t['id'])
        t['events'] = TicketRepository.get_events(t['id'])
        # Default priority if empty
        if not t.get('priority'):
            t['priority'] = 'High' if t.get('needs_human_review') or float(t.get('confidence', 100)) < 60.0 else 'Medium'
    
    # Calculate live production analytics directly from NeonDB
    metrics = TicketRepository.get_admin_metrics()
    resolution_history = TicketRepository.list_resolution_history(limit=50)

    return render_template(
        'admin_dashboard.html', 
        tickets=tickets_dict,
        metrics=metrics,
        resolution_history=resolution_history,
        kb_documents=kb_docs,
        categories=category_descriptions,
        status_filter=status_filter,
        category_filter=category_filter,
        priority_filter=priority_filter,
        method_filter=method_filter,
        search_query=search_query
    )

@app.route('/classify_preview', methods=['POST'])
def classify_preview_route():
    """Public lightweight inference endpoint for the landing page interactive sandbox."""
    try:
        data = request.get_json() or {}
        text = str(data.get('text', '') or data.get('ticket_text', '')).strip()
        if not text:
            return jsonify({'error': 'Please enter a message to classify'}), 400
        
        # Run through production model and triage layer
        decision = classify_ticket(text, return_structured=True)
        conf = decision['confidence']
        is_oos = bool(decision['is_out_of_scope'])
        needs_review = bool(decision['needs_human_review'] or (conf < 60.0))
        cat = 'Out of Scope' if is_oos else decision['predicted_category']

        return jsonify({
            'success': True,
            'predicted_category': cat,
            'raw_category': decision.get('raw_category', decision['predicted_category']),
            'display_category': cat,
            'confidence': conf,
            'confidence_level': 'Low (Review Suggested)' if (conf < 60.0 or needs_review) else decision['confidence_level'],
            'needs_human_review': needs_review,
            'is_out_of_scope': is_oos,
            'review_reason': decision['review_reason'],
            'margin': decision['margin'],
            'top_predictions': decision['top_predictions']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_ticket_route():
    if 'user_id' not in session or session.get('is_admin'):
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json() or {}
        ticket_text = data.get('ticket_text', '').strip()
        notification_email = data.get('notification_email', '').strip()
        
        if not ticket_text:
            return jsonify({'error': 'Please enter ticket text'}), 400
        
        # Validate email if provided
        if notification_email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', notification_email):
            return jsonify({'error': 'Please enter a valid email address'}), 400
        
        # Run unified TicketAI Orchestrator (ML Classification -> NeonDB -> RAG -> Gemini -> Validation)
        orchestrator = TicketOrchestrator()
        orch_res = orchestrator.process_ticket_submission(
            user_id=session['user_id'],
            ticket_text=ticket_text,
            notification_email=notification_email
        )

        ticket_id = orch_res['ticket_id']
        predicted_category = orch_res['category']
        raw_category = orch_res['raw_category']
        confidence = orch_res['confidence']
        is_oos = orch_res['is_out_of_scope']
        needs_review = orch_res['needs_human_review']
        review_reason = orch_res['review_reason']
        top_predictions = orch_res['top_predictions']

        top_categories = [
            {'category': p['category'], 'probability': p['confidence']}
            for p in top_predictions
        ]
        
        # Send email notification if email provided
        email_sent = False
        if notification_email:
            email_sent = send_ticket_creation_email(notification_email, ticket_id, predicted_category, ticket_text, confidence)
        
        desc = "This message is outside customer support scope and has been flagged for human review." if is_oos else category_descriptions.get(predicted_category, '')
        
        return jsonify({
            'category': predicted_category,
            'raw_category': raw_category,
            'display_category': predicted_category,
            'confidence': round(confidence, 2),
            'confidence_level': orch_res.get('confidence_level', 'Medium'),
            'status': orch_res.get('status', 'awaiting_customer'),
            'needs_human_review': needs_review,
            'ml_needs_review': orch_res.get('ml_needs_review', False),
            'is_out_of_scope': is_oos,
            'review_reason': review_reason,
            'top_predictions': top_predictions,
            'top_categories': top_categories,
            'description': desc,
            'ticket_id': ticket_id,
            'resolution': orch_res.get('resolution', {}),
            'sources': orch_res.get('sources', []),
            'email_sent': email_sent,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify_route():
    if 'user_id' not in session or session.get('is_admin'):
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json() or {}
        tickets = data.get('tickets', [])
        if not isinstance(tickets, list):
            return jsonify({'error': 'Input must be a JSON array of tickets'}), 400
        
        results = []
        for item in tickets:
            text = str(item.get('text', item) if isinstance(item, dict) else item).strip()
            if not text:
                continue
            decision = classify_ticket(text, return_structured=True)
            results.append({
                'text': text,
                'category': decision['predicted_category'],
                'confidence': round(decision['confidence'], 2),
                'confidence_level': decision['confidence_level'],
                'needs_human_review': decision['needs_human_review'],
                'is_out_of_scope': decision['is_out_of_scope'],
                'review_reason': decision['review_reason']
            })
        return jsonify({'results': results, 'total': len(results), 'success': True})
    except Exception as e:
        return jsonify({'error': f'Batch classification error: {str(e)}'}), 500

@app.route('/admin/resolve_manual', methods=['POST'])
def resolve_ticket_manually():
    """Admin route for submitting manual resolution to customer (Status -> awaiting_customer)"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized: Admin access required'}), 401
    
    try:
        data = request.get_json() or {}
        ticket_id = data.get('ticket_id')
        summary = str(data.get('summary', '')).strip()
        detailed_solution = str(data.get('detailed_solution', '') or data.get('resolution_notes', '')).strip()
        internal_notes = str(data.get('internal_notes', '')).strip()
        resolution_category = str(data.get('resolution_category', '')).strip()
        
        if not ticket_id:
            return jsonify({'error': 'Ticket ID is required'}), 400
        if not summary:
            summary = "Admin Manual Resolution"
        if not detailed_solution:
            return jsonify({'error': 'Detailed solution is required'}), 400
            
        ticket = TicketRepository.get_ticket(ticket_id)
        if not ticket:
            return jsonify({'error': f'Ticket #{ticket_id} not found'}), 404
            
        admin_id = session['user_id']
        res_id = TicketRepository.save_manual_resolution(
            ticket_id=ticket_id,
            admin_id=admin_id,
            summary=summary,
            detailed_solution=detailed_solution,
            internal_notes=internal_notes,
            resolution_category=resolution_category or ticket.get('predicted_category', 'General')
        )
        
        # Send resolution email to customer if notification email available
        email_sent = False
        dest_email = ticket.get('notification_email') or ticket.get('user_email')
        if dest_email:
            try:
                email_sent = send_ticket_resolution_email(
                    dest_email,
                    ticket_id,
                    ticket.get('predicted_category', 'General'),
                    ticket.get('ticket_text', ''),
                    detailed_solution
                )
            except Exception as mail_err:
                logger.warning(f"Failed sending manual resolution email for #{ticket_id}: {mail_err}")
                
        return jsonify({
            'success': True,
            'message': f'Manual resolution for Ticket #{ticket_id} successfully dispatched to customer. Status updated to Awaiting Customer.',
            'resolution_id': res_id,
            'status': 'awaiting_customer',
            'email_sent': email_sent
        })
    except Exception as e:
        logger.error(f"Error submitting manual resolution: {e}")
        return jsonify({'error': f'Error submitting manual resolution: {str(e)}'}), 500

@app.route('/admin/resolve_ticket', methods=['POST'])
def resolve_ticket():
    """Admin route to resolve a ticket manually (delegates to save_manual_resolution)"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        ticket_id = data.get('ticket_id')
        resolution_notes = str(data.get('resolution_notes', '')).strip()
        summary = str(data.get('summary', 'Admin Manual Resolution')).strip()
        internal_notes = str(data.get('internal_notes', '')).strip()
        
        if not ticket_id:
            return jsonify({'error': 'Ticket ID is required'}), 400
        if not resolution_notes:
            return jsonify({'error': 'Resolution notes are required'}), 400
        
        ticket = TicketRepository.get_ticket(ticket_id)
        if not ticket:
            return jsonify({'error': 'Ticket not found'}), 404
        
        admin_id = session['user_id']
        res_id = TicketRepository.save_manual_resolution(
            ticket_id=ticket_id,
            admin_id=admin_id,
            summary=summary,
            detailed_solution=resolution_notes,
            internal_notes=internal_notes,
            resolution_category=ticket.get('predicted_category', 'General')
        )
        
        email_sent = False
        dest_email = ticket.get('notification_email') or ticket.get('user_email')
        if dest_email:
            try:
                email_sent = send_ticket_resolution_email(
                    dest_email, ticket_id, ticket['predicted_category'], ticket['ticket_text'], resolution_notes
                )
            except Exception:
                pass
        
        return jsonify({
            'success': True,
            'message': 'Manual resolution submitted. Status updated to awaiting customer confirmation.',
            'status': 'awaiting_customer',
            'resolution_id': res_id,
            'email_sent': email_sent
        })
    except Exception as e:
        return jsonify({'error': f'Error resolving ticket: {str(e)}'}), 500

@app.route('/admin/ticket/<int:ticket_id>/generate_ai_solution', methods=['POST'])
def generate_ai_solution_api(ticket_id):
    """Generates an AI proposed solution using RAG + Gemini for admin review and editing."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized: Admin access required'}), 401

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': f'Ticket #{ticket_id} not found'}), 404

    from src.resolution.resolution_engine import ResolutionEngine
    res_engine = ResolutionEngine()
    result = res_engine.process_ticket(
        ticket_id=ticket_id,
        customer_message=ticket['ticket_text'],
        predicted_category=ticket['predicted_category'],
        confidence=float(ticket['confidence']),
        is_out_of_scope=bool(ticket.get('is_out_of_scope', 0)),
        needs_human_review_flag=bool(ticket.get('needs_human_review', 0)),
        review_reason=ticket.get('review_reason', '')
    )

    r_data = result.get('resolution_data', {})
    proposed = r_data.get('proposed_solution', [])
    if isinstance(proposed, list):
        formatted_proposed = "\n".join(f"{i+1}. {s}" for i, s in enumerate(proposed))
    else:
        formatted_proposed = str(proposed)

    return jsonify({
        'success': True,
        'message': 'AI proposed solution generated successfully.',
        'problem_summary': r_data.get('problem_summary', ''),
        'identified_issue': r_data.get('identified_issue', ''),
        'proposed_solution': formatted_proposed,
        'grounding_status': r_data.get('grounding_status', 'unverified'),
        'requires_human_review': r_data.get('requires_human_review', False),
        'escalation_reason': r_data.get('escalation_reason', ''),
        'sources': result.get('retrieved_documents', []),
        'model_name': r_data.get('model_name', 'gemini-3.8-flash'),
        'suggested_status': result.get('status')
    })

@app.route('/admin/ticket/<int:ticket_id>/approve_ai_solution', methods=['POST'])
def approve_ai_solution_api(ticket_id):
    """Admin approves, edits, and sends AI resolution to customer (Status -> awaiting_customer)."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized: Admin access required'}), 401

    data = request.get_json() or {}
    final_solution = str(data.get('final_solution', '')).strip()
    internal_notes = str(data.get('internal_notes', '')).strip()

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': f'Ticket #{ticket_id} not found'}), 404

    if not final_solution:
        latest_res = TicketRepository.get_latest_resolution(ticket_id)
        if latest_res and latest_res.get('proposed_solution'):
            final_solution = latest_res['proposed_solution']
        else:
            return jsonify({'error': 'Solution text is required'}), 400

    admin_id = session['user_id']
    TicketRepository.approve_and_send_ai_resolution(
        ticket_id=ticket_id,
        admin_id=admin_id,
        final_solution=final_solution,
        internal_notes=internal_notes
    )

    email_sent = False
    dest_email = ticket.get('notification_email') or ticket.get('user_email')
    if dest_email:
        try:
            email_sent = send_ticket_resolution_email(
                dest_email,
                ticket_id,
                ticket.get('predicted_category', 'General'),
                ticket.get('ticket_text', ''),
                final_solution
            )
        except Exception as mail_err:
            logger.warning(f"Failed sending email for approved AI resolution #{ticket_id}: {mail_err}")

    return jsonify({
        'success': True,
        'message': f'AI resolution for Ticket #{ticket_id} approved and sent to customer. Status set to Awaiting Customer.',
        'status': 'awaiting_customer',
        'email_sent': email_sent
    })

@app.route('/admin/resolution_history')
def get_resolution_history_api():
    """Returns resolution history across all tickets for the Admin console."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized: Admin access required'}), 401

    limit = request.args.get('limit', 100)
    try:
        limit = min(500, max(1, int(limit)))
    except ValueError:
        limit = 100

    history = TicketRepository.list_resolution_history(limit=limit)
    return jsonify({
        'success': True,
        'history': history,
        'count': len(history)
    })

@app.route('/api/ticket/<int:ticket_id>/reopen', methods=['POST'])
def reopen_ticket_api(ticket_id):
    """Customer or Admin reopens a ticket if problem is not resolved."""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404

    if not session.get('is_admin') and ticket.get('user_id') != session.get('user_id'):
        return jsonify({'error': 'Unauthorized to reopen this ticket'}), 403

    data = request.get_json() or {}
    reason = str(data.get('reason', 'Customer reported issue is not resolved.')).strip()

    TicketRepository.reopen_ticket(ticket_id, reason=reason, user_id=session.get('user_id'))
    return jsonify({
        'success': True,
        'message': f'Ticket #{ticket_id} has been reopened and queued for specialist review.',
        'status': 'reopened'
    })

@app.route('/api/tickets')
def api_tickets():
    """API endpoint to get tickets as JSON"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    db = get_db()
    
    if session.get('is_admin'):
        # Admin can see all tickets
        tickets = db.execute('''
            SELECT t.*, u.username, u.email 
            FROM tickets t 
            LEFT JOIN users u ON t.user_id = u.id 
            ORDER BY t.created_at DESC
        ''').fetchall()
    else:
        # Regular users only see their own tickets (both open and resolved)
        tickets = db.execute('''
            SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC
        ''', (session['user_id'],)).fetchall()
    
    db.close()
    
    # Convert Row objects to dictionaries for JSON serialization
    tickets_dict = rows_to_dict_list(tickets)
    
    return jsonify({'tickets': tickets_dict})

@app.route('/api/ticket/<int:ticket_id>/resolution')
def get_ticket_resolution_api(ticket_id):
    """Retrieve detailed AI resolution, sources, events, and feedback for a ticket."""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404

    # Permission check: user can only view their own ticket unless admin
    if not session.get('is_admin') and ticket.get('user_id') != session.get('user_id'):
        return jsonify({'error': 'Unauthorized to view this ticket'}), 403

    resolution = TicketRepository.get_latest_resolution(ticket_id)
    events = TicketRepository.get_events(ticket_id)

    # Fetch feedback if exists
    with get_db() as db:
        fb_rows = db.execute("SELECT * FROM ticket_feedback WHERE ticket_id = %s ORDER BY created_at DESC LIMIT 1;", (ticket_id,)).fetchall()
        feedback = dict(fb_rows[0]) if fb_rows else None

    return jsonify({
        'success': True,
        'ticket': ticket,
        'resolution': resolution,
        'events': events,
        'feedback': feedback
    })


@app.route('/api/ticket/<int:ticket_id>/approve_resolution', methods=['POST'])
def approve_ticket_resolution(ticket_id):
    """Admin approves and sends proposed AI solution to customer (Status -> awaiting_customer)."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    custom_notes = str(data.get('resolution_notes', '') or data.get('final_solution', '')).strip()
    internal_notes = str(data.get('internal_notes', '')).strip()

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404

    resolution = TicketRepository.get_latest_resolution(ticket_id)
    final_solution = custom_notes or (resolution.get('proposed_solution') if resolution else 'Approved by support supervisor.')

    admin_id = session['user_id']
    TicketRepository.approve_and_send_ai_resolution(
        ticket_id=ticket_id,
        admin_id=admin_id,
        final_solution=final_solution,
        internal_notes=internal_notes
    )

    # Send resolution email if address available
    email_sent = False
    dest_email = ticket.get('notification_email') or ticket.get('user_email')
    if dest_email:
        try:
            email_sent = send_ticket_resolution_email(
                dest_email, ticket_id, ticket['predicted_category'], ticket['ticket_text'], final_solution
            )
        except Exception as mail_err:
            logger.warning(f"Could not send approval email for #{ticket_id}: {mail_err}")

    return jsonify({
        'success': True,
        'message': f'Resolution for Ticket #{ticket_id} approved and dispatched to customer. Status set to Awaiting Customer.',
        'status': 'awaiting_customer',
        'email_sent': email_sent
    })


@app.route('/api/ticket/<int:ticket_id>/escalate', methods=['POST'])
def escalate_ticket_api(ticket_id):
    """Escalates a ticket for manual review."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    reason = data.get('reason', 'Escalated by user/admin for manual specialist inspection.').strip()

    with get_db() as db:
        db.execute(
            """
            UPDATE tickets
            SET status = 'needs_review', needs_human_review = 1, review_reason = %s
            WHERE id = %s;
            """,
            (reason, ticket_id)
        )
        db.commit()

    TicketRepository.update_status(ticket_id, 'needs_review', f"Escalated: {reason}")
    return jsonify({'success': True, 'message': f'Ticket #{ticket_id} escalated for human review.'})


@app.route('/api/ticket/<int:ticket_id>/regenerate', methods=['POST'])
def regenerate_resolution_api(ticket_id):
    """Regenerates AI resolution for a ticket."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404

    from src.resolution.resolution_engine import ResolutionEngine
    res_engine = ResolutionEngine()
    result = res_engine.process_ticket(
        ticket_id=ticket_id,
        customer_message=ticket['ticket_text'],
        predicted_category=ticket['predicted_category'],
        confidence=float(ticket['confidence']),
        is_out_of_scope=bool(ticket.get('is_out_of_scope', 0)),
        needs_human_review_flag=bool(ticket.get('needs_human_review', 0)),
        review_reason=ticket.get('review_reason', '')
    )

    return jsonify({
        'success': True,
        'message': 'Resolution regenerated successfully',
        'resolution': result.get('resolution_data', {}),
        'sources': result.get('retrieved_documents', []),
        'status': result.get('status')
    })


@app.route('/api/ticket/<int:ticket_id>/feedback', methods=['POST'])
def submit_ticket_feedback_api(ticket_id):
    """Customer submits feedback rating and confirms resolution status."""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401

    ticket = TicketRepository.get_ticket(ticket_id)
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404

    # Regular users can only submit feedback for their own tickets
    if not session.get('is_admin') and ticket.get('user_id') != session.get('user_id'):
        return jsonify({'error': 'Unauthorized to submit feedback for this ticket'}), 403

    data = request.get_json() or {}
    try:
        rating = int(data.get('rating', 5))
        rating = max(1, min(5, rating))
    except (ValueError, TypeError):
        rating = 5

    was_resolved = bool(data.get('was_resolved', False))
    comment = str(data.get('comment', '')).strip()

    latest_res = TicketRepository.get_latest_resolution(ticket_id)
    resolution_id = latest_res.get('id') if latest_res else None

    fb_id = TicketRepository.save_feedback(
        ticket_id=ticket_id,
        resolution_id=resolution_id,
        rating=rating,
        comment=comment,
        was_resolved=was_resolved
    )

    return jsonify({
        'success': True,
        'feedback_id': fb_id,
        'was_resolved': was_resolved,
        'message': 'Thank you! Your feedback has been recorded and ticket status updated.'
    })


# -----------------------------------------------------------------------------
# Knowledge Base Management Endpoints
# -----------------------------------------------------------------------------

@app.route('/admin/knowledge_base')
def list_knowledge_base_api():
    """Returns all knowledge base documents and their chunk stats."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    category = request.args.get('category')
    docs = KnowledgeRepository.list_documents(active_only=False, category=category)
    for d in docs:
        d['chunk_count'] = KnowledgeRepository.get_document_chunk_count(d['id'])

    return jsonify({'success': True, 'documents': docs, 'total': len(docs)})


@app.route('/admin/kb/upload', methods=['POST'])
def upload_kb_document_api():
    """Uploads and indexes PDF, TXT, or Markdown documents into NeonDB."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    category = request.form.get('category', 'General Inquiry').strip()
    title = request.form.get('title', '').strip() or file.filename

    filename = secure_filename(file.filename)
    upload_dir = os.path.join(BASE_DIR, 'knowledge_base')
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, f"upload_{int(datetime.now().timestamp())}_{filename}")

    try:
        file.save(temp_path)
        kb_manager = KnowledgeBaseManager()
        content = kb_manager.extract_text_from_file(temp_path)

        res = kb_manager.ingest_document(
            title=title,
            content=content,
            category=category,
            source=f"upload:{filename}",
            version="1.0"
        )
        return jsonify({
            'success': True,
            'message': f"Document '{title}' ingested successfully with {res.get('chunks_indexed', 0)} chunks.",
            'document': res
        })
    except Exception as e:
        return jsonify({'error': f"Failed to ingest document: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route('/admin/kb/add_faq', methods=['POST'])
def add_kb_faq_api():
    """Manually adds a verified FAQ article to knowledge base."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    question = str(data.get('question', '')).strip()
    answer = str(data.get('answer', '')).strip()
    category = str(data.get('category', 'General Inquiry')).strip()

    if not question or not answer:
        return jsonify({'error': 'Both question and answer are required.'}), 400

    try:
        kb_manager = KnowledgeBaseManager()
        res = kb_manager.ingest_faq(question=question, answer=answer, category=category)
        return jsonify({
            'success': True,
            'message': 'FAQ added and indexed into pgvector.',
            'document': res
        })
    except Exception as e:
        return jsonify({'error': f"Error adding FAQ: {str(e)}"}), 500


@app.route('/admin/kb/toggle/<int:doc_id>', methods=['POST'])
def toggle_kb_document_api(doc_id):
    """Activates or deactivates a document from retrieval."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    doc = KnowledgeRepository.get_document_by_id(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    new_active = not doc['is_active'] if 'is_active' not in data else bool(data['is_active'])
    KnowledgeRepository.set_document_active(doc_id, new_active)
    return jsonify({'success': True, 'document_id': doc_id, 'is_active': new_active})


@app.route('/admin/kb/delete/<int:doc_id>', methods=['POST'])
def delete_kb_document_api(doc_id):
    """Deletes a document and cascades deletion of its vector chunks."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    KnowledgeRepository.delete_document(doc_id)
    return jsonify({'success': True, 'message': f'Document #{doc_id} deleted.'})


@app.route('/admin/kb/reindex/<int:doc_id>', methods=['POST'])
def reindex_kb_document_api(doc_id):
    """Reindexes document chunks and embeddings."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    kb_manager = KnowledgeBaseManager()
    success = kb_manager.reindex_document(doc_id)
    if success:
        return jsonify({'success': True, 'message': f'Document #{doc_id} reindexed successfully.'})
    return jsonify({'error': 'Failed to reindex document'}), 400


# -----------------------------------------------------------------------------
# Analytics Endpoint (Computed from Real NeonDB Records)
# -----------------------------------------------------------------------------

@app.route('/admin/analytics')
def get_analytics_api():
    """Returns production analytics calculated directly from NeonDB."""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    metrics = TicketRepository.get_admin_metrics()

    category_distribution = []
    kb_docs_count = 0
    kb_chunks_count = 0

    try:
        with get_db() as db:
            # Category distribution
            cat_rows = db.execute("""
                SELECT predicted_category, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM tickets
                GROUP BY predicted_category
                ORDER BY count DESC;
            """).fetchall()
            category_distribution = [dict(r) for r in cat_rows]

            # Knowledge Base stats
            kb_docs_count = db.execute("SELECT COUNT(*) as cnt FROM knowledge_documents WHERE is_active = TRUE;").fetchone()["cnt"]
            kb_chunks_count = db.execute("SELECT COUNT(*) as cnt FROM knowledge_chunks kc JOIN knowledge_documents kd ON kc.document_id = kd.id WHERE kd.is_active = TRUE;").fetchone()["cnt"]
    except Exception as db_err:
        logger.warning(f"Could not query live category distribution: {db_err}")

    metrics['category_distribution'] = category_distribution
    metrics['active_kb_documents'] = kb_docs_count
    metrics['active_kb_chunks'] = kb_chunks_count

    return jsonify({
        'success': True,
        'metrics': metrics
    })


@app.route('/logout')
def logout():
    """Logout route - clear session and redirect to landing page"""
    session.clear()
    return redirect(url_for('landing'))

# Custom JSON encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)