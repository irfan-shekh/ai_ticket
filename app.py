from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sqlite3
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


# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-123')
app.config['DATABASE'] = os.getenv('DATABASE', 'tickets.db')

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

# Optional Hugging Face model for emotion classification
emotion_classifier = None
try:
    import importlib.util
    # Only load Hugging Face pipeline if PyTorch is present and explicitly enabled
    if importlib.util.find_spec('torch') is not None and os.environ.get('ENABLE_HF_EMOTION'):
        from transformers import pipeline
        emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            return_all_scores=True
        )
except Exception:
    emotion_classifier = None


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

def check_column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Get table info
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in c.fetchall()]
    
    conn.close()
    return column_name in columns

def init_db():
    """Initialize SQLite database with proper migration handling"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create tickets table with additional admin fields
    c.execute('''CREATE TABLE IF NOT EXISTS tickets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  ticket_text TEXT NOT NULL,
                  predicted_category TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  status TEXT DEFAULT 'Open',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  resolved_at TIMESTAMP NULL,
                  resolved_by INTEGER NULL,
                  resolution_notes TEXT NULL,
                  notification_email TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Check if notification_email column exists, if not add it
    if not check_column_exists('tickets', 'notification_email'):
        print("Adding notification_email column to tickets table...")
        c.execute('ALTER TABLE tickets ADD COLUMN notification_email TEXT')
    
    # Check if admin columns exist, if not add them
    if not check_column_exists('tickets', 'resolved_at'):
        print("Adding admin columns to tickets table...")
        c.execute('ALTER TABLE tickets ADD COLUMN resolved_at TIMESTAMP NULL')
        c.execute('ALTER TABLE tickets ADD COLUMN resolved_by INTEGER NULL')
        c.execute('ALTER TABLE tickets ADD COLUMN resolution_notes TEXT NULL')
    
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def row_to_dict(row):
    """Convert SQLite Row object to dictionary"""
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}

def rows_to_dict_list(rows):
    """Convert list of SQLite Row objects to list of dictionaries"""
    return [row_to_dict(row) for row in rows]

def classify_ticket(text, return_top_k=False):
    """Classify ticket using production trained pipeline with calibrated confidence and safety guards"""
    try:
        raw_text = str(text or "").strip()
        processed_text = preprocess_text(raw_text)
        raw_words = raw_text.split()
        is_ambiguous = len(raw_words) < 3 or len(raw_text) < 12
        
        # 1. Use high-accuracy production model if loaded
        if production_model is not None:
            # Handle empty/whitespace text edge case
            if not processed_text:
                default_cat = "General Inquiry"
                default_preds = [{'category': default_cat, 'confidence': 40.0}]
                return (default_cat, 40.0, default_preds) if return_top_k else (default_cat, 40.0)
            
            probs = production_model.predict_proba([processed_text])[0]
            classes = production_model.classes_
            
            # Rank top predictions
            sorted_indices = np.argsort(probs)[::-1]
            top_predictions = [
                {
                    'category': classes[idx],
                    'confidence': round(float(probs[idx]) * 100, 2)
                }
                for idx in sorted_indices[:3]
            ]
            
            best_category = top_predictions[0]['category']
            best_confidence = top_predictions[0]['confidence']
            
            # Production Guardrail: Cap confidence for extremely brief/vague queries (< 3 words)
            if is_ambiguous and best_confidence > 55.0:
                best_confidence = 50.0
                top_predictions[0]['confidence'] = 50.0
            
            if return_top_k:
                return best_category, best_confidence, top_predictions
            return best_category, best_confidence
            
        # 2. Check Hugging Face model as secondary alternative
        elif emotion_classifier is not None:
            emotion_results = emotion_classifier(text)
            emotion_scores = {result['label']: result['score'] for result in emotion_results[0]}
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            hf_category = emotion_to_category.get(dominant_emotion, 'General Inquiry')
            hf_confidence = round(emotion_scores[dominant_emotion] * 100, 2)
            
            top_predictions = [{'category': hf_category, 'confidence': hf_confidence}]
            if return_top_k:
                return hf_category, hf_confidence, top_predictions
            return hf_category, hf_confidence
            
        else:
            # Safe default fallback
            default_cat = "General Inquiry"
            default_preds = [{'category': default_cat, 'confidence': 60.0}]
            if return_top_k:
                return default_cat, 60.0, default_preds
            return default_cat, 60.0
            
    except Exception as e:
        print(f"Classification error: {e}")
        fallback_cat = "General Inquiry"
        fallback_preds = [{'category': fallback_cat, 'confidence': 50.0}]
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
        except sqlite3.IntegrityError:
            db.close()
            return render_template('register.html', error='Username or email already exists')
    
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
    db.close()
    
    # Convert Row objects to dictionaries for JSON serialization
    tickets_dict = rows_to_dict_list(tickets)
    
    return render_template('dashboard.html', 
                         username=session['username'], 
                         tickets=tickets_dict,
                         categories=category_descriptions)

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard to view and manage all tickets"""
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    
    db = get_db()
    
    # Get filter parameters
    status_filter = request.args.get('status', 'all')
    category_filter = request.args.get('category', 'all')
    
    # Build query based on filters
    query = '''
        SELECT t.*, u.username, u.email 
        FROM tickets t 
        LEFT JOIN users u ON t.user_id = u.id 
        WHERE 1=1
    '''
    params = []
    
    if status_filter != 'all':
        query += ' AND t.status = ?'
        params.append(status_filter)
    
    if category_filter != 'all':
        query += ' AND t.predicted_category = ?'
        params.append(category_filter)
    
    query += ' ORDER BY t.created_at DESC'
    
    tickets = db.execute(query, params).fetchall()
    
    # Get statistics
    stats = db.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) as open,
            SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved,
            predicted_category,
            COUNT(*) as category_count
        FROM tickets 
        GROUP BY predicted_category
    ''').fetchall()
    
    db.close()
    
    tickets_dict = rows_to_dict_list(tickets)
    stats_dict = rows_to_dict_list(stats)
    
    return render_template('admin_dashboard.html', 
                         tickets=tickets_dict,
                         stats=stats_dict,
                         categories=category_descriptions,
                         status_filter=status_filter,
                         category_filter=category_filter)

@app.route('/classify', methods=['POST'])
def classify_ticket_route():
    if 'user_id' not in session or session.get('is_admin'):
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        ticket_text = data.get('ticket_text', '').strip()
        notification_email = data.get('notification_email', '').strip()
        
        if not ticket_text:
            return jsonify({'error': 'Please enter ticket text'}), 400
        
        # Validate email if provided
        if notification_email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', notification_email):
            return jsonify({'error': 'Please enter a valid email address'}), 400
        
        # Classify the ticket using production model
        predicted_category, confidence, top_predictions = classify_ticket(ticket_text, return_top_k=True)
        
        # Save to database
        db = get_db()
        cursor = db.execute('''
            INSERT INTO tickets (user_id, ticket_text, predicted_category, confidence, notification_email)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], ticket_text, predicted_category, confidence, notification_email))
        
        ticket_id = cursor.lastrowid
        db.commit()
        db.close()
        
        # Send email notification if email provided
        email_sent = False
        if notification_email:
            email_sent = send_ticket_creation_email(notification_email, ticket_id, predicted_category, ticket_text, confidence)
        
        needs_human_review = bool(confidence < 60.0 or len(ticket_text.split()) < 3)
        confidence_level = "High" if confidence >= 75 else "Medium" if confidence >= 60 else "Low (Review Suggested)"
        
        return jsonify({
            'category': predicted_category,
            'confidence': round(confidence, 2),
            'confidence_level': confidence_level,
            'needs_human_review': needs_human_review,
            'top_predictions': top_predictions,
            'description': category_descriptions.get(predicted_category, ''),
            'ticket_id': ticket_id,
            'email_sent': email_sent,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/admin/resolve_ticket', methods=['POST'])
def resolve_ticket():
    """Admin route to resolve a ticket"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        ticket_id = data.get('ticket_id')
        resolution_notes = data.get('resolution_notes', '').strip()
        
        if not ticket_id:
            return jsonify({'error': 'Ticket ID is required'}), 400
        
        if not resolution_notes:
            return jsonify({'error': 'Resolution notes are required'}), 400
        
        db = get_db()
        
        # Get ticket details
        ticket = db.execute('''
            SELECT t.*, u.email, u.username 
            FROM tickets t 
            LEFT JOIN users u ON t.user_id = u.id 
            WHERE t.id = ?
        ''', (ticket_id,)).fetchone()
        
        if not ticket:
            db.close()
            return jsonify({'error': 'Ticket not found'}), 404
        
        # Update ticket status
        db.execute('''
            UPDATE tickets 
            SET status = 'Resolved', 
                resolved_at = CURRENT_TIMESTAMP,
                resolved_by = ?,
                resolution_notes = ?
            WHERE id = ?
        ''', (session['user_id'], resolution_notes, ticket_id))
        
        db.commit()
        db.close()
        
        # Send resolution email
        email_sent = False
        if ticket['notification_email']:
            email_sent = send_ticket_resolution_email(
                ticket['notification_email'],
                ticket_id,
                ticket['predicted_category'],
                ticket['ticket_text'],
                resolution_notes
            )
        elif ticket['email']:  # Use user's registered email if no notification email
            email_sent = send_ticket_resolution_email(
                ticket['email'],
                ticket_id,
                ticket['predicted_category'],
                ticket['ticket_text'],
                resolution_notes
            )
        
        return jsonify({
            'success': True,
            'message': 'Ticket resolved successfully',
            'email_sent': email_sent
        })
        
    except Exception as e:
        return jsonify({'error': f'Error resolving ticket: {str(e)}'}), 500

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