# REMOVE THESE TWO LINES IF THEY ARE PRESENT AT THE TOP OF YOUR FILE
# import nest_asyncio
# nest_asyncio.apply()


import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash
import logging
from datetime import datetime, timedelta # Import timedelta for time calculations
import re # Import for regular expressions
from functools import wraps # Import wraps for decorators
import base64 # Import base64 for image processing
import uuid # For generating unique reset tokens
import random # NEW: For generating random username suggestions
import string # NEW: For string manipulation in username generation
import json # NEW: For handling JSON responses for image/video prompting

# --- NEW IMPORTS FOR AUTHENTICATION ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# --- END NEW IMPORTS ---


# NEW: Import for specific Gemini API exceptions
from google.api_core import exceptions as google_api_exceptions


# NEW: Flask-Mail imports for email sending
from flask_mail import Mail, Message


app = Flask(__name__)


# --- NEW: Flask-SQLAlchemy Configuration ---
# Configure SQLite database. This file will be created in your project directory.
# On Render, this database file will be ephemeral unless you attach a persistent disk.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_that_should_be_changed')
db = SQLAlchemy(app)


# --- Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # NEW: Set the login view for redirection


# NEW: Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com' # Use your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER')
mail = Mail(app)


# --- Gemini API Configuration ---
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# Configure logging
logging.basicConfig(level=logging.INFO)


# --- NEW: Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(36), unique=True, nullable=True)
    token_expiration = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_reset_token(self):
        self.reset_token = str(uuid.uuid4())
        self.token_expiration = datetime.now() + timedelta(hours=1)
        db.session.commit()
        return self.reset_token


# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- NEW: Admin Required Decorator ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cv-maker')
@login_required # NEW: Protect this route
def cv_maker():
    return render_template('cv_maker.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html', username=username, email=email)
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('register.html', username=username, email=email)

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = user.get_reset_token()
            msg = Message('Password Reset Request', recipients=[user.email])
            msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
            try:
                mail.send(msg)
                flash('An email has been sent with instructions to reset your password.', 'info')
            except Exception as e:
                app.logger.error(f"Failed to send password reset email: {e}")
                flash('An error occurred while sending the email. Please try again later.', 'danger')
        else:
            flash('There is no account with that email.', 'warning')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or user.token_expiration < datetime.now():
        flash('That is an invalid or expired token.', 'warning')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password')
        user.set_password(password)
        user.reset_token = None
        user.token_expiration = None
        db.session.commit()
        flash('Your password has been updated! You are now able to log in.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html')


@app.route('/admin')
@login_required
@admin_required # NEW: Protect with admin decorator
def admin_panel():
    users = User.query.all()
    return render_template('admin_panel.html', users=users)


@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required # NEW: Protect with admin decorator
def delete_user(user_id):
    user_to_delete = User.query.get_or_404(user_id)
    if user_to_delete.is_admin:
        flash('Cannot delete an admin user.', 'danger')
    else:
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f'User {user_to_delete.username} has been deleted.', 'success')
    return redirect(url_for('admin_panel'))


@app.route('/generate_llm_response', methods=['POST'])
async def generate_llm_response():
    data = request.json
    prompt = data.get('prompt')
    is_json = data.get('is_json', False)
    prompt_mode = data.get('prompt_mode', 'text') # Get the new prompt_mode

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    parts = []

    # Handle image prompting logic
    if prompt_mode == 'image':
        image_data = data.get('image_data')
        if not image_data:
            return jsonify({'error': 'Image data is required for image mode.'}), 400
        
        parts.append(prompt)
        parts.append({
            'mime_type': 'image/jpeg',  # Assuming JPEG, you can generalize this
            'data': base64.b64decode(image_data.split(',')[1])
        })
        
        # Determine the response schema for image prompting
        if is_json:
            schema = {
                "type": "OBJECT",
                "properties": {
                    "meta": {
                        "type": "OBJECT",
                        "properties": {
                            "styleName": {"type": "STRING"},
                            "aspectRatio": {"type": "STRING"}
                        }
                    },
                    "camera": {
                        "type": "OBJECT",
                        "properties": {
                            "model": {"type": "STRING"},
                            "lens": {"type": "STRING"},
                            "focalLength": {"type": "STRING"}
                        }
                    },
                    "subject": {
                        "type": "OBJECT",
                        "properties": {
                            "details": {"type": "STRING"},
                            "pose": {"type": "STRING"}
                        }
                    },
                    "environment": {
                        "type": "OBJECT",
                        "properties": {
                            "setting": {"type": "STRING"},
                            "timeOfDay": {"type": "STRING"},
                            "era": {"type": "STRING"}
                        }
                    },
                    "lighting": {
                        "type": "OBJECT",
                        "properties": {
                            "source": {"type": "STRING"},
                            "direction": {"type": "STRING"},
                            "quality": {"type": "STRING"}
                        }
                    },
                    "fx": {
                        "type": "OBJECT",
                        "properties": {
                            "stylizations": {"type": "STRING"}
                        }
                    },
                    "colorGrading": {
                        "type": "OBJECT",
                        "properties": {
                            "palette": {"type": "STRING"}
                        }
                    },
                    "style": {
                        "type": "OBJECT",
                        "properties": {
                            "artDirection": {"type": "STRING"}
                        }
                    },
                    "rendering": {
                        "type": "OBJECT",
                        "properties": {
                            "engine": {"type": "STRING"}
                        }
                    },
                    "postEditing": {
                        "type": "OBJECT",
                        "properties": {
                            "treatments": {"type": "STRING"}
                        }
                    }
                }
            }
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        else:
            generation_config = {}
            
    # Handle video prompting logic
    elif prompt_mode == 'video':
        parts.append(prompt)
        # Determine the response schema for video prompting
        if is_json:
            schema = {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "duration": {"type": "STRING"},
                    "aspect_ratio": {"type": "STRING"},
                    "style": {
                        "type": "OBJECT",
                        "properties": {
                            "overall_aesthetic": {"type": "STRING"},
                            "genre": {"type": "STRING"},
                            "mood_tone": {"type": "STRING"}
                        }
                    },
                    "scenes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "scene_number": {"type": "NUMBER"},
                                "description": {"type": "STRING"},
                                "visuals": {"type": "STRING"},
                                "camera_work": {"type": "STRING"}
                            }
                        }
                    }
                }
            }
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        else:
            generation_config = {}

    # Handle text prompting logic (existing logic)
    else: # prompt_mode == 'text'
        parts.append(prompt)
        if is_json:
            # Generate a flexible JSON schema based on the prompt's intent
            schema = {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "A title for the content."},
                    "content": {"type": "STRING", "description": "The main content of the response."},
                    "keywords": {"type": "ARRAY", "items": {"type": "STRING"}}
                }
            }
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        else:
            generation_config = {}

    try:
        if generation_config:
            response = await asyncio.to_thread(model.generate_content, parts, generation_config=generation_config)
        else:
            response = await asyncio.to_thread(model.generate_content, parts)

        # Check for empty content
        if not response or not response.text:
            return jsonify({'error': 'LLM returned an empty response. Please try again.'}), 500

        # Attempt to parse JSON if requested, otherwise return raw text
        if is_json:
            try:
                # The model's response.text is already a JSON string
                parsed_json = json.loads(response.text)
                return jsonify({'content': parsed_json, 'is_json': True})
            except json.JSONDecodeError:
                app.logger.error("JSON decode error. Model did not return valid JSON.")
                # Fallback to returning raw text if JSON parsing fails
                return jsonify({'content': response.text, 'is_json': False})
        else:
            return jsonify({'content': response.text, 'is_json': False})

    except google_api_exceptions.ResourceExhausted as e:
        app.logger.error(f"API Resource Exhausted: {e}")
        return jsonify({'error': 'Resource Exhausted. Please try a shorter prompt or reduce API calls.'}), 429
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


@app.route('/subscribe_to_newsletter', methods=['POST'])
def subscribe_to_newsletter():
    # Placeholder for subscription logic
    # In a real app, you would save the email to a database
    email = request.form.get('email')
    if email:
        # Here you would save the email to your database
        app.logger.info(f"New newsletter subscription from: {email}")
        return jsonify({'success': True, 'message': 'Thank you for subscribing!'})
    return jsonify({'success': False, 'message': 'Invalid email address.'}), 400


# --- Database Initialization (Run once to create tables) ---
# This block ensures tables are created when the app starts.
# In production, you might use Flask-Migrate or a separate script.
with app.app_context():
    db.create_all()
    app.logger.info("Database tables created/checked.")

    # NEW: Create an admin user if one doesn't exist for easy testing
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', is_admin=True)
        admin_user.set_password('adminpass') # Set a default password for the admin
        # For admin, set a dummy email or leave None if not required for testing password reset
        admin_user.email = 'admin@example.com' # Assign a dummy email for admin
        db.session.add(admin_user)
        db.session.commit()
        app.logger.info("Default admin user 'admin' created with password 'adminpass'.")


# --- Main App Run ---
if __name__ == '__main__':
    # Important: For async Flask routes, you should use an ASGI server in production.
    # For local development with auto-reloading, Hypercorn is a good choice.
    # To run with Hypercorn:
    # 1. Install it: pip install hypercorn
    # 2. Run: hypercorn app:app --reload
    # If you must use app.run() for quick tests and encounter the 'event loop closed' error,
    # you can use `nest_asyncio.apply()` (install with `pip install nest-asyncio`), but this is
    # generally not recommended for production as it can hide underlying architectural issues.
    app.run(debug=True)
