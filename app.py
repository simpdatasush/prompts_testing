# REMOVE THESE TWO LINES IF THEY ARE PRESENT AT THE TOP OF YOUR FILE
# import nest_asyncio
# import nest_asyncio
# nest_asyncio.apply()


import asyncio
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, send_file
import logging
from datetime import datetime, timedelta # Import timedelta for time calculations
import re # Import for regular expressions
from functools import wraps # Import wraps for decorators
import base64 # Import base64 for image processing
import uuid # For generating unique reset tokens
import random # NEW: For generating random username suggestions
import string # NEW: For string manipulation in username generation
import json # NEW: For handling JSON responses for image/video prompting
import time # For latency tracking 
import requests # For Perplexity API HTTP calls (if we stick to that instead of the SDK) 
# If using the provided SDK:
from perplexity import Perplexity, APIError as PerplexityAPIError

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
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_should_be_in_env') # Needed for Flask-Login sessions
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30) # Remember user for 30 days, re-added from previous app.py
db = SQLAlchemy(app)
# --- END NEW: Flask-SQLAlchemy Configuration ---


# --- NEW: Flask-Login Configuration ---
# app.py (Around line 41-46)
# --- NEW: Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info' 

# NEW FIX: Custom handler for unauthorized AJAX requests
@login_manager.unauthorized_handler
def unauthorized():
    if request.blueprint == 'api' or request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"error": "Authentication required. Please log in.", "redirect": url_for('login')}), 401
    return redirect(url_for('login'))
# --- END NEW FIX ---
# --- END NEW: Flask-Login Configuration ---


# --- NEW: Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.hostinger.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'info@promptsgenerator.ai'
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD') # IMPORTANT: Set this environment variable!
app.config['MAIL_DEFAULT_SENDER'] = 'info@promptsgenerator.ai' # Set default sender
mail = Mail(app)
# --- END NEW: Flask-Mail Configuration ---


# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO) # Simplified logging setup


# --- Cooldown and Daily Limit Configuration ---
COOLDOWN_SECONDS = 60 # 60 seconds cooldown as requested
FREE_DAILY_LIMIT = 3 # New default for free users
PAID_DAILY_LIMIT = 10 # New default for paid users
PAYMENT_LINK = "https://buymeacoffee.com/simpaisush"


# --- Language Mapping for Gemini Instructions ---
LANGUAGE_MAP = {
 "en-US": "English",
 "en-GB": "English (UK)",
 "es-ES": "Spanish",
 "fr-FR": "French",
 "de-DE": "German",
 "it-IT": "Italian",
 "ja-JP": "Japanese",
 "ko-KR": "Korean",
 "zh-CN": "Simplified Chinese",
 "hi-IN": "Hindi"
}

# https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash

# --- Configure Google Gemini API ---
# Ensure your GOOGLE_API_KEY is set in your environment variables
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# --- NEW: Perplexity API Configuration ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
# Initialize the Perplexity Client
# This client will be used inside the new ask_perplexity function
if PERPLEXITY_API_KEY:
    PERPLEXITY_CLIENT = Perplexity(api_key=PERPLEXITY_API_KEY)
else:
    PERPLEXITY_CLIENT = None
# --- End Perplexity Configuration ---

# --- NEW: Three-Tier Dynamic Model Selection Logic ---
def get_dynamic_model_name(prompt_instruction: str) -> str:
    """
    Selects the best LLM (Gemini or Perplexity) based on the complexity (length)
    of the prompt instruction using character count thresholds.
    """
    prompt_length = len(prompt_instruction)
    
    # Tier 3: Very Complex (>900 chars) -> Use Perplexity Sonar Pro
    if prompt_length > 900:
        model_name = 'sonar-pro'
    
    # Tier 2: Moderately Complex (300 to 900 chars) -> Use Gemini 2.5 Flash
    elif prompt_length >= 300:
        model_name = 'gemini-2.5-flash'
        
    # Tier 1: Simple/Cost-Effective (<300 chars) -> Use Gemini 2.0 Flash
    else:
        model_name = 'gemini-2.0-flash'

    app.logger.info(f"LLM Selected: {model_name} (Prompt length: {prompt_length})")
    return model_name
# --- END NEW: Three-Tier Dynamic Model Selection Logic ---

vision_model = genai.GenerativeModel('gemini-2.5-flash') # KEEP this line
# DELETE the old text_model and structured_gen_model definitions

# --- UPDATED: User Model for SQLAlchemy and Flask-Login ---
# Added allowed_categories and allowed_personas columns for access control
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    last_generation_time = db.Column(db.DateTime, default=datetime.min) # Track last generation time for cooldown
    daily_generation_count = db.Column(db.Integer, default=0) # Track daily generation count
    daily_generation_date = db.Column(db.Date, default=datetime.now().date()) # Track date for daily count reset
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiration = db.Column(db.DateTime, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=True) # Email field for password reset
    
    # NEW: Columns for user management and payment
    is_locked = db.Column(db.Boolean, default=False)
    subscription_type = db.Column(db.String(50), nullable=False, default='free') # 'free', 'one-time', 'monthly'
    payment_date = db.Column(db.DateTime, nullable=True)
    daily_limit = db.Column(db.Integer, default=FREE_DAILY_LIMIT) # NEW: Per-user daily limit
    api_key = db.Column(db.String(100), unique=True, nullable=True) # NEW: API Key for each user

    # NEW: Column for Gamification Points
    total_points = db.Column(db.Integer, default=0)

    # NEW: Columns for category and persona access control (stored as JSON strings)
    # Default to empty JSON list to indicate no specific restrictions initially
    allowed_categories = db.Column(db.Text, nullable=False, default='[]')
    allowed_personas = db.Column(db.Text, nullable=False, default='[]')


    # Relationship for saved prompts (using SavedPrompt model)
    saved_prompts = db.relationship('SavedPrompt', backref='author', lazy=True)
    raw_prompts = db.relationship('RawPrompt', backref='requester', lazy=True)
    news_items = db.relationship('NewsItem', backref='admin_poster', lazy=True) # Renamed from News
    job_listings = db.relationship('JobListing', backref='admin_poster', lazy=True) # Renamed from Job
    api_logs = db.relationship('ApiRequestLog', backref='api_user', lazy=True) # NEW: API Request logs

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', Admin: {self.is_admin}')"

# --- SavedPrompt Model (retained from previous app.py for persistence) ---
class SavedPrompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False) # e.g., 'polished', 'creative', 'technical'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"SavedPrompt('{self.type}', '{self.timestamp}')"

# --- RawPrompt Model for storing user's raw input requests ---
class RawPrompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"RawPrompt('{self.raw_text[:30]}...', '{self.timestamp}')"

# --- NewsItem Model (renamed from News for consistency) ---
class NewsItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    published_date = db.Column(db.Date, nullable=True) # Changed to Date type
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Admin who added it

    def __repr__(self):
        return f"NewsItem('{self.title}', '{self.published_date}')"

# app.py (New Gift Model added around line 197)
class Gift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    points_required = db.Column(db.Integer, default=0, nullable=False)
    is_active = db.Column(db.Boolean, default=True) # Used instead of timestamp for sorting/reposting
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Admin who added it

    def __repr__(self):
        return f"Gift('{self.name}', '{self.points_required} points')"

# NOTE: The User model already has 'total_points' from the previous steps.

# --- JobListing Model (renamed from Job for consistency) ---
class JobListing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=True)
    url = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    published_date = db.Column(db.Date, nullable=True) # Changed to Date type
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Admin who added it

    def __repr__(self):
        return f"JobListing('{self.title}', '{self.company}', '{self.location}')"

# NEW: SamplePrompt Model for the new collection
class SamplePrompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    raw_prompt = db.Column(db.Text, nullable=False) # Changed to not nullable
    polished_prompt = db.Column(db.Text, nullable=False)
    creative_prompt = db.Column(db.Text, nullable=False)
    technical_prompt = db.Column(db.Text, nullable=False)
    display_type = db.Column(db.String(20), nullable=False, default='polished') # New column
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"SamplePrompt('{self.polished_prompt[:30]}...', '{self.timestamp}')"

# NEW: NewsletterSubscriber Model (from provided docx)
class NewsletterSubscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_on = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Subscriber {self.email}>'

# NEW: ApiRequestLog Model for tracking API usage
class ApiRequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    request_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    endpoint = db.Column(db.String(100), nullable=False)
    status_code = db.Column(db.Integer, nullable=False)
    latency_ms = db.Column(db.Float, nullable=False)
    raw_input = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"ApiRequestLog(user_id={self.user_id}, endpoint='{self.endpoint}', status_code={self.status_code})"

 # app.py (New model added around line 11331, after ApiRequestLog)

# ... (Existing ApiRequestLog model around line 11331) ...

# NEW: Model for Auto-Saved LLM Test Responses
class AutoSavedResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    test_text = db.Column(db.Text, nullable=False) # The returned sample response text
    input_prompt = db.Column(db.String(250), nullable=True) # The prompt used as input for context
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"AutoSavedResponse('{self.test_text[:30]}...', '{self.timestamp}')"

# --- Helper Function to Log LLM Responses/Errors (Retained from previous plan) ---
# ... (log_llm_response function definition must follow LLMResponseLog model definition) ...


# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Admin Required Decorator ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have administrative access.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- API Key Required Decorator ---
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        user = User.query.filter_by(api_key=api_key).first()
        if not user:
            return jsonify({"error": "Invalid or missing API key."}), 401
        
        # Check for locked status
        if user.is_locked:
            return jsonify({"error": "Account is locked. Please contact support."}), 403
            
        return f(user, *args, **kwargs)
    return decorated_function


# --- Helper Function for Exponential Backoff ---
async def generate_content_with_retry(model, prompt_parts, generation_config=None, stream=False):
    retries = 0
    max_retries = 5
    base_delay = 1 # seconds

    while retries < max_retries:
        try:
            # If generation_config is provided, use it. Otherwise, use default.
            if generation_config:
                response = await model.generate_content_async(prompt_parts, generation_config=generation_config, stream=stream)
            else:
                response = await model.generate_content_async(prompt_parts, stream=stream)

            # Accessing response.text or iterating response for stream will raise exceptions on error
            if stream:
                full_response = ""
                for chunk in response:
                    full_response += chunk.text
                return full_response
            else:
                return response.text

        except google_api_exceptions.ResourceExhausted as e:
            retries += 1
            delay = base_delay * (2 ** retries) + (random.randint(0, 1000) / 1000) # Add jitter
            logging.warning(f"ResourceExhausted error: {e}. Retrying in {delay:.2f} seconds... (Attempt {retries}/{max_retries})")
            await asyncio.sleep(delay)
        except google_api_exceptions.BlockedPromptException as e:
            logging.error(f"Prompt was blocked due to safety concerns: {e}")
            raise ValueError("Prompt blocked due to safety concerns. Please modify your input.")
        except google_api_exceptions.InternalServerError as e:
            retries += 1
            delay = base_delay * (2 ** retries) + (random.randint(0, 1000) / 1000)
            logging.error(f"Internal Server Error: {e}. Retrying in {delay:.2f} seconds... (Attempt {retries}/{max_retries})")
            await asyncio.sleep(delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred during content generation: {e}")
            raise

    raise Exception(f"Failed to generate content after {max_retries} retries.")


# --- Response Filtering Function (from provided docx) ---
def filter_gemini_response(text):
    unauthorized_message = "I am not authorised to answer this question. My purpose is solely to refine your raw prompt into a machine-readable format."
    text_lower = text.lower()

    # Generic unauthorized phrases
    unauthorized_phrases = [
        "as a large language model", "i am an ai", "i was trained by", "my training data",
        "this application was built using", "the code for this app", "i cannot fulfill this request because",
        "i apologize, but i cannot", "i'm sorry, but i cannot", "i am unable to", "i do not have access",
        "i am not able to", "i cannot process", "i cannot provide", "i am not programmed",
        "i cannot generate", "i cannot give you details about my internal workings",
        "i cannot discuss my creation or operation", "i cannot explain the development of this tool",
        "my purpose is to", "i am designed to", "i don't have enough information to"
    ]
    for phrase in unauthorized_phrases:
        if phrase in text_lower:
            if phrase == "i don't have enough information to" and \
               ("about the provided prompt" in text_lower or "based on your input" in text_lower or "to understand the context" in text_lower):
                continue
            return unauthorized_message

    # Generic bug/error phrases
    bug_phrases = [
        "a bug occurred", "i encountered an error", "there was an issue in my processing",
        "i made an error", "my apologies", "i cannot respond to that"
    ]
    for phrase in bug_phrases:
        if phrase in text_lower:
            return unauthorized_message

    # Specific filtering for Gemini API quota/internal errors
    if "you exceeded your current quota" in text_lower:
        return "You exceeded your current quota. Please try again later or check your plan and billing details."
    # Catch-all for any API-related error details
    if "error communicating with gemini api:" in text_lower or "no response from model." in text_lower:
        filtered_text = text
        filtered_text = re.sub(r"model: \"[a-zA-Z0-9-.]+\"", "model: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_metric: \"[^\"]+\"", "quota_metric: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_id: \"[^\"]+\"", "quota_id: \"[REDACTED]\"", filtered_text)
        filtered_text = re.sub(r"quota_dimensions \{[^\}]+\}", "quota_dimensions { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"links \{\s*description: \"[^\"]+\"\s*url: \"[^\"]+\"\s*\}", "links { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"retry_delay \{\s*seconds: \d+\s*\}", "retry_delay { [REDACTED] }", filtered_text)
        filtered_text = re.sub(r"\[violations \{.*?\}\s*,?\s*links \{.*?\}\s*,?\s*retry_delay \{.*?\}\s*\]", "", filtered_text, flags=re.DOTALL)
        filtered_text = re.sub(r"\[violations \{.*?\}\s*\]", "", filtered_text, flags=re.DOTALL) # In case only violations are present

        if "google.api_core.exceptions" in filtered_text.lower() or "api_key" in filtered_text.lower():
            return "There was an issue with the AI service. Please try again later."
        
        return filtered_text.strip()

    return text

# --- NEW: Perplexity SDK interaction function ---
def ask_perplexity_for_text_prompt(prompt_instruction, model_name='sonar-pro', max_output_tokens=8192):
    if not PERPLEXITY_CLIENT:
        app.logger.error("Perplexity Client not initialized. API Key is missing.")
        return "Error: Perplexity API key is not configured."
    
    start_time = time.time()
    try:
        completion = PERPLEXITY_CLIENT.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_instruction}
            ],
            max_tokens=max_output_tokens,
            temperature=0.1
        )
        end_time = time.time()
        latency = (end_time - start_time) * 1000 # Convert to milliseconds

        pplx_text = completion.choices[0].message.content
        app.logger.info(f"Perplexity call succeeded. Model: {model_name}, Latency: {latency:.2f}ms")
        return pplx_text

    except PerplexityAPIError as e:
        app.logger.error(f"DEBUG: Perplexity API Error ({model_name}): {e}", exc_info=True)
        return f"Error communicating with Perplexity API: {str(e)}"
    except Exception as e:
        app.logger.error(f"DEBUG: Unexpected Error calling Perplexity API: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

# --- NEW: Master LLM Routing Function ---
def route_and_call_llm(raw_input, prompt_mode, instruction, max_output_tokens=8192):
    """
    Dispatches the LLM call based on the selected model name from the complexity check.
    """
    
    # 1. Structured/Multimedia Modes MUST use Gemini's specialized structured generation
    if prompt_mode in ['image_gen', 'video_gen']:
        app.logger.info("Routing to Gemini: Structured/Multimedia Mode.")
        # We enforce Gemini 2.5 Flash for structured tasks in its function
        return ask_gemini_for_structured_prompt(instruction, max_output_tokens=max_output_tokens)

    # 2. Dynamic Routing for Text Mode
    model_name = get_dynamic_model_name(instruction)
    
    if model_name.startswith('gemini'):
        # Route to Gemini API (using the selected model_name)
        return ask_gemini_for_text_prompt(instruction, model_name=model_name, max_output_tokens=max_output_tokens)
    
    elif model_name == 'sonar-pro':
        # Route to Perplexity API
        return ask_perplexity_for_text_prompt(instruction, model_name=model_name, max_output_tokens=max_output_tokens)
    
    else:
        app.logger.error(f"Unknown model name selected: {model_name}. Defaulting to Gemini 2.0 Flash.")
        return ask_gemini_for_text_prompt(instruction, model_name='gemini-2.0-flash', max_output_tokens=max_output_tokens)
# --- End Master LLM Routing Function —

# --- CORRECTED: Perplexity Search API Function ---
def perform_perplexity_search(query_text: str):
    """
    Performs a synchronous search using the Perplexity Search API.
    Returns a list of dictionaries containing title and URL, or an error message.
    """
    if not PERPLEXITY_CLIENT:
        app.logger.error("Perplexity Client not initialized. API Key is missing.")
        return {"error": "Perplexity API key is not configured."}

    try:
        # The user's prompt text is used as the query
        search_results = PERPLEXITY_CLIENT.search.create(
            query=[query_text],
            # Removed the problematic parameter: search_mode='web'
            # The API will now use its default search mode (which is typically web)
            max_results=5 
        )
        
        # Format the results into a clean list of dictionaries
        formatted_results = []
        for result in search_results.results:
            formatted_results.append({
                "title": result.title,
                "url": result.url
            })
            
        app.logger.info(f"Perplexity Search succeeded for query: {query_text[:50]}...")
        return {"results": formatted_results}

    except PerplexityAPIError as e:
        app.logger.error(f"Perplexity Search API Error: {e}", exc_info=True)
        return {"error": f"Error communicating with Perplexity API: {str(e)}"}
    except Exception as e:
        app.logger.error(f"Unexpected Error during Perplexity Search: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}
# --- END CORRECTED Perplexity Search API Function ---


# --- Gemini API interaction function (Synchronous wrapper for text_model) ---
# NOTE: This function now requires the specific model_name be passed in from the router.
def ask_gemini_for_text_prompt(prompt_instruction, model_name, max_output_tokens=8192):

    # Model is instantiated dynamically based on model_name passed from router
    model = genai.GenerativeModel(model_name)

    try:
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1
        }
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return raw_gemini_text
    except ValueError as e: 
        app.logger.error(f"DEBUG: Unexpected ValueError from Gemini API ({model_name}): {e}", exc_info=True)
        return filter_gemini_response(f"AI response failed due to an issue: {str(e)}")
    except google_api_exceptions.ResourceExhausted as e: # Explicitly catch 429 ResourceExhausted
        app.logger.error(f"DEBUG: Gemini API ResourceExhausted error: {e}", exc_info=True)
        # Custom friendly message for the user
        return f"Error communicating with SuperPrompter AI, please try after sometime."
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"DEBUG: Google API Call Error ({model_name}): {e}", exc_info=True)
        # Apply filter, but for low-level connection/generic API errors, use the filter's output
        return filter_gemini_response(f"Error communicating with SuperPrompter AI, please try after sometime.")
    except Exception as e:
        app.logger.error(f"DEBUG: Unexpected Error calling Gemini API ({model_name}): {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred: {str(e)}")

# --- Gemini API interaction function (Synchronous wrapper for structured_gen_model) ---
def ask_gemini_for_structured_prompt(prompt_instruction, generation_config=None, max_output_tokens=8192):

    # We enforce Gemini 2.5 Flash for all structured/multimedia tasks for reliability.
    model_name = 'gemini-2.5-flash'
    model = genai.GenerativeModel(model_name)
    
    # --- FIX: Initialize current_generation_config to ensure it's defined in scope ---
    current_generation_config = {} 
    # --- END FIX ---

    try:
        # We explicitly set response_mime_type to 'application/json' for structured output
        
        # REFINED LOGIC: Use a copy of the passed config, or an empty dict if None
        current_generation_config = generation_config.copy() if generation_config else {}
        
        if "max_output_tokens" not in current_generation_config:
            current_generation_config["max_output_tokens"] = max_output_tokens

        # Ensure response_mime_type is set for structured output
        current_generation_config["response_mime_type"] = "application/json"

        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=current_generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return raw_gemini_text
    except ValueError as e:
        app.logger.error(f"DEBUG: Unexpected ValueError from Gemini API (structured_gen_model - {model_name}): {e}", exc_info=True)
        return filter_gemini_response(f"AI structured generation failed due to an issue: {str(e)}")
    except google_api_exceptions.ResourceExhausted as e: # Explicitly catch 429 ResourceExhausted
        app.logger.error(f"DEBUG: Gemini API ResourceExhausted error: {e}", exc_info=True)
        # Custom friendly message for the user
        return f"Error communicating with SuperPrompter AI, please try after sometime."
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"DEBUG: Google API Call Error (structured_gen_model - {model_name}): {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with SuperPrompter AI, please try after sometime.")
    except Exception as e:
        app.logger.error(f"DEBUG: Unexpected Error calling Gemini API (structured_gen_model - {model_name}): {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred: {str(e)}")


# --- NEW: Gemini API for Image Understanding (Synchronous wrapper for vision_model) ---
def ask_gemini_for_image_text(image_data_bytes):
    try:
        response = vision_model.generate_content(prompt_parts)
        extracted_text = response.text if response and response.text else ""
        return extracted_text # Return raw text for further processing/filtering
    except ValueError as e: # <--- CATCH THE VALUE ERROR HERE
        app.logger.error(f"DEBUG: Unexpected ValueError from Gemini API (vision_model): {e}", exc_info=True)
        return filter_gemini_response(f"AI image text extraction failed due to an issue: {str(e)}")
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"Error calling Gemini API for image text extraction: {e}", exc_info=True)
        return filter_gemini_response(f"Error extracting text from image: {str(e)}")
    except Exception as e:
        app.logger.error(f"Unexpected Error calling Gemini API for image text extraction: {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred during image text extraction: {str(e)}")

# --- NEW GAMIFICATION HELPER FUNCTIONS ---
def calculate_generation_points(raw_input, prompt_mode, language_code, category, persona):
    """Calculates points based on complexity and settings usage."""
    points = 0
    raw_length = len(raw_input)
    
    # 1. Complexity-Based Points (Max 150)
    if raw_length < 300:
        points += 50    # Tier 1
    elif raw_length <= 900:
        points += 100   # Tier 2
    else:
        points += 150   # Tier 3

    # 2. Settings Utilization Points (Max 35)
    
    # Prompt Mode Select (5 Points - awarded if not 'text')
    if prompt_mode in ['image_gen', 'video_gen']:
        points += 5
        
    # Language Select (5 Points - awarded if not 'en-US')
    if language_code != 'en-US':
        points += 5
        
    # Category Select (10 Points - awarded if selected)
    if category:
        points += 10
        
    # Persona Select (15 Points - awarded if selected)
    if persona:
        points += 15
        
    return points


def award_refinement_points(raw_input):
    """Awards 25 points if the input appears to be a previous AI response (refinement)."""
    # Simple heuristic: look for AI output markers that suggest refinement
    if raw_input.startswith(('<pre>', '{', 'Based on the previous prompt')) or \
       re.search(r'\n{2,}', raw_input): 
        return 25
    return 0
# --- END GAMIFICATION HELPER FUNCTIONS ---

# Helper function to remove nulls recursively
def remove_null_values(obj):
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_null_values(elem) for elem in obj if elem is not None]
    else:
        return obj

# Define the JSON templates for image and video generation
IMAGE_JSON_TEMPLATE = {
    "meta": {
        "styleName": "gemini should decide",
        "aspectRatio": "gemini should decide",
        "seed": "gemini should decide"
    },
    "camera": {
        "model": "gemini should decide",
        "lens": "gemini should decide",
        "focalLength": "gemini should decide"
    },
    "subject": {
        "primary": "A {topic}", # Placeholder for user's topic
        "emotion": "gemini should decide",
        "pose": "gemini should decide"
    },
    "character": {
        "identity": "gemini should decide",
        "appearance": "gemini should decide",
        "wardrobe": "gemini should decide"
    },
    "composition": {
        "theory": "gemini should decide",
        "visualHierarchy": "gemini should decide"
    },
    "setting": {
        "environment": "gemini should decide",
        "architecture": "gemini should decide",
        "era": "gemini should decide"
    },
    "lighting": {
        "source": "gemini should decide",
        "direction": "gemini should decide",
        "quality": "gemini should decide"
    },
    "fx": {
        "stylizations": "gemini should decide",
        "atmospheric": "gemini should decide"
    },
    "colorGrading": {
        "palette": "gemini should decide",
        "toneMapping": "gemini should decide",
        "mood": "gemini should decide",
        "skinTones": "gemini should decide",
        "temperature": "gemini should decide"
    },
    "style": {
        "artDirection": "gemini should decide",
        "photographerReference": "gemini should decide",
        "overlay": "gemini should decide"
    },
    "rendering": {
        "engine": "gemini should decide",
        "fidelitySpec": "gemini should decide"
    },
    "postEditing": {
        "treatments": "gemini should decide",
        "filmStock": "gemini should decide"
    }
}

VIDEO_JSON_TEMPLATE = {
    "title": "Gemini should decide",
    "duration": "Gemini should decide",
    "aspect_ratio": "Gemini should decide",
    "model": "veo-3.0-fast", # This model is fixed as per your template
    "style": {
        "overall_aesthetic": "Gemini should decide",
        "visual_language": "Gemini should decide",
        "genre": "Gemini should decide",
        "mood_tone": "Gemini should decide"
    },
    "camera_style": {
        "lens_type": "Gemini should decide",
        "depth_of_field": "Gemini should decide",
        "framing_composition": "Gemini should decide",
        "film_properties": "Gemini should decide"
    },
    "camera_direction": [
        "Gemini should decide",
        "Gemini should decide",
        "Gemini should decide",
        "Gemini should decide",
        "Gemini should decide"
    ],
    "pacing": {
        "cut_frequency": "Gemini should decide",
        "motion_speed": "Gemini should decide",
        "rhythm": "Gemini should decide"
    },
    "special_effects": {
        "visual_effects": [
            "Gemini should decide",
            "Gemini should decide",
            "Gemini should decide"
        ],
        "atmospheric_effects": [
            "Gemini should decide",
            "Gemini should decide"
        ],
        "overlays_graphics": [
            "Gemini should decide",
            "Gemini should decide"
        ],
        "character_fx": "Gemini should decide"
    },
    "scenes": [
        {
            "time_range": "Gemini should decide",
            "description": "Gemini should decide",
            "specific_camera_action": "Gemini should decide"
        },
        {
            "time_range": "Gemini should decide",
            "description": "Gemini should decide",
            "specific_camera_action": "Gemini should decide"
        },
        {
            "time_range": "Gemini should decide",
            "description": "Gemini should decide.",
            "specific_camera_action": "Gemini should decide"
        },
        {
            "time_range": "Gemini should decide",
            "description": "Gemini should decide",
            "specific_camera_action": "Gemini should decide"
        }
    ],
    "audio": {
        "music": "Gemini should decide",
        "ambient_sounds": [
            "Gemini should decide",
            "Gemini should decide",
            "Gemini should decide"
        ],
        "sound_effects": [
            "Gemini should decide",
            "Gemini should decide",
            "Gemini should decide",
            "Gemini should decide"
        ],
        "mix_level": "Gemini should decide"
    },
    "voiceover": {
        "language": "Gemini should decide",
        "tone": "Gemini should decide",
        "script": "Gemini should decide"
    },
    "branding": {
        "product_name": "Gemini should decide",
        "brand_color": "Gemini should decide",
        "logo_display": "Gemini should decide"
    },
    "custom_elements": {
        "prohibited_elements": [
            "Gemini should decide",
            "Gemini should decide",
            "Gemini should decide"
        ],
        "specific_character_details": "Gemini should decide",
        "unique_physics": "Gemini should decide",
        "visual_banding": "gemini should decide"
    }
}


# --- generate_prompts_async function (main async logic for prompt variations) ---
async def generate_prompts_async(raw_input, language_code="en-US", prompt_mode='text', category=None, subcategory=None, persona=None): # NEW: Added persona parameter
    if not raw_input.strip():
        return {
            "polished": "Please enter some text to generate prompts.",
            "creative": "",
            "technical": "",
        }

    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "
    
    generation_config = {
        "temperature": 0.1 # Default temperature for all generations
    }
    
    polished_output = ""
    creative_output = ""
    technical_output = ""

    if prompt_mode == 'image_gen':
        # Create a copy of the template to fill
        current_image_template = IMAGE_JSON_TEMPLATE.copy()
        # Insert the user's raw input into the primary subject
        current_image_template["subject"]["primary"] = f"A {raw_input}"

        base_instruction = (
            "You are an expert AI image prompt generator. A user has provided a simple topic. "
            "Your task is to fill out the following JSON template to create a complete and detailed image prompt. "
            "For each field where 'gemini should decide' is written, you must provide a specific, creative, and plausible value "
            "that aligns with the topic. Ensure the primary subject is centered around the given topic. "
            "The output should be ONLY the completed JSON object, with no extra text or explanations. "
            "Analyze the following raw user input to block for explicit signs of malicious activity, illegal content, self-harm/suicide, or severe bad intent (e.g., hate speech)."
            "Make sure the JSON is perfectly valid and ready for parsing.\n\n"
            f"User topic: '{raw_input}'\n\n"
            f"JSON Template to fill:\n{json.dumps(current_image_template, indent=2)}"
        )
        
        main_prompt_result = await asyncio.to_thread(ask_gemini_for_structured_prompt, base_instruction, generation_config)

        if "Error" in main_prompt_result or "not configured" in main_prompt_result or "quota" in main_prompt_result.lower():
            return {
                "polished": main_prompt_result,
                "creative": "",
                "technical": "",
            }
        
        try:
            # Strip markdown code block fences before JSON parsing
            match = re.search(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", main_prompt_result, re.DOTALL)
            json_string_to_parse = match.group(1) if match else main_prompt_result

            parsed_json_obj = json.loads(json_string_to_parse)

            # Remove 'model' field if it exists and apply null value removal
            if "model" in parsed_json_obj:
                del parsed_json_obj["model"]
                logging.info(f"Removed 'model' field from generated image JSON.")
            
            cleaned_json_obj = remove_null_values(parsed_json_obj)
            formatted_json = json.dumps(cleaned_json_obj, indent=2)
            
            creative_output = formatted_json # Image JSON goes into creative output
            polished_output = ""
            technical_output = ""

        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON for image mode: {json_string_to_parse}")
            polished_output = creative_output = technical_output = f"Error: Failed to parse JSON response for image. Raw: {json_string_to_parse}"
        
        return {
            "polished": polished_output,
            "creative": creative_output,
            "technical": technical_output,
        }

    elif prompt_mode == 'video_gen':
        # Create a copy of the template to fill
        current_video_template = VIDEO_JSON_TEMPLATE.copy()
        # The 'model' field is fixed in the template, no need to change it here.

        base_instruction = (
            "You are an expert AI video prompt generator. A user has provided a simple topic. "
            "Your task is to fill out the following JSON template to create a complete and detailed video prompt. "
            "For each field where 'Gemini should decide' is written, you must provide a specific, creative, and plausible value "
            "that aligns with the topic. Think about how a professional video production would be structured. "
            "Analyze the following raw user input to block explicit signs of malicious activity, illegal content, self-harm/suicide, or severe bad intent (e.g., hate speech)."
            "The output should be ONLY the completed JSON object, with no extra text or explanations. "
            "Make sure the JSON is perfectly valid and ready for parsing. Ensure that the 'model' field remains 'veo-3.0-fast'.\n\n"
            f"User topic: '{raw_input}'\n\n"
            f"JSON Template to fill:\n{json.dumps(current_video_template, indent=2)}"
        )
        
        main_prompt_result = await asyncio.to_thread(ask_gemini_for_structured_prompt, base_instruction, generation_config)

        if "Error" in main_prompt_result or "not configured" in main_prompt_result or "quota" in main_prompt_result.lower():
            return {
                "polished": main_prompt_result,
                "creative": "",
                "technical": "",
            }

        try:
            # Strip markdown code block fences before JSON parsing
            match = re.search(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", main_prompt_result, re.DOTALL)
            json_string_to_parse = match.group(1) if match else main_prompt_result

            parsed_json_obj = json.loads(json_string_to_parse)
            
            cleaned_json_obj = remove_null_values(parsed_json_obj)
            formatted_json = json.dumps(cleaned_json_obj, indent=2)

            creative_output = formatted_json # Video JSON goes into creative output
            polished_output = ""
            technical_output = ""

        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON for video mode: {json_string_to_parse}")
            polished_output = creative_output = technical_output = f"Error: Failed to parse JSON response for video. Raw: {json_string_to_parse}"
        
        return {
            "polished": polished_output,
            "creative": creative_output,
            "technical": technical_output,
        }

    else: # Default text mode (contextual)
        context_str = ""
        if category:
            context_str += f"The user is looking for help with the category '{category}'"
            if subcategory:
                context_str += f" and the subcategory '{subcategory}'."
            else:
                context_str += "."
        if persona: # NEW: Add persona to context string
            if context_str: # If category/subcategory already added, append with "as"
                context_str += f" The response should be crafted from the perspective of a '{persona}'."
            else: # If no category/subcategory, start with persona
                context_str += f"Craft the response from the perspective of a '{persona}'."
             
        base_instruction = language_instruction_prefix + f"""Refine the following text into a clear, concise, and effective prompt for a large language model. {context_str} Improve grammar, clarity, and structure. Do not add external information, only refine the given text. Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided raw text into a better prompt. Avoid explicit signs of malicious activity, illegal content, self-harm/suicide, or severe bad intent (e.g., hate speech) Raw Text: {raw_input}"""
     
     # --- NEW: Master LLM Router Call (Three-Tier Dynamic Selection) ---
        main_prompt_result = await asyncio.to_thread(route_and_call_llm, raw_input=raw_input, prompt_mode=prompt_mode, instruction=base_instruction, max_output_tokens=8192)
     
     # --- END NEW ROUTER CALL ---
        # Apply filter_gemini_response here for all main prompt generations
        main_prompt_result = filter_gemini_response(main_prompt_result)

        if "Error" in main_prompt_result or "not configured" in main_prompt_result or "quota" in main_prompt_result.lower(): # Check for quota error
            return {
                "polished": main_prompt_result,
                "creative": "",
                "technical": "",
            }

        # For text mode, we expect a single string response that contains all three variants
        # This is a heuristic and might need refinement based on actual model output patterns.
        # For now, let's assume the model will output clearly labeled sections.
        polished_match = re.search(r"Polished Version:\s*(.*?)(?=\nCreative Version:|\nTechnical Version:|\Z)", main_prompt_result, re.DOTALL)
        creative_match = re.search(r"Creative Version:\s*(.*?)(?=\nTechnical Version:|\Z)", main_prompt_result, re.DOTALL)
        technical_match = re.search(r"Technical Version:\s*(.*)", main_prompt_result, re.DOTALL)

        polished_output = polished_match.group(1).strip() if polished_match else "Could not extract polished version."
        creative_output = creative_match.group(1).strip() if creative_match else "Could not extract creative version."
        technical_output = technical_match.group(1).strip() if technical_match else "Could not extract technical version."

        # Fallback if no specific sections are found
        if not (polished_match and creative_match and technical_match):
            # If the model just gives a single block of text, use it for all three
            polished_output = main_prompt_result.strip()
            creative_output = main_prompt_result.strip()
            technical_output = main_prompt_result.strip()
            logging.warning("Could not parse distinct polished, creative, technical sections. Using full response for all.")

        # Only generate creative/technical for text mode
        strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

        creative_max_output_tokens = 8192 
        technical_max_output_tokens = 8192 
     
        # Create coroutines for parallel execution, running synchronous calls in threads
        creative_instruction = (language_instruction_prefix + f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_output}{strict_instruction_suffix}")
        technical_instruction = (language_instruction_prefix + f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_output}{strict_instruction_suffix}")
     
        # 2. Define the coroutines using the new Master LLM Router
        # NOTE: Using 'route_and_call_llm' ensures dynamic model selection (including Sonar Pro)
        # based on the length of the instruction.
        creative_coroutine = asyncio.to_thread(route_and_call_llm,raw_input=raw_input,prompt_mode=prompt_mode,instruction=creative_instruction, max_output_tokens=creative_max_output_tokens)
        technical_coroutine = asyncio.to_thread(route_and_call_llm,raw_input=raw_input,prompt_mode=prompt_mode,instruction=technical_instruction,max_output_tokens=technical_max_output_tokens)     

        # This line remains the same
        creative_output_raw, technical_output_raw = await asyncio.gather(
            creative_coroutine, technical_coroutine
        )
        # Apply filter_gemini_response to creative and technical outputs as well
        creative_output = filter_gemini_response(creative_output_raw)
        technical_output = filter_gemini_response(technical_output_raw)


        return {
            "polished": polished_output,
            "creative": creative_output,
            "technical": technical_output,
        }


# --- NEW: Reverse Prompting function ---
async def generate_reverse_prompt_async(input_text, language_code="en-US", prompt_mode='text'):
    if not input_text.strip():
        return "Please provide text or code to infer a prompt from."

    # --- NEW: Disable reverse prompting for image_gen and video_gen modes ---
    if prompt_mode in ['image_gen', 'video_gen']:
        return "Reverse prompting is not applicable for image or video generation modes."
    # --- END NEW ---

    # Enforce character limit
    MAX_REVERSE_PROMPT_CHARS = 10000
    if len(input_text) > MAX_REVERSE_PROMPT_CHARS:
        return f"Input for reverse prompting exceeds the {MAX_REVERSE_PROMPT_CHARS} character limit. Please shorten your input."

    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    # Escape curly braces in input_text to prevent f-string parsing errors
    escaped_input_text = input_text.replace('{', '{{').replace('}', '}}')

    if prompt_mode == 'text':
        prompt_instruction = f"Analyze the following text/code and infer a concise, high-level prompt idea that could have generated it. Respond in {language_code}. Input: {escaped_input_text}"
        # The image_gen and video_gen cases below are now effectively unreachable due to the early return above.
        # However, keeping them for clarity of original intent if the restriction were to be lifted.
    elif prompt_mode == 'image_gen':
        prompt_instruction = f"The user has provided a natural language description for an image. Infer a concise, natural language prompt idea for image generation based on this input. Input: {escaped_input_text}"
    elif prompt_mode == 'video_gen':
        prompt_instruction = f"The user has provided a natural language description for a video. Infer a concise, natural language prompt idea for video generation based on this input. Input: {escaped_input_text}"
    else:
        prompt_instruction = f"Analyze the following text/code and infer a concise, high-level prompt idea that could have generated it. Respond in {language_code}. Input: {escaped_input_text}"

    app.logger.info(f"Sending reverse prompt instruction to Gemini (length: {len(prompt_instruction)} chars))")

    reverse_prompt_result = await asyncio.to_thread(ask_gemini_for_text_prompt, prompt_instruction, model_name='gemini-2.0-flash', max_output_tokens=8912)

    return filter_gemini_response(reverse_prompt_result) # Apply filter here

# --- NEW LEADERBOARD HELPER FUNCTION ---
def mask_username(username):
    """Masks a username: shows first 3 characters, replaces rest with asterisks."""
    if len(username) <= 3:
        return username
    return username[:3] + '*' * (len(username) - 3)
# --- END NEW HELPER ---

# --- Flask Routes ---

# UPDATED: Landing page route to fetch more news AND jobs
@app.route('/')
def landing():
    # Fetch latest 6 news items for the landing page
    news_items = NewsItem.query.order_by(NewsItem.timestamp.desc()).limit(6).all()
    # Fetch latest 6 job listings for the landing page
    job_listings = JobListing.query.order_by(JobListing.timestamp.desc()).limit(6).all()
    # NEW: Fetch latest 3 SamplePrompts for the landing page
    sample_prompts = SamplePrompt.query.order_by(SamplePrompt.timestamp.desc()).limit(3).all()

    # 1. Fetch Top 5 Users for Leaderboard
    top_users = User.query.with_entities(User.username, User.total_points).order_by(User.total_points.desc()).limit(5).all()

    leaderboard_data = []
    for rank, (username, points) in enumerate(top_users):
        leaderboard_data.append({
            'rank': rank + 1,
            'username': mask_username(username), # Now correctly defined
            'points': points
        })

    # 2. Fetch all active gifts
    gifts = Gift.query.filter_by(is_active=True).order_by(Gift.points_required.asc()).all()

    # 3. Process sample_prompts
    display_prompts = []
    for prompt in sample_prompts:
        display_prompt_text = getattr(prompt, prompt.display_type + '_prompt', prompt.polished_prompt)
        display_prompts.append({
            'id': prompt.id,
            'raw_prompt': prompt.raw_prompt,
            'display_prompt_text': display_prompt_text,
            'display_type': prompt.display_type,
            'timestamp': prompt.timestamp
        })

    return render_template('landing.html', 
                           news_items=NewsItem.query.order_by(NewsItem.timestamp.desc()).limit(6).all(), 
                           job_listings=JobListing.query.order_by(JobListing.timestamp.desc()).limit(6).all(), 
                           sample_prompts=display_prompts, 
                           leaderboard_data=leaderboard_data,
                           gifts=gifts,
                           current_user=current_user)

# UPDATED: Route to view a specific news item (using NewsItem model)
@app.route('/view_news/<int:news_id>')
def view_news(news_id):
    item = NewsItem.query.get_or_404(news_id)
    return render_template('shared_content_landing.html', item=item, item_type='news')


# UPDATED: Route to view a specific job listing (using JobListing model)
@app.route('/view_job/<int:job_id>')
def view_job(job_id):
    item = JobListing.query.get_or_404(job_id)
    return render_template('shared_content_landing.html', item=item, item_type='job')

# NEW: Route to view a specific sample prompt
@app.route('/view_prompt/<int:prompt_id>')
def view_prompt(prompt_id):
    item = SamplePrompt.query.get_or_404(prompt_id)
    return render_template('shared_content_landing.html', item=item, item_type='prompt')

# Renamed original index route to /app_home
@app.route('/app_home')
@login_required # REQUIRE LOGIN FOR APP HOME PAGE
def app_home():
    # Pass current_user object to the template to show login/logout status
    # Also pass allowed_categories and allowed_personas (parsed from JSON string)
    allowed_categories_list = json.loads(current_user.allowed_categories)
    allowed_personas_list = json.loads(current_user.allowed_personas)

    return render_template('index.html',
                           current_user=current_user,
                           allowed_categories=allowed_categories_list,
                           allowed_personas=allowed_personas_list)
# The link needs to be updated in index.html (not app.py route itself)

# NEW: LLM Benchmark Page Route
@app.route('/llm_benchmark')
def llm_benchmark():
    return render_template('llm_benchmark.html', current_user=current_user)


@app.route('/generate', methods=['POST'])
@login_required # Protect this route
def generate(): # CHANGED FROM ASYNC
    user = current_user # Get the current user object
    now = datetime.utcnow() # Use utcnow for consistency with database default

    # --- Check if the user is locked out ---
    if user.is_locked:
        return jsonify({
            "error": "Your account is locked. Please contact support.",
            "account_locked": True
        }), 403 # Forbidden


        
    # --- Cooldown Check using database timestamp ---
    if user.last_generation_time:
        time_since_last_request = (now - user.last_generation_time).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before generating new prompts.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429 # 429 Too Many Requests
    # --- END UPDATED ---

    # --- Daily Limit Check ---
    if not user.is_admin: # Admins are exempt from the daily limit
        today = now.date()
        if user.daily_generation_date != today:
            user.daily_generation_count = 0
            user.daily_generation_date = today
            db.session.add(user) # Mark user as modified
            db.session.commit() # Commit reset immediately to prevent race conditions on count

        if user.daily_generation_count >= user.daily_limit: # Check against per-user limit
            app.logger.info(f"User {user.username} exceeded their daily prompt limit of {user.daily_limit}.")
            # NEW: Return a specific payment message instead of just an error
            return jsonify({
                "error": f"You have reached your daily limit of {user.daily_limit} prompt generations. If you are looking for more prompts, kindly make a payment to increase your limit.",
                "daily_limit_reached": True,
                "payment_link": PAYMENT_LINK
            }), 429 # 429 Too Many Requests
    # --- END NEW: Daily Limit Check ---

    prompt_input = request.form.get('prompt_input', '').strip()
    language_code = request.form.get('language_code', 'en-US')
    is_json_mode = request.form.get('is_json_mode') == 'true'
    prompt_mode = request.form.get('prompt_mode', 'text') # 'text', 'image_gen', 'video_gen'
    category = request.form.get('category')
    subcategory = request.form.get('subcategory')
    persona = request.form.get('persona') # NEW

    # --- NEW: Server-side validation for allowed categories/personas ---
    # Convert stored JSON strings to Python lists for checks
    user_allowed_categories = json.loads(user.allowed_categories)
    user_allowed_personas = json.loads(user.allowed_personas)

    if not user.is_admin:
        if category and category not in user_allowed_categories:
            return jsonify({"error": "Selected category is not allowed for your account."}), 403
        if persona and persona not in user_allowed_personas:
            return jsonify({"error": "Selected persona is not allowed for your account."}), 403
    # --- END NEW: Server-side validation ---


    if not prompt_input:
        return jsonify({
            "polished": "Please enter some text to generate prompts.",
            "creative": "",
            "technical": "",
        })

    try:
        results = asyncio.run(generate_prompts_async(prompt_input, language_code, prompt_mode, category, subcategory, persona)) # NEW: Pass persona

    # --- GAMIFICATION: Award points for complexity, settings, and refinement ---
        points_awarded = calculate_generation_points(prompt_input, prompt_mode, language_code, category, persona)
        points_awarded += award_refinement_points(prompt_input)
    
        user.total_points += points_awarded
        app.logger.info(f"User {user.username} awarded {points_awarded} points for generation. Total: {user.total_points}")
    # --- END GAMIFICATION ---

        # --- Update last_generation_time in database and Save raw_input ---
        user.last_generation_time = now # Record the time of this successful request
        if not user.is_admin: # Only increment count for non-admin users
            user.daily_generation_count += 1
        db.session.add(user) # Add the user object back to the session to mark it as modified
        db.session.commit()
        app.logger.info(f"User {user.username}'s last prompt request time updated and count incremented. (Forward Prompt)")

        if current_user.is_authenticated:
            try:
                new_raw_prompt = RawPrompt(user_id=current_user.id, raw_text=prompt_input)
                db.session.add(new_raw_prompt)
                db.session.commit()
                app.logger.info(f"Raw prompt saved for user {current_user.username}")
            except Exception as e:
                app.logger.error(f"Error saving raw prompt for user {current_user.username}: {e}")
                db.session.rollback() # Rollback in case of error
        # --- END UPDATED ---

        return jsonify(results)
    except Exception as e:
        app.logger.exception("Error during prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500


# --- NEW: Reverse Prompt Endpoint ---
@app.route('/reverse_prompt', methods=['POST'])
@login_required
def reverse_prompt(): # CHANGED FROM ASYNC
    user = current_user
    now = datetime.utcnow()

    # --- Check if the user is locked out ---
    if user.is_locked:
        return jsonify({
            "error": "Your account is locked. Please contact support.",
            "account_locked": True
        }), 403 # Forbidden

    # Apply cooldown to reverse prompting as well
    if user.last_generation_time: # Changed from last_prompt_request
        time_since_last_request = (now - user.last_generation_time).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown for reverse prompt. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before performing another reverse prompt.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # --- NEW: Daily Limit Check for Reverse Prompt ---
    if not user.is_admin: # Admins are exempt from the daily limit
        today = now.date()
        if user.daily_generation_date != today: # Changed from last_count_reset_date
            user.daily_generation_count = 0
            user.daily_generation_date = today # Changed from last_count_reset_date
            db.session.add(user)
            db.session.commit()

        if user.daily_generation_count >= user.daily_limit: # Check against per-user limit
            app.logger.info(f"User {user.username} exceeded their daily reverse prompt limit of {user.daily_limit}.")
            # NEW: Return a specific payment message instead of just an error
            return jsonify({
                "error": f"You have reached your daily limit of {user.daily_limit} generations. If you are looking for more prompts, kindly make a payment to increase your limit.",
                "daily_limit_reached": True,
                "payment_link": PAYMENT_LINK
            }), 429
    # --- END NEW: Daily Limit Check ---

    data = request.get_json()
    input_text = data.get('input_text', '').strip()
    language_code = data.get('language_code', 'en-US')
    prompt_mode = data.get('prompt_mode', 'text')


    if not input_text:
        return jsonify({"error": "Please provide text or code to infer a prompt from."}), 400

    if prompt_mode in ['image_gen', 'video_gen']:
        return jsonify({"inferred_prompt": "Reverse prompting is not applicable for image or video generation modes."}), 200

    try:
        inferred_prompt = asyncio.run(generate_reverse_prompt_async(input_text, language_code, prompt_mode))

     # --- GAMIFICATION: Award points for reverse prompt ---
        points_awarded = 75 # Flat rate for reverse prompting
        user.total_points += points_awarded
        app.logger.info(f"User {user.username} awarded {points_awarded} points for reverse prompt. Total: {user.total_points}")
    # --- END GAMIFICATION ---

    # Update user stats after successful reverse prompt
        user.last_generation_time = now
        if not user.is_admin:
            user.daily_generation_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"API user {user.username}'s last prompt request time updated and count incremented. (API Reverse Prompt)")

        return jsonify({"inferred_prompt": inferred_prompt})
    except Exception as e:
        app.logger.exception("Error during reverse prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500


# NEW: Image Processing Endpoint ---
@app.route('/process_image_prompt', methods=['POST'])
@login_required
def process_image_prompt(): # CHANGED FROM ASYNC
    user = current_user
    now = datetime.utcnow()
    
    # --- Check if the user is locked out ---
    if user.is_locked:
        return jsonify({
            "error": "Your account is locked. Please contact support.",
            "account_locked": True
        }), 403 # Forbidden

    # Apply cooldown to image processing as well
    if user.last_generation_time: # Changed from last_prompt_request
        time_since_last_request = (now - user.last_generation_time).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown for image processing. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before processing another image.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # Daily limit check for image processing
    if not user.is_admin:
        today = now.date()
        if user.daily_generation_date != today: # Changed from last_count_reset_date
            user.daily_generation_count = 0
            user.daily_generation_date = today # Changed from last_count_reset_date
            db.session.add(user)
            db.session.commit()

        if user.daily_generation_count >= user.daily_limit:
            app.logger.info(f"User {user.username} exceeded their daily image processing limit of {user.daily_limit}.")
            return jsonify({
                "error": f"You have reached your daily limit of {user.daily_limit} generations. If you are looking for more prompts, kindly make a payment to increase your limit.",
                "daily_limit_reached": True,
                "payment_link": PAYMENT_LINK
            }), 429

    data = request.get_json()
    image_data_b64 = data.get('image_data')
    language_code = data.get('language_code', 'en-US') # Not directly used by Gemini Vision, but good to pass


    if not image_data_b64:
        return jsonify({"error": "No image data provided."}), 400

    try:
        image_data_bytes = base64.b64decode(image_data_b64) # Decode base64 string to bytes
        
        # Call the Gemini API for image understanding
        recognized_text = asyncio.run(ask_gemini_for_image_text(image_data_bytes))

        # --- GAMIFICATION: Award points for image processing (multimodal input) ---
        points_awarded = 50 # Flat rate for processing an image/multimodal input
        user.total_points += points_awarded
        app.logger.info(f"User {user.username} awarded {points_awarded} points for image processing. Total: {user.total_points}")
        # --- END GAMIFICATION ---

        # Update last_generation_time after successful image processing
        user.last_generation_time = now
        if not user.is_admin:
            user.daily_generation_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"User {user.username}'s last prompt request time updated and count incremented after image processing.")

        return jsonify({"recognized_text": recognized_text})
    except Exception as e:
        app.logger.exception("Error during image processing endpoint:")
        return jsonify({"error": f"An unexpected server error occurred during image processing: {e}. Please check server logs for details."}), 500


# --- NEW: Route to Handle Perplexity Search Request from Frontend ---
# app.py (Modified /search_perplexity route)

@app.route('/search_perplexity', methods=['POST'])
@login_required
async def search_perplexity():
    # Expects JSON data: {"prompt_text": "the published prompt content"}
    user = current_user # Get current user object
    data = request.get_json()
    prompt_text = data.get('prompt_text', '').strip()

    if not prompt_text:
        return jsonify({"error": "No prompt text provided for search."}), 400

    # Execute the synchronous search function in a separate thread
    search_response = await asyncio.to_thread(perform_perplexity_search, prompt_text)

    # The function already returns a dictionary with 'results' or 'error'
    if "error" in search_response:
        app.logger.error(f"Perplexity search failed for user {user.id}: {search_response['error']}")
        return jsonify(search_response), 500
    
    # --- GAMIFICATION: Award points for successful search ---
    # Points awarded: +20
    points_awarded = 20
    user.total_points += points_awarded
    db.session.add(user)
    
    try:
        # Commit the point change to the database
        db.session.commit()
        app.logger.info(f"User {user.username} awarded {points_awarded} points for Web Search. Total: {user.total_points}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error awarding points to user {user.id} after Perplexity search: {e}")
        # Continue execution even if point commit fails
    # --- END GAMIFICATION ---

    # Log the search activity (optional, but good practice)
    app.logger.info(f"User {user.id} performed Perplexity search: {prompt_text[:50]}...")
    
    # Return the successful search results
    return jsonify(search_response), 200
# --- END NEW Search Route ---


# NEW: Endpoint to test a prompt against the LLM and return a sample response
@app.route('/test_llm_response', methods=['POST'])
@login_required
async def test_llm_response(): # CHANGED to async def
    user = current_user
    now = datetime.utcnow()

    # --- Check if the user is locked out ---
    if user.is_locked:
        return jsonify({
            "error": "Your account is locked. Please contact support.",
            "account_locked": True
        }), 403 # Forbidden

    # Apply cooldown to test prompt as well
    if user.last_generation_time:
        time_since_last_request = (now - user.last_generation_time).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS: # Corrected variable name
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
            app.logger.info(f"User {user.username} is on cooldown for test prompt. Remaining: {remaining_time}s")
            return jsonify({
                "error": f"Please wait {remaining_time} seconds before testing another prompt.",
                "cooldown_active": True,
                "remaining_time": remaining_time
            }), 429

    # Daily limit check for test prompt
    if not user.is_admin:
        today = now.date()
        if user.daily_generation_date != today:
            user.daily_generation_count = 0
            user.daily_generation_date = today
            db.session.add(user)
            db.session.commit()

        if user.daily_generation_count >= user.daily_limit:
            app.logger.info(f"User {user.username} exceeded their daily test prompt limit of {user.daily_limit}.")
            return jsonify({
                "error": f"You have reached your daily limit of {user.daily_limit} generations. If you are looking for more prompts, kindly make a payment to increase your limit.",
                "daily_limit_reached": True,
                "payment_link": PAYMENT_LINK
            }), 429

    data = request.get_json()
    prompt_text = data.get('prompt_text', '').strip()
    language_code = data.get('language_code', 'en-US')
    prompt_mode = data.get('prompt_mode', 'text')
    category = data.get('category')
    subcategory = data.get('subcategory')
    persona = data.get('persona')

    if not prompt_text:
        return jsonify({"error": "No prompt text provided for testing."}), 400

    # --- Server-side validation for allowed categories/personas for test prompts ---
    user_allowed_categories = json.loads(user.allowed_categories)
    user_allowed_personas = json.loads(user.allowed_personas)

    if not user.is_admin:
        if category and category not in user_allowed_categories:
            return jsonify({"error": "Selected category is not allowed for your account."}), 403
        if persona and persona not in user_allowed_personas:
            return jsonify({"error": "Selected persona is not allowed for your account."}), 403
    # --- END Server-side validation ---

    # Construct the instruction for the LLM, including context if provided
    context_str = ""
    if category:
        context_str += f"The user requested this for the category '{category}'"
        if subcategory:
            context_str += f" and subcategory '{subcategory}'."
        else:
            context_str += "."
    if persona:
        if context_str:
            context_str += f" The response should be from the perspective of a '{persona}'."
        else:
            context_str += f"Craft the response from the perspective of a '{persona}'."

    # Define the model and temperature to be used for the test response
    llm_model_name = "gemini-2.5-flash" # As defined globally
    llm_temperature = 0.1 # As defined globally for text_model

    llm_instruction = (
        f"Generate a concise sample response to the following prompt, as if you are the AI model "
        f"receiving this prompt. Keep the response brief and to the point, demonstrating how you would "
        f"interpret and fulfill the prompt. The response MUST be entirely in {LANGUAGE_MAP.get(language_code, 'English')}. "
        f"Crucially, **DO NOT attempt to refine, rewrite, rephrase, or critique the input prompt**; only provide the requested answer or output based on the prompt's content. "
        f"{context_str}\n\nPrompt: {prompt_text}"
    )

    try:
        # Generate LLM response with a specific model
        # For testing/admin purposes, we default to the fastest, cheapest model: gemini-2.0-flash.
        TEST_MODEL = 'gemini-2.0-flash'
        llm_response_text_raw = await asyncio.to_thread(ask_gemini_for_text_prompt, llm_instruction, model_name=TEST_MODEL, max_output_tokens=8192)
        
        # Apply filter_gemini_response to the LLM's raw text before returning
        filtered_llm_response_text = filter_gemini_response(llm_response_text_raw)

        # --- GAMIFICATION: Award points for successful test ---
        points_awarded = 10
        user.total_points += points_awarded
        app.logger.info(f"User {user.username} awarded {points_awarded} points for Test LLM. Total: {user.total_points}")
        # --- END GAMIFICATION ---

        # --- PREPARE FOR FINAL DATABASE COMMIT (Successful Path) ---

        # 1. Update user stats 
        user.last_generation_time = now
        if not user.is_admin:
            user.daily_generation_count += 1
        db.session.add(user)
        
        # 2. AUTO-SAVE the successful Test Response
        if current_user.is_authenticated:
            new_auto_save = AutoSavedResponse(
                user_id=current_user.id,
                test_text=filtered_llm_response_text,
                input_prompt=prompt_text[:250]
            )
            db.session.add(new_auto_save)
        
        # 3. Log the successful LLM response/answer
        log_llm_response(
            current_user.id, 
            'test_llm_response', 
            prompt_mode, 
            prompt_text, # raw input
            filtered_llm_response_text, # full output
            model_name=TEST_MODEL
        )

        # 4. Commit ALL changes (stats, points, auto-save, log) atomically
        db.session.commit()
        app.logger.info(f"User {user.username}'s last prompt request time updated, count incremented, and LLM response logged.")

        return jsonify({
            "sample_response": filtered_llm_response_text,
            "model_name": llm_model_name,
            "temperature": llm_temperature
        })
    except Exception as e:
        app.logger.exception("Error during LLM sample response generation:")
        
        # --- LOGGING FAILURE AND ROLLBACK ---
        
        # 1. Rollback any pending user stat/point changes
        db.session.rollback() 
        
        # 2. Log the failure
        error_output = f"An unexpected server error occurred during sample response generation: {e}"
        log_llm_response(
            current_user.id, 
            'test_llm_response_FAIL', 
            prompt_mode, 
            prompt_text, 
            error_output, 
            model_name='N/A'
        )
        
        # 3. Commit only the failure log entry
        try:
            db.session.commit() 
        except Exception as commit_e:
            app.logger.error(f"Failed to commit FAILURE log for user {current_user.id}: {commit_e}")

        # 4. Return filtered error message
        filtered_error_message = filter_gemini_response(f"An unexpected server error occurred during sample response generation: {e}. Please check server logs for details.")
        return jsonify({"error": filtered_error_message}), 500


 # --- UPDATED: Endpoint to check cooldown status for frontend ---
@app.route('/check_cooldown', methods=['GET'])
@login_required
def check_cooldown():
    user = current_user
    now = datetime.utcnow() # Use utcnow for consistency

    cooldown_active = False
    remaining_time = 0
    if user.last_generation_time: # Changed from last_prompt_request
        time_since_last_request = (now - user.last_generation_time).total_seconds()
        if time_since_last_request < COOLDOWN_SECONDS:
            cooldown_active = True
            remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)

    daily_limit_reached = False
    daily_count = 0
    
    user_daily_limit = user.daily_limit if not user.is_admin else 999999
    
    if user.is_locked:
        daily_limit_reached = True
    elif not user.is_admin: # Check daily limit only for non-admins
        today = now.date()
        if user.daily_generation_date != today: # Changed from last_count_reset_date
            # If the last reset date is not today, reset the count for the current session's check
            daily_count = 0
        else:
            daily_count = user.daily_generation_count # Changed from daily_prompt_count
        
        if daily_count >= user_daily_limit:
            daily_limit_reached = True

    return jsonify({
        "cooldown_active": cooldown_active,
        "remaining_time": remaining_time,
        "daily_limit_reached": daily_limit_reached,
        "daily_count": daily_count,
        "user_daily_limit": user_daily_limit,
        "is_admin": user.is_admin
    }), 200

@app.route('/save_prompt', methods=['POST'])
@login_required
def save_prompt():
    data = request.get_json()
    prompt_text = data.get('prompt_text')
    prompt_type = data.get('prompt_type')

    if not prompt_text or not prompt_type:
        return jsonify({'success': False, 'message': 'Missing prompt text or type.'}), 400

    try:
        new_saved_prompt = SavedPrompt(
            user_id=current_user.id,
            text=prompt_text,
            type=prompt_type,
            timestamp=datetime.utcnow()
        )
        db.session.add(new_saved_prompt)

        # --- GAMIFICATION: Award points for saving (CORRECTLY INDENTED) ---
        points_awarded = 10
        current_user.total_points += points_awarded
        db.session.add(current_user)
        app.logger.info(f"User {current_user.username} awarded {points_awarded} points for saving. Total: {current_user.total_points}")
        # --- END GAMIFICATION ---

        db.session.commit()
        return jsonify({'success': True, 'message': 'Prompt saved successfully!', 'new_points': points_awarded, 'total_points': current_user.total_points})
    except Exception as e:
        logging.error(f"Error saving prompt: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Database error: {e}'}), 500

# app.py (New route added around line 12526)

# --- Get Auto-Saved Responses Endpoint ---
@app.route('/get_auto_saved_responses', methods=['GET'])
@login_required
def get_auto_saved_responses():
    auto_saved_responses = AutoSavedResponse.query.filter_by(user_id=current_user.id).order_by(AutoSavedResponse.timestamp.desc()).limit(10).all()
    responses_data = []
    for response in auto_saved_responses:
        responses_data.append({
            'test_text': response.test_text,
            'input_prompt': response.input_prompt,
            'timestamp': response.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(responses_data)

# --- NEW: Endpoint for Social Sharing Points (Gamification) ---
@app.route('/award_share_points', methods=['POST'])
@login_required
def award_share_points():
    points_awarded = 15
    try:
        current_user.total_points += points_awarded
        db.session.add(current_user)
        db.session.commit()
        app.logger.info(f"User {current_user.username} awarded {points_awarded} points for sharing. Total: {current_user.total_points}")
        return jsonify({'success': True, 'new_points': points_awarded, 'total_points': current_user.total_points})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error awarding share points: {e}")
        return jsonify({'success': False, 'message': 'Failed to award points.'}), 500
# --- END NEW: Endpoint for Social Sharing Points ---

# --- Database Initialization (Run once to create tables) ---
# ...

# --- Get Saved Prompts Endpoint (using SavedPrompt model) ---
@app.route('/get_saved_prompts', methods=['GET'])
@login_required
def get_saved_prompts():
    saved_prompts = SavedPrompt.query.filter_by(user_id=current_user.id).order_by(SavedPrompt.timestamp.desc()).all()
    prompts_data = []
    for prompt in saved_prompts:
        prompts_data.append({
            'text': prompt.text,
            'type': prompt.type,
            'timestamp': prompt.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(prompts_data)

# --- Get Raw Prompts Endpoint (using RawPrompt model) ---
@app.route('/get_raw_prompts', methods=['GET'])
@login_required
def get_raw_prompts():
    raw_prompts = RawPrompt.query.filter_by(user_id=current_user.id).order_by(RawPrompt.timestamp.desc()).limit(10).all()
    prompts_data = []
    for prompt in raw_prompts:
        prompts_data.append({
            'raw_text': prompt.raw_text,
            'timestamp': prompt.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(prompts_data)


# --- Download Prompts as TXT Endpoint (using SavedPrompt model) ---
@app.route('/download_prompts_txt', methods=['GET'])
@login_required
def download_prompts_txt():
    saved_prompts = SavedPrompt.query.filter_by(user_id=current_user.id).order_by(SavedPrompt.timestamp.desc()).all()
    
    output = []
    output.append("--- Your Saved Prompts ---\n\n")
    for prompt in saved_prompts:
        output.append(f"Type: {prompt.type.capitalize()}\n")
        output.append(f"Timestamp: {prompt.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.append("Prompt:\n")
        output.append(f"{prompt.text}\n")
        output.append("-" * 30 + "\n\n")

    response = make_response("".join(output))
    response.headers["Content-Disposition"] = "attachment; filename=saved_prompts.txt"
    response.headers["Content-type"] = "text/plain"
    return response

# NEW: Admin Prompts Management Routes (using SamplePrompt model)
@app.route('/admin/prompts', methods=['GET'])
@admin_required
def admin_prompts():
    sample_prompts = SamplePrompt.query.order_by(SamplePrompt.timestamp.desc()).all()
    return render_template('admin_prompts.html', sample_prompts=sample_prompts, current_user=current_user)

@app.route('/admin/prompts/add', methods=['POST'])
@admin_required
def add_prompt():
    raw_prompt = request.form.get('raw_prompt')
    polished_prompt = request.form.get('polished_prompt')
    creative_prompt = request.form.get('creative_prompt')
    technical_prompt = request.form.get('technical_prompt')
    display_type = request.form.get('display_type', 'polished') # NEW: Get display_type from form

    if not raw_prompt or not polished_prompt or not creative_prompt or not technical_prompt:
        flash('All prompt fields are required.', 'danger')
        return redirect(url_for('admin_prompts'))

    try:
        new_prompt = SamplePrompt(
            raw_prompt=raw_prompt,
            polished_prompt=polished_prompt,
            creative_prompt=creative_prompt,
            technical_prompt=technical_prompt,
            display_type=display_type # NEW: Save display_type
        )
        db.session.add(new_prompt)
        db.session.commit()
        flash('Sample prompt added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding sample prompt: {e}', 'danger')
        app.logger.error(f"Error adding sample prompt: {e}")
    return redirect(url_for('admin_prompts'))

@app.route('/admin/prompts/delete/<int:prompt_id>', methods=['POST'])
@admin_required
def delete_prompt(prompt_id):
    prompt = SamplePrompt.query.get_or_404(prompt_id)
    try:
        db.session.delete(prompt)
        db.session.commit()
        flash('Sample prompt deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting sample prompt: {e}', 'danger')
        app.logger.error(f"Error deleting sample prompt: {e}")
    return redirect(url_for('admin_prompts'))

@app.route('/admin/prompts/repost/<int:prompt_id>', methods=['POST'])
@admin_required
def repost_prompt(prompt_id):
    prompt = SamplePrompt.query.get_or_404(prompt_id)
    try:
        # Update timestamp to now to bring it to the top of the list
        prompt.timestamp = datetime.utcnow()
        db.session.commit()
        flash('Sample prompt reposted successfully (timestamp updated)!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting sample prompt: {e}', 'danger')
        app.logger.error(f"Error reposting sample prompt: {e}")
    return redirect(url_for('admin_prompts'))


# --- Admin News Management Routes (using NewsItem model) ---
@app.route('/admin/news', methods=['GET'])
@admin_required
def admin_news():
    news_items = NewsItem.query.order_by(NewsItem.timestamp.desc()).all()
    return render_template('admin_news.html', news_items=news_items, current_user=current_user)

@app.route('/admin/news/add', methods=['POST'])
@admin_required
def add_news():
    title = request.form.get('title')
    url = request.form.get('url')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date')

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d').date() # Changed to .date()
        except ValueError:
            flash('Invalid published date format. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_news'))

    if not title or not url:
        flash('Title and URL are required for news items.', 'danger')
        return redirect(url_for('admin_news'))

    try:
        new_news = NewsItem(
            title=title,
            url=url,
            description=description,
            published_date=published_date,
            user_id=current_user.id # Assign current admin user
        )
        db.session.add(new_news)
        db.session.commit()
        flash('News item added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding news item: {e}', 'danger')
        app.logger.error(f"Error adding news item: {e}")
    return redirect(url_for('admin_news'))

@app.route('/admin/news/delete/<int:news_id>', methods=['POST'])
@admin_required
def delete_news(news_id):
    news_item = NewsItem.query.get_or_404(news_id)
    try:
        db.session.delete(news_item)
        db.session.commit()
        flash('News item deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting news item: {e}', 'danger')
        app.logger.error(f"Error deleting news item: {e}")
    return redirect(url_for('admin_news'))

@app.route('/admin/news/repost/<int:news_id>', methods=['POST'])
@admin_required
def repost_news(news_id):
    news_item = NewsItem.query.get_or_404(news_id)
    try:
        # Update timestamp to now to bring it to the top of the list
        news_item.timestamp = datetime.utcnow()
        db.session.commit()
        flash('News item reposted successfully (timestamp updated)!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting news item: {e}', 'danger')
        app.logger.error(f"Error reposting news item: {e}")
    return redirect(url_for('admin_news'))


# --- Admin Jobs Management Routes (using JobListing model) ---
@app.route('/admin/jobs', methods=['GET'])
@admin_required
def admin_jobs():
    job_listings = JobListing.query.order_by(JobListing.timestamp.desc()).all()
    return render_template('admin_jobs.html', job_listings=job_listings, current_user=current_user)

@app.route('/admin/jobs/add', methods=['POST'])
@admin_required
def add_job():
    title = request.form.get('title')
    company = request.form.get('company')
    location = request.form.get('location')
    url = request.form.get('url')
    description = request.form.get('description')
    published_date_str = request.form.get('published_date')

    published_date = None
    if published_date_str:
        try:
            published_date = datetime.strptime(published_date_str, '%Y-%m-%d').date() # Changed to .date()
        except ValueError:
            flash('Invalid published date format. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('admin_jobs'))

    if not title or not company or not url:
        flash('Job Title, Company, and URL are required for job listings.', 'danger')
        return redirect(url_for('admin_jobs'))

    try:
        new_job = JobListing(
            title=title,
            company=company,
            location=location,
            url=url,
            description=description,
            published_date=published_date,
            user_id=current_user.id # Assign current admin user
        )
        db.session.add(new_job)
        db.session.commit()
        flash('Job listing added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding job listing: {e}', 'danger')
        app.logger.error(f"Error adding job listing: {e}")
    return redirect(url_for('admin_jobs'))

@app.route('/admin/jobs/delete/<int:job_id>', methods=['POST'])
@admin_required
def delete_job(job_id):
    job_listing = JobListing.query.get_or_404(job_id)
    try:
        db.session.delete(job_listing)
        db.session.commit()
        flash('Job listing deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting job listing: {e}', 'danger')
        app.logger.error(f"Error deleting job listing: {e}")
    return redirect(url_for('admin_jobs'))

@app.route('/admin/jobs/repost/<int:job_id>', methods=['POST'])
@admin_required
def repost_job(job_id):
    job_listing = JobListing.query.get_or_404(job_id)
    try:
        # Update timestamp to now to bring it to the top of the list
        job_listing.timestamp = datetime.utcnow()
        db.session.commit()
        flash('Job listing reposted successfully (timestamp updated)!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting job listing: {e}', 'danger')
        app.logger.error(f"Error reposting job listing: {e}")
    return redirect(url_for('admin_jobs'))


# --- Change Password Route ---
@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')

        if not current_password or not new_password or not confirm_new_password:
            flash('All fields are required.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if new_password != confirm_new_password:
            flash('New password and confirmation do not match.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        if len(new_password) < 6: # Example: enforce minimum password length
            flash('New password must be at least 6 characters long.', 'danger')
            return render_template('change_password.html', current_user=current_user)

        try:
            current_user.set_password(new_password)
            db.session.commit()
            flash('Your password has been changed successfully!', 'success')
            return redirect(url_for('app_home')) # Redirect to app_home
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while changing your password: {e}', 'danger')
            app.logger.error(f"Error changing password for user {current_user.username}: {e}")

    return render_template('change_password.html', current_user=current_user)

# app.py (New Admin Gift Management Routes)

@app.route('/admin/gifts', methods=['GET'])
@admin_required
def admin_gifts():
    gifts = Gift.query.order_by(Gift.timestamp.desc()).all()
    return render_template('admin_gifts.html', gifts=gifts, current_user=current_user)

@app.route('/admin/gifts/add', methods=['POST'])
@admin_required
def add_gift():
    name = request.form.get('name')
    description = request.form.get('description')
    points_required_str = request.form.get('points_required')
    
    try:
        points_required = int(points_required_str) if points_required_str else 0
        if points_required < 0: raise ValueError
    except ValueError:
        flash('Points Required must be a non-negative integer.', 'danger')
        return redirect(url_for('admin_gifts'))

    if not name:
        flash('Gift Name is required.', 'danger')
        return redirect(url_for('admin_gifts'))

    try:
        new_gift = Gift(
            name=name,
            description=description,
            points_required=points_required,
            user_id=current_user.id
        )
        db.session.add(new_gift)
        db.session.commit()
        flash('Gift added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding gift: {e}', 'danger')
        app.logger.error(f"Error adding gift: {e}")
    return redirect(url_for('admin_gifts'))

@app.route('/admin/gifts/delete/<int:gift_id>', methods=['POST'])
@admin_required
def delete_gift(gift_id):
    gift = Gift.query.get_or_404(gift_id)
    try:
        db.session.delete(gift)
        db.session.commit()
        flash('Gift deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting gift: {e}', 'danger')
    return redirect(url_for('admin_gifts'))

@app.route('/admin/gifts/repost/<int:gift_id>', methods=['POST'])
@admin_required
def repost_gift(gift_id):
    gift = Gift.query.get_or_404(gift_id)
    try:
        # Repost logic updates the timestamp and ensures it is active
        gift.timestamp = datetime.utcnow()
        gift.is_active = True
        db.session.commit()
        flash('Gift reposted successfully (timestamp updated)!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error reposting gift: {e}', 'danger')
    return redirect(url_for('admin_gifts'))

# --- Forgot Password Routes ---
@app.route('/forgot_password', methods=['GET'])
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/send_reset_link', methods=['POST'])
async def send_reset_link():
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    if user:
        if not user.email:
            flash('This account does not have an email address associated for password reset. Please contact support.', 'danger')
            return redirect(url_for('forgot_password'))

        # Generate a unique token
        token = str(uuid.uuid4())
        # Set token expiration (e.g., 1 hour from now)
        expiration = datetime.utcnow() + timedelta(hours=1)
        
        user.reset_token = token # Changed from password_reset_token
        user.reset_token_expiration = expiration # Changed from password_reset_expiration
        db.session.commit()
        
        reset_link = url_for('reset_password', token=token, _external=True)
        
        try:
            msg = Message('Password Reset Request for SuperPrompter',
                          sender=app.config['MAIL_USERNAME'],
                          recipients=[user.email])
            msg.body = f"""
Dear {user.username},

You have requested a password reset for your SuperPrompter account.

Please click on the following link to reset your password:
{reset_link}

This link will expire in 1 hour.

If you did not request a password reset, please ignore this email.

Sincerely,
The SuperPrompter Team
"""
            mail.send(msg)
            app.logger.info(f"Password reset email sent to {user.email} for user {user.username}")
            flash('A password reset link has been sent to your email address. Please check your inbox (and spam folder).', 'info')
        except Exception as e:
            app.logger.error(f"Failed to send password reset email to {user.email}: {e}", exc_info=True)
            flash('Failed to send password reset email. Please try again later or contact support.', 'danger')
    else:
        # For security, always give a generic success message even if the user doesn't exist
        flash('If an account with that username exists, a password reset link has been sent to the associated email address.', 'info')
    return redirect(url_for('login'))

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first() # Changed from password_reset_token
    now = datetime.utcnow()

    if not user or user.reset_token_expiration < now: # Changed from password_reset_expiration
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')

        if not new_password or not confirm_new_password:
            flash('Both new password fields are required.', 'danger')
            return render_template('reset_password.html', token=token)

        if new_password != confirm_new_password:
            flash('New password and confirmation do not match.', 'danger')
            return render_template('reset_password.html', token=token)

        if len(new_password) < 6: # Example: enforce minimum password length
            flash('New password must be at least 6 characters long.', 'danger')
            return render_template('reset_password.html', token=token)

        user.set_password(new_password)
        user.reset_token = None # Invalidate the token after use
        user.reset_token_expiration = None
        db.session.commit()
        flash('Your password has been reset successfully! Please log in with your new password.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)


# NEW: Newsletter Subscription Route
@app.route('/subscribe_newsletter', methods=['POST'])
def subscribe_newsletter():
    email = request.form.get('email')
    if not email:
        flash('Email address is required to subscribe.', 'danger')
        return redirect(url_for('landing'))
    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        flash('Invalid email address format.', 'danger')
        return redirect(url_for('landing'))

    existing_subscriber = NewsletterSubscriber.query.filter_by(email=email).first()
    if existing_subscriber:
        flash('You are already subscribed to our newsletter!', 'info')
    else:
        try:
            new_subscriber = NewsletterSubscriber(email=email)
            db.session.add(new_subscriber)
            db.session.commit()
            flash('Successfully subscribed to our newsletter! Thank a you!', 'success')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error subscribing email {email} to newsletter: {e}")
            flash('Failed to subscribe to newsletter. Please try again later.', 'danger')
        
    return redirect(url_for('landing'))


# --- UPDATED: Authentication Routes for automatic redirect ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('You are already registered and logged in.', 'info')
        return redirect(url_for('app_home')) # Redirect to app home if already logged in

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # NEW: Get email from registration form
        email = request.form.get('email') # Make sure to add this field to your register.html

        user = User.query.filter_by(username=username).first()
        if user:
            flash('this username already exists', 'danger') # Updated message
            # Generate suggestions if username exists
            suggestions = generate_unique_username_suggestions(username)
            return render_template('register.html', suggestions=suggestions, username=username, email=email) # Pass suggestions and original inputs
        else:
            # NEW: Check if email already exists
            if email and User.query.filter_by(email=email).first():
                flash('Email already registered. Please use a different email or log in.', 'danger')
                return render_template('register.html', username=username, email=email) # Re-render without suggestions for email conflict
            else:
                new_user = User(username=username, email=email) # Pass email to User constructor
                new_user.set_password(password)
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user) # Automatically log in the new user
                flash('Registration successful! You are now logged in.', 'success')
                return redirect(url_for('app_home')) # Redirect to app home after registration
    return render_template('register.html') # Initial GET request, no suggestions

# NEW: Helper function to generate unique username suggestions
def generate_unique_username_suggestions(base_username, num_suggestions=3):
    suggestions = []
    attempts = 0
    max_attempts_per_suggestion = 10 # Prevent infinite loops

    while len(suggestions) < num_suggestions and attempts < num_suggestions * max_attempts_per_suggestion:
        suffix = ''.join(random.choices(string.digits, k=4)) # 4 random digits
        new_username = f"{base_username}{suffix}"
        
        # Ensure the suggestion is not too long
        if len(new_username) > 80: # Max length for username field
            new_username = f"{base_username[:76]}{suffix}" # Truncate base_username if needed

        if not User.query.filter_by(username=new_username).first():
            suggestions.append(new_username)
        attempts += 1
    # If we still don't have enough suggestions, try more generic ones
    while len(suggestions) < num_suggestions:
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        generic_username = f"user_{random_suffix}"
        if not User.query.filter_by(username=generic_username).first():
            suggestions.append(generic_username)
    return suggestions


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('app_home')) # Redirect to app home if already logged in

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember_me = 'remember_me' in request.form

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('app_home')) # Redirect to app home after login
        else:
            flash('Login Unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required # Only logged-in users can log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing')) # Redirect to landing page after logout

# NEW: Admin route for downloading the database
@app.route('/admin/download_database', methods=['GET'])
@admin_required
def download_database():
    try:
        db_path = os.path.join(app.root_path, 'site.db')
        return send_file(db_path, as_attachment=True, download_name='site.db')
    except Exception as e:
        flash(f"Error downloading database: {e}", 'danger')
        return redirect(url_for('admin_jobs'))


# NEW: Admin User Management Routes
@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    # Attempt to parse allowed_categories and allowed_personas from JSON strings
    # Handle cases where they might be malformed or non-existent (e.g., for new users before first admin edit)
    users_data = []
    for user in users:
        try:
            allowed_categories = json.loads(user.allowed_categories)
        except json.JSONDecodeError:
            allowed_categories = [] # Default to empty list if JSON is invalid
        try:
            allowed_personas = json.loads(user.allowed_personas)
        except json.JSONDecodeError:
            allowed_personas = [] # Default to empty list if JSON is invalid
        
        users_data.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'daily_limit': user.daily_limit,
            'api_key': user.api_key,
            'is_admin': user.is_admin,
            'is_locked': user.is_locked,
            'allowed_categories': allowed_categories,
            'allowed_personas': allowed_personas
        })
    
    return render_template('admin_users.html', users=users_data, current_user=current_user)

@app.route('/admin/users/toggle_access/<int:user_id>', methods=['POST'])
@admin_required
def toggle_user_access(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash("Cannot lock or unlock an admin account.", "danger")
    else:
        user.is_locked = not user.is_locked
        if not user.is_locked:
            flash(f"User {user.username} has been unlocked.", "success")
        else:
            flash(f"User {user.username} has been locked.", "info")
        db.session.commit()
    return redirect(url_for('admin_users'))

# NEW: Admin route to update user's daily limit
@app.route('/admin/users/update_quota/<int:user_id>', methods=['POST'])
@admin_required
def update_user_quota(user_id):
    user = User.query.get_or_404(user_id)
    new_limit_str = request.form.get('new_limit')

    try:
        new_limit = int(new_limit_str)
        if new_limit < 0:
            raise ValueError("Limit cannot be negative.")
        user.daily_limit = new_limit
        db.session.commit()
        flash(f"Daily prompt limit for {user.username} has been updated to {new_limit}.", "success")
    except (ValueError, TypeError) as e:
        flash(f"Invalid limit value. Please enter a positive integer. Error: {e}", "danger")

    return redirect(url_for('admin_users'))

# NEW: Admin API Performance Route
@app.route('/admin/api_performance')
@admin_required
def admin_api_performance():
    api_logs = ApiRequestLog.query.order_by(ApiRequestLog.request_timestamp.desc()).limit(100).all()
    # To get usernames for the logs
    users = {user.id: user for user in User.query.all()}
    return render_template('admin_api_performance.html', api_logs=api_logs, users=users, current_user=current_user)

@app.route('/admin/users/generate_api_key/<int:user_id>', methods=['POST'])
@admin_required
def generate_api_key(user_id):
    user = User.query.get_or_404(user_id)
    new_api_key = str(uuid.uuid4())
    user.api_key = new_api_key
    db.session.commit()
    flash(f"New API key generated for {user.username}.", "success")
    return redirect(url_for('admin_users'))

# NEW: Admin route to update allowed categories and personas for a user
@app.route('/admin/users/update_access/<int:user_id>', methods=['POST'])
@admin_required
def update_user_access(user_id):
    user = User.query.get_or_404(user_id)
    
    # Get selected categories and personas from the form
    # request.form.getlist will return a list of values for multiple select inputs
    selected_categories = request.form.getlist('allowed_categories')
    selected_personas = request.form.getlist('allowed_personas')

    try:
        # Store them as JSON strings in the database
        user.allowed_categories = json.dumps(selected_categories)
        user.allowed_personas = json.dumps(selected_personas)
        db.session.commit()
        flash(f"Access permissions for {user.username} updated successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating access permissions: {e}", "danger")
        app.logger.error(f"Error updating user access for {user.username}: {e}")
    
    return redirect(url_for('admin_users'))


# NEW: API endpoint for external clients using API keys to generate prompts
@app.route('/api/v1/generate', methods=['POST'])
@api_key_required
def api_generate(user):
    """
    API endpoint to generate polished, creative, and technical prompts.
    Requires an API key in the 'X-API-KEY' header.
    """
    start_time = datetime.utcnow() # Record start time
    status_code = 500 # Default to 500 for errors
    response_data = {}
    
    try:
        now = datetime.utcnow()

        # --- Check if the user is locked out ---
        if user.is_locked:
            status_code = 403
            response_data = {
                "error": "Your account is locked. Please contact support.",
                "account_locked": True
            }
            return jsonify(response_data), status_code
            
        # --- Cooldown Check using database timestamp ---
        if user.last_generation_time:
            time_since_last_request = (now - user.last_generation_time).total_seconds()
            if time_since_last_request < COOLDOWN_SECONDS:
                remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
                app.logger.info(f"API user {user.username} is on cooldown. Remaining: {remaining_time}s")
                status_code = 429
                response_data = {
                    "error": f"Please wait {remaining_time} seconds before generating new prompts.",
                    "cooldown_active": True,
                    "remaining_time": remaining_time
                }
                return jsonify(response_data), status_code
        
        # --- Daily Limit Check ---
        if not user.is_admin:
            today = now.date()
            if user.daily_generation_date != today:
                user.daily_generation_count = 0
                user.daily_generation_date = today
                db.session.add(user)
                db.session.commit()

            if user.daily_generation_count >= user.daily_limit:
                app.logger.info(f"API user {user.username} exceeded their daily prompt limit of {user.daily_limit}.")
                status_code = 429
                response_data = {
                    "error": f"You have reached your daily limit of {user.daily_limit} prompt generations. Please upgrade your plan.",
                    "daily_limit_reached": True,
                    "payment_link": PAYMENT_LINK
                }
                return jsonify(response_data), status_code

        data = request.get_json()
        prompt_input = data.get('raw_input', '').strip()
        language_code = data.get('language_code', 'en-US')
        prompt_mode = data.get('prompt_mode', 'text')
        category = data.get('category')
        subcategory = data.get('subcategory')
        persona = data.get('persona')

        # --- NEW: Server-side validation for allowed categories/personas for API users ---
        user_allowed_categories = json.loads(user.allowed_categories)
        user_allowed_personas = json.loads(user.allowed_personas)

        if not user.is_admin:
            if category and category not in user_allowed_categories:
                status_code = 403
                response_data = {"error": "Selected category is not allowed for your account via API."}
                return jsonify(response_data), status_code
            if persona and persona not in user_allowed_personas:
                status_code = 403
                response_data = {"error": "Selected persona is not allowed for your account via API."}
                return jsonify(response_data), status_code
        # --- END NEW: Server-side validation ---


        if not prompt_input:
            status_code = 400
            response_data = {
                "error": "Please provide some text to generate prompts."
            }
            return jsonify(response_data), status_code

        results = asyncio.run(generate_prompts_async(raw_input=prompt_input, language_code=language_code, prompt_mode=prompt_mode, category=category, subcategory=subcategory, persona=persona))

        # --- Update user stats and save raw_input ---
        user.last_generation_time = now
        if not user.is_admin:
            user.daily_generation_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"API user {user.username}'s last prompt request time updated and count incremented. (API Forward Prompt)")

        status_code = 200
        response_data = results
        return jsonify(response_data), status_code
    except Exception as e:
        app.logger.exception("Error during API prompt generation in /api/v1/generate:")
        status_code = 500
        # Ensure error message is filtered before sending to frontend
        filtered_error_message = filter_gemini_response(f"An unexpected server error occurred: {e}")
        response_data = {"error": filtered_error_message}
        return jsonify(response_data), status_code
    finally:
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        raw_input_log = request.get_data(as_text=True) # Get raw request body for logging
        
        try:
            log_entry = ApiRequestLog(
                user_id=user.id,
                endpoint='/api/v1/generate',
                request_timestamp=start_time,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_input=raw_input_log
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as log_e:
            app.logger.error(f"Error saving API request log for /api/v1/generate: {log_e}")
            db.session.rollback()


# NEW: API endpoint for reverse prompting
@app.route('/api/v1/reverse', methods=['POST'])
@api_key_required
def api_reverse_prompt(user):
    """
    API endpoint to infer a prompt from a given text or code.
    Requires an API key in the 'X-API-KEY' header.
    """
    start_time = datetime.utcnow() # Record start time
    status_code = 500 # Default to 500 for errors
    response_data = {}

    try:
        now = datetime.utcnow()

        # --- Check if the user is locked out ---
        if user.is_locked:
            status_code = 403
            response_data = {
                "error": "Your account is locked. Please contact support.",
                "account_locked": True
            }
            return jsonify(response_data), status_code

        # Apply cooldown to reverse prompting as well
        if user.last_generation_time:
            time_since_last_request = (now - user.last_generation_time).total_seconds()
            if time_since_last_request < COOLDOWN_SECONDS:
                remaining_time = int(COOLDOWN_SECONDS - time_since_last_request)
                app.logger.info(f"API user {user.username} is on cooldown for reverse prompt. Remaining: {remaining_time}s")
                status_code = 429
                response_data = {
                    "error": f"Please wait {remaining_time} seconds before performing another reverse prompt.",
                    "cooldown_active": True,
                    "remaining_time": remaining_time
                }
                return jsonify(response_data), status_code

        # --- Daily Limit Check for Reverse Prompt ---
        if not user.is_admin:
            today = now.date()
            if user.daily_generation_date != today:
                user.daily_generation_count = 0
                user.daily_generation_date = today
                db.session.add(user)
                db.session.commit()

            if user.daily_generation_count >= user.daily_limit:
                app.logger.info(f"API user {user.username} exceeded their daily reverse prompt limit of {user.daily_limit}.")
                status_code = 429
                response_data = {
                    "error": f"You have reached your daily limit of {user.daily_limit} generations. Please upgrade your plan.",
                    "daily_limit_reached": True,
                    "payment_link": PAYMENT_LINK
                }
                return jsonify(response_data), status_code

        data = request.get_json()
        input_text = data.get('input_text', '').strip()
        language_code = data.get('language_code', 'en-US')
        prompt_mode = data.get('prompt_mode', 'text')

        if not input_text:
            status_code = 400
            response_data = {"error": "Please provide text or code to infer a prompt from."}
            return jsonify(response_data), status_code

        if prompt_mode in ['image_gen', 'video_gen']:
            status_code = 200 # This is a specific message, not an error
            response_data = {"inferred_prompt": "Reverse prompting is not applicable for image or video generation modes."}
            return jsonify(response_data), status_code

        inferred_prompt = asyncio.run(generate_reverse_prompt_async(input_text, language_code, prompt_mode))

        # Update user stats after successful reverse prompt
        user.last_generation_time = now
        if not user.is_admin:
            user.daily_generation_count += 1
        db.session.add(user)
        db.session.commit()
        app.logger.info(f"API user {user.username}'s last prompt request time updated and count incremented. (API Reverse Prompt)")

        status_code = 200
        response_data = {"inferred_prompt": inferred_prompt}
        return jsonify(response_data), status_code
    except Exception as e:
        app.logger.exception("Error during API reverse prompt generation in /api/v1/reverse:")
        status_code = 500
        # Ensure error message is filtered before sending to frontend
        filtered_error_message = filter_gemini_response(f"An unexpected server error occurred: {e}")
        response_data = {"error": filtered_error_message}
        return jsonify(response_data), status_code
    finally:
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        raw_input_log = request.get_data(as_text=True) # Get raw request body for logging

        try:
            log_entry = ApiRequestLog(
                user_id=user.id,
                endpoint='/api/v1/reverse',
                request_timestamp=start_time,
                latency_ms=latency_ms,
                status_code=status_code,
                raw_input=raw_input_log
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as log_e:
            app.logger.error(f"Error saving API request log for /api/v1/reverse: {log_e}")
            db.session.rollback()


# --- Database Initialization (Run once to create tables) ---
# This block ensures tables are created when the app starts.
# In production, you might use Flask-Migrate or a separate script.
with app.app_context():
    db.create_all()
    app.logger.info("Database tables created/checked.")

    # NEW: Create an admin user if one doesn't exist for easy testing
    # Also ensure initial admin gets all categories and personas
    if not User.query.filter_by(username='admin').first():
        admin_user = User(
            username='admin', 
            is_admin=True, 
            daily_limit=999999,
            email = 'admin@example.com', # Assign a dummy email for admin
            # Populate allowed_categories and allowed_personas with all possible options
            # This is a placeholder for initial admin setup. In a real app, you might fetch all distinct values.
            # For now, we'll use a hardcoded list that includes all values from CATEGORIES_AND_SUBCATEGORIES and CATEGORY_PERSONAS
            allowed_categories=json.dumps([
                "General Writing & Editing", "Programming & Code", "Business & Finance",
                "Education & Learning", "Technical Writing & Explanation", "Customer Support",
                "Research & Information Retrieval", "Data Analysis & Interpretation",
                "Productivity & Planning", "Creative Writing", "Marketing & Advertising",
                "Multilingual & Translation", "Entertainment & Media", "Career & Resume",
                "Legal & Compliance", "Healthcare & Wellness", "Image Generation & Visual Design",
                "Event Planning", "UX/UI & Product Design", "Spirituality & Self-Reflection",
                "Gaming", "Voice, Audio & Podcasting", "AI & Prompt Engineering",
                "News & Current Affairs", "Travel & Culture", "Other"
            ]),
            allowed_personas=json.dumps([
                "Author", "Editor", "Copywriter", "Content Creator", "Blogger",
                "Software Developer", "Frontend Engineer", "Backend Engineer", "Data Scientist", "DevOps Engineer",
                "Entrepreneur", "Business Analyst", "Financial Advisor", "Investor", "Startup Founder", "Director", "CEO",
                "Student", "Teacher", "Tutor", "Curriculum Designer", "Lifelong Learner",
                "Technical Writer", "System Architect", "Engineer", "Product Manager", "Compliance Officer",
                "Support Agent", "Customer Success Manager", "Helpdesk Analyst", "Call Center Manager", "Chatbot Designer",
                "Researcher", "Scientist", "Academic", "Policy Analyst", "Librarian",
                "Data Analyst", "BI Analyst", "Statistician", "Data Engineer", "Operations Manager",
                "Project Manager", "Life Coach", "Executive Assistant", "Scrum Master", "Productivity Hacker",
                "Novelist", "Poet", "Screenwriter", "Songwriter", "Creative Director",
                "Marketing Manager", "Brand Strategist", "SEO Specialist", "Content Marketer", "Media Planner",
                "Translator", "Interpreter", "Language Teacher", "Localization Specialist", "Multilingual Blogger",
                "YouTuber", "Streamer", "Podcaster", "Critic", "Fan Fiction Author",
                "Job Seeker", "Career Coach", "HR Recruiter", "Hiring Manager", "Resume Writer",
                "Lawyer", "Paralegal", "Compliance Officer", "Policy Advisor", "Contract Manager",
                "Nutritionist", "Fitness Coach", "Therapist", "Health Blogger", "Wellness Consultant",
                "Graphic Designer", "Concept Artist", "Art Director", "Photographer", "AI Image Prompt Engineer",
                "Event Planner", "Wedding Coordinator", "Conference Organizer", "Marketing Executive", "Venue Manager",
                "UX Designer", "UI Designer", "Product Designer", "Interaction Designer", "Design Researcher",
                "Meditation Coach", "Spiritual Guide", "Mindfulness Blogger", "Philosopher", "Self-help Author",
                "Game Developer", "Game Designer", "Gamer", "Stream Host", "Lore Writer",
                "Voice Actor", "Podcaster", "Audio Engineer", "Narrator", "Sound Designer",
                "Prompt Engineer", "ML Engineer", "AI Researcher", "NLP Scientist", "Chatbot Developer",
                "Journalist", "News Curator", "Political Analyst", "Opinion Writer", "Debater",
                "Itineraries", "Local tips", "Cultural do’s and don’ts",
                "General", "Custom", "Uncategorized", "Other"
            ])
        )
        admin_user.set_password('adminpass') # Set a default password for the admin
        db.session.add(admin_user)
        db.session.commit()
        app.logger.info("Default admin user 'admin' created with password 'adminpass'.")


# --- Main App Run ---
if __name__ == '__main__':
    # The following line has been removed to allow the async routes to work
    # in a proper ASGI environment.
    # If you must use app.run() for quick tests and encounter the 'event loop closed' error,
    # you can use `nest_asyncio.apply()` (install with `pip install nest-asyncio`), but this is
    # generally not recommended for production as it can hide underlying architectural issues.
    app.run(debug=True)

