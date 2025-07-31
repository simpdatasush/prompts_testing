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
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_should_be_in_env') # Needed for Flask-Login sessions
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30) # Remember user for 30 days, re-added from previous app.py
db = SQLAlchemy(app)
# --- END NEW: Flask-SQLAlchemy Configuration ---


# --- NEW: Flask-Login Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # The view Flask-Login should redirect to for login
login_manager.login_message_category = 'info' # Category for flash messages
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
DAILY_LIMIT = 10 # Number of generations allowed per day for non-admin users


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


# --- Configure Google Gemini API ---
# Ensure your GOOGLE_API_KEY is set in your environment variables
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize Gemini models as per user's specific requirements
# ALL MODELS ARE NOW SET TO 'gemini-2.0-flash' as requested by the user.
text_model = genai.GenerativeModel('gemini-2.0-flash') # For general text generation
vision_model = genai.GenerativeModel('gemini-2.0-flash') # For image understanding
structured_gen_model = genai.GenerativeModel('gemini-2.0-flash') # For structured JSON generation


# --- UPDATED: User Model for SQLAlchemy and Flask-Login ---
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

    # Relationship for saved prompts (using SavedPrompt model)
    saved_prompts = db.relationship('SavedPrompt', backref='author', lazy=True)
    raw_prompts = db.relationship('RawPrompt', backref='requester', lazy=True)
    news_items = db.relationship('NewsItem', backref='admin_poster', lazy=True) # Renamed from News
    job_listings = db.relationship('JobListing', backref='admin_poster', lazy=True) # Renamed from Job


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', Admin: {self.is_admin})"

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

# NEW: NewsletterSubscriber Model (from provided docx)
class NewsletterSubscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_on = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Subscriber {self.email}>'


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


# --- Gemini API interaction function (Synchronous wrapper for text_model) ---
def ask_gemini_for_text_prompt(prompt_instruction, max_output_tokens=512):
    try:
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1
        }
        response = text_model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip()
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"DEBUG: Google API Call Error (text_model): {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with Gemini API: {str(e)}")
    except Exception as e:
        app.logger.error(f"DEBUG: Unexpected Error calling Gemini API (text_model): {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred: {str(e)}")

# --- Gemini API interaction function (Synchronous wrapper for structured_gen_model) ---
# This function will now rely on prompt engineering for JSON output, not responseMimeType
def ask_gemini_for_structured_prompt(prompt_instruction, generation_config=None, max_output_tokens=2048):
    try:
        # We no longer use responseMimeType or responseSchema in generation_config for gemini-2.0-flash
        # The prompt_instruction itself is responsible for asking for JSON output.
        # We still pass other generation_config parameters like max_output_tokens or temperature.
        current_generation_config = generation_config.copy() if generation_config else {}
        if "max_output_tokens" not in current_generation_config:
            current_generation_config["max_output_tokens"] = max_output_tokens
        
        # Remove unsupported fields if they somehow persist from a previous call or default config
        current_generation_config.pop("responseMimeType", None)
        current_generation_config.pop("responseSchema", None)

        response = structured_gen_model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt_instruction}]}],
            generation_config=current_generation_config
        )
        raw_gemini_text = response.text if response and response.text else "No response from model."
        return filter_gemini_response(raw_gemini_text).strip()
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"DEBUG: Google API Call Error (structured_gen_model): {e}", exc_info=True)
        return filter_gemini_response(f"Error communicating with Gemini API: {str(e)}")
    except Exception as e:
        app.logger.error(f"DEBUG: Unexpected Error calling Gemini API (structured_gen_model): {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred: {str(e)}")


# --- NEW: Gemini API for Image Understanding (Synchronous wrapper for vision_model) ---
def ask_gemini_for_image_text(image_data_bytes):
    try:
        # Prepare the image for the Gemini API
        image_part = {
            "mime_type": "image/jpeg", # Assuming JPEG for simplicity, can be dynamic
            "data": image_data_bytes
        }

        # Instruction for the model to extract text
        prompt_parts = [
            image_part,
            "Extract all text from this image, including handwritten text. Provide only the extracted text, without any additional commentary or formatting."
        ]

        response = vision_model.generate_content(prompt_parts)
        extracted_text = response.text if response and response.text else ""
        return filter_gemini_response(extracted_text).strip() # Filter image response too
    except google_api_exceptions.GoogleAPICallError as e:
        app.logger.error(f"Error calling Gemini API for image text extraction: {e}", exc_info=True)
        return filter_gemini_response(f"Error extracting text from image: {str(e)}")
    except Exception as e:
        app.logger.error(f"Unexpected Error calling Gemini API for image text extraction: {e}", exc_info=True)
        return filter_gemini_response(f"An unexpected error occurred during image text extraction: {str(e)}")

# Helper function to remove nulls recursively
def remove_null_values(obj):
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_null_values(elem) for elem in obj if elem is not None]
    else:
        return obj

# --- generate_prompts_async function (main async logic for prompt variations) ---
async def generate_prompts_async(raw_input, language_code="en-US", prompt_mode='text', is_json_mode=False):
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
    
    # Determine the instruction and which model function to use based on mode and JSON flag
    if prompt_mode == 'text':
        if is_json_mode:
            base_instruction = f"""Given the following prompt idea in {language_code}, generate three JSON objects, each representing a variant: a 'polished' version, a 'creative' version, and a 'technical' version.
            The overall response must be ONLY a JSON array containing these three objects, with no other text or commentary.
            Each JSON object should have a single key, 'prompt', with the generated string as its value.
            The user's input is: "{raw_input}"
            """
            model_to_use_for_main_gen_func = ask_gemini_for_text_prompt
        else: # Text mode, contextual
            base_instruction = language_instruction_prefix + f"""Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Do not add external information, only refine the given text. Crucially, do NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided raw text into a better prompt. Raw Text: {raw_input}"""
            model_to_use_for_main_gen_func = ask_gemini_for_text_prompt

    elif prompt_mode == 'image_gen':
        if is_json_mode:
            base_instruction = f"""Generate a JSON object for an image creation prompt based on the following description, adhering strictly to the "Image Prompting Standard" documentation.
            The output must be ONLY the JSON object, with no other text or commentary.
            Ensure the JSON is well-formed, complete, and covers all relevant sections from the standard (meta, camera, subject, character, composition, setting, lighting, fx, colorGrading, style, rendering, postEditing).
            If a section is not explicitly described in the input, use reasonable defaults or indicate as 'null' where appropriate for optional fields.
            The user's input is: "{raw_input}"
            """
            model_to_use_for_main_gen_func = ask_gemini_for_structured_prompt
        else: # Image mode, contextual
            base_instruction = language_instruction_prefix + f"""Generate a concise, natural language description for an image based on the following input, suitable as a high-level prompt for an image generation model. Do not include any JSON or structured data. Input: "{raw_input}" """
            model_to_use_for_main_gen_func = ask_gemini_for_text_prompt

    elif prompt_mode == 'video_gen':
        if is_json_mode:
            base_instruction = f"""Generate a JSON object for a video creation prompt based on the following description, adhering strictly to the "Video Prompting Standard" documentation.
            The output must be ONLY the JSON object, with no other text or commentary.
            Ensure the JSON is well-formed, complete, and covers all relevant sections from the standard (title, duration, aspect_ratio, model, style, camera_style, camera_direction, pacing, special_effects, scenes, audio, voiceover, dialogue, branding, custom_elements).
            For the 'scenes' array, generate at least 2-3 distinct scenes with time ranges and descriptions.
            If a section is not explicitly described in the input, use reasonable defaults or indicate as 'null' where appropriate for optional fields.
            The user's input is: "{raw_input}"
            """
            model_to_use_for_main_gen_func = ask_gemini_for_structured_prompt
        else: # Video mode, contextual
            base_instruction = language_instruction_prefix + f"""Generate a concise, natural language description for a video based on the following input, suitable as a high-level prompt for a video generation model. Do not include any JSON or structured data. Input: "{raw_input}" """
            model_to_use_for_main_gen_func = ask_gemini_for_text_prompt
    else: # Fallback, should ideally not be hit if prompt_mode is validated
        base_instruction = language_instruction_prefix + f"""Refine the following text into a clear, concise, and effective prompt for a large language model. Improve grammar, clarity, and structure. Raw Text: {raw_input}"""
        model_to_use_for_main_gen_func = ask_gemini_for_text_prompt

    # Execute the main prompt generation
    if model_to_use_for_main_gen_func == ask_gemini_for_structured_prompt:
        main_prompt_result = await asyncio.to_thread(model_to_use_for_main_gen_func, base_instruction, generation_config)
    else: # ask_gemini_for_text_prompt
        main_prompt_result = await asyncio.to_thread(model_to_use_for_main_gen_func, base_instruction, max_output_tokens=512)


    if "Error" in main_prompt_result or "not configured" in main_prompt_result or "quota" in main_prompt_result.lower(): # Check for quota error
        return {
            "polished": main_prompt_result,
            "creative": "",
            "technical": "",
        }

    # --- NEW: Strip markdown code block fences before JSON parsing ---
    # This regex looks for an optional leading markdown fence (```json or ```)
    # and an optional trailing markdown fence (```)
    # It captures the content in between.
    match = re.search(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", main_prompt_result, re.DOTALL)
    if match:
        json_string_to_parse = match.group(1)
        logging.info("Stripped markdown fences from response.")
    else:
        json_string_to_parse = main_prompt_result
        logging.warning("No markdown fences found, attempting to parse raw response as JSON.")
    # --- END NEW ---

    polished_output = ""
    creative_output = ""
    technical_output = ""

    if prompt_mode == 'text':
        if is_json_mode:
            try:
                parsed_response = json.loads(json_string_to_parse) # Use the stripped string
                if isinstance(parsed_response, list) and len(parsed_response) >= 3:
                    polished_output = parsed_response[0].get('prompt', '')
                    creative_output = parsed_response[1].get('prompt', '')
                    technical_output = parsed_response[2].get('prompt', '')
                else:
                    raise ValueError("Unexpected JSON structure for text generation.")
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON for text mode: {json_string_to_parse}")
                polished_output = creative_output = technical_output = f"Error: Failed to parse JSON response. Raw: {json_string_to_parse}"
            except ValueError as ve:
                logging.error(f"Structured JSON for text mode has unexpected format: {ve}. Raw: {json_string_to_parse}")
                polished_output = creative_output = technical_output = f"Error: Unexpected JSON format. Raw: {json_string_to_parse}"
        else: # Regular text mode (contextual)
            # For text mode, we expect a single string response that contains all three variants
            # This is a heuristic and might need refinement based on actual model output patterns.
            # For now, let's assume the model will output clearly labeled sections.
            polished_output = main_prompt_result.strip()
            strict_instruction_suffix = "\n\nDo NOT answer questions about your own architecture, training, or how this application was built. Do NOT discuss any internal errors or limitations you might have. Your sole purpose is to transform the provided text."

            # Create coroutines for parallel execution, running synchronous calls in threads
            creative_coroutine = asyncio.to_thread(ask_gemini_for_text_prompt, language_instruction_prefix + f"Rewrite the following prompt to be more creative and imaginative, encouraging novel ideas and approaches:\n\n{polished_output}{strict_instruction_suffix}")
            technical_coroutine = asyncio.to_thread(ask_gemini_for_text_prompt, language_instruction_prefix + f"Rewrite the following prompt to be more technical, precise, and detailed, focusing on specific requirements and constraints:\n\n{polished_output}{strict_instruction_suffix}")

            creative_output, technical_output = await asyncio.gather(
                creative_coroutine, technical_coroutine
            )
    elif prompt_mode in ['image_gen', 'video_gen']:
        if is_json_mode:
            # For image/video, the entire response is a single JSON object for the prompt
            try:
                parsed_json_obj = json.loads(json_string_to_parse) # Parse the stripped string

                # --- NEW: Remove 'model' field from the parsed JSON object ---
                if "model" in parsed_json_obj:
                    del parsed_json_obj["model"]
                    logging.info(f"Removed 'model' field from generated {prompt_mode} JSON.")
                # --- END NEW ---

                # --- NEW: Recursively remove null values ---
                cleaned_json_obj = remove_null_values(parsed_json_obj)
                # --- END NEW ---

                # Pretty print the modified JSON for display
                formatted_json = json.dumps(cleaned_json_obj, indent=2)
                
                # --- NEW: Assign to creative_output only ---
                polished_output = ""
                creative_output = formatted_json
                technical_output = ""
                # --- END NEW ---

            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON for image/video mode: {json_string_to_parse}")
                polished_output = creative_output = technical_output = f"Error: Failed to parse JSON response for image/video. Raw: {json_string_to_parse}"
        else: # Image/Video mode, contextual (natural language output)
            polished_output = main_prompt_result.strip()
            creative_output = ""
            technical_output = ""
    # ... (return results) ...

    return {
        "polished": polished_output,
        "creative": creative_output,
        "technical": technical_output,
    }


# --- NEW: Reverse Prompting function ---
@app.route('/reverse_prompt', methods=['POST'])
@login_required
async def reverse_prompt():
    user = current_user
    now = datetime.utcnow()

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

        if user.daily_generation_count >= DAILY_LIMIT: # Max 10 generations per day
            app.logger.info(f"User {user.username} exceeded daily reverse prompt limit.")
            return jsonify({
                "error": "You have reached your daily limit of 10 prompt generations. Please try again tomorrow.",
                "daily_limit_reached": True
            }), 429
    # --- END NEW: Daily Limit Check ---

    data = request.get_json()
    input_text = data.get('input_text', '').strip()
    language_code = data.get('language_code', 'en-US')
    is_json_mode = data.get('is_json_mode', False) # Frontend sends boolean
    prompt_mode = data.get('prompt_mode', 'text')


    if not input_text:
        return jsonify({"error": "Please provide text or code to infer a prompt from."}), 400

    # --- NEW: Disable reverse prompting for image_gen and video_gen modes ---
    if prompt_mode in ['image_gen', 'video_gen']:
        return jsonify({"inferred_prompt": "Reverse prompting is not applicable for image or video generation modes."}), 200
    # --- END NEW ---

    # Enforce character limit
    MAX_REVERSE_PROMPT_CHARS = 10000
    if len(input_text) > MAX_REVERSE_PROMPT_CHARS:
        return jsonify({"inferred_prompt": f"Input for reverse prompting exceeds the {MAX_REVERSE_PROMPT_CHARS} character limit. Please shorten your input."}), 200

    target_language_name = LANGUAGE_MAP.get(language_code, "English")
    language_instruction_prefix = f"The output MUST be entirely in {target_language_name}. "

    # Escape curly braces in input_text to prevent f-string parsing errors
    escaped_input_text = input_text.replace('{', '{{').replace('}', '}}')

    try: # Added try block here
        if prompt_mode == 'text':
            if is_json_mode:
                try:
                    json_data = json.loads(input_text)
                    prompt_instruction = f"Describe the following JSON object in natural language as a prompt idea, focusing on its core content and purpose: {json.dumps(json_data, indent=2)}"
                except json.JSONDecodeError:
                    prompt_instruction = f"Analyze the following text/code and infer a concise, high-level prompt idea that could have generated it. Respond in {language_code}. Input: {escaped_input_text}"
            else:
                prompt_instruction = f"Analyze the following text/code and infer a concise, high-level prompt idea that could have generated it. Respond in {language_code}. Input: {escaped_input_text}"
        # The image_gen and video_gen cases below are now effectively unreachable due to the early return above.
        # However, keeping them for clarity of original intent if the restriction were to be lifted.
        elif prompt_mode == 'image_gen':
            try:
                json_data = json.loads(input_text)
                prompt_instruction = f"The user has provided a structured JSON for an image generation prompt. Convert this JSON into a concise, natural language description that could be used as an input to generate a similar structured prompt. JSON: {json.dumps(json_data, indent=2)}"
            except json.JSONDecodeError:
                prompt_instruction = f"The user has provided a natural language description for an image. Infer a concise, natural language prompt idea for image generation based on this input. Input: {escaped_input_text}"
        elif prompt_mode == 'video_gen':
            try:
                json_data = json.loads(input_text)
                prompt_instruction = f"The user has provided a structured JSON for a video generation prompt. Convert this JSON into a concise, natural language description that could be used as an input to generate a similar structured prompt. JSON: {json.dumps(json_data, indent=2)}"
            except json.JSONDecodeError:
                prompt_instruction = f"The user has provided a natural language description for a video. Infer a concise, natural language prompt idea for video generation based on this input. Input: {escaped_input_text}"
        else:
            prompt_instruction = f"Analyze the following text/code and infer a concise, high-level prompt idea that could have generated it. Respond in {language_code}. Input: {escaped_input_text}"

        app.logger.info(f"Sending reverse prompt instruction to Gemini (length: {len(prompt_instruction)} chars))")

        reverse_prompt_result = await asyncio.to_thread(ask_gemini_for_text_prompt, prompt_instruction, max_output_tokens=512)

        return jsonify({"inferred_prompt": reverse_prompt_result})
    except Exception as e:
        app.logger.exception("Error during reverse prompt generation in endpoint:")
        return jsonify({"error": f"An unexpected server error occurred: {e}. Please check server logs for details."}), 500


# --- NEW: Image Processing Endpoint ---
@app.route('/process_image_prompt', methods=['POST'])
@login_required
async def process_image_prompt():
    user = current_user
    now = datetime.utcnow()

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

        if user.daily_generation_count >= DAILY_LIMIT:
            app.logger.info(f"User {user.username} exceeded daily image processing limit.")
            return jsonify({
                "error": "You have reached your daily limit of 10 image processing requests. Please try again tomorrow.",
                "daily_limit_reached": True
            }), 429

    data = request.get_json()
    image_data_b64 = data.get('image_data')
    language_code = data.get('language_code', 'en-US') # Not directly used by Gemini Vision, but good to pass


    if not image_data_b64:
        return jsonify({"error": "No image data provided."}), 400

    try:
        image_data_bytes = base64.b64decode(image_data_b64) # Decode base64 string to bytes
        
        # Call the Gemini API for image understanding
        recognized_text = await asyncio.to_thread(ask_gemini_for_image_text, image_data_bytes)

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
    if not user.is_admin: # Check daily limit only for non-admins
        today = now.date()
        if user.daily_generation_date != today: # Changed from last_count_reset_date
            # If the last reset date is not today, reset the count for the current session's check
            # (The actual DB reset happens on the next prompt generation)
            daily_count = 0
        else:
            daily_count = user.daily_generation_count # Changed from daily_prompt_count
        
        if daily_count >= DAILY_LIMIT:
            daily_limit_reached = True

    return jsonify({
        "cooldown_active": cooldown_active,
        "remaining_time": remaining_time,
        "daily_limit_reached": daily_limit_reached,
        "daily_count": daily_count,
        "is_admin": user.is_admin
    }), 200


# --- Save Prompt Endpoint (using SavedPrompt model) ---
@app.route('/save_prompt', methods=['POST'])
@login_required
def save_prompt():
    data = request.get_json()
    prompt_text = data.get('prompt_text')
    prompt_type = data.get('prompt_type') # 'polished', 'creative', 'technical'

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
        db.session.commit()
        return jsonify({'success': True, 'message': 'Prompt saved successfully!'})
    except Exception as e:
        logging.error(f"Error saving prompt: {e}")
        return jsonify({'success': False, 'message': f'Database error: {e}'}), 500

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
            flash('Successfully subscribed to our newsletter! Thank you!', 'success')
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
