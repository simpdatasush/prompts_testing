from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, SubmitField, DateField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, NumberRange

# -------------------------------------------------------------
# 1. RegistrationForm (The missing class that caused the error)
# -------------------------------------------------------------
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField(
        'Confirm Password', 
        validators=[DataRequired(), EqualTo('password', message='Passwords must match')]
    )
    submit = SubmitField('Register')

# -------------------------------------------------------------
# 2. AddLibraryPromptForm (The new class we added)
# -------------------------------------------------------------
class AddLibraryPromptForm(FlaskForm):
    title = StringField('Prompt Title / Use Case', validators=[DataRequired(), Length(max=255)])
    description = TextAreaField('Detailed Prompt Text', validators=[DataRequired()])
    # Category choices are dynamically set in the route
    category = SelectField('Category (Where to be used)', validators=[DataRequired()])
    submit = SubmitField('Add Prompt to Library')

# -------------------------------------------------------------
# 3. AddNewsArticleForm (Used by /admin/library_news)
# -------------------------------------------------------------
class AddNewsArticleForm(FlaskForm):
    title = StringField('Article Title', validators=[DataRequired(), Length(max=255)])
    summary = TextAreaField('Summary/Snippet', validators=[DataRequired()])
    # FIX: URL is now imported
    source_url = StringField('Source URL', validators=[DataRequired()]) 
    date_published = StringField('Date Published (YYYY-MM-DD)')
    submit = SubmitField('Add News Article')

# -------------------------------------------------------------
# 4. AddJobPostingForm (Used by /admin/library_jobs)
# -------------------------------------------------------------
class AddJobPostingForm(FlaskForm):
    title = StringField('Job Title', validators=[DataRequired(), Length(max=255)])
    company = StringField('Company Name', validators=[DataRequired(), Length(max=100)])
    location = StringField('Location', validators=[DataRequired(), Length(max=100)])
    description_summary = TextAreaField('Description Summary', validators=[DataRequired()])
    # FIX: URL is now imported
    job_url = StringField('Job URL (Application Link)', validators=[DataRequired()]) 
    date_posted = StringField('Date Posted (YYYY-MM-DD)')
    submit = SubmitField('Add Job Posting')

# forms.py (New forms added at the end)

# ... (Existing forms retained) ...

# -------------------------------------------------------------
# 6. AddAiAppForm (For Admin AI App Management)
# -------------------------------------------------------------
class AddAiAppForm(FlaskForm):
    title = StringField('Application Name', validators=[DataRequired(), Length(max=200)])
    description = TextAreaField('Description/Summary', validators=[DataRequired()])
    app_url = StringField('App URL (Official Link)', validators=[DataRequired()]) 
    submit = SubmitField('Add AI Application')

# -------------------------------------------------------------
# 7. AddAiAppFeatureForm (For Admin AI App Feature Management)
# -------------------------------------------------------------
class AddAiAppFeatureForm(FlaskForm):
    app_name = StringField('Application Name (e.g., Claude, Midjourney)', validators=[DataRequired(), Length(max=100)])
    feature_summary = TextAreaField('Feature Summary/Detail', validators=[DataRequired()])
    release_date = StringField('Release Date (YYYY-MM-DD)', validators=[Optional()])
    submit = SubmitField('Add App Feature')
