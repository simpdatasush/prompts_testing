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

# -------------------------------------------------------------
# 5. AddAIAppForm (NEW CLASS FOR LATEST AI APPS)
# -------------------------------------------------------------
class AddAIAppForm(FlaskForm):
    name = StringField('App Name', validators=[DataRequired(), Length(max=255)])
    developer = StringField('Developer/Company', validators=[DataRequired(), Length(max=100)])
    summary = TextAreaField('Short Summary', validators=[DataRequired()])
    app_url = StringField('App URL (Website/Store)', validators=[DataRequired()])
    category = SelectField('Category (e.g., Image Gen, Text Editor)', validators=[DataRequired()], 
                            choices=[('ImageGen', 'Image Generation'), 
                                     ('TextGen', 'Text Generation'),
                                     ('Audio', 'Audio/Voice'),
                                     ('Data', 'Data Analysis'),
                                     ('Other', 'Other')]) # Simple fixed choices
    date_launched = StringField('Launch Date (YYYY-MM-DD)', validators=[Optional()])
    submit = SubmitField('Add AI App')

# __forms.py__ (Append the new form after AddAIAppForm)

# -------------------------------------------------------------
# 6. AddAIBookForm (NEW CLASS FOR LATEST AI BOOKS)
# -------------------------------------------------------------
class AddAIBookForm(FlaskForm):
    title = StringField('Book Title', validators=[DataRequired(), Length(max=255)])
    author = StringField('Author', validators=[DataRequired(), Length(max=100)])
    summary = TextAreaField('Short Summary/Synopsis', validators=[DataRequired()])
    purchase_url = StringField('Purchase URL (Amazon/Publisher)', validators=[DataRequired()])
    topic = SelectField('Primary Topic', validators=[DataRequired()], 
                            choices=[('PromptEng', 'Prompt Engineering'), 
                                     ('MLOps', 'ML/MLOps'),
                                     ('Ethics', 'AI Ethics'),
                                     ('GenAI', 'General Generative AI'),
                                     ('Other', 'Other')]) # Simple fixed choices
    date_published = StringField('Published Date (YYYY-MM-DD)', validators=[Optional()])
    submit = SubmitField('Add AI Book')
