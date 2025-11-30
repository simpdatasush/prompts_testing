from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length 

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
# 5. NEW PollQuestionForm (For Admin Poll Management)
# -------------------------------------------------------------
class PollQuestionForm(FlaskForm):
    question_text = TextAreaField(
        'Question Text', 
        validators=[DataRequired()],
        description='The main prediction question (e.g., analyse quantum computing ...?)'
    )
    option_A_text = StringField(
        'Option A Text', 
        validators=[DataRequired(), Length(max=100)],
        description='Text for Option A (e.g., Yes/Texans)'
    )
    option_B_text = StringField(
        'Option B Text', 
        validators=[DataRequired(), Length(max=100)],
        description='Text for Option B (e.g., No/Colts)'
    )
    category = StringField(
        'Category', 
        validators=[Optional(), Length(max=50)], 
        description='Optional: e.g., AI Policy, NFL'
    )
    total_volume = IntegerField(
        'Total Volume (Mock $)', 
        validators=[DataRequired(), NumberRange(min=0)], 
        default=0,
        description='Mock volume or investment amount for display ($)'
    )
    closing_date = DateField(
        'Closing Date', 
        validators=[Optional()], 
        format='%Y-%m-%d',
        description='When voting should stop (YYYY-MM-DD)'
    )
    submit = SubmitField('Add Poll Question')
