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
# 3. NEW: AddNewsArticleForm
# -------------------------------------------------------------
class AddNewsArticleForm(FlaskForm):
    title = StringField('Article Title', validators=[DataRequired(), Length(max=255)])
    summary = TextAreaField('Summary/Snippet', validators=[DataRequired()])
    source_url = StringField('Source URL', validators=[DataRequired(), URL()])
    date_published = StringField('Date Published (YYYY-MM-DD)', validators=[Optional()])
    submit = SubmitField('Add News Article')

# -------------------------------------------------------------
# 4. NEW: AddJobPostingForm
# -------------------------------------------------------------
class AddJobPostingForm(FlaskForm):
    title = StringField('Job Title', validators=[DataRequired(), Length(max=255)])
    company = StringField('Company Name', validators=[DataRequired(), Length(max=100)])
    location = StringField('Location', validators=[Optional(), Length(max=100)])
    description_summary = TextAreaField('Description Summary', validators=[DataRequired()])
    job_url = StringField('Job URL (Application Link)', validators=[DataRequired(), URL()])
    date_posted = StringField('Date Posted (YYYY-MM-DD)', validators=[Optional()])
    submit = SubmitField('Add Job Posting')
