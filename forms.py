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
