# forms.py (Insert this new form class)

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length

class AddLibraryPromptForm(FlaskForm):
    # This will be the main heading/use case shown on the /all_prompts page
    title = StringField('Prompt Title / Use Case', validators=[DataRequired(), Length(max=255)])
    
    # The full, detailed prompt text (The Prompt Description)
    description = TextAreaField('Detailed Prompt Text', validators=[DataRequired()])
    
    # Category (Where to be used) - Populate choices dynamically in the route
    category = SelectField('Category (Where to be used)', validators=[DataRequired()], 
                           choices=[('General', 'General'), ('Code', 'Code'), ('Creative', 'Creative'), ('Other', 'Other')]) 
                           # NOTE: You must populate the real choices from CATEGORIES_AND_SUBCATEGORIES in the route.
    
    submit = SubmitField('Add Prompt to Library')
