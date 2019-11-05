from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    search = StringField('Search', validators=[DataRequired()])
    submit = SubmitField('Search')

class BigSearchForm(FlaskForm):
    search = TextAreaField('Search', validators=[DataRequired()])
    submit = SubmitField('Search')