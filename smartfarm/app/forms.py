from flask_wtf import FlaskForm
from wtforms.fields.simple import StringField, TextAreaField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, Email

class CustomerCreateForm(FlaskForm):
    id = StringField('아이디', validators=[DataRequired(), Length(min=5, max=20)])
    password1 = PasswordField('비밀번호', validators=[DataRequired(), EqualTo('password2', '비밀번호가 일치하지않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    name = StringField('이름', validators=[DataRequired()])
    address = StringField('주소', validators=[DataRequired()])
    phone = StringField('전화번호', validators=[DataRequired()])
    ## 전화번호 타입검증 버전
    # phone = StringField(
    #     '전화번호',
    #     validators=[
    #         DataRequired(),
    #         Regexp(r'^01[0-9]-?\d{3,4}-?\d{4}$', message='올바른 전화번호 형식이 아닙니다.')
    #     ]
    # )
    email = StringField('이메일', validators=[DataRequired(), Email()])

class CustomerLoginForm(FlaskForm):
    id = StringField('아이디', validators=[DataRequired(), Length(min=5, max=20)])
    password = PasswordField('비밀번호', validators=[DataRequired()])