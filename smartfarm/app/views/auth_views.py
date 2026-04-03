from os import name

from flask import Blueprint, url_for, render_template, request, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect

from app import db
from app.forms import CustomerCreateForm, CustomerLoginForm
from app.models import Customer


bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/signin/', methods=['GET', 'POST'])
def signin():
    form = CustomerLoginForm()

    # 요청이 POST이고 사용자 정보를 입력한 경우
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        customer = Customer.query.filter_by(id=form.id.data).first()
        if not customer:
            error = "존재하지 않는 사용자입니다."
        elif not check_password_hash(customer.password, form.password.data):
            error = "비밀번호가 올바르지 않습니다."
        if error is None:
            session.clear()
            session['customer_seq'] = customer.seq
            return redirect(url_for('main.index'))
        flash(error)

    return render_template('auth/signin.html', form=form)

@bp.route('/signup/', methods=['GET','POST'])
def signup():
    form = CustomerCreateForm()

    # POST 요청이면 계정등록을 수행
    if request.method == 'POST' and form.validate_on_submit():
        customer = Customer.query.filter_by(id=form.id.data).first()
        if not customer:
            customer = Customer(id=form.id.data,
                                password=generate_password_hash(form.password1.data),
                                name=form.name.data,
                                address=form.address.data,
                                phone=form.phone.data,
                                email=form.email.data)
            db.session.add(customer)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            flash('이미 존재하는 사용자입니다.')

    # GET 요청이면 계정등록을 하는 화면을 띄워줌
    return render_template('auth/signup.html', form=form)

@bp.before_app_request
def load_signed_in_customer():
    customer_seq = session.get('customer_seq')
    if customer_seq is None:
        g.customer = None
    else:
        g.customer = Customer.query.get(customer_seq)


@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))
