from flask import Blueprint, url_for, render_template, request
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/contact', methods=['GET', 'POST'])
def contact():
    #폼을 띄워주는 코드가 올 자리
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        #받은 데이터를 처리하는 코드(DB저장 등)
        print(name, email, message)

        return redirect(url_for('main.index'))

    return render_template('contact.html')
