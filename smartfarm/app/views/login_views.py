from flask import Blueprint, render_template, request, redirect, url_for

bp = Blueprint('login', __name__, url_prefix='/login')

@bp.route('/')
def index():
    return render_template('sign_in.html')


#로그인 화면에서 form 제출시 오는 곳
@bp.route('/', methods=['GET', 'POST'])
def sign_in():
    # 예제코드(추후 수정 필요)
    if request.method == 'POST':
        username = request.form['username'] #id가 아닌 name을 호출해야함
        password = request.form['password'] # ex) id=name, name=username이면 'username'으로 값을 호출

        # 로그인 처리하는 코드가 올 공간
        print(username, password)

        return redirect(url_for('main.index'))

    return render_template('login.html')