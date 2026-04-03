from flask import Blueprint, render_template

bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@bp.route('/')
def index():
    has_uploaded_data = False # 업로드 받은 데이터가 있는지 여부에 따라 다른 화면을 보여주기 위해서 필요한 변수
    return render_template('analysis.html', has_uploaded_data=has_uploaded_data)
