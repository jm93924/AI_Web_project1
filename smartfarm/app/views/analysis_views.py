from flask import Blueprint, render_template

bp = Blueprint('analysis', __name__, url_prefix='/analysis')

@bp.route('/')
def index():
    return render_template('analysis.html')