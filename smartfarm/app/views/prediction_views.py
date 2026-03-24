from flask import Blueprint, render_template

bp = Blueprint('prediction', __name__, url_prefix='/prediction')

@bp.route('/')
def index():
    return render_template('prediction.html')