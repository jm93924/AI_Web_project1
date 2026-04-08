from flask import Blueprint, render_template
from app.services.kamis_api import get_today_strawberry_wholesale_price

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@bp.route('/')
def index():
    wholesale_info = None

    try:
        wholesale_info = get_today_strawberry_wholesale_price()
    except Exception as e:
        print("KAMIS API 호출 실패:", e)

    return render_template(
        'dashboard.html',
        wholesale_info=wholesale_info
    )