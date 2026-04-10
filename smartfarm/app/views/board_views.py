from flask import Blueprint, render_template
from app.services.price_service import (
    get_or_create_today_price,
    get_price_chart_data,
    get_price_chart_data_weekly,
    get_price_chart_data_monthly
)

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')


@bp.route('/')
def index():
    wholesale_info = None
    price_daily = {'labels': [], 'values': []}
    price_weekly = {'labels': [], 'values': []}
    price_monthly = {'labels': [], 'values': []}

    try:
        wholesale_info = get_or_create_today_price(grade='중')
    except Exception as e:
        print("오늘 가격 조회 실패:", e)

    try:
        price_daily = get_price_chart_data(limit=14, grade='중')
        price_weekly = get_price_chart_data_weekly(weeks=8, grade='중')
        price_monthly = get_price_chart_data_monthly(months=6, grade='중')
    except Exception as e:
        print("가격 차트 데이터 조회 실패:", e)

    return render_template(
        'dashboard.html',
        wholesale_info=wholesale_info,
        price_daily=price_daily,
        price_weekly=price_weekly,
        price_monthly=price_monthly
    )