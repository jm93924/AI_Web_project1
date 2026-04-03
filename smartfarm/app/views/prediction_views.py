from flask import Blueprint, render_template, request

bp = Blueprint('prediction', __name__, url_prefix='/prediction')

@bp.route('/')
def index():
    #더미 데이터 사용하는 코드(추후 수정 필요)
    selected_crop = request.args.get('crop', '딸기')
    selected_variety = request.args.get('variety', '설향')
    selected_region = request.args.get('region', '전국')
    selected_period = request.args.get('period', 'short')

    current_price = 10500
    predicted_tomorrow_price = 10600
    price_diff = 100
    price_diff_rate = 1.0
    confidence = 88
    current_date = '2026년 3월 13일'

    if selected_period == 'short':
        chart_labels = ['03/13', '03/14', '03/15', '03/16', '03/17']
        chart_values = [10500, 10600, 10650, 10750, 10800]
        forecast_list = [
            {'date': '03/14', 'price': 10600, 'confidence': 88},
            {'date': '03/15', 'price': 10650, 'confidence': 85},
            {'date': '03/16', 'price': 10750, 'confidence': 82},
        ]
    elif selected_period == 'long':
        chart_labels = ['4월', '5월', '6월', '7월', '8월', '9월']
        chart_values = [10400, 10350, 10200, 10150, 10080, 10200]
        forecast_list = [
            {'date': '4월', 'price': 10400, 'confidence': 80},
            {'date': '5월', 'price': 10350, 'confidence': 78},
            {'date': '6월', 'price': 10200, 'confidence': 76},
        ]
    else:
        chart_labels = ['2025/11', '2025/12', '2026/01', '2026/02', '2026/03']
        chart_values = [9800, 10100, 10300, 10400, 10500]
        forecast_list = [
            {'date': '2025/12', 'price': 10100, 'confidence': 100},
            {'date': '2026/01', 'price': 10300, 'confidence': 100},
            {'date': '2026/02', 'price': 10400, 'confidence': 100},
        ]

    weather_impact_labels = ['습도', '일조량', '강수량', '기온']
    weather_impact_values = [8, 15, 12, 18]

    best_shipping_period = '2026년 3월 16일 – 3월 17일'
    best_shipping_price = 10750

    market_trend_text = '다음 주 날씨가 맑을 것으로 예상되어 딸기 당도가 높아질 것입니다. 고품질 딸기는 프리미엄 가격을 받을 수 있습니다.'
    long_term_strategy_text = '3~4월 가격이 가장 높은 시기입니다. 품질 관리를 철저히 하여 이 시기에 집중 출하하면 최대 수익을 얻을 수 있습니다.'

    return render_template(
        'prediction.html',
        selected_crop=selected_crop,
        selected_variety=selected_variety,
        selected_region=selected_region,
        selected_period=selected_period,
        current_price=current_price,
        predicted_tomorrow_price=predicted_tomorrow_price,
        price_diff=price_diff,
        price_diff_rate=price_diff_rate,
        confidence=confidence,
        current_date=current_date,
        chart_labels=chart_labels,
        chart_values=chart_values,
        forecast_list=forecast_list,
        weather_impact_labels=weather_impact_labels,
        weather_impact_values=weather_impact_values,
        best_shipping_period=best_shipping_period,
        best_shipping_price=best_shipping_price,
        market_trend_text=market_trend_text,
        long_term_strategy_text=long_term_strategy_text
    )