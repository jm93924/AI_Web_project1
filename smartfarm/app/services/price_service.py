from collections import defaultdict
from app.models import Price


# 일간 가격 데이터
def get_price_chart_data(limit=14, grade='중'):
    """
    최근 limit일 가격 데이터를 조회해서
    Chart.js에 바로 넣을 수 있는 형태로 반환
    """

    rows = (
        Price.query
        .filter(Price.grade == grade)
        .order_by(Price.trade_date.desc())
        .limit(limit)
        .all()
    )

    # 최신순으로 가져왔으니, 차트용으로 다시 날짜 오름차순 정렬
    rows = list(reversed(rows))

    labels = [row.trade_date.strftime('%m/%d') for row in rows]
    values = [row.avg_price for row in rows]

    return {
        'labels': labels,
        'values': values
    }


def get_price_chart_data_weekly(weeks=8, grade='중'):
    """
    일별 데이터를 주차별 평균가로 묶어서 최근 weeks주 반환
    """
    query = Price.query

    if grade:
        query = query.filter(Price.grade == grade)

    rows = (
        query
        .order_by(Price.trade_date.asc())
        .all()
    )

    weekly_map = defaultdict(list)

    for row in rows:
        year, week, _ = row.trade_date.isocalendar()
        key = (year, week)
        weekly_map[key].append(row.avg_price)

    weekly_items = []
    for (year, week), prices in weekly_map.items():
        avg_price = round(sum(prices) / len(prices))
        label = f'{str(year)[2:]}년 {week}주'
        weekly_items.append((year, week, label, avg_price))

    weekly_items = weekly_items[-weeks:]

    labels = [item[2] for item in weekly_items]
    values = [item[3] for item in weekly_items]

    return {
        'labels': labels,
        'values': values
    }


def get_price_chart_data_monthly(months=6, grade='중'):
    """
    일별 데이터를 월별 평균가로 묶어서 최근 months개월 반환
    """
    query = Price.query

    if grade:
        query = query.filter(Price.grade == grade)

    rows = (
        query
        .order_by(Price.trade_date.asc())
        .all()
    )

    monthly_map = defaultdict(list)

    for row in rows:
        key = (row.trade_date.year, row.trade_date.month)
        monthly_map[key].append(row.avg_price)

    monthly_items = []
    for (year, month), prices in monthly_map.items():
        avg_price = round(sum(prices) / len(prices))
        label = f'{month:02d}월'
        monthly_items.append((year, month, label, avg_price))

    monthly_items = monthly_items[-months:]

    labels = [item[2] for item in monthly_items]
    values = [item[3] for item in monthly_items]

    return {
        'labels': labels,
        'values': values
    }