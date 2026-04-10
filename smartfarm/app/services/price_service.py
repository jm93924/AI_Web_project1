from collections import defaultdict
from datetime import datetime, date, time

from sqlalchemy import func

from app import db
from app.models import Price
from app.services.kamis_api import get_today_strawberry_wholesale_price


def get_latest_price_from_db(grade='중'):
    """
    특정 등급의 가장 최근 가격 1건 조회
    """
    return (
        Price.query
        .filter(Price.grade == grade)
        .order_by(Price.trade_date.desc())
        .first()
    )


def get_price_by_date_and_grade(target_date, grade='중'):
    """
    특정 날짜 + 등급 데이터 1건 조회
    Oracle DATE 비교를 위해 trunc 사용
    """
    return (
        Price.query
        .filter(func.trunc(Price.trade_date) == target_date)
        .filter(Price.grade == grade)
        .first()
    )


def save_price_to_db(trade_date, grade, avg_price):
    """
    같은 날짜/등급 데이터가 이미 있으면 기존값 반환,
    없으면 새로 저장
    """
    existing = get_price_by_date_and_grade(trade_date.date(), grade)
    if existing:
        return existing

    new_price = Price(
        trade_date=trade_date,
        grade=grade,
        avg_price=avg_price
    )

    db.session.add(new_price)
    db.session.commit()

    return new_price


def get_or_create_today_price(grade='중'):
    """
    1) DB에서 해당 등급의 가장 최근 데이터 조회
    2) 오늘 데이터면 그대로 반환
    3) 오늘 데이터가 아니면 API 호출
    4) API 결과를 DB에 저장 후 반환
    """
    latest = get_latest_price_from_db(grade=grade)
    today = date.today()

    if latest and latest.trade_date.date() == today:
        return {
            "price": latest.avg_price,
            "grade": latest.grade,
            "trade_date": latest.trade_date,
            "regday": latest.trade_date.strftime('%m/%d'),
            "source": "db"
        }

    api_data = get_today_strawberry_wholesale_price()

    if not api_data:
        if latest:
            return {
                "price": latest.avg_price,
                "grade": latest.grade,
                "trade_date": latest.trade_date,
                "regday": latest.trade_date.strftime('%m/%d'),
                "source": "db_old"
            }
        return None

    if not api_data.get("trade_date"):
        if latest:
            return {
                "price": latest.avg_price,
                "grade": latest.grade,
                "trade_date": latest.trade_date,
                "regday": latest.trade_date.strftime('%m/%d'),
                "source": "db_old"
            }
        return None

    saved = save_price_to_db(
        trade_date=api_data["trade_date"],
        grade=grade,
        avg_price=api_data["price"]
    )

    return {
        "price": saved.avg_price,
        "grade": saved.grade,
        "trade_date": saved.trade_date,
        "regday": saved.trade_date.strftime('%m/%d'),
        "source": "api"
    }


# 일간 가격 데이터
def get_price_chart_data(limit=14, grade='중'):
    rows = (
        Price.query
        .filter(Price.grade == grade)
        .order_by(Price.trade_date.desc())
        .limit(limit)
        .all()
    )

    rows = list(reversed(rows))

    labels = [row.trade_date.strftime('%m/%d') for row in rows]
    values = [row.avg_price for row in rows]

    return {
        'labels': labels,
        'values': values
    }


def get_price_chart_data_weekly(weeks=8, grade='중'):
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