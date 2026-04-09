import requests
from datetime import datetime, timedelta, date

KAMIS_API_KEY = "3cd65fc0-a7f8-4802-a429-57a9f0ef749c"
KAMIS_CERT_ID = "7375"
KAMIS_URL = "http://www.kamis.or.kr/service/price/xml.do"

def get_today_strawberry_wholesale_price():
    """
    KAMIS에서 딸기 도매가격을 조회해서
    가장 최근 날짜의 평균 가격 1건을 반환한다.
    """

    today = datetime.today().date()
    start_day = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_day = today.strftime("%Y-%m-%d")

    params = {
        "action": "periodProductList",  # 일별 품목별 도소매가격 정보
        "p_cert_key": KAMIS_API_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_returntype": "json",
        "p_startday": start_day,
        "p_endday": end_day,
        "p_productclscode": "02",     # 도매
        "p_itemcategorycode": "200",  # 분류코드
        "p_itemcode": "226",          # 딸기 품목코드
        "p_kindcode": "00",           # 품종코드
        "p_productrankcode": "05",    # 등급코드(중품)
        "p_countrycode": "1101",      # 서울
        "p_convert_kg_yn": "Y"
    }

    response = requests.get(KAMIS_URL, params=params, timeout=10)
    response.raise_for_status()

    result = response.json()

    # 실제 응답 구조: result -> data -> item
    data_block = result.get("data", {})
    error_code = data_block.get("error_code", "")

    if error_code != "000":
        print("KAMIS error_code:", error_code)
        return None

    items = data_block.get("item", [])

    if not items:
        return None

    # item이 dict 하나로 올 수도 있으니 리스트로 보정
    if isinstance(items, dict):
        items = [items]

    # '평균' 데이터만 우선 사용
    avg_items = [x for x in items if isinstance(x, dict) and x.get("countyname") == "평균"]

    target_items = avg_items if avg_items else [x for x in items if isinstance(x, dict)]

    if not target_items:
        return None

    # regday가 MM/DD 형식이라 yyyy와 합쳐서 정렬
    def sort_key(x):
        yyyy = str(x.get("yyyy", "")).strip()
        regday = str(x.get("regday", "")).strip()
        return f"{yyyy}-{regday}"

    target_items = sorted(target_items, key=sort_key, reverse=True)
    latest = target_items[0]

    price_str = str(latest.get("price", "0")).replace(",", "").strip()
    price = int(price_str) if price_str.isdigit() else 0

    return {
        "itemname": latest.get("itemname", "딸기"),
        "kindname": latest.get("kindname", ""),
        "countyname": latest.get("countyname", ""),
        "marketname": latest.get("marketname", ""),
        "regday": latest.get("regday", ""),
        "yyyy": latest.get("yyyy", ""),
        "price": price
    }

def fetch_kamis_period_prices(
    start_day: str,
    end_day: str,
    product_cls_code: str = "02",      # 도매
    item_category_code: str = "200",
    item_code: str = "226",            # 딸기
    kind_code: str = "00",
    product_rank_code: str = "05",     # 중품
    country_code: str = "1101",        # 서울
    convert_kg_yn: str = "Y"
):
    """
    지정 기간의 KAMIS 가격 데이터를 가져와 raw json item 리스트를 반환
    """

    params = {
        "action": "periodProductList",
        "p_cert_key": KAMIS_API_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_returntype": "json",
        "p_startday": start_day,
        "p_endday": end_day,
        "p_productclscode": product_cls_code,
        "p_itemcategorycode": item_category_code,
        "p_itemcode": item_code,
        "p_kindcode": kind_code,
        "p_productrankcode": product_rank_code,
        "p_countrycode": country_code,
        "p_convert_kg_yn": convert_kg_yn,
    }

    response = requests.get(KAMIS_URL, params=params, timeout=20)
    response.raise_for_status()

    print("요청 URL:", response.url)

    result = response.json()
    print("condition:", result.get("condition"))

    data_block = result.get("data", {})
    error_code = data_block.get("error_code", "")

    if error_code != "000":
        raise ValueError(f"KAMIS API error_code={error_code}, response={result}")

    items = data_block.get("item", [])

    if isinstance(items, dict):
        items = [items]

    if items:
        print("첫 행:", items[0])
        print("마지막 행:", items[-1])

    return items


def parse_kamis_item(item: dict, meta: dict) -> dict:
    """
    KAMIS item 1건을 DB 저장용 dict로 변환
    """
    yyyy = str(item.get("yyyy", "")).strip()
    regday = str(item.get("regday", "")).strip()   # 예: 04/07

    price_date = None
    if yyyy and regday:
        month, day = regday.split("/")
        price_date = date(int(yyyy), int(month), int(day))

    price_str = str(item.get("price", "0")).replace(",", "").strip()
    price = int(price_str) if price_str.isdigit() else None

    return {
        "price_date": price_date,
        "price": price,
    }