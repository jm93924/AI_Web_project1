from datetime import date
from app import create_app, db
from app.models import WholesalePrice
from app.services.kamis_api import fetch_kamis_period_prices, parse_kamis_item
from app.utils.date_ranges import split_date_ranges


def upsert_wholesale_price(row: dict):
    """
    중복키 기준으로 있으면 스킵, 없으면 insert
    """
    exists = WholesalePrice.query.filter_by(
        price_date=row["price_date"]
    ).first()

    if exists:
        return False

    obj = WholesalePrice(**row)
    db.session.add(obj)
    return True


def load_kamis_prices():
    start_date = date(2017, 1, 1)
    end_date = date.today()

    date_ranges = split_date_ranges(start_date, end_date, months_per_chunk=6)

    total_inserted = 0
    total_skipped = 0

    meta = {
        "product_cls_code": "02",      # 도매
        "item_category_code": "200",
        "item_code": "226",            # 딸기
        "kind_code": "00",
        "product_rank_code": "05",
        "country_code": "1101",
        "convert_kg_yn": "Y",
    }

    for chunk_start, chunk_end in date_ranges:
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        print(f"[조회] {start_str} ~ {end_str}")

        try:
            raw_items = fetch_kamis_period_prices(
                start_day=start_str,
                end_day=end_str,
                product_cls_code=meta["product_cls_code"],
                item_category_code=meta["item_category_code"],
                item_code=meta["item_code"],
                kind_code=meta["kind_code"],
            )

            print(f"  raw_items 개수: {len(raw_items)}")

            inserted_in_chunk = 0
            skipped_in_chunk = 0

            for item in raw_items:
                if not isinstance(item, dict):
                    continue

                # 평균 데이터만 저장
                if item.get("countyname") != "평균":
                    skipped_in_chunk += 1
                    continue

                row = parse_kamis_item(item, meta)

                # 날짜 없으면 스킵
                if row["price_date"] is None:
                    skipped_in_chunk += 1
                    continue

                # 요청 구간 밖 데이터는 저장하지 않음
                if not (chunk_start <= row["price_date"] <= chunk_end):
                    print(f"  [범위밖 스킵] 요청:{chunk_start}~{chunk_end}, 응답:{row['price_date']}")
                    skipped_in_chunk += 1
                    continue

                # 가격 없으면 스킵할지 여부는 정책에 따라 결정
                if row["price"] is None:
                    skipped_in_chunk += 1
                    continue

                inserted = upsert_wholesale_price(row)

                if inserted:
                    inserted_in_chunk += 1
                else:
                    skipped_in_chunk += 1

            db.session.commit()

            total_inserted += inserted_in_chunk
            total_skipped += skipped_in_chunk

            print(f"  inserted: {inserted_in_chunk}, skipped: {skipped_in_chunk}")

        except Exception as e:
            db.session.rollback()
            print(f"[오류] {start_str} ~ {end_str} 적재 실패: {e}")

    print("=" * 60)
    print(f"총 inserted: {total_inserted}")
    print(f"총 skipped : {total_skipped}")


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        load_kamis_prices()