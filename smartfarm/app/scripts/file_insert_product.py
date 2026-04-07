from app import create_app
import pandas as pd
import os
import re

from app import db
from app.models import FarmInfo, ProductData

PRODUCT_DATA_PATH = "D:/취업반/1차프로젝트/raw_datas/생산_딸기_2018.csv"


def load_strawberry_product_csv(csv_path):
    """
    생산 csv를 읽어서 product_data 테이블에 저장한다.

    생산 CSV에는 farm_id가 없으므로,
    파일명에서 추출한 연도 + 품목 + 농가명 으로 farm_info를 조회해서
    해당 farm_id를 찾아 product_data에 넣는다.

    같은 날짜의 출하량/판매금액은 날짜별로 합산하여 1행으로 저장한다.
    """

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 컬럼명 리네임
    df = df.rename(columns={
        '도': 'district',
        '시군': 'city',
        '품목': 'item',
        '농가명': 'farm_num',
        '출하일자': 'production_date',
        '총출하량': 'total_quantity',
        '판매금액': 'total_sales'
    })

    # 필요한 컬럼만 사용
    df = df[['item', 'farm_num', 'production_date', 'total_quantity', 'total_sales']]

    # NaN -> None 전에 날짜/숫자 처리
    df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
    df['total_quantity'] = pd.to_numeric(df['total_quantity'], errors='coerce')
    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')

    # 파일명에서 연도 추출
    filename = os.path.basename(csv_path)
    match = re.search(r'\d{4}', filename)
    if not match:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {filename}")

    survey_year = int(match.group())
    df['survey_year'] = survey_year

    # 품목 문자열 정리
    df['item'] = df['item'].astype(str).str.strip()

    # 농가명 숫자 변환
    df['farm_num'] = pd.to_numeric(df['farm_num'], errors='coerce')

    # 필수값 없는 행 제거
    invalid_mask = (
        df['survey_year'].isna() |
        df['item'].isna() |
        df['farm_num'].isna() |
        df['production_date'].isna() |
        df['total_quantity'].isna()
    )

    skipped_count = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    # farm_num 정수화
    df['farm_num'] = df['farm_num'].astype(int)

    # 판매금액이 비어 있으면 0으로 둘지, None으로 둘지 선택 가능
    # 여기서는 합산 편의상 0으로 채움
    df['total_sales'] = df['total_sales'].fillna(0)

    # 날짜별 취합
    grouped_df = (
        df.groupby(['survey_year', 'item', 'farm_num', 'production_date'], as_index=False)
          .agg({
              'total_quantity': 'sum',
              'total_sales': 'sum'
          })
    )

    inserted_count = 0

    try:
        # farm_info 전체를 미리 읽어서 매핑 딕셔너리 생성
        farm_rows = FarmInfo.query.all()
        farm_map = {}

        for farm in farm_rows:
            key = (
                farm.survey_year,
                farm.item.strip(),
                farm.farm_num
            )
            farm_map[key] = farm.farm_id

        for row in grouped_df.itertuples(index=False):
            survey_year = int(row.survey_year)
            item = row.item.strip()
            farm_num = int(row.farm_num)

            # farm_info에서 farm_id 찾기
            farm_key = (survey_year, item, farm_num)
            farm_id = farm_map.get(farm_key)

            if farm_id is None:
                print(f"farm_info 매칭 실패 -> 연도:{survey_year}, 품목:{item}, 농가명:{farm_num}")
                skipped_count += 1
                continue

            production_date = row.production_date.to_pydatetime()

            # PK 중복 체크 (farm_id, production_date)
            existing_product = ProductData.query.filter_by(
                farm_id=farm_id,
                production_date=production_date
            ).first()

            if existing_product:
                print(f"이미 존재하는 생산데이터 -> farm_id:{farm_id}, production_date:{production_date}")
                skipped_count += 1
                continue

            product = ProductData(
                farm_id=farm_id,
                production_date=production_date,
                total_quantity=float(row.total_quantity),
                total_sales=None if pd.isna(row.total_sales) else int(round(row.total_sales))
            )

            db.session.add(product)
            inserted_count += 1

        db.session.commit()
        print(f"총 {inserted_count}건 insert 완료")
        print(f"총 {skipped_count}건 skip")

    except Exception as e:
        db.session.rollback()
        print(f"에러 발생, rollback 수행: {e}")
        raise


if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        load_strawberry_product_csv(PRODUCT_DATA_PATH)