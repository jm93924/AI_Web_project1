from app import create_app
import pandas as pd
from datetime import datetime

from app import db
from app.models import FarmInfo, CultivationInfo, CultivationVariety

CULTIVATION_DATA_PATH = "D:/취업반/1차프로젝트/raw_datas/재배정보_딸기_2018.csv"

def load_strawberry_cultivation_csv(csv_path):
    """
    재배정보 csv를 읽어서
    1) farm_info
    2) cultivation_info
    3) cultivation_variety
    테이블에 한 번에 저장한다.

    farm_id는 farm_info insert 시 시퀀스로 자동 생성되고,
    생성된 farm_id를 cultivation_info, cultivation_variety에 재사용한다.
    """

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 파일마다 컬럼명이 다를때
    df.columns = df.columns.str.replace("총면적", "전체면적")

    # 컬럼명 리네임
    df = df.rename(columns={
        '연도': 'survey_year',
        '품목': 'item',
        '농가명': 'farm_num',
        '지역(도)': 'district',
        '시군': 'city',
        '온실종류': 'house_type',
        '온실유형': 'house_form',
        '전체면적': 'total_area',
        '식부면적': 'planting_area',
        '재식밀도': 'planting_density',
        '정식일': 'planting_date',
        '품종': 'item_variety'
    })

    # NaN -> None 치환
    df = df.where(pd.notnull(df), None)

    # 진행상황 카운트
    inserted_count = 0

    try:
        for row in df.itertuples(index=False):
            # ---------------------------
            # 1. farm_info 생성
            # ---------------------------

            # 연도, 품목, 농가명 3개 컬럼 조합의 중복검사
            existing_farm = FarmInfo.query.filter_by(
                # 공백, null, nan 등등 모두 DB에는 "null"로 넣는다
                survey_year=None if pd.isna(row.survey_year) else int(row.survey_year),
                item=None if pd.isna(row.item) else str(row.item).strip(),
                farm_num=None if pd.isna(row.farm_num) else int(row.farm_num)
            ).first()

            # 재배정보 하나만 넣어서 3개의 테이블을 만들고 있으므로, 시작이 되는 부분만 중복검사.
            if existing_farm:
                print("이미 존재하는 데이터입니다.")
                continue # 이 부분이 중복이면 나머지도 중복이라고 가정.

            # 새로운 데이터면 집어넣기
            else:
                farm_info = FarmInfo(
                    survey_year=None if pd.isna(row.survey_year) else int(row.survey_year),
                    item=None if pd.isna(row.item) else str(row.item).strip(),
                    farm_num=None if pd.isna(row.farm_num) else int(row.farm_num),
                    district=None if pd.isna(row.district) else str(row.district).strip(),
                    city=None if pd.isna(row.city) else str(row.city).strip()
                )

                db.session.add(farm_info)

                # commit 전에도 시퀀스로 생성된 PK 값을 받아오기 위해 flush
                db.session.flush()

                generated_farm_id = farm_info.farm_id

            # ---------------------------
            # 2. cultivation_info 생성
            # ---------------------------
            planting_date = None
            if row.planting_date is not None:
                # csv 값 예: 2022-07-28
                planting_date = pd.to_datetime(row.planting_date, errors='coerce')
                if pd.notnull(planting_date):
                    planting_date = planting_date.to_pydatetime()
                else:
                    planting_date = None

            cultivation_info = CultivationInfo(
                farm_id=generated_farm_id,
                house_type=None if pd.isna(row.house_type) else str(row.house_type).strip(),
                house_form=None if pd.isna(row.house_form) else str(row.house_form).strip(),
                total_area=None if pd.isna(row.total_area) else float(row.total_area),
                planting_area=None if pd.isna(row.planting_area) else float(row.planting_area),
                planting_density = None if pd.isna(row.planting_density) else float(row.planting_density),
                planting_date=planting_date
            )

            db.session.add(cultivation_info)

            # ---------------------------
            # 3. cultivation_variety 생성
            # ---------------------------
            cultivation_variety = CultivationVariety(
                farm_id=generated_farm_id,
                item_variety=None if pd.isna(row.item_variety) else str(row.item_variety).strip()
            )

            db.session.add(cultivation_variety)

            inserted_count += 1

        # 모든 row 처리 후 한 번에 commit
        db.session.commit()
        print(f"총 {inserted_count}건 insert 완료")

    except Exception as e:
        db.session.rollback()
        print(f"에러 발생, rollback 수행: {e}")
        raise

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        load_strawberry_cultivation_csv(CULTIVATION_DATA_PATH)