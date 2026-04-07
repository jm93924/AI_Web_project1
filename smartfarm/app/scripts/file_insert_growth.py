from app import create_app
import pandas as pd
import os
import re

from app import db
from app.models import FarmInfo, GrowthData

GROWTH_DATA_PATH = "D:/취업반/1차프로젝트/raw_datas/생육_딸기_2019.csv"


def load_strawberry_growth_csv(csv_path):
    """
    생육 csv를 읽어서 growth_data 테이블에 저장한다.

    생육 CSV에는 farm_id가 없으므로,
    파일명에서 추출한 연도 + 품목 + 농가명 으로 farm_info를 조회해서
    해당 farm_id를 찾아 growth_data에 넣는다.
    """

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 컬럼명 리네임
    df = df.rename(columns={
        '품목': 'item',
        '농가명': 'farm_num',
        '조사일자': 'survey_date',
        '개체번호': 'plant_num',
        '액아구분': 'axillary_branch',
        '초장': 'plant_height',
        '엽수': 'leaf_count',
        '엽장': 'leaf_length',
        '엽폭': 'leaf_width',
        '엽병장': 'petiole_length',
        '관부직경': 'crown_diameter',
        '화방번호': 'flower_cluster_no',
        '화방별착과수': 'fruits_per_cluster'
    })

    # NaN -> None
    df = df.where(pd.notnull(df), None)

    # 조사일자 datetime 변환
    df['survey_date'] = pd.to_datetime(df['survey_date'], errors='coerce')

    # 파일명에서 연도 추출
    filename = os.path.basename(csv_path)
    match = re.search(r'\d{4}', filename)
    if not match:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {filename}")

    survey_year = int(match.group())
    df['survey_year'] = survey_year

    inserted_count = 0
    skipped_count = 0

    try:
        # farm_info 미리 읽어서 매핑 딕셔너리 생성
        farm_rows = FarmInfo.query.all()
        farm_map = {}

        for farm in farm_rows:
            key = (
                farm.survey_year,
                farm.item.strip(),
                farm.farm_num
            )
            farm_map[key] = farm.farm_id

        for row in df.itertuples(index=False):
            # 필수값 체크
            if pd.isna(row.survey_date):
                print("조사일자가 없어 skip")
                skipped_count += 1
                continue

            if pd.isna(row.plant_num):
                print("개체번호가 없어 skip")
                skipped_count += 1
                continue

            if pd.isna(row.plant_height):
                print("초장이 없어 skip")
                skipped_count += 1
                continue

            if pd.isna(row.crown_diameter):
                print("관부직경이 없어 skip")
                skipped_count += 1
                continue

            if row.axillary_branch != "본주":
                print("본주 이외에는 skip")
                skipped_count += 1
                continue

            if pd.isna(row.flower_cluster_no) or not str(row.flower_cluster_no).isdigit():
                print(f"화방번호 형식 오류 -> {row.flower_cluster_no}")
                skipped_count += 1
                continue

            # 여기서 row.xxx은 DB에 넣을때 이미 검사했다.
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

            survey_date = row.survey_date.to_pydatetime()
            plant_num = int(row.plant_num)

            # 중복 체크 (PK: farm_id, survey_date, plant_num)
            existing_growth = GrowthData.query.filter_by(
                farm_id=farm_id,
                survey_date=survey_date,
                plant_num=plant_num
            ).first()

            if existing_growth:
                print(f"이미 존재하는 생육데이터 -> farm_id:{farm_id}, survey_date:{survey_date}, plant_num:{plant_num}")
                skipped_count += 1
                continue

            growth = GrowthData(
                farm_id=farm_id,
                survey_date=survey_date,
                plant_num=plant_num,
                axillary_branch=None if pd.isna(row.axillary_branch) else str(row.axillary_branch).strip(),
                plant_height=None if pd.isna(row.plant_height) else float(row.plant_height),
                leaf_count=None if pd.isna(row.leaf_count) else int(row.leaf_count),
                leaf_length=None if pd.isna(row.leaf_length) else float(row.leaf_length),
                leaf_width=None if pd.isna(row.leaf_width) else float(row.leaf_width),
                petiole_length=None if pd.isna(row.petiole_length) else float(row.petiole_length),
                crown_diameter=None if pd.isna(row.crown_diameter) else float(row.crown_diameter),
                flower_cluster_no = int(row.flower_cluster_no),
                fruits_per_cluster=None if pd.isna(row.fruits_per_cluster) else int(row.fruits_per_cluster)
            )

            db.session.add(growth)
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
        load_strawberry_growth_csv(GROWTH_DATA_PATH)