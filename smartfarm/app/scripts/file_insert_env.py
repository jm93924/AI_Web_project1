from app import create_app
import pandas as pd
import os, re

from app import db
from app.models import FarmInfo, Environment

ENV_DATA_PATH = "D:/취업반/1차프로젝트/raw_datas/환경_딸기_2022.csv"


def load_strawberry_environment_csv(csv_path):
    """
    환경 csv를 읽어서 environment 테이블에 저장한다.

    환경 CSV에는 farm_id가 없으므로,
    측정시간의 연도 + 품목 + 농가명 으로 farm_info를 조회해서
    해당 farm_id를 찾아 environment에 넣는다.
    """

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 파일마다 컬럼명이 다를때
    df.columns = df.columns.str.replace("잔존이산화탄소(CO2)", "잔존CO2")

    # 컬럼명 리네임
    df = df.rename(columns={
        '도': 'district',
        '시군': 'city',
        '품목': 'item',
        '농가명': 'farm_num',
        '측정시간': 'measure_time',
        '온도_외부': 'out_temp',
        '풍향_외부': 'out_wind_direction',
        '풍속_외부': 'out_wind_speed',
        '일사량_외부': 'solar_radiation',
        '누적일사량_외부': 'solar_radiation_sum',
        '강우감지': 'rain',
        '온도_내부': 'inside_temp',
        '상대습도_내부': 'relative_humidity',
        '잔존CO2': 'carbon_dioxide',
        '토양온도': 'soil_temp'
    })

    # NaN -> None
    df = df.where(pd.notnull(df), None)

    # 측정시간 datetime 변환
    df['measure_time'] = pd.to_datetime(df['measure_time'], errors='coerce')

    # 연도 컬럼 생성 (farm_info 매칭용)
    filename = os.path.basename(csv_path)
    survey_year = int(re.search(r'\d{4}', filename).group()) # 정규식; 숫자 4개 찾기
    # split 방식
    # survey_year = int(filename.split('_')[2].split('.')[0])
    df['survey_year'] = survey_year

    inserted_count = 0
    skipped_count = 0

    try:
        # farm_info를 미리 전부 읽어서 매핑 딕셔너리 생성
        farm_rows = FarmInfo.query.all()
        farm_map = {}

        for farm in farm_rows:
            key = (
                farm.survey_year,
                farm.item.strip(), # 품목이 애초에 DB에 null로 들어갈 수 없으므로 검사는 따로 안함
                farm.farm_num
            )
            # 조사연도, 품목, 농가번호 3개를 알면 farm_id 조회가 가능하게끔
            farm_map[key] = farm.farm_id

        for row in df.itertuples(index=False):
            # measure_time이 없으면 PK 구성 불가
            if pd.isna(row.measure_time):
                print("측정시간이 없어 skip")
                skipped_count += 1
                continue

            survey_year = None if pd.isna(row.survey_year) else int(row.survey_year)
            item = None if pd.isna(row.item) else str(row.item).strip()
            farm_num = None if pd.isna(row.farm_num) else int(row.farm_num)

            # farm_info에서 farm_id 찾기
            farm_key = (survey_year, item, farm_num)
            farm_id = farm_map.get(farm_key)

            if farm_id is None:
                print(f"farm_info 매칭 실패 -> 연도:{survey_year}, 품목:{item}, 농가명:{farm_num}")
                skipped_count += 1
                continue

            measure_time = row.measure_time.to_pydatetime()

            # 중복 체크 (PK: farm_id, measure_time)
            existing_env = Environment.query.filter_by(
                farm_id=farm_id,
                measure_time=measure_time
            ).first()

            if existing_env:
                print(f"이미 존재하는 환경데이터 -> farm_id:{farm_id}, measure_time:{measure_time}")
                skipped_count += 1
                continue

            env = Environment(
                farm_id=farm_id,
                measure_time=measure_time,
                out_temp=None if pd.isna(row.out_temp) else float(row.out_temp),
                out_wind_direction=None if pd.isna(row.out_wind_direction) else int(row.out_wind_direction),
                out_wind_speed=None if pd.isna(row.out_wind_speed) else float(row.out_wind_speed),
                solar_radiation=None if pd.isna(row.solar_radiation) else int(row.solar_radiation),
                solar_radiation_sum=None if pd.isna(row.solar_radiation_sum) else int(row.solar_radiation_sum),
                rain=None if pd.isna(row.rain) else int(row.rain),
                inside_temp=None if pd.isna(row.inside_temp) else float(row.inside_temp),
                relative_humidity=None if pd.isna(row.relative_humidity) else float(row.relative_humidity),
                carbon_dioxide=None if pd.isna(row.carbon_dioxide) else int(row.carbon_dioxide),
                soil_temp=None if pd.isna(row.soil_temp) else float(row.soil_temp)
            )

            db.session.add(env)
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
        load_strawberry_environment_csv(ENV_DATA_PATH)