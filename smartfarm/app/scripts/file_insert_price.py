from app import create_app, db
from app.models import Price
import pandas as pd


CSV_PATH = "D:/취업반/1차프로젝트/raw_datas/가격정보_딸기_통합.csv"


def load_price_csv(csv_path):
    """
    가격 CSV를 읽어서 price 테이블에 저장한다.

    CSV 예시 컬럼:
    - 일자
    - 등급
    - 평균가
    """

    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 필요한 컬럼 확인
    required_cols = ['일자', '등급', '평균가']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    # 사용할 컬럼만 추출
    df = df[required_cols].copy()

    # 값 정리
    df['일자'] = df['일자'].astype(str).str.strip()
    df['등급'] = df['등급'].astype(str).str.strip()
    df['평균가'] = df['평균가'].astype(str).str.replace(',', '', regex=False).str.strip()

    # 날짜 변환
    # 예: 2021.01.04
    df['일자'] = pd.to_datetime(df['일자'], format='%Y.%m.%d', errors='coerce')

    # 평균가 숫자 변환
    df['평균가'] = pd.to_numeric(df['평균가'], errors='coerce')

    # 결측 제거
    df = df.dropna(subset=['일자', '등급', '평균가'])

    # int 변환
    df['평균가'] = df['평균가'].astype(int)

    return df


def insert_price_data(df):
    """
    DataFrame 데이터를 price 테이블에 insert
    """
    insert_count = 0

    for _, row in df.iterrows():
        price = Price(
            trade_date=row['일자'].to_pydatetime(),
            grade=row['등급'],
            avg_price=int(row['평균가'])
        )
        db.session.add(price)
        insert_count += 1

    db.session.commit()
    print(f"총 {insert_count}건 insert 완료")


if __name__ == "__main__":
    app = create_app()

    with app.app_context():
        try:
            df = load_price_csv(CSV_PATH)
            insert_price_data(df)
        except Exception as e:
            db.session.rollback()
            print("에러 발생, rollback 수행:", e)