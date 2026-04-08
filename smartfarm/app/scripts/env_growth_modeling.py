import os, re

from app import create_app, db
from app.models import FarmInfo, Environment, GrowthData, Analysis
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)


# =========================
# DB 저장
# =========================
def save_analysis_result_to_db(
    fitted_models,
    result_df,
    feature_cols,
    target_col,
    analyzed_type='기본'
):
    """
    fitted_models: 학습 완료 모델 dict
    result_df: 모델 성능 비교 DataFrame (model, MAE, R2)
    feature_cols: 실제 학습에 사용한 입력 컬럼 리스트
    target_col: 실제 타겟 컬럼명
    analyzed_type: 분석 유형 (기본값='기본')
    """

    if not analyzed_type or not str(analyzed_type).strip():
        analyzed_type = '기본'
    else:
        analyzed_type = str(analyzed_type).strip()

    analyzed_name = f"env_growth_analysis_{analyzed_type}"
    analyze_input = ",".join(feature_cols)
    analyze_target = str(target_col)

    rf_model = fitted_models['RandomForest']

    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_cols
    )

    def get_imp(col_name):
        return float(rf_importance.get(col_name, 0.0))

    score_map = dict(zip(result_df['model'], result_df['R2']))

    row = Analysis(
        analyzed_type=analyzed_type,
        analyzed_name=analyzed_name,
        analyze_input=analyze_input,
        analyze_target=analyze_target,

        temp_mean_importance=get_imp('x_temp_mean'),
        hum_mean_importance=get_imp('x_hum_mean'),
        co2_mean_importance=get_imp('x_co2_mean'),
        rad_per_day_importance=get_imp('x_rad_per_day'),
        high_temp_hours_importance=get_imp('x_high_temp_hours'),
        low_temp_hours_importance=get_imp('x_low_temp_hours'),
        vpd_importance=get_imp('x_vpd'),
        gdd_cum_importance=get_imp('x_gdd_cum'),
        prev_plant_height_importance=get_imp('prev_초장'),

        random_forest_score=float(score_map.get('RandomForest', 0.0)),
        extra_trees_score=float(score_map.get('ExtraTrees', 0.0)),
        gradient_boosting_score=float(score_map.get('GradientBoosting', 0.0)),
        hist_gradient_boosting_score=float(score_map.get('HistGradientBoosting', 0.0)),

        analyzed_date=datetime.now()
    )

    db.session.add(row)
    db.session.commit()

    print("analysis 테이블 저장 완료")
    print(f"analyzed_type   : {analyzed_type}")
    print(f"analyzed_name   : {analyzed_name}")
    print(f"analyze_input   : {analyze_input}")
    print(f"analyze_target  : {analyze_target}")


# =========================
# 0. 기본 설정
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

MIN_HOURLY_COUNT_PER_DAY = 18
MIN_VALID_DAY_RATIO_IN_INTERVAL = 0.8
MIN_INTERVAL_DAYS = 2
MAX_INTERVAL_DAYS = 10

USE_MAIN_STEM_ONLY = True

RAD_DAY_MIN = 100
RAD_DAY_MAX = 5000

LOW_TEMP_THRESHOLD = 5
HIGH_TEMP_THRESHOLD = 28



# =========================
# 1. DB에서 환경 데이터 로드
# =========================
def load_env_from_db():
    """
    environment + farm_info 조인해서 DataFrame으로 가져온다.
    """
    query = (
        db.session.query(
            FarmInfo.survey_year.label('season_year'),
            FarmInfo.district.label('도'),
            FarmInfo.city.label('시군'),
            FarmInfo.item.label('품목'),
            FarmInfo.farm_num.label('농가명'),
            Environment.measure_time.label('측정시간'),
            Environment.inside_temp.label('온도_내부'),
            Environment.relative_humidity.label('상대습도_내부'),
            Environment.carbon_dioxide.label('잔존CO2'),
            Environment.solar_radiation_sum.label('누적일사량_외부')
        )
        .join(Environment, FarmInfo.farm_id == Environment.farm_id)
    )

    result = db.session.execute(query.statement)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        raise ValueError("environment 테이블에 데이터가 없습니다.")

    df['측정시간'] = pd.to_datetime(df['측정시간'], errors='coerce')
    df = df.dropna(subset=['측정시간'])

    num_cols = ['온도_내부', '상대습도_내부', '잔존CO2', '누적일사량_외부']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# 업로드 파일 검사
def read_uploaded_csv(file_storage):
    """
    Flask request.files 로 받은 FileStorage 객체를
    DataFrame으로 읽어온다.
    """
    if file_storage is None:
        raise ValueError("파일이 전달되지 않았습니다.")

    if not file_storage.filename:
        raise ValueError("선택된 파일이 없습니다.")

    if not file_storage.filename.lower().endswith('.csv'):
        raise ValueError("CSV 파일만 업로드 가능합니다.")

    # 한글 CSV 인코딩 대응
    encodings = ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']

    for enc in encodings:
        try:
            file_storage.stream.seek(0)   # 다시 처음부터 읽기
            df = pd.read_csv(file_storage, encoding=enc)
            df.columns = df.columns.str.strip()   # 컬럼명 공백 제거
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"CSV 읽기 중 오류가 발생했습니다: {e}")

    raise ValueError("CSV 인코딩을 읽을 수 없습니다. utf-8 또는 cp949 형식인지 확인하세요.")

# 환경 파일을 df로 변환
def load_env_from_upload(env_file):
    """
    업로드된 환경 CSV 파일을 DataFrame으로 읽어서
    기존 DB 로드 결과와 비슷한 형태로 맞춘다.
    """
    # 업로드 파일 가져와서 df에 넣기
    df = read_uploaded_csv(env_file)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 컬럼명 리네임
    df = df.rename(columns={
        '연도': 'season_year',
        '지역(도)': '도',
        '잔존이산화탄소(CO2)': '잔존CO2',
    })

    # 파일명에서 연도 추출
    filename = os.path.basename(env_file.filename)
    survey_year = int(re.search(r'\d{4}', filename).group())  # 정규식; 숫자 4개 찾기
    df['season_year'] = survey_year

    required_cols = [
        'season_year', '도', '시군', '품목', '농가명',
        '측정시간', '온도_내부', '상대습도_내부', '잔존CO2', '누적일사량_외부'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"환경 데이터에 필요한 컬럼이 없습니다: {missing_cols}")

    df = df[required_cols].copy()

    df['측정시간'] = pd.to_datetime(df['측정시간'], errors='coerce')
    df = df.dropna(subset=['측정시간'])

    df['season_year'] = pd.to_numeric(df['season_year'], errors='coerce')

    num_cols = ['온도_내부', '상대습도_내부', '잔존CO2', '누적일사량_외부']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df



# =========================
# 2. 환경 일별 집계
# =========================
def build_env_daily_from_db(upload=False, env_file=None):
    if not upload:
        df = load_env_from_db()
    else:
        df = load_env_from_upload(env_file)

    df['date'] = df['측정시간'].dt.floor('D')
    df['hour'] = df['측정시간'].dt.hour

    key_cols = ['season_year', '도', '시군', '품목', '농가명', 'date']

    day_count = (
        df.groupby(key_cols)
          .size()
          .reset_index(name='n_records')
    )

    day_mean = (
        df.groupby(key_cols)[['온도_내부', '상대습도_내부', '잔존CO2']]
          .mean()
          .reset_index()
          .rename(columns={
              '온도_내부': 'temp_mean_day',
              '상대습도_내부': 'hum_mean_day',
              '잔존CO2': 'co2_mean_day'
          })
    )

    df['low_temp_stress'] = (df['온도_내부'] < LOW_TEMP_THRESHOLD).astype(int)
    df['high_temp_stress'] = (df['온도_내부'] > HIGH_TEMP_THRESHOLD).astype(int)

    day_stress = (
        df.groupby(key_cols)[['low_temp_stress', 'high_temp_stress']]
          .sum()
          .reset_index()
          .rename(columns={
              'low_temp_stress': 'low_temp_hours_day',
              'high_temp_stress': 'high_temp_hours_day'
          })
    )

    last_rad = (
        df.sort_values('측정시간')
          .groupby(key_cols, as_index=False)
          .tail(1)[key_cols + ['측정시간', '누적일사량_외부']]
          .rename(columns={
              '누적일사량_외부': 'rad_cum_day',
              '측정시간': 'last_timestamp'
          })
    )

    env_daily = day_count.merge(day_mean, on=key_cols, how='left')
    env_daily = env_daily.merge(last_rad, on=key_cols, how='left')
    env_daily = env_daily.merge(day_stress, on=key_cols, how='left')

    env_input_cols = ['temp_mean_day', 'hum_mean_day', 'co2_mean_day', 'rad_cum_day']
    env_daily['has_null_input'] = env_daily[env_input_cols].isnull().any(axis=1)

    env_daily['is_rad_outlier'] = (
        (env_daily['rad_cum_day'] <= RAD_DAY_MIN) |
        (env_daily['rad_cum_day'] >= RAD_DAY_MAX)
    )

    env_daily['is_valid_day'] = (
        (env_daily['n_records'] >= MIN_HOURLY_COUNT_PER_DAY) &
        (~env_daily['has_null_input']) &
        (~env_daily['is_rad_outlier'])
    )

    # VPD
    svp = 0.6108 * np.exp(
        (17.27 * env_daily['temp_mean_day']) /
        (env_daily['temp_mean_day'] + 237.3)
    )
    avp = svp * (env_daily['hum_mean_day'] / 100)
    env_daily['vpd_day'] = svp - avp

    # GDD
    BASE_TEMP = 5
    env_daily['gdd_day'] = np.maximum(env_daily['temp_mean_day'] - BASE_TEMP, 0)

    env_daily = env_daily.sort_values(['season_year', '농가명', 'date']).copy()

    env_daily['gdd_cum'] = (
        env_daily.groupby(['season_year', '농가명'])['gdd_day']
                 .cumsum()
    )

    dedup_cols = ['season_year', '도', '시군', '품목', '농가명', 'date']
    env_daily = env_daily.sort_values(dedup_cols + ['last_timestamp'])
    env_daily = env_daily.drop_duplicates(subset=dedup_cols, keep='last')

    return env_daily


# =========================
# 3. DB에서 생육 데이터 로드
# =========================
def load_growth_from_db():
    """
    growth_data + farm_info 조인해서 DataFrame으로 가져온다.
    """
    query = (
        db.session.query(
            FarmInfo.survey_year.label('season_year'),
            FarmInfo.district.label('도'),
            FarmInfo.city.label('시군'),
            FarmInfo.item.label('품목'),
            FarmInfo.farm_num.label('농가명'),
            GrowthData.survey_date.label('조사일자'),
            GrowthData.plant_num.label('개체번호'),
            GrowthData.axillary_branch.label('액아구분'),
            GrowthData.plant_height.label('초장'),
            GrowthData.leaf_length.label('엽장'),
            GrowthData.leaf_width.label('엽폭'),
            GrowthData.petiole_length.label('엽병장')
        )
        .join(GrowthData, FarmInfo.farm_id == GrowthData.farm_id)
    )

    result = db.session.execute(query.statement)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        raise ValueError("growth_data 테이블에 데이터가 없습니다.")

    df['조사일자'] = pd.to_datetime(df['조사일자'], errors='coerce')
    df = df.dropna(subset=['조사일자'])

    for col in ['개체번호', '초장', '엽장', '엽폭', '엽병장']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if USE_MAIN_STEM_ONLY:
        df = df[df['액아구분'].astype(str).str.strip() == '본주'].copy()

    return df


# 생육파일을 df로 변환
def load_growth_from_upload(growth_file):
    """
    업로드된 생육 CSV 파일을 DataFrame으로 읽어서
    기존 DB 로드 결과와 비슷한 형태로 맞춘다.
    """
    # 업로드 파일을 df로 변환
    df = read_uploaded_csv(growth_file)

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 컬럼명 변경
    df = df.rename(columns={
        '지역(도)': '도'
    })

    # NaN -> None
    df = df.where(pd.notnull(df), None)

    # 파일명에서 연도 추출
    filename = os.path.basename(growth_file.filename)
    match = re.search(r'\d{4}', filename)
    if not match:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {filename}")

    survey_year = int(match.group())
    df['season_year'] = survey_year

    required_cols = [
        'season_year', '도', '시군', '품목', '농가명',
        '조사일자', '개체번호', '액아구분', '초장', '엽장', '엽폭', '엽병장'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"생육 데이터에 필요한 컬럼이 없습니다: {missing_cols}")

    df = df[required_cols].copy()

    df['조사일자'] = pd.to_datetime(df['조사일자'], errors='coerce')
    df = df.dropna(subset=['조사일자'])

    df['season_year'] = pd.to_numeric(df['season_year'], errors='coerce')

    for col in ['개체번호', '초장', '엽장', '엽폭', '엽병장']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if USE_MAIN_STEM_ONLY:
        df = df[df['액아구분'].astype(str).str.strip() == '본주'].copy()

    return df

# =========================
# 4. 생육 정리
# =========================
def load_growth_clean(upload=False, growth_file=None):
    if not upload:
        df = load_growth_from_db()
    else:
        df = load_growth_from_upload(growth_file)

    key_cols = ['season_year', '도', '시군', '품목', '농가명', '조사일자', '개체번호', '액아구분']

    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) > 0 else np.nan

    grow_clean = (
        df.groupby(key_cols, as_index=False)
          .agg({
              '초장': first_valid,
              '엽장': first_valid,
              '엽폭': first_valid,
              '엽병장': first_valid
          })
    )

    grow_clean = grow_clean.dropna(subset=['초장']).copy()

    return grow_clean


# =========================
# 5. 생육 변화량 생성
# =========================
def make_growth_intervals(grow_all):
    sort_cols = ['season_year', '도', '시군', '품목', '농가명', '개체번호', '조사일자']
    grow_all = grow_all.sort_values(sort_cols).copy()

    group_cols = ['season_year', '도', '시군', '품목', '농가명', '개체번호']

    grow_all['prev_date'] = grow_all.groupby(group_cols)['조사일자'].shift(1)
    grow_all['prev_초장'] = grow_all.groupby(group_cols)['초장'].shift(1)

    grow_all['interval_days'] = (grow_all['조사일자'] - grow_all['prev_date']).dt.days

    grow_all['target_dheight_per_day'] = (
        (grow_all['초장'] - grow_all['prev_초장']) / grow_all['interval_days']
    )

    interval_df = grow_all.dropna(subset=['prev_date']).copy()

    interval_df = interval_df[
        (interval_df['interval_days'] >= MIN_INTERVAL_DAYS) &
        (interval_df['interval_days'] <= MAX_INTERVAL_DAYS)
    ].copy()

    interval_df = interval_df.dropna(subset=[
        'prev_초장',
        '초장',
        'target_dheight_per_day'
    ]).copy()

    return interval_df


# =========================
# 6. 조사구간별 환경 집계
# =========================
def aggregate_env_for_intervals(interval_df, env_daily):
    merged_rows = []

    env_group_cols = ['season_year', '도', '시군', '품목', '농가명']

    env_dict = {
        key: sub.sort_values('date').copy()
        for key, sub in env_daily.groupby(env_group_cols)
    }

    for _, row in interval_df.iterrows():
        key = (
            row['season_year'],
            row['도'],
            row['시군'],
            row['품목'],
            row['농가명']
        )

        if key not in env_dict:
            continue

        env_sub = env_dict[key]

        start_date = row['prev_date']
        end_date = row['조사일자']

        mask = (env_sub['date'] > start_date) & (env_sub['date'] <= end_date)
        interval_env = env_sub.loc[mask].copy()

        if interval_env.empty:
            continue

        total_days = (end_date - start_date).days
        if total_days <= 0:
            continue

        valid_env = interval_env[interval_env['is_valid_day']].copy()
        valid_days = valid_env['date'].nunique()

        valid_ratio = valid_days / total_days
        if valid_ratio < MIN_VALID_DAY_RATIO_IN_INTERVAL:
            continue

        temp_mean_interval = valid_env['temp_mean_day'].mean()
        hum_mean_interval = valid_env['hum_mean_day'].mean()
        co2_mean_interval = valid_env['co2_mean_day'].mean()

        rad_per_day_interval = valid_env['rad_cum_day'].sum() / total_days
        vpd_interval = valid_env['vpd_day'].mean()

        if len(valid_env) >= 2:
            gdd_interval = valid_env['gdd_cum'].iloc[-1] - valid_env['gdd_cum'].iloc[0]
        else:
            gdd_interval = valid_env['gdd_day'].sum()

        low_temp_hours_interval = valid_env['low_temp_hours_day'].sum()
        high_temp_hours_interval = valid_env['high_temp_hours_day'].sum()

        out = row.to_dict()
        out['x_temp_mean'] = temp_mean_interval
        out['x_hum_mean'] = hum_mean_interval
        out['x_co2_mean'] = co2_mean_interval
        out['x_rad_per_day'] = rad_per_day_interval
        out['x_vpd'] = vpd_interval
        out['x_gdd_cum'] = gdd_interval
        out['n_valid_env_days'] = valid_days
        out['env_valid_ratio'] = valid_ratio
        out['x_low_temp_hours'] = low_temp_hours_interval
        out['x_high_temp_hours'] = high_temp_hours_interval

        merged_rows.append(out)

    return pd.DataFrame(merged_rows)


# =========================
# 7. 최종 데이터셋 생성
# =========================
def build_dataset_from_db(upload=False, env_file=None, growth_file=None):
    grow_all = load_growth_clean(upload, growth_file)
    interval_df = make_growth_intervals(grow_all)
    env_daily = build_env_daily_from_db(upload, env_file)

    dataset = aggregate_env_for_intervals(interval_df, env_daily)

    # analysis 테이블에 저장하는 9개 변수만 사용
    feature_cols = [
        'x_temp_mean',
        'x_hum_mean',
        'x_co2_mean',
        'x_rad_per_day',
        'x_vpd',
        'x_gdd_cum',
        'prev_초장',
        'x_low_temp_hours',
        'x_high_temp_hours'
    ]

    target_col = 'target_dheight_per_day'

    dataset = dataset.dropna(subset=feature_cols + [target_col]).copy()

    # 초장 변화율 이상치 제거
    dataset = dataset[dataset['target_dheight_per_day'] < 1].copy()

    return dataset, feature_cols, target_col


# =========================
# 8. 여러 모델 학습/평가
# =========================
def train_and_evaluate_models(dataset, feature_cols, target_col):
    dataset = dataset.copy()

    dataset['group_id'] = (
        dataset['season_year'].astype(str) + "_" +
        dataset['농가명'].astype(str) + "_" +
        dataset['개체번호'].astype(str)
    )

    X = dataset[feature_cols]
    y = dataset[target_col]
    groups = dataset['group_id']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=15,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=6,
            max_iter=300,
            random_state=42
        )
    }

    results = []
    fitted_models = {}
    pred_table = y_test.to_frame(name='actual').copy()

    print("\n[전체 샘플 수]")
    print(len(dataset))

    print("\n[Train / Test]")
    print(len(X_train), len(X_test))

    print("\n[모델별 성능]")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results.append({
            'model': name,
            'MAE': mae,
            'R2': r2
        })

        pred_table[f'pred_{name}'] = pred
        fitted_models[name] = model

        print(f"{name:>20} | MAE={mae:.4f} | R2={r2:.4f}")

    result_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)

    print("\n[모델 성능 비교표]")
    print(result_df)

    print("\n[예측 샘플]")
    print(pred_table.head(10))

    return fitted_models, result_df, X_train, X_test, y_train, y_test, pred_table


# =========================
# 9. 중요도 출력
# =========================
def print_feature_importance(fitted_models, feature_cols):
    tree_model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting']

    for name in tree_model_names:
        if name not in fitted_models:
            continue

        model = fitted_models[name]

        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)

            print(f"\n[Feature importance - {name}]")
            print(importance)
            print(f"[합계 - {name}] {importance.sum():.6f}")

    if 'HistGradientBoosting' in fitted_models:
        print("\n[Feature importance - HistGradientBoosting]")
        print("HistGradientBoostingRegressor는 기본 feature_importances_를 직접 제공하지 않습니다.")


def run_analysis(env_file=None, growth_file=None):
    dataset, feature_cols, target_col = build_dataset_from_db(True, env_file, growth_file)

    print("\n[최종 데이터셋 컬럼]")
    print(dataset.columns.tolist())

    print("\n[최종 데이터셋 앞부분]")
    print(dataset.head())

    print("\n[입력 변수 기초통계]")
    print(dataset[feature_cols].describe())

    print("\n[타겟 기초통계]")
    print(dataset[[target_col]].describe())

    fitted_models, result_df, X_train, X_test, y_train, y_test, pred_table = train_and_evaluate_models(
        dataset, feature_cols, target_col
    )

    print_feature_importance(fitted_models, feature_cols)

    return fitted_models, result_df, feature_cols, target_col


# =========================
# 10. 실행
# =========================
if __name__ == "__main__":
    app = create_app()

    with app.app_context():
        dataset, feature_cols, target_col = build_dataset_from_db()

        print("\n[최종 데이터셋 컬럼]")
        print(dataset.columns.tolist())

        print("\n[최종 데이터셋 앞부분]")
        print(dataset.head())

        print("\n[입력 변수 기초통계]")
        print(dataset[feature_cols].describe())

        print("\n[타겟 기초통계]")
        print(dataset[[target_col]].describe())

        fitted_models, result_df, X_train, X_test, y_train, y_test, pred_table = train_and_evaluate_models(
            dataset, feature_cols, target_col
        )

        print_feature_importance(fitted_models, feature_cols)

        save_analysis_result_to_db(
            fitted_models=fitted_models,
            result_df=result_df,
            feature_cols=feature_cols,
            target_col=target_col,
            analyzed_type='기본'
        )