from app import create_app, db
from app.models import FarmInfo, Environment, GrowthData

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 0. 기본 설정
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

MIN_HOURLY_COUNT_PER_DAY = 18           # 해당날짜에 시간 데이터가 18개 이상이어야만 사용
MIN_VALID_DAY_RATIO_IN_INTERVAL = 0.8   # 조사 기간에 데이터가 80%이상 존재해야 사용
MIN_INTERVAL_DAYS = 2                   # 조사 기간이 2일 이하면 스킵
MAX_INTERVAL_DAYS = 10                  # 조사 기간이 10일 이상이면 스킵

USE_MAIN_STEM_ONLY = True               # 액아 구분에서 본주만 사용

RAD_DAY_MIN = 100                       # 하루의 누적 일사량이 100 미만이면 이상치로 보고 스킵
RAD_DAY_MAX = 5000                      # 하루의 누적 일사량이 5000 초과면 이상치로 보고 스킵

LOW_TEMP_THRESHOLD = 5                  # 저온 스트레스 기준치
HIGH_TEMP_THRESHOLD = 28                # 고온 스트레스 기준치


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


# =========================
# 2. 환경 일별 집계
# =========================
def build_env_daily_from_db():
    df = load_env_from_db()

    df['date'] = df['측정시간'].dt.floor('D')
    df['hour'] = df['측정시간'].dt.hour

    key_cols = ['season_year', '도', '시군', '품목', '농가명', 'date']

    # 하루 기록 수
    day_count = (
        df.groupby(key_cols)
          .size()
          .reset_index(name='n_records')
    )

    # 일평균
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

    # 스트레스 시간
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

    # 날짜별 마지막 누적일사량
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
    svp = 0.6108 * np.exp((17.27 * env_daily['temp_mean_day']) /
                          (env_daily['temp_mean_day'] + 237.3))
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

    env_daily['rad_3day'] = (
        env_daily.groupby(['season_year', '농가명'])['rad_cum_day']
                 .rolling(3)
                 .mean()
                 .reset_index(level=[0, 1], drop=True)
    )

    # 같은 날짜 중복 방지
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
            GrowthData.crown_diameter.label('관부직경'),
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

    for col in ['개체번호', '초장', '관부직경', '엽장', '엽폭', '엽병장']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if USE_MAIN_STEM_ONLY:
        df = df[df['액아구분'].astype(str).str.strip() == '본주'].copy()

    return df


# =========================
# 4. 생육 정리
# =========================
def load_growth_clean():
    df = load_growth_from_db()

    key_cols = ['season_year', '도', '시군', '품목', '농가명', '조사일자', '개체번호', '액아구분']

    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) > 0 else np.nan

    grow_clean = (
        df.groupby(key_cols, as_index=False)
          .agg({
              '초장': first_valid,
              '관부직경': first_valid,
              '엽장': first_valid,
              '엽폭': first_valid,
              '엽병장': first_valid
          })
    )

    grow_clean['has_null_target_base'] = grow_clean[['초장', '관부직경']].isnull().any(axis=1)
    grow_clean = grow_clean[~grow_clean['has_null_target_base']].copy()
    grow_clean = grow_clean.drop(columns=['has_null_target_base'])

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
    grow_all['prev_관부직경'] = grow_all.groupby(group_cols)['관부직경'].shift(1)

    grow_all['interval_days'] = (grow_all['조사일자'] - grow_all['prev_date']).dt.days

    grow_all['target_dheight_per_day'] = (
        (grow_all['초장'] - grow_all['prev_초장']) / grow_all['interval_days']
    )
    grow_all['target_dcrown_per_day'] = (
        (grow_all['관부직경'] - grow_all['prev_관부직경']) / grow_all['interval_days']
    )

    grow_all['prev_엽장'] = grow_all.groupby(group_cols)['엽장'].shift(1)
    grow_all['prev_엽폭'] = grow_all.groupby(group_cols)['엽폭'].shift(1)
    grow_all['prev_엽병장'] = grow_all.groupby(group_cols)['엽병장'].shift(1)

    interval_df = grow_all.dropna(subset=['prev_date']).copy()

    interval_df = interval_df[
        (interval_df['interval_days'] >= MIN_INTERVAL_DAYS) &
        (interval_df['interval_days'] <= MAX_INTERVAL_DAYS)
    ].copy()

    interval_df = interval_df.dropna(subset=[
        'prev_초장', 'prev_관부직경',
        '초장', '관부직경',
        'target_dheight_per_day',
        'target_dcrown_per_day'
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

        # 평균계산은 valid_env 기준으로 유지
        temp_mean_interval = valid_env['temp_mean_day'].mean()
        hum_mean_interval = valid_env['hum_mean_day'].mean()
        co2_mean_interval = valid_env['co2_mean_day'].mean()

        rad_per_day_interval = valid_env['rad_cum_day'].sum() / total_days
        vpd_interval = valid_env['vpd_day'].mean()
        rad3_interval = valid_env['rad_3day'].mean()

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
        out['x_rad_3day'] = rad3_interval
        out['x_gdd_cum'] = gdd_interval
        out['n_valid_env_days'] = valid_days
        out['env_valid_ratio'] = valid_ratio
        out['x_low_temp_hours'] = low_temp_hours_interval
        out['x_high_temp_hours'] = high_temp_hours_interval
        out['x_low_temp_hours_per_day'] = low_temp_hours_interval / total_days
        out['x_high_temp_hours_per_day'] = high_temp_hours_interval / total_days

        merged_rows.append(out)

    return pd.DataFrame(merged_rows)


# =========================
# 7. 최종 데이터셋 생성
# =========================
def build_dataset_from_db():
    grow_all = load_growth_clean()
    interval_df = make_growth_intervals(grow_all)
    env_daily = build_env_daily_from_db()

    dataset = aggregate_env_for_intervals(interval_df, env_daily)

    feature_cols = [
        'x_temp_mean',
        'x_hum_mean',
        'x_co2_mean',
        'x_rad_per_day',
        'x_vpd',
        'x_rad_3day',
        'x_gdd_cum',
        'prev_초장',
        'prev_관부직경',
        'prev_엽장',
        'prev_엽폭',
        'prev_엽병장',
        'x_low_temp_hours',
        'x_high_temp_hours'
    ]

    target_cols = [
        'target_dheight_per_day',
        'target_dcrown_per_day'
    ]

    dataset = dataset.dropna(subset=feature_cols + target_cols).copy()

    dataset = dataset[dataset['target_dheight_per_day'] < 1].copy()
    dataset = dataset[dataset['target_dcrown_per_day'] < 0.5].copy()

    return dataset, feature_cols, target_cols


# =========================
# 8. 모델 학습/평가
# =========================
def train_and_evaluate(dataset, feature_cols, target_cols):
    dataset = dataset.copy()

    dataset['group_id'] = (
        dataset['season_year'].astype(str) + "_" +
        dataset['농가명'].astype(str) + "_" +
        dataset['개체번호'].astype(str)
    )

    X = dataset[feature_cols]
    y = dataset[target_cols]
    groups = dataset['group_id']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=15,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    pred_df = pd.DataFrame(pred, columns=target_cols, index=y_test.index)

    print("\n[전체 샘플 수]")
    print(len(dataset))

    print("\n[Train / Test]")
    print(len(X_train), len(X_test))

    print("\n[타겟별 성능]")
    for col in target_cols:
        mae = mean_absolute_error(y_test[col], pred_df[col])
        r2 = r2_score(y_test[col], pred_df[col])
        print(f"{col:>25} | MAE={mae:.4f} | R2={r2:.4f}")

    print("\n[예측 샘플]")
    sample = y_test.copy()
    for col in target_cols:
        sample[f"pred_{col}"] = pred_df[col]
    print(sample.head(10))

    return model, X_train, X_test, y_train, y_test, pred_df


# =========================
# 9. 실행
# =========================
if __name__ == "__main__":
    app = create_app()

    with app.app_context():
        df = load_env_from_db()

        dataset, feature_cols, target_cols = build_dataset_from_db()

        print("\n[최종 데이터셋 컬럼]")
        print(dataset.columns.tolist())

        print("\n[최종 데이터셋 앞부분]")
        print(dataset.head())

        print("\n[입력 변수 기초통계]")
        print(dataset[feature_cols].describe())

        print("\n[타겟 기초통계]")
        print(dataset[target_cols].describe())

        model, X_train, X_test, y_train, y_test, pred_df = train_and_evaluate(
            dataset, feature_cols, target_cols
        )

        for i, target in enumerate(target_cols):
            rf = model.estimators_[i]

            importance = pd.Series(
                rf.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)

            print(f"\n[Feature importance - {target}]")
            print(importance)