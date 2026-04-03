import pandas as pd
import numpy as np
import glob
import os
import re

from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 0. 기본 설정
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# 파일 패턴
ENV_PATTERN = "D:/취업반/1차프로젝트/raw_datas/환경_딸기_*.csv"
GROW_PATTERN = "D:/취업반/1차프로젝트/raw_datas/생육_딸기_*.csv"

# 환경 데이터 일단위 유효성 기준
MIN_HOURLY_COUNT_PER_DAY = 18   # 하루 24시간 중 6개 이상 빠지면 제외
MAX_MISSING_PER_DAY = 6

# 구간(생육 조사일 사이) 품질 기준
MIN_VALID_DAY_RATIO_IN_INTERVAL = 0.8   # 구간 일수 중 80% 이상 유효한 날짜만 사용
MIN_INTERVAL_DAYS = 2                   # 조사간격이 너무 짧으면 제외
MAX_INTERVAL_DAYS = 10   # 추가: 조사 간격 10일 초과 제거

# 액아는 제외하고 본주만 볼지 여부
USE_MAIN_STEM_ONLY = True

# 추가: 일별 누적일사량 이상치 기준
RAD_DAY_MIN = 100
RAD_DAY_MAX = 5000

LOW_TEMP_THRESHOLD = 5     # 5도 미만 = 저온 스트레스
HIGH_TEMP_THRESHOLD = 28   # 28도 초과 = 고온 스트레스

# =========================
# 1. 공통 유틸
# =========================
def read_csv_auto(path):
    """
    CSV 인코딩 자동 시도
    """
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr' ]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise ValueError(f"읽기 실패: {path}")

def extract_season_year_from_filename(path):
    """
    파일명에서 season_year 추출
    예: 환경_딸기_2018.csv -> 2018
    """
    fname = os.path.basename(path)
    m = re.search(r'(\d{4})', fname)
    if not m:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {fname}")
    return int(m.group(1))

def standardize_columns(df):
    df.columns = df.columns.str.strip()

    rename_map = {
        '잔존이산화탄소(CO2)': '잔존CO2',
        '잔존 이산화탄소(CO2)': '잔존CO2',
        '잔존CO₂': '잔존CO2',
        'CO2': '잔존CO2',
        '잔존 co2': '잔존CO2',
        '측정 시간': '측정시간',
        '조사 날짜': '조사일자',
    }

    df = df.rename(columns=rename_map)
    return df

# =========================
# 2. 환경 데이터 로드 및 일별 집계
# =========================
def load_env_files(env_pattern):
    env_files = sorted(glob.glob(env_pattern))
    if not env_files:
        raise FileNotFoundError(f"환경 파일 없음: {env_pattern}")

    env_list = []

    for path in env_files:
        season_year = extract_season_year_from_filename(path)
        df = read_csv_auto(path)
        df = standardize_columns(df)

        required_cols = [
            '도', '시군', '품목', '작기', '농가명', '측정시간',
            '온도_내부', '상대습도_내부', '잔존CO2', '누적일사량_외부'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} 에 필요한 컬럼이 없습니다: {missing}")

        df['season_year'] = season_year
        df['측정시간'] = pd.to_datetime(df['측정시간'], errors='coerce')
        df = df.dropna(subset=['측정시간'])

        # 날짜/시간 분리
        df['date'] = df['측정시간'].dt.floor('D')
        df['hour'] = df['측정시간'].dt.hour

        # 수치형 변환
        num_cols = ['온도_내부', '상대습도_내부', '잔존CO2', '누적일사량_외부']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 키
        key_cols = ['season_year', '도', '시군', '품목', '작기', '농가명', 'date']

        # 하루 기록 수
        day_count = (
            df.groupby(key_cols)
              .size()
              .reset_index(name='n_records')
        )

        # 평균온도 / 평균습도 / 평균CO2
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

        # 저온/고온 스트레스 여부 (시간 단위 원본 기준)
        df['low_temp_stress'] = (df['온도_내부'] < LOW_TEMP_THRESHOLD).astype(int)
        df['high_temp_stress'] = (df['온도_내부'] > HIGH_TEMP_THRESHOLD).astype(int)

        # 날짜별 스트레스 노출 시간 합계
        # 현재 데이터가 시간 단위라면 sum = 시간 수 로 해석 가능
        day_stress = (
            df.groupby(key_cols)[['low_temp_stress', 'high_temp_stress']]
            .sum()
            .reset_index()
            .rename(columns={
                'low_temp_stress': 'low_temp_hours_day',
                'high_temp_stress': 'high_temp_hours_day'
            })
        )

        # 하루 마지막 누적일사량
        # 날짜별로 측정시간 기준 마지막 행
        last_rad = (
            df.sort_values('측정시간')
              .groupby(key_cols, as_index=False)
              .tail(1)[key_cols + ['측정시간', '누적일사량_외부']]
              .rename(columns={
                  '누적일사량_외부': 'rad_cum_day',
                  '측정시간': 'last_timestamp'
              })
        )

        day_agg = day_count.merge(day_mean, on=key_cols, how='left')
        day_agg = day_agg.merge(last_rad, on=key_cols, how='left')
        day_agg = day_agg.merge(day_stress, on=key_cols, how='left')

        # 환경 4개 입력 변수 중 하나라도 결측이면 그 날짜는 사용 불가
        env_input_cols = ['temp_mean_day', 'hum_mean_day', 'co2_mean_day', 'rad_cum_day']
        day_agg['has_null_input'] = day_agg[env_input_cols].isnull().any(axis=1)

        # 추가: 일별 누적일사량 이상치 여부
        day_agg['is_rad_outlier'] = (
          (day_agg['rad_cum_day'] <= RAD_DAY_MIN) |
          (day_agg['rad_cum_day'] >= RAD_DAY_MAX)
        )

        # 유효 일자 판단:
        # 1) 하루 기록 수 >= 18
        # 2) 입력 4개 null 없음
        # 3) 누적일사량 이상치 아님
        day_agg['is_valid_day'] = (
            (day_agg['n_records'] >= MIN_HOURLY_COUNT_PER_DAY) &
            (~day_agg['has_null_input']) &
            (~day_agg['is_rad_outlier'])
        )

        #실제 사용할 날짜만 남김
        day_agg = day_agg[day_agg['is_valid_day']].copy()

        env_list.append(day_agg)

    env_daily = pd.concat(env_list, ignore_index=True)

    # 혹시 중복되면 제거
    dedup_cols = ['season_year', '도', '시군', '품목', '작기', '농가명', 'date']
    env_daily = env_daily.sort_values(dedup_cols + ['last_timestamp'])
    env_daily = env_daily.drop_duplicates(subset=dedup_cols, keep='last')

    return env_daily

# =========================
# 3. 생육 데이터 로드 및 개체별 정리
# =========================
def load_growth_files(grow_pattern):
    grow_files = sorted(glob.glob(grow_pattern))
    if not grow_files:
        raise FileNotFoundError(f"생육 파일 없음: {grow_pattern}")

    grow_list = []

    for path in grow_files:
        season_year = extract_season_year_from_filename(path)
        df = read_csv_auto(path)
        df = standardize_columns(df)

        required_cols = [
            '도', '시군', '품목', '작기', '농가명', '조사일자',
            '개체번호', '액아구분',
            '초장', '관부직경',
            '엽장', '엽폭', '엽병장'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} 에 필요한 컬럼이 없습니다: {missing}")

        df['season_year'] = season_year
        df['조사일자'] = pd.to_datetime(df['조사일자'], errors='coerce')
        df = df.dropna(subset=['조사일자'])

        # 수치형 변환
        for col in ['개체번호', '초장', '관부직경', '엽장', '엽폭', '엽병장']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 본주만 사용할 경우 필터
        if USE_MAIN_STEM_ONLY:
            df = df[df['액아구분'].astype(str).str.strip() == '본주'].copy()

        # 같은 조사일자/개체에 화방별로 여러 행이 있을 수 있으므로
        # 생육값만 뽑아 개체 단위 1행으로 정리
        key_cols = ['season_year', '도', '시군', '품목', '작기', '농가명', '조사일자', '개체번호', '액아구분']

        # 각 생육값이 들어있는 행을 우선 사용
        # 동일 key 내에서 첫 유효값 선택
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

        # 생육 3개 중 하나라도 null이면 해당 조사 record 제거
        grow_clean['has_null_target_base'] = grow_clean[['초장', '관부직경']].isnull().any(axis=1)
        grow_clean = grow_clean[~grow_clean['has_null_target_base']].copy()

        # 보조 컬럼 제거
        grow_clean = grow_clean.drop(columns=['has_null_target_base'])

        grow_list.append(grow_clean)

    grow_all = pd.concat(grow_list, ignore_index=True)

    return grow_all

# =========================
# 4. 생육 변화량(타겟) 생성
# =========================
def make_growth_intervals(grow_all):
    sort_cols = ['season_year', '도', '시군', '품목', '작기', '농가명', '개체번호', '조사일자']
    grow_all = grow_all.sort_values(sort_cols).copy()

    group_cols = ['season_year', '도', '시군', '품목', '작기', '농가명', '개체번호']

    # 이전 조사값
    grow_all['prev_date'] = grow_all.groupby(group_cols)['조사일자'].shift(1)
    grow_all['prev_초장'] = grow_all.groupby(group_cols)['초장'].shift(1)
    grow_all['prev_관부직경'] = grow_all.groupby(group_cols)['관부직경'].shift(1)

    # 조사 간격
    grow_all['interval_days'] = (grow_all['조사일자'] - grow_all['prev_date']).dt.days

    # 변화량/day
    grow_all['target_dheight_per_day'] = (grow_all['초장'] - grow_all['prev_초장']) / grow_all['interval_days']
    grow_all['target_dcrown_per_day'] = (grow_all['관부직경'] - grow_all['prev_관부직경']) / grow_all['interval_days']

    grow_all['prev_엽장'] = grow_all.groupby(group_cols)['엽장'].shift(1)
    grow_all['prev_엽폭'] = grow_all.groupby(group_cols)['엽폭'].shift(1)
    grow_all['prev_엽병장'] = grow_all.groupby(group_cols)['엽병장'].shift(1)

    # 첫 조사 제거
    interval_df = grow_all.dropna(subset=['prev_date']).copy()
    # 조사 간격 조건: 최소 2일, 최대 10일
    interval_df = interval_df[
        (interval_df['interval_days'] >= MIN_INTERVAL_DAYS) &
        (interval_df['interval_days'] <= MAX_INTERVAL_DAYS)
    ].copy()

    # 타겟 계산에 필요한 현재값/이전값 중 하나라도 null이면 제거
    interval_df = interval_df.dropna(subset=[
        'prev_초장', 'prev_관부직경',
        '초장', '관부직경',
        'target_dheight_per_day',
        'target_dcrown_per_day'
    ]).copy()

    # # 추가: 타겟 변화속도가 0 이하인 경우 제거
    # interval_df = interval_df[
    #     (interval_df['target_dheight_per_day'] > 0) &
    #     (interval_df['target_dcrown_per_day'] > 0)
    # ].copy()

    return interval_df

# =========================
# 5. 환경 데이터를 생육 조사구간에 맞춰 집계
# =========================
def aggregate_env_for_intervals(interval_df, env_daily):
    """
    각 생육 조사구간 (prev_date, 조사일자] 에 대해
    해당 기간의 일별 환경을 집계
    """
    merged_rows = []

    env_group_cols = ['season_year', '도', '시군', '품목', '작기', '농가명']

    # env_daily를 그룹 dict로 미리 분할해 속도 개선
    env_dict = {
        key: sub.sort_values('date').copy()
        for key, sub in env_daily.groupby(env_group_cols)
    }

    for idx, row in interval_df.iterrows():
        key = (
            row['season_year'], row['도'], row['시군'],
            row['품목'], row['작기'], row['농가명']
        )

        if key not in env_dict:
            continue

        env_sub = env_dict[key]

        start_date = row['prev_date']
        end_date = row['조사일자']

        # 조사구간: (이전 조사일, 현재 조사일] 대신
        # 실무적으로는 이전 조사 다음날부터 현재 조사일까지 많이 씀
        # 여기서는 날짜 단위 집계이므로 prev_date 다음날 ~ 조사일자 당일까지 사용
        mask = (env_sub['date'] > start_date) & (env_sub['date'] <= end_date)
        interval_env = env_sub.loc[mask].copy()

        if interval_env.empty:
            continue

        total_days = (end_date - start_date).days
        valid_env = interval_env[interval_env['is_valid_day']].copy()
        valid_days = valid_env['date'].nunique()

        # 유효 날짜 비율 검사
        if total_days <= 0:
            continue

        valid_ratio = valid_days / total_days
        if valid_ratio < MIN_VALID_DAY_RATIO_IN_INTERVAL:
            continue

        # 집계
        temp_mean_interval = valid_env['temp_mean_day'].mean()
        hum_mean_interval = valid_env['hum_mean_day'].mean()
        co2_mean_interval = valid_env['co2_mean_day'].mean()

        temp_mean_interval = interval_env['temp_mean_day'].mean()
        hum_mean_interval = interval_env['hum_mean_day'].mean()
        co2_mean_interval = interval_env['co2_mean_day'].mean()

        # 누적일사량은 일별 마지막값(rad_cum_day)을 구간 전체 합산 후 날짜수로 나눔
        rad_per_day_interval = interval_env['rad_cum_day'].sum() / total_days

        vpd_interval = interval_env['vpd_day'].mean()
        rad3_interval = interval_env['rad_3day'].mean()
        gdd_interval = interval_env['gdd_cum'].iloc[-1] - interval_env['gdd_cum'].iloc[0]

        low_temp_hours_interval = interval_env['low_temp_hours_day'].sum()
        high_temp_hours_interval = interval_env['high_temp_hours_day'].sum()

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

    result = pd.DataFrame(merged_rows)
    return result


def build_env_daily(env_pattern):
    env_daily = load_env_files(env_pattern)

    # VPD
    svp = 0.6108 * np.exp((17.27 * env_daily['temp_mean_day']) /
                          (env_daily['temp_mean_day'] + 237.3))
    avp = svp * (env_daily['hum_mean_day'] / 100)
    env_daily['vpd_day'] = svp - avp

    # GDD
    BASE_TEMP = 5
    env_daily['gdd_day'] = np.maximum(env_daily['temp_mean_day'] - BASE_TEMP, 0)

    env_daily['gdd_cum'] = (
        env_daily
        .sort_values(['season_year', '농가명', 'date'])
        .groupby(['season_year', '농가명'])['gdd_day']
        .cumsum()
    )

    # 최근 3일 평균 일사량
    env_daily['rad_3day'] = (
        env_daily
        .sort_values(['season_year', '농가명', 'date'])
        .groupby(['season_year', '농가명'])['rad_cum_day']
        .rolling(3)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    return env_daily

# =========================
# 6. 최종 데이터셋 생성
# =========================
def build_dataset(env_pattern, grow_pattern):

    grow_all = load_growth_files(grow_pattern)
    interval_df = make_growth_intervals(grow_all)
    dataset = aggregate_env_for_intervals(interval_df, build_env_daily(env_pattern))

    # 최종 사용 컬럼
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

    #하루에 성장하는 길이 제한
    dataset = dataset[
    dataset['target_dheight_per_day'] < 1
    ]

    #하루에 성장하는 관부직경 제한
    dataset = dataset[
    dataset['target_dcrown_per_day'] < 0.5
    ].copy()

    return dataset, feature_cols, target_cols

# =========================
# 7. 모델 학습/평가
# =========================
def train_and_evaluate(dataset, feature_cols, target_cols):
    """
    같은 개체번호가 train/test에 동시에 들어가면 데이터 누수 가능성이 있어서
    group split 사용
    """
    dataset = dataset.copy()

    dataset['group_id'] = (
        dataset['season_year'].astype(str) + "_" +
        dataset['농가명'].astype(str) + "_" +
        dataset['개체번호'].astype(str)
    )

    X = dataset[feature_cols]
    y = dataset[target_cols]
    groups = dataset['group_id']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
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
    display_cols = []
    for col in target_cols:
        display_cols.append(col)
        display_cols.append(f"pred_{col}")

    sample = y_test.copy()
    for col in target_cols:
        sample[f"pred_{col}"] = pred_df[col]

    print(sample.head(10))

    return model, X_train, X_test, y_train, y_test, pred_df

# =========================
# 8. 실행
# =========================
if __name__ == "__main__":
    dataset, feature_cols, target_cols = build_dataset(ENV_PATTERN, GROW_PATTERN)

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

    # =========================
    # 변수 중요도 출력
    # =========================

    for i, target in enumerate(target_cols):
        rf = model.estimators_[i]

        importance = pd.Series(
            rf.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)

        print(f"\n[Feature importance - {target}]")
        print(importance)
