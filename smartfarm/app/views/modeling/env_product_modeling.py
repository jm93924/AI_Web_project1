import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 원본 데이터들이 들어있는 폴더
BASE = Path("D:/취업반/1차프로젝트/raw_datas")


def read_csv_auto(path):
    """
    CSV 인코딩이 파일마다 다를 수 있어서
    여러 인코딩을 순서대로 시도해서 읽는 함수
    """
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read {path}")


def load_yearly(prefix):
    """
    예:
    prefix='환경' 이면
    환경_딸기_2018*, 환경_딸기_2019* ... 환경_딸기_2022*
    파일을 찾아서 모두 읽고 하나로 합침
    """
    dfs = []
    for year in range(2018, 2023):
        matches = list(BASE.glob(f"{prefix}_딸기_{year}*"))
        if not matches:
            continue

        df = read_csv_auto(matches[0])
        df.columns = [str(c).strip() for c in df.columns]

        # 연도 컬럼이 없으면 파일명 기준 연도 추가
        if "연도" not in df.columns:
            df["연도"] = year

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False)


def load_all():
    """
    환경 / 생육 / 생산 / 재배정보 4종 데이터를 모두 읽어오는 함수
    """
    env = load_yearly("환경")
    growth = load_yearly("생육")
    prod = load_yearly("생산")
    cult = load_yearly("재배정보")

    # 컬럼명 통일
    cult = cult.rename(columns={"지역(도)": "도"})

    return env, growth, prod, cult


def make_daily_env(env):
    """
    시간 단위 환경 데이터를 '일 단위'로 집계하는 함수

    출력 예:
    연도, 도, 시군, 농가명, date 기준으로
    temp_mean, rh_mean, rad_sum, vpd_mean ...
    """
    env = env.copy()

    # 시간 컬럼 datetime 변환
    env["측정시간"] = pd.to_datetime(env["측정시간"], errors="coerce")
    env["date"] = env["측정시간"].dt.floor("D")

    # 농가명과 환경 수치형 컬럼 숫자로 변환
    env["농가명"] = pd.to_numeric(env["농가명"], errors="coerce")
    for c in ["온도_내부", "상대습도_내부", "잔존CO2", "토양온도", "일사량_외부"]:
        env[c] = pd.to_numeric(env[c], errors="coerce")

    # 기본 이상치 제거
    env = env[
        (env["온도_내부"].between(-10, 50) | env["온도_내부"].isna()) &
        (env["상대습도_내부"].between(0, 100) | env["상대습도_내부"].isna()) &
        (env["일사량_외부"].between(0, 5000) | env["일사량_외부"].isna())
    ].copy()

    # VPD 계산
    env["vpd"] = (
        0.6108
        * np.exp((17.27 * env["온도_내부"]) / (env["온도_내부"] + 237.3))
        * (1 - env["상대습도_내부"] / 100)
    )

    # 일 단위 집계 키
    keys = ["연도", "도", "시군", "농가명", "date"]

    # 날짜별 집계
    daily = env.groupby(keys, as_index=False).agg(
        temp_mean=("온도_내부", "mean"),                  # 일평균 내부온도
        rh_mean=("상대습도_내부", "mean"),               # 일평균 상대습도
        co2_mean=("잔존CO2", "mean"),                   # 일평균 CO2
        soil_temp_mean=("토양온도", "mean"),            # 일평균 토양온도
        rad_sum=("일사량_외부", "sum"),                 # 일 누적 일사량
        vpd_mean=("vpd", "mean"),                      # 일평균 VPD
        low_temp_hours=("온도_내부", lambda s: np.sum(s < 10)),   # 저온 노출 횟수
        high_temp_hours=("온도_내부", lambda s: np.sum(s > 30)),  # 고온 노출 횟수
    )

    return daily.sort_values(["연도", "도", "시군", "농가명", "date"]).reset_index(drop=True)


def add_ship_window_env(weekly, daily_env):
    """
    주간 출하 데이터(weekly)에 대해
    각 출하 주의 시작일 직전 7/14/21일 환경값을 붙이는 함수

    예:
    temp_mean_7d_before_ship
    rad_sum_14d_before_ship
    """
    key_cols = ["연도", "도", "시군", "농가명"]
    env_cols = [
        "temp_mean", "rh_mean", "co2_mean", "soil_temp_mean",
        "rad_sum", "vpd_mean", "low_temp_hours", "high_temp_hours"
    ]
    out = []

    # 농가별 환경 데이터 사전 생성
    env_groups = {
        k: g.sort_values("date").reset_index(drop=True)
        for k, g in daily_env.groupby(key_cols)
    }

    # 주간 출하 데이터도 같은 농가 단위로 처리
    for k, g in weekly.groupby(key_cols):
        ge = env_groups.get(k)
        g = g.sort_values("week_start").copy()

        # 7/14/21일 전 환경 변수용 빈 칼럼 생성
        for w in [7, 14, 21]:
            for c in env_cols:
                g[f"{c}_{w}d_before_ship"] = np.nan

        # 해당 농가의 환경기록이 있으면
        if ge is not None and len(ge):
            for idx, wk in g["week_start"].items():

                # 현재 출하 주 시작일보다 이전 환경만 사용
                mask = ge["date"] < wk
                hist = ge.loc[mask]

                if len(hist) == 0:
                    continue

                # 직전 7/14/21일 구간 환경 집계
                for w in [7, 14, 21]:
                    sub = hist[hist["date"] >= (wk - pd.Timedelta(days=w))]
                    if len(sub) == 0:
                        continue

                    # 평균으로 넣을 변수
                    for c in ["temp_mean", "rh_mean", "co2_mean", "soil_temp_mean", "vpd_mean"]:
                        g.at[idx, f"{c}_{w}d_before_ship"] = sub[c].mean()

                    # 합계로 넣을 변수
                    for c in ["rad_sum", "low_temp_hours", "high_temp_hours"]:
                        g.at[idx, f"{c}_{w}d_before_ship"] = sub[c].sum()

        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_growth_farm_daily(growth):
    """
    개체별 생육 데이터를 농가-조사일 단위 평균으로 집계

    즉, 개체 단위 → 농가 평균 생육 상태
    """
    growth = growth.copy()

    growth["조사일자"] = pd.to_datetime(growth["조사일자"], errors="coerce")
    growth["농가명"] = pd.to_numeric(growth["농가명"], errors="coerce")

    num_cols = ["초장", "엽수", "엽장", "엽폭", "엽병장", "관부직경", "화방번호", "화방별착과수"]
    for c in num_cols:
        growth[c] = pd.to_numeric(growth[c], errors="coerce")

    # 농가-날짜 단위 평균 생육
    farm_daily = growth.groupby(["연도", "도", "시군", "농가명", "조사일자"], as_index=False).agg(
        height_mean=("초장", "mean"),
        leaf_count_mean=("엽수", "mean"),
        leaf_len_mean=("엽장", "mean"),
        leaf_width_mean=("엽폭", "mean"),
        petiole_len_mean=("엽병장", "mean"),
        crown_diameter_mean=("관부직경", "mean"),
        flower_cluster_mean=("화방번호", "mean"),
        fruit_count_mean=("화방별착과수", "mean"),
    )

    return farm_daily.sort_values(["연도", "도", "시군", "농가명", "조사일자"]).reset_index(drop=True)


def attach_latest_growth(weekly, farm_growth):
    """
    각 주간 출하 데이터에 대해
    그 주 시작일 이전의 '가장 최근 생육 관측값'을 붙이는 함수

    단, 너무 오래된 생육값은 제외 (21일 이내만 허용)
    """
    key_cols = ["연도", "도", "시군", "농가명"]
    add_cols = [
        "height_mean", "leaf_count_mean", "leaf_len_mean", "leaf_width_mean", "petiole_len_mean",
        "crown_diameter_mean", "flower_cluster_mean", "fruit_count_mean"
    ]

    fg_groups = {
        k: g.sort_values("조사일자").reset_index(drop=True)
        for k, g in farm_growth.groupby(key_cols)
    }

    out = []

    for k, g in weekly.groupby(key_cols):
        gf = fg_groups.get(k)
        g = g.sort_values("week_start").copy()

        # 생육 관련 칼럼 초기화
        for c in add_cols:
            g[c] = np.nan
        g["days_since_growth_obs"] = np.nan

        if gf is not None and len(gf):
            for idx, wk in g["week_start"].items():
                # week_start 이전의 최근 생육 조사일 찾기
                pos = gf["조사일자"].searchsorted(wk, side="right") - 1

                if pos >= 0:
                    diff = (wk - gf.iloc[pos]["조사일자"]).days

                    # 최근 21일 이내인 경우만 붙임
                    if diff <= 21:
                        for c in add_cols:
                            g.at[idx, c] = gf.iloc[pos][c]
                        g.at[idx, "days_since_growth_obs"] = diff

        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_weekly_yield(prod, cult):
    """
    생산 데이터를 날짜별 합산 → 주단위 합산
    그리고 식부면적으로 나눠 yield_per_area 생성
    """
    prod = prod.copy()
    cult = cult.copy()

    # 날짜 / 숫자형 변환
    prod["출하일자"] = pd.to_datetime(prod["출하일자"], errors="coerce")
    prod["농가명"] = pd.to_numeric(prod["농가명"], errors="coerce")
    prod["총출하량"] = pd.to_numeric(prod["총출하량"], errors="coerce")
    prod["판매금액"] = pd.to_numeric(prod["판매금액"], errors="coerce")

    cult["농가명"] = pd.to_numeric(cult["농가명"], errors="coerce")
    cult["정식일"] = pd.to_datetime(cult["정식일"], errors="coerce")
    cult["식부면적"] = pd.to_numeric(cult["식부면적"], errors="coerce")
    cult["재식밀도"] = pd.to_numeric(cult["재식밀도"], errors="coerce")

    key_cols = ["연도", "도", "시군", "농가명"]

    # 같은 날짜에 여러 줄 있으면 합산
    prod_day = prod.groupby(key_cols + ["출하일자"], as_index=False).agg(
        total_yield=("총출하량", "sum"),
        total_sales=("판매금액", "sum"),
    )

    # 평균단가 계산
    prod_day["avg_price"] = prod_day["total_sales"] / prod_day["total_yield"]

    # 재배정보 붙이기
    prod_day = prod_day.merge(
        cult[key_cols + ["품종", "식부면적", "재식밀도", "정식일"]],
        on=key_cols,
        how="left"
    )

    # 면적 보정 출하량
    prod_day["yield_per_area"] = prod_day["total_yield"] / prod_day["식부면적"]

    # 주 시작일 계산 (월요일 기준)
    prod_day["week_start"] = prod_day["출하일자"] - pd.to_timedelta(prod_day["출하일자"].dt.weekday, unit="D")

    # 주 단위 집계
    weekly = prod_day.groupby(key_cols + ["week_start"], as_index=False).agg(
        total_yield=("total_yield", "sum"),
        yield_per_area=("yield_per_area", "sum"),
        total_sales=("total_sales", "sum"),
        days_with_ship=("출하일자", "nunique"),
        cultivar=("품종", "first"),
        density=("재식밀도", "first"),
        transplant_date=("정식일", "first"),
    )

    # 주 평균 단가
    weekly["avg_price"] = weekly["total_sales"] / weekly["total_yield"]

    # 정식 후 경과일
    weekly["days_from_transplant"] = (weekly["week_start"] - weekly["transplant_date"]).dt.days

    return weekly.sort_values(key_cols + ["week_start"]).reset_index(drop=True)


def add_lag_features(weekly):
    """
    직전 출하 흐름 feature 생성
    """
    key_cols = ["연도", "도", "시군", "농가명"]
    weekly = weekly.sort_values(key_cols + ["week_start"]).copy()

    # 출하량 / 평균단가의 1주전, 2주전 값 생성
    for col in ["yield_per_area", "avg_price"]:
        weekly[f"prev_{col}_1w"] = weekly.groupby(key_cols)[col].shift(1)
        weekly[f"prev_{col}_2w"] = weekly.groupby(key_cols)[col].shift(2)

    # 이동평균
    weekly["yield_ma_3w"] = (
        weekly.groupby(key_cols)["yield_per_area"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    weekly["yield_ma_5w"] = (
        weekly.groupby(key_cols)["yield_per_area"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    return weekly


def add_flower_features(df):
    """
    화방 기반 파생변수 생성
    """
    key_cols = ["연도", "도", "시군", "농가명"]
    df = df.sort_values(key_cols + ["week_start"]).copy()

    def stage_bucket(x):
        if pd.isna(x):
            return np.nan
        if x < 2:
            return "early"
        elif x < 4:
            return "mid"
        return "late"

    # 지금까지의 최대 화방번호
    df["flower_cluster_max"] = df.groupby(key_cols)["flower_cluster_mean"].cummax()

    # 직전 대비 화방 변화량
    df["flower_cluster_diff"] = df.groupby(key_cols)["flower_cluster_mean"].diff()

    # 주 간격
    df["week_diff_days"] = df.groupby(key_cols)["week_start"].diff().dt.days

    # 화방 진행 속도
    df["flower_cluster_velocity"] = df["flower_cluster_diff"] / df["week_diff_days"]

    # 화방 구간 bucket
    df["cluster_stage_bucket"] = df["flower_cluster_mean"].apply(stage_bucket)

    # 화방당 착과수
    df["fruit_count_per_cluster"] = df["fruit_count_mean"] / df["flower_cluster_mean"]

    # 단순 fruit density
    df["fruit_density"] = df["fruit_count_mean"] * df["density"]

    # 주차 누적 느낌의 변수
    df["days_since_cluster_start"] = df.groupby(key_cols).cumcount() * 7

    # inf 제거
    for c in ["flower_cluster_velocity", "fruit_count_per_cluster"]:
        df.loc[np.isinf(df[c]), c] = np.nan

    return df


def make_group_folds(group_array, n_splits=5, seed=42):
    """
    그룹 단위로 랜덤 셔플 후 5등분하는 함수
    같은 그룹이 train/test에 동시에 들어가지 않게 함

    여기서 그룹 = 연도 + 농가명
    """
    unique_groups = np.unique(group_array)

    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    group_folds = np.array_split(shuffled, n_splits)

    folds = []
    for test_groups in group_folds:
        test_mask = np.isin(group_array, test_groups)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        folds.append((train_idx, test_idx))

    return folds


def rmse(y_true, y_pred):
    """
    RMSE 계산 함수
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # -----------------------------------------------------
    # 1) 데이터 로드
    # -----------------------------------------------------
    env, growth, prod, cult = load_all()

    # -----------------------------------------------------
    # 2) 환경 데이터를 일단위로 변환
    # -----------------------------------------------------
    daily_env = make_daily_env(env)

    # -----------------------------------------------------
    # 3) 생산 데이터를 주단위로 변환
    # -----------------------------------------------------
    weekly = build_weekly_yield(prod, cult)

    # -----------------------------------------------------
    # 4) 각 출하 주 직전 7/14/21일 환경 붙이기
    # -----------------------------------------------------
    weekly = add_ship_window_env(weekly, daily_env)

    # -----------------------------------------------------
    # 5) 생육 데이터를 농가-날짜 평균으로 만들고
    #    주간 출하 데이터에 최근 생육값 붙이기
    # -----------------------------------------------------
    farm_growth = build_growth_farm_daily(growth)
    weekly = attach_latest_growth(weekly, farm_growth)

    # -----------------------------------------------------
    # 6) lag feature 추가
    # -----------------------------------------------------
    weekly = add_lag_features(weekly)

    # -----------------------------------------------------
    # 7) 화방 feature 추가
    # -----------------------------------------------------
    weekly = add_flower_features(weekly)

    # -----------------------------------------------------
    # 8) 이번 실험에서 사용할 입력 feature 목록
    #    = 화방 + 환경 + 생육 + lag
    # -----------------------------------------------------
    numeric_features = [
        # lag
        "prev_yield_per_area_1w", "prev_yield_per_area_2w", "yield_ma_3w", "yield_ma_5w",
        "prev_avg_price_1w", "prev_avg_price_2w", "days_with_ship",

        # 환경
        "temp_mean_7d_before_ship", "temp_mean_14d_before_ship",
        "rad_sum_7d_before_ship", "rad_sum_14d_before_ship",
        "vpd_mean_7d_before_ship", "low_temp_hours_14d_before_ship", "high_temp_hours_21d_before_ship",

        # 생육
        "height_mean", "leaf_count_mean", "crown_diameter_mean",

        # 화방
        "flower_cluster_mean", "flower_cluster_max", "flower_cluster_diff", "flower_cluster_velocity",
        "fruit_count_mean", "fruit_count_per_cluster", "fruit_density", "days_since_cluster_start",

        # 기타
        "density", "days_from_transplant", "days_since_growth_obs"
    ]

    categorical_features = ["cultivar", "cluster_stage_bucket"]

    # -----------------------------------------------------
    # 9) 최종 학습용 데이터셋 구성
    # -----------------------------------------------------
    use_cols = ["연도", "농가명", "week_start", "yield_per_area"] + numeric_features + categorical_features
    df = weekly[use_cols].copy()

    # 그룹 ID = 연도 + 농가명
    df["group_id"] = df["연도"].astype(str) + "_" + df["농가명"].astype(str)

    # 결측이 있으면 제거
    req = ["yield_per_area"] + numeric_features + categorical_features
    df = df.dropna(subset=req).reset_index(drop=True)

    # 범주형 원핫인코딩
    df_enc = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    # feature / target 분리
    feature_cols = [
        c for c in df_enc.columns
        if c not in ["연도", "농가명", "week_start", "yield_per_area", "group_id"]
    ]

    X = df_enc[feature_cols]
    y = df_enc["yield_per_area"]
    groups = df_enc["group_id"].values

    # -----------------------------------------------------
    # 10) 그룹 기반 5-fold 분할
    # -----------------------------------------------------
    folds = make_group_folds(groups, n_splits=5, seed=42)

    results = []
    fi_list = []
    oof = np.zeros(len(df_enc))

    # -----------------------------------------------------
    # 11) fold별 학습 / 평가
    # -----------------------------------------------------
    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof[te_idx] = pred

        # 평가결과 저장
        results.append({
            "fold": fold,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse(y_test, pred),
        })

        # feature importance 저장
        fi_list.append(pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold
        }))

    # -----------------------------------------------------
    # 12) 결과 정리
    # -----------------------------------------------------
    result_df = pd.DataFrame(results)

    fi_df = pd.concat(fi_list, ignore_index=True)
    fi_mean = (
        fi_df.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    oof_df = df_enc[["연도", "농가명", "week_start", "yield_per_area", "group_id"]].copy()
    oof_df["pred_yield_per_area"] = oof
    oof_df["abs_error"] = (oof_df["yield_per_area"] - oof_df["pred_yield_per_area"]).abs()

    # -----------------------------------------------------
    # 13) 결과 파일 저장
    # -----------------------------------------------------
    result_df.to_csv(BASE / "yield_flower_grouped_cv_results_exact.csv", index=False, encoding="utf-8-sig")
    fi_mean.to_csv(BASE / "yield_flower_feature_importance_exact.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(BASE / "yield_flower_oof_predictions_exact.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # 14) 화면 출력
    # -----------------------------------------------------
    print(result_df)
    print()
    print("R2 mean:", round(result_df["R2"].mean(), 4))
    print("MAE mean:", round(result_df["MAE"].mean(), 4))
    print("RMSE mean:", round(result_df["RMSE"].mean(), 4))
    print()
    print(fi_mean.head(15))


if __name__ == "__main__":
    main()