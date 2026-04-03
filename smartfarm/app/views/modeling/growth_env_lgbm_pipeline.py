import re
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

BASE = Path('D:/취업반/1차프로젝트/raw_datas')
OUT_DIR = BASE


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8', 'latin1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f'Cannot read {path}')


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns)
    dupes = cols[cols.duplicated()].unique()
    for c in dupes:
        same = df.loc[:, df.columns == c]
        first = same.iloc[:, 0].copy()
        for j in range(1, same.shape[1]):
            first = first.combine_first(same.iloc[:, j])
        df = df.loc[:, df.columns != c]
        df[c] = first
    return df


def load_yearly(prefix: str) -> pd.DataFrame:
    dfs = []
    for year in range(2018, 2023):
        matches = list(BASE.glob(f'{prefix}_딸기_{year}*'))
        if not matches:
            continue
        df = read_csv_auto(matches[0])
        df.columns = [str(c).strip() for c in df.columns]
        if '연도' not in df.columns:
            df['연도'] = year
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(prefix)
    return pd.concat(dfs, ignore_index=True, sort=False)


def make_group_folds(group_array, n_splits=5, seed=42):
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
    return np.sqrt(mean_squared_error(y_true, y_pred))


def sanitize_feature_names(columns):
    mapping = {}
    used = set()
    for c in columns:
        s = re.sub(r'[^0-9A-Za-z_]+', '_', str(c))
        if re.match(r'^\d', s):
            s = 'f_' + s
        base = s
        i = 1
        while s in used:
            i += 1
            s = f'{base}_{i}'
        used.add(s)
        mapping[c] = s
    return mapping


def load_all():
    env = load_yearly('환경')
    growth = load_yearly('생육')
    cult = load_yearly('재배정보')

    env.columns = [str(c).strip() for c in env.columns]
    growth.columns = [str(c).strip() for c in growth.columns]
    cult.columns = [str(c).strip() for c in cult.columns]

    env = env.rename(columns={'잔존이산화탄소(CO2)': '잔존CO2'})
    cult = cult.rename(columns={'지역(도)': '도'})

    env = coalesce_duplicate_columns(env)
    cult = coalesce_duplicate_columns(cult)

    return env, growth, cult


def make_daily_env(env: pd.DataFrame) -> pd.DataFrame:
    env = env.copy()
    env['농가명'] = pd.to_numeric(env['농가명'], errors='coerce')
    env['도'] = env['도'].astype(str).str.strip()
    env['시군'] = env['시군'].astype(str).str.strip()
    env['측정시간'] = pd.to_datetime(env['측정시간'], errors='coerce')
    env['date'] = env['측정시간'].dt.floor('D')

    for c in ['온도_내부', '상대습도_내부', '잔존CO2', '토양온도', '일사량_외부']:
        env[c] = pd.to_numeric(env[c], errors='coerce')

    env = env[
        (env['온도_내부'].between(-10, 50) | env['온도_내부'].isna()) &
        (env['상대습도_내부'].between(0, 100) | env['상대습도_내부'].isna()) &
        (env['일사량_외부'].between(0, 5000) | env['일사량_외부'].isna())
    ].copy()

    env['vpd'] = 0.6108 * np.exp((17.27 * env['온도_내부']) / (env['온도_내부'] + 237.3)) * (1 - env['상대습도_내부'] / 100)

    key_cols = ['연도', '도', '시군', '농가명']
    daily = env.groupby(key_cols + ['date'], as_index=False).agg(
        temp_mean=('온도_내부', 'mean'),
        rh_mean=('상대습도_내부', 'mean'),
        co2_mean=('잔존CO2', 'mean'),
        soil_temp_mean=('토양온도', 'mean'),
        rad_sum=('일사량_외부', 'sum'),
        vpd_mean=('vpd', 'mean'),
        low_temp_hours=('온도_내부', lambda s: np.sum(s < 10)),
        high_temp_hours=('온도_내부', lambda s: np.sum(s > 30)),
    )
    return daily.sort_values(key_cols + ['date']).reset_index(drop=True)


def make_env_windows(daily_env: pd.DataFrame) -> pd.DataFrame:
    key_cols = ['연도', '도', '시군', '농가명']
    mean_cols = ['temp_mean', 'rh_mean', 'co2_mean', 'soil_temp_mean', 'vpd_mean']
    sum_cols = ['rad_sum', 'low_temp_hours', 'high_temp_hours']
    out = []

    for k, g in daily_env.groupby(key_cols):
        g = g.sort_values('date').set_index('date')
        idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        r = g.reindex(idx)
        for i, col in enumerate(key_cols):
            r[col] = k[i]
        for c in mean_cols:
            r[c] = pd.to_numeric(r[c], errors='coerce')
        for c in sum_cols:
            r[c] = pd.to_numeric(r[c], errors='coerce').fillna(0)

        feat = r[key_cols + mean_cols + sum_cols].copy()
        for w in [3, 7, 14]:
            for c in mean_cols:
                feat[f'{c}_{w}d'] = feat[c].shift(1).rolling(window=w, min_periods=1).mean()
            for c in sum_cols:
                feat[f'{c}_{w}d'] = feat[c].shift(1).rolling(window=w, min_periods=1).sum()

        feat = feat.reset_index().rename(columns={'index': '조사일자'})
        keep_cols = key_cols + ['조사일자'] + [c for c in feat.columns if c.endswith(('3d', '7d', '14d'))]
        out.append(feat[keep_cols])

    return pd.concat(out, ignore_index=True)


def build_growth_targets(growth: pd.DataFrame, cult: pd.DataFrame) -> pd.DataFrame:
    key_cols = ['연도', '도', '시군', '농가명']

    growth = growth.copy()
    cult = cult.copy()

    for df in [growth, cult]:
        df['농가명'] = pd.to_numeric(df['농가명'], errors='coerce')
        df['도'] = df['도'].astype(str).str.strip()
        df['시군'] = df['시군'].astype(str).str.strip()

    growth['조사일자'] = pd.to_datetime(growth['조사일자'], errors='coerce')
    for c in ['초장', '엽수', '엽장', '엽폭', '엽병장', '관부직경', '화방번호', '화방별착과수', '개체번호']:
        growth[c] = pd.to_numeric(growth[c], errors='coerce')

    cult['정식일'] = pd.to_datetime(cult['정식일'], errors='coerce')
    cult['재식밀도'] = pd.to_numeric(cult['재식밀도'], errors='coerce')

    cult_small = cult[key_cols + ['품종', '재식밀도', '정식일']].drop_duplicates(key_cols)

    # 같은 개체-같은 조사일이 여러 줄이면 평균 처리
    plant_daily = growth.groupby(key_cols + ['개체번호', '조사일자'], as_index=False).agg(
        액아구분=('액아구분', 'first'),
        초장=('초장', 'mean'),
        엽수=('엽수', 'mean'),
        엽장=('엽장', 'mean'),
        엽폭=('엽폭', 'mean'),
        엽병장=('엽병장', 'mean'),
        관부직경=('관부직경', 'mean'),
        화방번호=('화방번호', 'mean'),
        화방별착과수=('화방별착과수', 'mean'),
    )

    df = plant_daily.merge(cult_small, on=key_cols, how='left')
    df = df.sort_values(key_cols + ['개체번호', '조사일자']).reset_index(drop=True)

    plant_key = key_cols + ['개체번호']
    prev_map = {
        '초장': 'prev_height',
        '엽수': 'prev_leaf_count',
        '엽장': 'prev_leaf_len',
        '엽폭': 'prev_leaf_width',
        '엽병장': 'prev_petiole_len',
        '관부직경': 'prev_crown',
        '화방번호': 'prev_flower_cluster',
        '화방별착과수': 'prev_fruit_count',
    }
    for src, dst in prev_map.items():
        df[dst] = df.groupby(plant_key)[src].shift(1)

    df['prev_date'] = df.groupby(plant_key)['조사일자'].shift(1)
    df['days_diff'] = (df['조사일자'] - df['prev_date']).dt.days
    df['days_from_transplant'] = (df['조사일자'] - df['정식일']).dt.days

    # target: 초장 변화량 / 시간
    df['height_growth_rate'] = (df['초장'] - df['prev_height']) / df['days_diff']
    return df


def prepare_model_dataset():
    env, growth, cult = load_all()
    daily_env = make_daily_env(env)
    env_win = make_env_windows(daily_env)
    growth_df = build_growth_targets(growth, cult)

    key_cols = ['연도', '도', '시군', '농가명']
    df = growth_df.merge(env_win, on=key_cols + ['조사일자'], how='left')

    # 기본 필터
    df = df[(df['days_diff'] >= 3) & (df['days_diff'] <= 21)].copy()
    df = df[np.isfinite(df['height_growth_rate'])].copy()

    # 극단치 제거: 상하위 1%
    q01, q99 = df['height_growth_rate'].quantile([0.01, 0.99])
    df = df[(df['height_growth_rate'] >= q01) & (df['height_growth_rate'] <= q99)].copy()

    env_features = [c for c in df.columns if c.endswith(('3d', '7d', '14d'))]
    prev_features = [
        'prev_height', 'prev_leaf_count', 'prev_leaf_len', 'prev_leaf_width',
        'prev_petiole_len', 'prev_crown', 'prev_flower_cluster', 'prev_fruit_count'
    ]
    numeric_features = prev_features + ['days_diff', 'days_from_transplant', '재식밀도'] + env_features
    categorical_features = ['품종', '액아구분']

    use_cols = key_cols + ['개체번호', '조사일자', 'height_growth_rate'] + numeric_features + categorical_features
    df = df[use_cols].copy()
    df = df.dropna(subset=['height_growth_rate'] + numeric_features + categorical_features).reset_index(drop=True)

    return df, numeric_features, categorical_features


def main():
    df, numeric_features, categorical_features = prepare_model_dataset()

    # 범주형 원핫 인코딩
    df_enc = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    feature_cols = [
        c for c in df_enc.columns
        if c not in ['연도', '도', '시군', '농가명', '개체번호', '조사일자', 'height_growth_rate']
    ]

    mapping = sanitize_feature_names(feature_cols)
    df_enc = df_enc.rename(columns=mapping)
    safe_feature_cols = [mapping[c] for c in feature_cols]

    X = df_enc[safe_feature_cols]
    y = df_enc['height_growth_rate']
    groups = (
        df_enc['연도'].astype(str) + '_' +
        df_enc['농가명'].astype(str) + '_' +
        df_enc['개체번호'].astype(str)
    ).values

    folds = make_group_folds(groups, n_splits=5, seed=42)

    results = []
    fi_list = []
    oof = np.zeros(len(df_enc))

    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='regression',
            verbose=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof[te_idx] = pred

        results.append({
            'fold': fold,
            'n_train': len(tr_idx),
            'n_test': len(te_idx),
            'R2': r2_score(y_test, pred),
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': rmse(y_test, pred),
        })

        fold_fi = pd.DataFrame({
            'feature_safe': safe_feature_cols,
            'importance': model.feature_importances_,
            'fold': fold,
        })
        inv_map = {v: k for k, v in mapping.items()}
        fold_fi['feature'] = fold_fi['feature_safe'].map(inv_map)
        fi_list.append(fold_fi[['feature', 'importance', 'fold']])

    result_df = pd.DataFrame(results)
    fi_df = pd.concat(fi_list, ignore_index=True)
    fi_mean = fi_df.groupby('feature', as_index=False)['importance'].mean().sort_values('importance', ascending=False)

    oof_df = df[['연도', '도', '시군', '농가명', '개체번호', '조사일자', 'height_growth_rate']].copy()
    oof_df['pred_height_growth_rate'] = oof
    oof_df['abs_error'] = (oof_df['height_growth_rate'] - oof_df['pred_height_growth_rate']).abs()

    result_df.to_csv(OUT_DIR / 'growth_env_lgbm_fold_results.csv', index=False, encoding='utf-8-sig')
    fi_mean.to_csv(OUT_DIR / 'growth_env_lgbm_feature_importance.csv', index=False, encoding='utf-8-sig')
    oof_df.to_csv(OUT_DIR / 'growth_env_lgbm_oof_predictions.csv', index=False, encoding='utf-8-sig')

    # feature importance 그래프
    topk = fi_mean.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(topk['feature'].astype(str), topk['importance'])
    plt.xlabel('Mean Importance')
    plt.title('Top 20 Feature Importance (Height Growth Rate Model)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'growth_env_lgbm_feature_importance_top20.png', dpi=160)
    plt.close()

    print(result_df)
    print('\nR2 mean:', round(result_df['R2'].mean(), 4))
    print('MAE mean:', round(result_df['MAE'].mean(), 4))
    print('RMSE mean:', round(result_df['RMSE'].mean(), 4))
    print('\nTop 15 features:')
    print(fi_mean.head(15))


if __name__ == '__main__':
    main()
