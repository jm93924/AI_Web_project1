from flask import Blueprint, render_template, request, redirect, url_for
import os
from app.scripts import env_growth_modeling
from app.models import Analysis
from werkzeug.utils import secure_filename

bp = Blueprint('analysis', __name__, url_prefix='/analysis')

UPLOAD_FOLDER = 'uploads'

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':        # 업로드를 했을때 오는 곳
        db_input = False

        env_file = request.files.get('env_file')
        growth_file = request.files.get('growth_file')
        yield_file = request.files.get('yield_file')

        for f in [env_file, growth_file, yield_file]:
            if not f.filename.lower().endswith('.csv'):
                return "CSV 파일만 업로드 가능합니다."

        env_growth_modeling.run_analysis(env_file, growth_file)

        #DB 저장이 TRUE일때
        if db_input == True:
            env_growth_modeling.save_analysis_result_to_db(
                fitted_models=env_growth_modeling.fitted_models,
                result_df=env_growth_modeling.result_df,
                feature_cols=env_growth_modeling.feature_cols,
                target_col=env_growth_modeling.target_col,
                analyzed_type='기본'
            )

        return redirect(url_for('analysis.index'))

    # 가장 최근 분석 결과 1건
    latest = (
        Analysis.query
        .order_by(Analysis.analyzed_date.desc())
        .first()
    )

    if latest is None:
        chart_labels = []
        relative_values = []
        top_factor = "-"
        top_factor_ratio = 0
        factor_count = 0
        top3_ratio = 0
        top3_names = []
    else:
        importance_map = {
            '온도': float(latest.temp_mean_importance),
            '습도': float(latest.hum_mean_importance),
            'CO2 농도': float(latest.co2_mean_importance),
            '일조량': float(latest.rad_per_day_importance),
            '고온 노출시간': float(latest.high_temp_hours_importance),
            '저온 노출시간': float(latest.low_temp_hours_importance),
            'VPD': float(latest.vpd_importance),
            '누적 GDD': float(latest.gdd_cum_importance),
            '이전 초장': float(latest.prev_plant_height_importance),
        }

        # 중요도 순으로 정렬
        sorted_items = sorted(
            importance_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        chart_labels = [label for label, value in sorted_items]

        total_importance = sum(value for label, value in sorted_items)

        if total_importance > 0:
            relative_values = [
                round((value / total_importance) * 100, 2)
                for label, value in sorted_items
            ]
        else:
            relative_values = [0.0 for _ in sorted_items]

        top_factor = chart_labels[0] if chart_labels else "-"
        top_factor_ratio = relative_values[0] if relative_values else 0
        factor_count = len(chart_labels)
        top3_ratio = round(sum(relative_values[:3]), 1)
        top3_names = chart_labels[:3]

    return render_template(
        'analysis.html',
        analysis=latest,
        chart_labels=chart_labels,
        relative_values=relative_values,
        top_factor=top_factor,
        top_factor_ratio=top_factor_ratio,
        factor_count=factor_count,
        top3_ratio=top3_ratio,
        top3_names=top3_names,
        has_uploaded_data=False
    )
