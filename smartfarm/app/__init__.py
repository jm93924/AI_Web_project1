from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config
db = SQLAlchemy()
migrate = Migrate()

def create_app():   # 관례적으로 사용하는 이름
    app = Flask(__name__)
    app.config.from_object(config)

    #ORM Initialization
    db.init_app(app)
    migrate.init_app(app, db)

    #해당 파일에 정의된 모든 모델 클래스를 애플리케이션에 등록하게 됨.
    from . import models

    # Blueprint
    from .views import main_views,analysis_views, board_views, login_views, prediction_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(analysis_views.bp)
    app.register_blueprint(board_views.bp)
    app.register_blueprint(login_views.bp)
    app.register_blueprint(prediction_views.bp)

    return app
