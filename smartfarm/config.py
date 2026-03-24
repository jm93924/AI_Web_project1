import os
import oracledb
from flask_sqlalchemy import SQLAlchemy

# oracle 11g 연결을 위한 Thick 모드 초기화
oracledb.init_oracle_client(lib_dir=r"C:\Users\jm939\Downloads\instantclient_19_25\instantclient_19_25")


BASE_DIR = os.path.dirname(__file__) # 현재 파일의 경로
print("BASE_DIR", BASE_DIR)

# URI 다이얼렉트를 oracledb로 설정
SQLALCHEMY_DATABASE_URI = "oracle+oracledb://smart:farm@localhost:1521/xe"
print("SQLALCHEMY_DATABASE_URI", SQLALCHEMY_DATABASE_URI)

SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = "dev"