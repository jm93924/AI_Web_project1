from app import db
from sqlalchemy import Sequence

class Customer(db.Model):
    __tablename__ = 'customer'

    seq = db.Column(db.Integer, db.Sequence('customer_seq', start=1, increment=1),primary_key=True)

    id = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(40), nullable=False)
    email = db.Column(db.String(100), nullable=False)

class FarmInfo(db.Model):
    __tablename__ = 'farm_info'

    farm_id = db.Column(db.Integer, db.Sequence('farm_info_seq', start=1, increment=1), primary_key=True)
    survey_year = db.Column(db.Integer, nullable=False)
    item = db.Column(db.String(40), nullable=False)
    farm_num = db.Column(db.Integer, nullable=False)
    district = db.Column(db.String(40), nullable=False)
    city = db.Column(db.String(40), nullable=False)

class Environment(db.Model):
    __tablename__ = 'environment'

    farm_id = db.Column(db.Integer, db.ForeignKey('farm_info.farm_id'), nullable=False, primary_key=True)
    measure_time = db.Column(db.DateTime, nullable=False, primary_key=True)
    out_temp = db.Column(db.Float)
    out_wind_direction = db.Column(db.Integer)
    out_wind_speed = db.Column(db.Float)
    solar_radiation = db.Column(db.Integer)
    solar_radiation_sum = db.Column(db.Integer, nullable=False)
    rain = db.Column(db.Integer)
    inside_temp = db.Column(db.Float, nullable=False)
    relative_humidity = db.Column(db.Float, nullable=False)
    carbon_dioxide = db.Column(db.Integer, nullable=False)
    soil_temp = db.Column(db.Float)

class GrowthData(db.Model):
    __tablename__ = 'growth_data'

    farm_id = db.Column(db.Integer, db.ForeignKey('farm_info.farm_id'), primary_key=True, nullable=False)
    survey_date = db.Column(db.DateTime, primary_key=True, nullable=False)
    plant_num = db.Column(db.Integer, primary_key=True, nullable=False)
    axillary_branch = db.Column(db.String)
    plant_height = db.Column(db.Float, nullable=False)
    leaf_count = db.Column(db.Integer)
    leaf_length = db.Column(db.Float)
    leaf_width = db.Column(db.Float)
    petiole_length = db.Column(db.Float)
    crown_diameter = db.Column(db.Float, nullable=False)
    flower_cluster_no = db.Column(db.Integer)
    fruits_per_cluster = db.Column(db.Integer)

class ProductData(db.Model):
    __tablename__ = 'product_data'

    farm_id = db.Column(db.Integer, db.ForeignKey('farm_info.farm_id'), primary_key=True, nullable=False)
    production_date = db.Column(db.DateTime, primary_key=True, nullable=False)
    total_quantity = db.Column(db.Float, nullable=False)
    total_sales = db.Column(db.Integer)

class CultivationInfo(db.Model):
    __tablename__ = 'cultivation_info'

    farm_id = db.Column(db.Integer, db.ForeignKey('farm_info.farm_id'), nullable=False, primary_key=True)
    house_type = db.Column(db.String, nullable=False)
    house_form = db.Column(db.String, nullable=False)
    total_area = db.Column(db.Float, nullable=False)
    planting_area = db.Column(db.Float, nullable=False)
    planting_density = db.Column(db.Float, nullable=False)
    planting_date = db.Column(db.DateTime, nullable=False)

class CultivationVariety(db.Model):
    __tablename__ = 'cultivation_variety'

    farm_id = db.Column(db.Integer, db.ForeignKey('farm_info.farm_id'), nullable=False, primary_key=True)
    item_variety = db.Column(db.String, nullable=False, primary_key=True)

class Analysis(db.Model):
    __tablename__ = 'analysis'

    analyzed_seq = db.Column(db.Integer,
                             db.Sequence('analyzed_seq', start=1, increment=1),
                             primary_key=True,
                             nullable=False)

    analyzed_type = db.Column(db.String(40), nullable=False)
    analyzed_name = db.Column(db.String(100), nullable=False)
    analyze_input = db.Column(db.String(1000), nullable=False)
    analyze_target = db.Column(db.String(200), nullable=False)

    temp_mean_importance =db.Column(db.Float, nullable=False)
    hum_mean_importance =db.Column(db.Float, nullable=False)
    co2_mean_importance =db.Column(db.Float, nullable=False)
    rad_per_day_importance =db.Column(db.Float, nullable=False)
    high_temp_hours_importance =db.Column(db.Float, nullable=False)
    low_temp_hours_importance =db.Column(db.Float, nullable=False)
    vpd_importance =db.Column(db.Float, nullable=False)
    gdd_cum_importance =db.Column(db.Float, nullable=False)
    prev_plant_height_importance =db.Column(db.Float, nullable=False)

    random_forest_score = db.Column(db.Float, nullable=False)
    extra_trees_score = db.Column(db.Float, nullable=False)
    gradient_boosting_score = db.Column(db.Float, nullable=False)
    hist_gradient_boosting_score = db.Column(db.Float, nullable=False)

    analyzed_date = db.Column(db.DateTime, nullable=False)