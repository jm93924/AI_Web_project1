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
