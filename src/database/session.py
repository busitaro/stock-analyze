from os import environ

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


db_user = environ['MYSQL_USER']
db_pass = environ['MYSQL_PASSWORD']
db_name = environ['MYSQL_DATABASE']
db_str = 'mysql+pymysql://{db_user}:{db_pass}@db/{db_name}?charset=utf8' \
    . format(db_user=db_user, db_pass=db_pass, db_name=db_name)

engine=create_engine(db_str, echo=True)


def make_session():
    sessionClass = sessionmaker(engine)
    session = sessionClass()
    return session
