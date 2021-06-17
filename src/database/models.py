from datetime import date

import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import DECIMAL, VARCHAR, DATE

from database import make_session


Base = declarative_base()


class DailyChart(Base):
    __tablename__ = 'daily_chart'
    chart_date = Column(DATE(), primary_key=True, nullable=False)
    description_code = Column(DECIMAL(5, 0), primary_key=True, nullable=False)
    open = Column(DECIMAL(8, 1), primary_key=True, nullable=False)
    high = Column(DECIMAL(8, 1), primary_key=True, nullable=False)
    low = Column(DECIMAL(8, 1), primary_key=True, nullable=False)
    close = Column(DECIMAL(8, 1), primary_key=True, nullable=False)
    turnover = Column(DECIMAL(10, 0), primary_key=True, nullable=False)
    vwap = Column(DECIMAL(10, 3), primary_key=True, nullable=False)
    execution_count = Column(DECIMAL(10, 0), primary_key=True, nullable=False)

    def __repr__(self):
        return '<daily_chart chart_date={chart_date} description_code={description_code}>' \
            .format(chart_date=self.chart_date, description_code=self.description_code)

    @staticmethod
    def columns():
        return (
            DailyChart.chart_date,
            DailyChart.description_code,
            DailyChart.open,
            DailyChart.high,
            DailyChart.low,
            DailyChart.close,
            DailyChart.turnover,
            DailyChart.vwap,
            DailyChart.execution_count,
        )

    @staticmethod
    def column_names():
        return DailyChart.__table__.c.keys()

    @staticmethod
    def all():
        return pd.DataFrame(make_session().query(*DailyChart.columns()).all())

    @staticmethod
    def date_between(bgn_date: date, end_date: date):
        rs = make_session() \
            .query(*DailyChart.columns()) \
            .filter(DailyChart.chart_date.between(bgn_date, end_date)) \
            .all()

        return pd.DataFrame(rs, columns=DailyChart.column_names())
