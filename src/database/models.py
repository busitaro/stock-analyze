from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import DECIMAL, VARCHAR, DATE

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
