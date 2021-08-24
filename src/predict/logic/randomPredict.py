from random import choice
from typing import Tuple
from datetime import datetime, date

from predict.predict import Predict
from database.models import DailyChart
from data_control.data import separate_by_code, combine_dict, calc_moving_average, filter_missing_stocks, filter_no_exec_stocks,filter_disignated_stocks
from util.date import BusinessDay


# MEMO: とりあえず
bgn_date = datetime(2020, 3, 1)
end_date = datetime(2021, 3, 31)

std_days = 10


class RandomPredict(Predict):
    def __init__(self):
        dailyChart = DailyChart()
        chart_data = dailyChart.date_between(bgn_date, end_date)
        data = separate_by_code(chart_data)
        data = calc_moving_average(data, days=std_days, column='open')
        # filter
        data = filter_missing_stocks(data, bgn_date, end_date)
        data = filter_no_exec_stocks(data, bgn_date, end_date)
        data = filter_disignated_stocks(data)
        if len(data) == 0:
            raise ValueError('価格データが揃った銘柄がありません。')
        self.__data = combine_dict(data)

    def select_purchase_stock(self, date: datetime) -> Tuple[int, float, float]:
        # 終値が1,000円以下の銘柄を対象
        target_data = self.__data[(self.__data['chart_date'] == date) & (self.__data['close'] <= 1000)]
        # 銘柄を選択
        if len(target_data) == 0:
            return None
        else:
            # ランダムに1銘柄を選択
            code = choice(list(target_data['description_code']))
            # 予測買付、売却価格は購入日の始値、vwapをそのまま返す
            d = target_data[target_data['description_code'] == code].iloc[0]
            predict_buy_price = float(d['open'])
            predict_sell_price = float(d['vwap'])

            return code, predict_buy_price, predict_sell_price

    def get_price_and_std(self, date: datetime, code: int) -> Tuple[float, float]:
        # 指定日付、指定銘柄のデータ
        target_data = self.__data[
            (self.__data['chart_date'] == date) &
            (self.__data['description_code'] == code)
        ].iloc[0]

        return target_data['open'], target_data['m_std_{}_{}'.format(std_days, 'open')]

    def calc_profit(self, date: datetime, code: int, buy_price: float) -> float:
        # 非営業日の場合、前営業日ベースで評価
        bs = BusinessDay()
        if not bs.is_businessday(date):
            date = bs.previous(date)

        # 指定日付、指定銘柄のデータ
        target_data = self.__data[
            (self.__data['chart_date'] == date) &
            (self.__data['description_code'] == code)
        ]
        if len(target_data) == 0:
            raise ValueError('指定日付のデータがありません code: {}, date: {}'.format(code, date))
        else:
            return target_data.iloc[0]['vwap'] - buy_price
