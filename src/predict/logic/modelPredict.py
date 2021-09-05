from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from util.date import BusinessDay
from data_control import data as dt
from predict.predict import Predict
from database.models import DailyChart


stock_model = 'stock_model_202108290146.pickle'
bgn_datetime = datetime(2020, 3, 1)
end_datetime = datetime(2021, 3, 31)


class ModelPredict(Predict):
    scaling_target_columns = ['open', 'high', 'low', 'close', 'turnover', 'vwap']
    predict_columns = ['open', 'vwap']

    def __init__(self):
        # modelの読込
        # self.__model = load_model(stock_model)
        # DBデータの読込
        data = self.__prepare_data()
        self.__data = dt.calc_moving_average(data, days=10, column='open')
        # modelに投入できる形に変形
        prev_length=5
        self.__data_for_predict = self.__prepare_data_for_predict(data, prev_length)
        # 日付インデックス
        first_data = self.__data[next(iter(self.__data))]
        self.__index = list(first_data['chart_date'])[prev_length:]

    def __prepare_data(self):
        all_data = DailyChart.date_between(bgn_datetime, end_datetime)
        data = dt.separate_by_code(all_data)
        
        data = dt.filter_missing_stocks(data, bgn_datetime, end_datetime)
        data = dt.filter_no_exec_stocks(data, bgn_datetime, end_datetime)
        data = dt.filter_disignated_stocks(data)
        # 銘柄のフィルタリング
        # 直近終値が1000円以下の銘柄を対象とする
        data = dt.filter_data(data, lambda d: int(d[-1:]['close']) <= 1000)
        return data

    def __prepare_data_for_predict(self, data, prev_length):
        data = self.__setup_columns(data)
        data, self.__scaler_dict = self.__scaling_data(data)

        data = self.__load_past_data(data, prev_length)
        data = self.__convert_to_ndarray(data)
        return data

    def __setup_columns(self, data: dict):
        target_columns = ['open', 'high', 'low', 'close', 'turnover', 'vwap']
        data = dt.to_float_type(data, target_columns)
        data = dt.filter_columns(data, target_columns)
        return data

    def __scaling_data(self, data: dict):
        """
        データのスケーリングを行う

        Params
        --------
        data: dict
            スケーリングを行うdata (key: 銘柄コード, value: pandas Dataframe)

        Returns
        ---------
        0 : dict
            スケーリング済のdata (key: 銘柄コード, value: pandas Dataframe)
        1 : dict
            scalerの辞書(key: 銘柄コード)
        """
        # scalingの実施
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_dict = {code: MinMaxScaler(feature_range=(0, 1)) for code in data}
        scaling_data, scaler_dict = dt.scaling(data, self.scaling_target_columns, scaler_dict)

        return scaling_data, scaler_dict

    def __reverse_scaling_data(self, data: dict):
        """
        スケーリングされたデータを元に戻す

        Params
        ---------
        data: dict
            元に戻すデータ

        Returns
        ---------
        0: dict
            元に戻したデータ
        """
        # 元に戻すデータに対応した銘柄コードのscalerを取得
        scaler_dict = {code: scaler for code, scaler in self.__scaler_dict.items() if code in data.keys()}
        # reverse scaling
        predict_data, _ = dt.scaling(data, self.scaling_target_columns, scaler_dict, inverse_option=True)

        return predict_data

    def __load_past_data(self, data: dict, n_prev: int):
        """
        データの各行に、過去データ行を追加

        Params
        --------
        data: dict
            過去データを追加するdata (key: 銘柄コード, value: pandas.DataFrame)
        n_prev: int
            過去データのload日数

        Returns
        ---------
        0: dict
            過去データを追加したdata(key: 銘柄コード, value: pandas.DataFrame)
        """

        past_add_dict = dict()

        for code, df in data.items():
            # 過去データを取得
            for prev in range(1, n_prev + 1):
                past_columns = ['{}_t-{}'.format(colname, prev) for colname in df.columns]
                past_df = df.shift(prev)
                past_df.rename(columns=dict(zip(past_df.columns, past_columns)), inplace=True)
                df = pd.concat([df, past_df], axis=1)
            # 過去データがない部分を削除
            df = df.drop(df.index[range(n_prev)])
            past_add_dict[code] = df
        return past_add_dict

    def __convert_to_ndarray(self, data):
        """
        必要なカラムを抽出し、ndarrayへ変換する

        """
        need_cols = ['open', 'high', 'low', 'close', 'turnover', 'vwap', \
                            'open_t-1', 'high_t-1', 'low_t-1', 'close_t-1', 'turnover_t-1', 'vwap_t-1', \
                            'open_t-2', 'high_t-2', 'low_t-2', 'close_t-2', 'turnover_t-2', 'vwap_t-2', \
                            'open_t-3', 'high_t-3', 'low_t-3', 'close_t-3', 'turnover_t-3', 'vwap_t-3', \
                            'open_t-4', 'high_t-4', 'low_t-4', 'close_t-4', 'turnover_t-4', 'vwap_t-4']
        need_col_df_list = [d[need_cols] for d in data.values()]
        org_shape = need_col_df_list[0].shape
        nd_data = np.vstack(need_col_df_list).reshape((len(need_col_df_list),) + org_shape)
        return nd_data

    def __create_dataset(self, data, batch_size=1, length=1):
        if data.shape[0] % batch_size != 0:
            raise ValueError('バッチサイズが不正です。')
        else:
            batch_num = data.shape[0] / batch_size

        if data.shape[1] % length != 0:
            raise ValueError('レングスが不正です。')
        else:
            split_num = data.shape[1] / length

        batch_list = np.split(data, batch_num, axis=0)
        split_list = [np.concatenate(np.split(batch, split_num, axis=1)) for batch in batch_list]
        data = np.concatenate(split_list)
        return data

    def __padding_dummy(self, data, length, position: Tuple):
        org_shape = data.shape
        after_shape = org_shape[:-1] + (length,)
        after = np.ones(after_shape)

        for idx, p in enumerate(position):
            after[:, :, p] = data[:, :, idx]
        return after


    def select_purchase_stock(self, date: datetime) -> Tuple[int, float, float]:
        """
        指定された日付のデータ買付銘柄を選定
        (翌営業日に買い付けるべき銘柄を予測)
        銘柄コード、買付価格、買付時の始値の予測を返却する
        存在しない場合にはNoneを返す

        実装はかなり分かりにくい
        modelへのデータ投入の為の整形、
        predictデータの再整形あたりが原因
        改善案があれば対応
        """

        #
        # データの存在確認
        #
        if not date in self.__index:
            raise ValueError('指定日付の価格データが存在しません。')

        #
        # modelからpredict
        #

        # indexから該当日付が何番目のデータか判別する
        data_position = self.__index.index(date)
        # 当該日のデータを取得
        data_of_target_date = self.__data_for_predict[:, data_position:data_position + 1, :]
        # model用に整形
        dataset = self.__create_dataset(data_of_target_date, 1, 1)

        # 予測
        __model = load_model(stock_model)
        predict = __model.predict(dataset, batch_size=1)

        # shapeを戻す
        predict = predict.reshape(data_of_target_date.shape[:2] + (len(self.predict_columns), ))
        # reverse scalingのためにdummyでレングスを合わせる
        predict_with_dummy = self.__padding_dummy(predict, len(self.scaling_target_columns), (0, 5))
        
        # dictに変換
        predict_scaling_dict = dict()
        target_code_list = list(self.__data.keys())
        for idx, d in enumerate(predict_with_dummy):
            predict_scaling_dict[target_code_list[idx]] = pd.DataFrame(d, columns=self.scaling_target_columns)
        # reverse scalinig
        predict_data = self.__reverse_scaling_data(predict_scaling_dict)

        # 始値と、vwapの差を算出
        price_gap_dict = dict()
        for code in predict_data:
            code_data = predict_data[code].tail(1)
            gap = int(code_data['vwap'] - code_data['open'])
            price_gap_dict[code] = gap
        # 最も差が大きい銘柄を抽出
        max_code = max(price_gap_dict, key=price_gap_dict.get)
        # 予測値を抽出
        predict_open = int(predict_data[max_code]['open'])
        predict_vwap = int(predict_data[max_code]['vwap'])

        return max_code, predict_open, predict_vwap

    def get_price_and_std(self, date: datetime, code: int) -> Tuple[float, float]:
        """
        指定日付における銘柄の購入金額、購入時標準偏差を返却する

        Params
        ---------

        """
        code_data = self.__data[code]
        target_data = code_data[
            code_data['chart_date'] == date
        ].iloc[0]
        return target_data['open'], target_data['m_std_{}_{}'.format(10, 'open')]

    def calc_profit(self, date: datetime, code: int, buy_price: float) -> float:
        """
        指定日の1株あたり評価益(終値ベース)を算出する

        Params
        ---------
        date: datetime.datetime
            損益を評価する日付

        code: int
            銘柄コード

        buy_price: int
            購入金額

        Returns
        ---------
        0: int
            1株あたり評価益
        """
        # 非営業日の場合、前営業日ベースで評価
        bs = BusinessDay()
        if not bs.is_businessday(date):
            date = bs.previous(date)

        # 指定日付、指定銘柄のデータ
        code_data = self.__data[code]
        target_data = code_data[
            code_data['chart_date'] == date
        ]
        if len(target_data) == 0:
            raise ValueError('指定日付のデータがありません code: {}, date: {}'.format(code, date))
        else:
            return target_data.iloc[0]['vwap'] - buy_price
