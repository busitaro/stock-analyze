from datetime import datetime
from typing import Tuple
from pickle import load

import numpy as np
import pandas as pd
from keras.models import load_model

from util.date import BusinessDay
from data_control import data as dt
from predict.predict import Predict
from database.models import DailyChart


stock_model = 'stock_model_202108290146.pickle'
stored_scaler = 'scaler_dict_202109110933.pickle'
bgn_datetime = datetime(2020, 3, 1)
end_datetime = datetime(2021, 3, 31)


class ModelPredict(Predict):
    scaling_target_columns = ['open', 'high', 'low', 'close', 'turnover', 'vwap']

    def __init__(self):
        self.__prev_length = 5

        # DBデータの読込
        data = self.__prepare_data()
        self.__data = dt.calc_moving_average(data, days=10, column='open')
        # 日付インデックス (必要か確認)
        first_data = self.__data[next(iter(self.__data))]
        self.__index = list(first_data['chart_date'])

    def __prepare_data(self):
        all_data = DailyChart.date_between(bgn_datetime, end_datetime)
        data = dt.separate_by_code(all_data)
        
        data = dt.filter_missing_stocks(data, bgn_datetime, end_datetime)
        data = dt.filter_no_exec_stocks(data, bgn_datetime, end_datetime)
        data = dt.filter_disignated_stocks(data)

        return data

    def __setup_columns(self, data: dict) -> dict:
        target_columns = ['open', 'high', 'low', 'close', 'turnover', 'vwap']
        data = dt.to_float_type(data, target_columns)
        data = dt.filter_columns(data, target_columns)
        return data


    def __prepare_data_for_predict(self, date: datetime):
        # 指定日のデータを作成
        data_for_predict = self.__extract_data_of_date(date)

        # 必要カラムを抽出
        data_for_predict = self.__setup_columns(data_for_predict)

        # スケーリング
        data_for_predict = self.__scaling_data(data_for_predict)

        # 過去データを1レコードへ連結
        data_for_predict = self.__load_past_data(data_for_predict)

        # ndarrayへ変換
        data_for_predict = self.__convert_to_ndarray(data_for_predict)

        return data_for_predict


    def __extract_data_of_date(self, date: datetime):
        """
        指定日付の予測に必要なデータを抽出する


        """
        extract_data = dict()
        for code in self.__data:
            code_data = self.__data[code].reset_index()
            data_of_date = code_data[code_data['chart_date'] == date]
            if len(data_of_date) == 0:
                raise ValueError('指定日付の価格データが存在しません。 {}'.format(date))
            data_position = data_of_date.index[0]
            extract_data[code] = code_data[data_position - self.__prev_length + 1:data_position + 1]
        return extract_data


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
        """
        # scalerのload
        with open(stored_scaler, mode='rb') as f:
            scaler_dict = load(f)

        # scalerにある銘柄のみに絞り込む
        filtered_data = dict()
        code_list = list(data.keys())
        for code in scaler_dict:
            if code in code_list:
                filtered_data[code] = data[code]

        # scalingの実施
        scaling_data, _ = dt.scaling(filtered_data, self.scaling_target_columns, scaler_dict, fit=False)

        return scaling_data

    def __reverse_scaling_data(self, data: np.array):
        """
        スケーリングされたデータを元に戻す

        Params
        ---------
        data: dict
            元に戻すデータ

        Returns
        ---------
        0: np.array
            元に戻したデータ
        """

        # scalerのload
        with open(stored_scaler, mode='rb') as f:
            scaler_dict = load(f)

        # dataをdictへ
        data_dict = dict()
        code_list = list(scaler_dict.keys())
        for idx, code_data in enumerate(data):
            data_dict[code_list[idx]] = pd.DataFrame(code_data, columns=self.scaling_target_columns)

        # reverse scaling
        predict_data, _ = dt.scaling(data_dict, self.scaling_target_columns, scaler_dict, inverse_option=True)

        return predict_data

    def __load_past_data(self, data: dict):
        """
        データの各行に、過去データ行を追加

        Params
        --------
        data: dict
            過去データを追加するdata (key: 銘柄コード, value: pandas.DataFrame)

        Returns
        ---------
        0: dict
            過去データを追加したdata(key: 銘柄コード, value: pandas.DataFrame)
        """

        past_add_dict = dict()

        for code, df in data.items():
            org_columns = df.columns
            # 過去データを取得
            for prev in range(1, self.__prev_length):
                past_columns = ['{}_t-{}'.format(colname, prev) for colname in org_columns]
                past_df = df[org_columns].shift(prev)
                past_df.rename(columns=dict(zip(past_df.columns, past_columns)), inplace=True)
                df = pd.concat([df, past_df], axis=1)
            # 過去データがない部分を削除
            df = df.drop(df.index[range(self.__prev_length - 1)])
            past_add_dict[code] = df
        return past_add_dict

    def __convert_to_ndarray(self, data):
        """
        必要なカラムを抽出し、ndarrayへ変換する

        """
        df_list = list(data.values())
        org_shape = df_list[0].shape
        nd_data = np.vstack(df_list).reshape((len(df_list),) + org_shape)
        return nd_data


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

        # データの準備
        dataset = self.__prepare_data_for_predict(date)
        # modelからpredict
        __model = load_model(stock_model)
        predict = __model.predict(dataset, batch_size=1)
        # reverse scalingのためにdummyでレングスを合わせる
        predict_with_dummy = self.__padding_dummy(predict, len(self.scaling_target_columns), (0, 5))
        # reverse scalinig
        predict_data = self.__reverse_scaling_data(predict_with_dummy)

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
