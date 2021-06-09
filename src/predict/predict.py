import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from analyze import data as dt


class Predict:
    scaling_target_columns = ['begin', 'high', 'low', 'end', 'turnover']
    predict_columns = ['begin', 'high']

    def __init__(self, data):
        # csv データ
        self.__data = self.__prepare_data(data)
        # 日付インデックス
        first_data = self.__data[list(self.__data.keys())[0]]
        self.__index = list(first_data.index)
        # model用データ(ndrray)
        scaling_data, self.__scaler_dict = self.__scaling_data(self.__data.copy())
        n_prev = 5
        past_data = self.__load_past_data(scaling_data, n_prev)
        self.__index = self.__index[n_prev:]
        self.__data_for_predict = self.__convert_to_ndarray(past_data)
        # model
        self.__model = self.__load_model()

    def __prepare_data(self, data):
        # dataにstdを追加
        return dt.calc_std(data, 'begin', days=10)

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
        need_cols = ['begin', 'high', 'low', 'end', 'turnover', \
                            'begin_t-1', 'high_t-1', 'low_t-1', 'end_t-1', 'turnover_t-1', \
                            'begin_t-2', 'high_t-2', 'low_t-2', 'end_t-2', 'turnover_t-2', \
                            'begin_t-3', 'high_t-3', 'low_t-3', 'end_t-3', 'turnover_t-3', \
                            'begin_t-4', 'high_t-4', 'low_t-4', 'end_t-4', 'turnover_t-4',]
        need_col_df_list = [d[need_cols] for d in data.values()]
        org_shape = need_col_df_list[0].shape
        nd_data = np.vstack(need_col_df_list).reshape((len(need_col_df_list),) + org_shape)
        return nd_data

    def __filter_data(self, data: np.ndarray, target_code_list: list):
        """
        dataからtarget_codeのみを抽出する
        dataはndarrayで、第一成分方向にself.__dataのkeyの順で、
        各銘柄データが存在する前提

        Params
        ---------
        data: np.ndarray
            フィルタリング対象データ
        target_code: list
            対象銘柄のリスト
        """
        # 対象の銘柄コードのデータ位置を取得
        code_list = list(self.__data.keys())
        code_position_list = [code_list.index(code) for code in target_code_list]

        # データのフィルタリング
        filtered_data = data[code_position_list, :, :]

        return filtered_data

    # bath_sizeは1固定だし、多分lengthも1固定出てつかうことになると思う
    def __create_dataset(self, data, batch_size, length):
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

    def __load_model(self):
        model_name = 'stock_model_202103231540.pickle'
        model = load_model(model_name)
        return model

    def __padding_dummy(self, data, length):
        org_shape = data.shape
        padding_length = length - org_shape[-1]

        dummy = np.ones(org_shape[:-1] + (padding_length,))
        padding_data = np.append(data, dummy, axis=len(data.shape) - 1)
        return padding_data

    def select_purchase_stock(self, date: datetime):
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
        # 当日終値が1000円以下の銘柄に限定
        #
        target_code_list = list()
        for code, code_data in self.__data.items():
            if code_data.loc[date, 'end'] <= 1000:
                target_code_list.append(code)

        #
        # modelからpredict
        #

        # indexから該当日付が何番目のデータか判別する
        data_position = self.__index.index(date)
        # 当該日のデータを取得
        data_of_target_date = self.__data_for_predict[:, data_position:data_position + 1, :]
        # データを指定コードのみにフィルタリング
        data_of_target_date = self.__filter_data(data_of_target_date, target_code_list)
        # model用に整形
        dataset = self.__create_dataset(data_of_target_date, 1, 1)

        # 予測
        predict = self.__model.predict(dataset, batch_size=1)

        # shapeを戻す
        predict = predict.reshape(data_of_target_date.shape[:2] + (len(self.predict_columns), ))
        # reverse scalingのためにdummyでレングスを合わせる
        predict_with_dummy = self.__padding_dummy(predict, len(self.scaling_target_columns))
        # dictに変換
        predict_scaling_dict = dict()
        for idx, d in enumerate(predict_with_dummy):
            predict_scaling_dict[target_code_list[idx]] = pd.DataFrame(d, columns=self.scaling_target_columns)
        # reverse scalinig
        predict_data = self.__reverse_scaling_data(predict_scaling_dict)

        # 始値と、高値の差を算出
        price_gap_dict = dict()
        for code in predict_data:
            code_data = predict_data[code].tail(1)
            gap = int(code_data['high'] - code_data['begin'])
            price_gap_dict[code] = gap
        # 最も差が大きい銘柄を抽出
        max_code = max(price_gap_dict, key=price_gap_dict.get)
        # 予測値を抽出
        predict_begin = int(predict_data[max_code]['begin'])
        predict_high = int(predict_data[max_code]['high'])

        return max_code, predict_begin, predict_high

    def get_price_and_std(self, date, code):
        """
        指定日付における銘柄の購入金額、購入時標準偏差を返却する

        Params
        ---------

        """
        # 指定日付のデータから、買い付け価格、買付時の価格標準偏差を取得
        buy_price = self.__data[code].loc[date, 'begin']
        buy_std = self.__data[code].loc[date, 'begin_std_10days']

        return buy_price, buy_std

    def calc_profit(self, date, code, buy_price) -> int:
        """
        指定日の1株あたり評価益(高値ベース)を算出する

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
        price_of_date = self.__data[code].loc[date, 'high']
        return buy_price - price_of_date


def input_data():
    data = dt.input_price_file(exclude=True, exclude_by_count=True)
    data = dt.to_float_type(data, ['begin', 'high', 'low', 'end', 'turnover'])

    if len(data) == 0:
        raise ValueError('data for simulation is empty')
    return data


def main(date: datetime):
    d = input_data()
    predict = Predict(d)

    print(predict.select_purchase_stock(date))


if __name__ == '__main__':
    """
    Params
    ---------
    $1: 予測日付(yyyymmdd)
    """
    date = datetime.strptime(sys.argv[1], '%Y%m%d')
    main(date)
