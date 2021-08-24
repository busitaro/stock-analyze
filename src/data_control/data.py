import os
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime, timedelta

from util.date import BusinessDay


def separate_by_code(data: pd.DataFrame):
    """
    銘柄コード毎に分かれていないpandasデータを
    銘柄コードをkeyとしdictに分割する

    Params
    ---------
    data: pd.DataFrame
        分割対象のデータ

    Returns
    ---------
    0: dict
        分割した辞書データ
    """
    data_dict = {}

    for code, d in data.groupby('description_code'):
        data_dict[code] = d

    return data_dict


def combine_dict(data: dict):
    """
    銘柄コード毎に分かれたdictデータを
    結合したpandasにする

    Params
    ---------
    data: dict
        結合対象の辞書データ

    Returns
    ---------
    0: pd.DataFrame
        結合したpandasデータ
    """
    data_list = list()
    for code, code_data in data.items():
        code_data['description_code'] = code
        data_list.append(code_data)

    return pd.concat(data_list)


def filter_missing_stocks(data:dict, bgn_date: datetime, end_date: datetime):
    """
    全営業日分のデータがそろっていない銘柄を除去する

    Params
    ---------
    data: dict
        フィルタリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    bgn_date: datetime.datetime
        確認開始日
    end_date: datetime.datetime
        確認終了日

    Returns
    ---------
    0: dict
        フィルタリングした辞書データ
    """
    filtered_data = dict()

    # 営業日のリストを取得
    bs = BusinessDay()
    business_days = set(bs.between(bgn_date, end_date))

    for code, code_data in data.items():
        dates = set(code_data['chart_date'])
        if business_days.issubset(dates):
            # すべての営業日をデータの日付が含んでいた場合
            filtered_data[code] = code_data

    return filtered_data


def filter_no_exec_stocks(data, bgn_date: datetime, end_date: datetime):
    """
    出来がない(価格データがない)日が存在する銘柄を除外する

    Params
    ---------
    data: dict
        フィルタリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    bgn_date: datetime.datetime
        確認開始日
    end_date: datetime.datetime
        確認終了日

    Returns
    ---------
    0: dict
        フィルタリングした辞書データ
    """
    filtered_data = dict()

    for code, code_data in data.items():
        # パラメータで指定された日付範囲に絞る
        check_data = code_data[(bgn_date <= code_data['chart_date']) & (code_data['chart_date'] <= end_date)]
        if \
        (
            len(check_data[check_data['turnover'] == 0]) == 0 
            and len(check_data[check_data['open'] == -1]) == 0
        ):
            # 出来高が0の日がない かつ 始値が-1の日がない
            filtered_data[code] = code_data

    return filtered_data


def filter_disignated_stocks(data):
    exclude_file = 'file/data_exclude.lst'
    with open(exclude_file, 'r') as f:
        exclude_list = [code.replace('\n', '') for code in f.readlines()]

    filtered_data = dict(filter(lambda item: str(item[0]) not in exclude_list, data.items()))
    return filtered_data


def change_data_length(data, length, from_top=False):
    """
    データの長さを調節する

    Parameters
    ------------
    length : int
        調節後のデータの長さ
    from_top : bool
        データを前から切り出す(true) / 後ろから切り出す(false)

    Returns
    ---------
    0 :
        長さ調節後データ
    """
    for code in data:
        code_data = data[code]

        # 指定長のデータを切り出す
        if from_top:
            clipped_data = code_data[:length]
        else:
            clipped_data = code_data[-1 * length:]

        data[code] = clipped_data
    return data


def to_float_type(data, columns):
    """
    指定カラムをfloat型(dtype)に変換する

    Parameters
    -------------
    data : dict
        スケーリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    columns : list
        対象のカラム

    Returns
    ---------
    0 :
        変換済みのdata (key: 銘柄コード, value: pandas Dataframe)
    """
    for code in data:
        code_data = data[code]

        # 欠損値を直前の値で埋める
        code_data[columns] = code_data[columns].replace('－', np.nan)
        code_data = code_data.fillna(method='ffill')

        # さらに欠損値を直後の値で埋める
        code_data = code_data.fillna(method='bfill')

        # floatへ変換
        not_float_datapart = code_data[columns].select_dtypes(exclude=float)
        for column in not_float_datapart.columns:
            # replaceの為に、一度strを経由する
            code_data[column] = code_data[column].astype(str) \
                                                    .str.replace(',', '').astype(float)
        data[code] = code_data
    return data


def scaling(data, columns, scaler_dict, inverse_option=False, fit=True):
    """
    データのスケーリングを行う

    Parameters
    -------------
    data : dict
        スケーリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    columns : list
        対象のカラム
    scaler_dict : dict
        スケーラーの辞書 (key: 銘柄コード, value: scalerオブジェクト)
    inverse_option : bool
        False: スケーリングを行う True: スケーリングされたデータを復元する
    fit : bool
        スケーラーに対して、fitを行うか

    Returns
    ---------
    0 : dict
        スケーリング済のdata (key: 銘柄コード, value: pandas Dataframe)
    1 : dict
        scalerの辞書(key: 銘柄コード)
    """
    ret_scaler_dict = dict()
    for code in data:
        scaler = scaler_dict[code]
        code_data = data[code].copy()

        if inverse_option:
            processed_data = scaler.inverse_transform(code_data[columns])
        else:
            if fit:
                scaler.fit(code_data[columns])
            processed_data = scaler.transform(code_data[columns])

        # 指定カラムをスケーリング済値で上書き
        code_data.loc[:, tuple(columns)] = processed_data

        data[code] = code_data
        ret_scaler_dict[code] = scaler
    return data, ret_scaler_dict


def filter_columns(data, keep_columns):
    """
    カラムのフィルタリングを行う
    (指定カラム以外を削除する
    Parameters
    -------------
    data : dict
        フィルタリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    keep_columns : list
        残すカラム

    Returns
    ---------
    0 : dict
        フィルタリング済のdata (key: 銘柄コード, value: pandas Dataframe)
    """
    ret_data = data.copy()

    for code in data:
        code_data = data[code][keep_columns].copy()
        ret_data[code] = code_data
    return ret_data


def filter_data(data, condition_func):
    """
    データのフィルタリングを行う
    指定条件に合致する銘柄のデータのみ残す

    Prameters
    ---------
    data : dict
        フィルタリングを行うdata (key: 銘柄コード, value: pandas Dataframe)
    condition_func : function
        データ残存条件

    Returns
    ---------
    0 : dict
        フィルタリング済のdata (key: 銘柄コード, value: pandas Dataframe)
    -------
    """

    ret_data = dict()

    for code in data:
        code_data = data[code].copy()
        if condition_func(code_data):
            ret_data[code] = code_data
    return ret_data
    

def calc_price_difference(data, days=1, earlier_days=0):
    for code in data:
        tmp = data[code]

        for idx in range(1, days + 1):
            tmp['pd-{}'.format(idx)] = tmp['end'].diff(idx)
            for idx2 in range(1, idx):
                tmp['pd-{}'.format(idx)] = tmp['pd-{}'.format(idx)] - tmp['pd-{}'.format(idx2)]
        for idx in range(1, earlier_days + 1):
            tmp['pd+{}'.format(idx)] = -1 * tmp.sort_index(ascending=False)['end'].diff(idx)
            for idx2 in range(1, idx):
                tmp['pd+{}'.format(idx)] = tmp['pd+{}'.format(idx)] - tmp['pd+{}'.format(idx2)]
    return data


def get_other_day(data, column_list, prev_days=0, later_days=0):
    def get(d):
        for idx in range(-1 * prev_days, later_days + 1):
            if idx == 0:
                continue
            tmp = d.shift(-1 * idx)[column_list].rename(columns=dict(zip(column_list, map(lambda c: '{}{:+}'.format(c, idx), column_list))))
            d = pd.concat([d, tmp], axis=1)
        return d

    if isinstance(data, dict):
        for key, d in data.items():
            data[key] = get(d)
    else:
        data = get(data)
    return data


def calc_profit(data, days=1):
    for code in data:
        tmp = data[code]

        exec_price = tmp['begin'].replace('－', np.nan).fillna(method = 'ffill').astype(str)
        exec_price = exec_price.shift(-1)
        exec_price = exec_price.str.replace(',', '').astype(float)

        settle_price = pd.DataFrame()
        for idx in range(1, days + 1):
            settle_price = pd.concat([settle_price, tmp.shift(-1 * idx)[['end']].rename(columns={'end': 't+{}'.format(idx)})], axis=1) 
        settle_price = settle_price.max(axis=1)
        tmp['profit_{}'.format(days)] = settle_price - exec_price
    return data


def input_pbr():
    pbr = input_pbr_file()
    pbr = add_price_to_pbr('2020-08-06')
    pbr = delete_abnormal_value_from_pbr()
    pbr = calc_fair_price_from_pbr()
    return pbr


def input_pbr_file():
    global pbr
    pbr = pd.read_csv('save/pbr.csv', names=['code', 'pbr'], index_col=['code'])
    return pbr


# dateは yyyy-mm-dd
# 指定された日付の価格をもってきてpbrと連結したい
def add_price_to_pbr(date):
    global pbr

    if len(data) == 0:
        return pbr
    price_data = pd.DataFrame(index=[], columns=['code', 'price'])
    for code in data:
        try:
            df = pd.DataFrame({'code': [code], 'price': [data[code].at[date, 'end']]})
            price_data = pd.concat([price_data, df])
        except Exception as e:
            print(code)
            print('type:{}'.format(str(type(e))))
            print('args:' + str(e.args))
            continue

    pbr = pd.merge(pbr, price_data, on='code', how='left').set_index('code')
    return pbr


def delete_abnormal_value_from_pbr():
    global pbr
    pbr.loc[pbr['pbr'] == '－', ['pbr']] = None
    pbr = pbr.dropna()
    pbr = pbr.astype({'pbr': float})
    return pbr


def calc_fair_price_from_pbr():
    global pbr
    pbr['fair_price'] = pbr.price / pbr.pbr
    return pbr


def calc_moving_average(data: dict, days: int=10, column: str='end'):
    """
    移動平均とその標準偏差を算出する

    Params
    ---------
    data: dict
        算出対象のデータ
    days: int
        移動平均、標準偏差の算出日数
    column: str
        算出対象カラム

    Returns
    ---------
    算出後のデータ
        移動平均 => m_avg_日数_カラム名
        標準偏差 => m_std_日数_カラム名
    """
    def calc(d):
        rolling = d.rolling(days)
        d = pd.concat([d, rolling.mean()[[column]].rename(columns={column: 'm_avg_{}_{}'.format(days, column)})], axis=1)
        d = pd.concat([d, rolling.std()[[column]].rename(columns={column: 'm_std_{}_{}'.format(days, column)})], axis=1)
        return d

    for key, d in data.items():
        data[key] = calc(d)

    return data


def calc_std(data: dict, column: str, days: int=10):
    """
    指定カラムの標準偏差カラムを末尾に追加する

    Params
    ---------
    data: dict
        データ
    column: str
        カラム名
    days: int
        標準偏差算出日数
    """
    for code, d in data.items():
        rolling = d.rolling(days)
        std = rolling.std()[[column]].rename(columns={column: '{}_std_{}days'.format(column, days)})
        data[code] = pd.concat([d, std], axis=1)
    return data


#######################
# 指定カラムについて、直前データとの差分算出する
# data: data(dict形 or 単数pandas)
# column: 対象カラム
#######################
def calc_sub(data, column):
    def calc(d):
        d['diff_{}'.format(column)] = d[[column]].diff()
        return d

    if isinstance(data, dict):
        for key, d in data.items():
            data[key] = calc(d)
    else:
        data = calc(data)
    return data

#######################
# 指定日数分の過去データについて、条件を満たすデータ数をカウントする
# data: data(dict形 or 単数pandas)
# condition: 対象適合条件
# days: 検索対象日数(当日含む)
#######################
def count_compatible_data(data, condition, days):
    def count(d):
        d['condition'] = condition(d)
        d['compatible_count'] = d['condition'].rolling(days).sum()
        d = d.drop('condition', axis=1)
        return d

    if isinstance(data, dict):
        for key, d in data.items():
            data[key] = count(d)
    else:
        data = count(data)
    return data


def flatten_data(data):
    # コードの追加
    for code, val in data.items():
        val['code'] = code
    # 平面化    
    merge_data = pd.concat(data.values())
    return merge_data
