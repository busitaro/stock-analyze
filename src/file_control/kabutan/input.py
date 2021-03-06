import glob
import re
import warnings
from os.path import basename
from datetime import datetime
from statistics import mode

import numpy as np
import pandas as pd

from setting.config import Config


exclude_file = 'file/exclude.lst'


def input_price_file(exclude=False, exclude_by_count=False, raw=False):
    """
    price.csvファイルを読み込む

    Parameters
    -------------
    exclude : 
        exclude_fileに記載の銘柄コードを除外するかのフラグ
    exclude_by_count : 
        データ数不正の銘柄を除外するかのフラグ

    Returns
    ---------
    0 :
        読み込み結果の辞書 key: 銘柄コード, value: pandas Dataframe
    """
    config = Config()
    data = dict()
    # 除外銘柄ファイルの読込
    if exclude:
        with open(exclude_file, 'r') as f:
            exclude_list = [code.replace('\n', '') for code in f.readlines()]

    for file in glob.glob('{}/{}'.format(config.kabutan_input_path, 'price_*.csv')):
        # ファイル名からコードを取得
        code = re.sub('price_|.csv', '', basename(file))
    
        if exclude and code in exclude_list:
            continue
        try:
            d = pd.read_csv(
                    file, 
                    names=['date', 'open', 'high', 'low', 'close', 'compare', 'compare_rate', 'turnover'])

            if not raw:
            # 型変換等
                # indexの設定
                d['date'] = pd.to_datetime(d['date'], format='%y/%m/%d')
                d = d.set_index('date').sort_index()
                # 各項目の設定
                object_columns = d.select_dtypes(object).columns
                if 'open' in object_columns:
                    d['open'] = d['open'].str.replace('－', '-1')
                    d['open'] = d['open'].str.replace(',', '').astype(float)
                if 'high' in object_columns:
                    d['high'] = d['high'].str.replace('－', '-1')
                    d['high'] = d['high'].str.replace(',', '').astype(float)
                if 'low' in object_columns:
                    d['low'] = d['low'].str.replace('－', '-1')
                    d['low'] = d['low'].str.replace(',', '').astype(float)
                if 'close' in object_columns:
                    d['close'] = d['close'].str.replace(',', '').astype(float)
                if 'compare' in object_columns:
                    d['compare'] = d['compare'].str.replace('－', '0')
                    d['compare'] = d['compare'].str.replace(',', '').astype(float)
                if 'compare_rate' in object_columns:
                    d['compare_rate'] = d['compare_rate'].str.replace('－', '0')
                    d['compare_rate'] = d['compare_rate'].str.replace(',', '').astype(float)
                if 'turnover' in object_columns:
                    d['turnover'] = d['turnover'].str.replace(',', '').astype(int)

            data[int(code)] = d
        except Exception as ex:
            print(ex)
            print('input error code: {}'.format(code))

    if exclude_by_count:
        data = exclude_incorrect_data_count(data)

    return data


def exclude_incorrect_data_count(data, chk_rate=0.95):
    """
    データ数が不正の銘柄を除外する
    (データ数の最頻値と一致しない物を対象)

    Parameters
    -------------
    data : dict
        処理対象データ
    chk_rate : float
        警告発生除外率

    Returns
    ---------
    0 :
        除外済みのデータ

    Note
    -----
    除外率が警告発生除外率を超えた場合に、警告を発する
    """

    num_of_code = len(data.keys())                                                  # 総銘柄数
    mode_of_data_num = mode([len(data[code]) for code in data])      # 各銘柄ごとのデータ数の最頻値

    # データ数が最頻値と異なるデータを除外
    exclude_data = {code: data[code] for code in data if len(data[code]) == mode_of_data_num}

    # chk_rate以上に除外された場合、警告を表示する
    remain_rate = len(exclude_data) / num_of_code
    if remain_rate < chk_rate:
        warnings.warn(
           'データ不足により除外された銘柄コードが設定率を超えています。 残存率: {:.2f}'.format(remain_rate), 
            stacklevel=2
        )

    return exclude_data


def input_trade_file(raw=False, exclude=False):
    """
    trade.csvファイルを読み込む

    Parameters
    -------------
    exclude : 
        exclude_fileに記載の銘柄コードを除外するかのフラグ

    Returns
    ---------
    0 :
        読み込み結果の辞書 key: 銘柄コード, value: pandas Dataframe
    """
    config = Config()
    data = dict()
    date_parser=lambda date: datetime.strptime(date, '%Y-%m-%d')
    # 除外銘柄ファイルの読込
    if exclude:
        with open(exclude_file, 'r') as f:
            exclude_list = [code.replace('\n', '') for code in f.readlines()]

    for file in glob.glob('{}/{}'.format(config.kabutan_input_path, 'trade_*.csv')):
        # ファイル名からコードを取得
        code = re.sub('trade_|.csv', '', basename(file))
    
        if exclude and code in exclude_list:
            continue
        try:
            d = pd.read_csv(
                    file, 
                    names=['date', 'turnover', 'trading_price', 'vwap', 'execution_count', 'min_price', 'unit', 'market_capitalization', 'shares_outstanding'],
                )
            if not raw:
            # 型変換等
                # indexの設定
                d['date'] = pd.to_datetime(d['date'], format='%Y-%m-%d')
                d = d.set_index('date').sort_index()
                # 各項目の設定
                d['turnover'] = d['turnover'].str.replace('\xa0株', '').str.replace(',', '')
                d['turnover'] = d['turnover'].replace('－', np.nan).fillna('0').astype(int)
                d['trading_price'] = d['trading_price'].str.replace('\xa0百万円', '').str.replace(',', '')
                d['trading_price'] = d['trading_price'].replace('－', np.nan).fillna('0').astype(int)
                d['vwap'] = d['vwap'].str.replace('\xa0円', '').str.replace(',', '')
                d['vwap'] = d['vwap'].replace('－', np.nan).fillna('0').astype(float)
                d['execution_count'] = d['execution_count'].str.replace('\xa0回', '').str.replace(',', '').astype(int)
                d['min_price'] = d['min_price'].str.replace('\xa0円', '').str.replace(',', '').astype(int)
                d['unit'] = d['unit'].str.replace('\xa0株', '').str.replace(',', '').astype(int)

            data[int(code)] = d
        except:
            print('input error code: {}'.format(code))

    return data