import glob
import re
import warnings
from os.path import basename
from datetime import datetime

import numpy as np
import pandas as pd

from setting.config import Config


exclude_file = 'file/exclude.lst'


def input_file(exclude=False):
    """
    SBIのprice.csvファイルを読み込む

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
    date_parser=lambda date: datetime.strptime(date, '%Y%m%d')
    # 除外銘柄ファイルの読込
    if exclude:
        with open(exclude_file, 'r') as f:
            exclude_list = [code.replace('\n', '') for code in f.readlines()]

    for file in glob.glob('{}/{}'.format(config.sbi_input_path, '*.csv')):
        # ファイル名からコードを取得
        code = re.sub('.csv', '', basename(file))
    
        if exclude and code in exclude_list:
            continue
        try:
            data[int(code)] = \
                pd.read_csv(
                    file, 
                    names=['date', 'open', 'high', 'low', 'close', 'close_avg_5', 'close_avg_25', 'close_avg_75', 'vwap', 'turnover', 'turnover_avg_5', 'turnover_avg_25'],
                    parse_dates=['date'], 
                    index_col=['date'], 
                    date_parser=date_parser, 
                    thousands=',',
                    skiprows=1,
                    encoding = 'shift-jis').sort_index()
        except:
            print('input error code: {}'.format(code))

    return data
