from csv import QUOTE_ALL

import pandas as pd

from setting.config import Config
from database.control import insert
from database.models import DailyChart
from .input import input_price_file, input_trade_file


def delete_error_data_of_trade_file():
    """
    trade_code.csvから、日付インデックスが空の行を削除する

    """
    data = input_trade_file(raw=True)

    for code, d in data.items():
        # dateがNanの行を削除
        d = d[~d['date'].isnull()]

        # 削除後のデータをファイル出力
        config = Config()
        d.to_csv(
            '{}/{}'.format(config.kabutan_input_path, 'trade_{}.csv'.format(code)),
            sep=',', encoding='utf-8', index=False, header=False, quotechar='"', quoting=QUOTE_ALL
        )


def import_db():
    # ファイルを読み込む
    price_data = input_price_file(exclude=True)
    trade_data = input_trade_file(exclude=True)

    # 読み込んだファイルを結合し、DBカラムに合わせる
    insert_data = pd.DataFrame()
    for code in price_data:
        # データを取得し、インデックスでマージ
        price = price_data[code]
        if len(price) == 0:
            continue
        trade = trade_data.get(code, None)
        if trade is None:
            continue
        trade = trade.drop('turnover', axis=1)
        merge_pd = pd.merge(price, trade, left_index=True, right_index=True, how='inner').reset_index()

        # 必要カラムを抽出
        merge_pd = merge_pd[['date', 'open', 'high', 'low', 'close', 'turnover', 'vwap', 'execution_count']]
        merge_pd = merge_pd.rename(columns={'date': 'chart_date'})
        merge_pd['description_code'] = code

        insert_data = pd.concat([insert_data, merge_pd])

    # DBに存在するデータを取得し、不足分のみインサート
    begin_date = insert_data.min()['chart_date']
    end_date = insert_data.max()['chart_date']
    stored_data = DailyChart.date_between(begin_date, end_date)[['chart_date']]
    stored_data['exist'] = True
    insert_data = pd.merge(insert_data, stored_data, how='left', left_on='chart_date', right_on='chart_date')
    insert_data = insert_data[insert_data['exist'].isnull()].drop('exist', axis=1)

    # DBへ挿入
    insert('daily_chart', insert_data)
