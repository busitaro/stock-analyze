import pandas as pd

from database.control import insert
from .input import input_file


def import_db():
    # MEMO: SBIファイルは、出来がない日はレコードがない

    # input file
    data_dict = input_file(exclude=False)

    insert_data = pd.DataFrame()

    for code, d in data_dict.items():
        # pickup columns
        d = d.reset_index()
        d = d[['date', 'open', 'high', 'low', 'close', 'turnover', 'vwap']]
        d = d.rename(columns={'date': 'chart_date'})
        d['description_code'] = code
        d['execution_count'] = -1
        # add to insert data
        insert_data = pd.concat([insert_data, d])

    # vwap が空白の箇所を-1に置換
    insert_data.loc[insert_data['vwap'] == '         --', 'vwap'] = -1
    # DBへ挿入
    insert('daily_chart', insert_data)
