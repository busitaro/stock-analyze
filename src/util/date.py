from datetime import datetime, timedelta

import pandas as pd


class BusinessDay:
    file_path = 'file/business_day.csv'

    def __init__(self):
        #csvを読み込む
        self.__file = pd.read_csv(
                self.file_path, 
                names=['date', 'business'],
                parse_dates=['date'], 
                date_parser=lambda date: datetime.strptime(date, '%Y/%m/%d')).sort_index()

    def next(self, date):
        """
        指定された日付の翌営業日を返却
        (dateより大きい日付 かつ 営業日の最小日付を返す)

        Params
        ---------
        date: datetime.datetime
            翌営業日を探す基準日付

        Returns
        ---------
        0: datetime.datetime
            翌営業日
        """
        # date以降で、営業日のデータを抽出
        after_dates = self.__file[(self.__file.date > date) & (self.__file.business)]

        if len(after_dates) == 0:
            raise ValueError('指定された日付の営業日データがありません')

        # datetime.datetimeにして返却
        return after_dates.iloc[0].date.to_pydatetime()

    def previous(self, date):
        """
        指定された日付の前営業日を返却
        (dateより小さい日付 かつ 最大の営業日を返す)

        Params
        ---------
        date: datetime.datetime
            前営業日を探す基準日付

        Returns
        ---------
        0: datetime.datetime
            前営業日
        """
        # date以前の営業日のデータを抽出
        before_dates = self.__file[(self.__file.date < date) & (self.__file.business)]

        if len(before_dates) == 0:
            raise ValueError('指定された日付の営業日データがありません')

        # datetime.datetimeにして返却
        return before_dates.iloc[-1].date.to_pydatetime()


def between(self, bgn_date: datetime, end_date: datetime):
        """
        指定された日付間の営業日のリストを取得する

        Params
        ---------
        bgn_date: datetime.datetime
            開始日
        end_date: datetime.datetime
            終了日

        Returns
        ---------
        0: list
            営業日のリスト
        """
        business_day_list = list(
            self.__file[
                (self.__file.date >= bgn_date) & (self.__file.date <= end_date) & (self.__file.business)
            ]['date']
        )
        return business_day_list
