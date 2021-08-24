from typing import Tuple
from datetime import datetime
from abc import ABCMeta, abstractmethod


class Predict(metaclass=ABCMeta):
    """
    予測の抽象クラス

    """

    @abstractmethod
    def select_purchase_stock(self, date: datetime) -> Tuple[int, float, float]:
        """
        指定された日付のデータ買付銘柄を選定
        (翌営業日に買い付けるべき銘柄を予測)
        銘柄コード、予測買付価格、予測売却価格
        存在しない場合にはNoneを返す

        Params
        ---------
        date: datetime
            買付銘柄選定日付

        Returns
        ---------
        0: int
            購入銘柄コード
        1: float
            予測買付価格
        2: float
            予測売却価格
        """
        raise NotImplementedError()

    def get_price_and_std(self, date: datetime, code: int) -> Tuple[float, float]:
        """
        指定日付における銘柄の購入金額、購入時標準偏差を返却する

        Params
        ---------
        date: datetime
            購入日付
        code: 
            購入銘柄コード

        Returns
        ---------
        0: float
            購入価格
        1: float
            購入時の標準偏差
        """
        raise NotImplementedError()

    def calc_profit(self, date: datetime, code: int, buy_price: float) -> float:
        """
        指定日の1株あたり評価益(終値ベース)を算出する

        Params
        ---------
        date: datetime
            損益を評価する日付
        code: int
            銘柄コード
        buy_price: float
            購入金額

        Returns
        ---------
        0: float
            1株あたり評価益
        """
        raise NotImplementedError()
