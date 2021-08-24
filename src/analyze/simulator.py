import csv
import sys
import traceback
from os import getpid
from datetime import datetime

from injector import Injector, inject
from operator import add
from multiprocessing import Pool, Manager
from itertools import product
from psutil import cpu_count
import pandas as pd

from util.date import BusinessDay
from predict.predict import Predict
from di.modules import PredictDIModule
from database.control import truncate, insert


class Simulation():

    @inject
    def __init__(self, predict: Predict, output_path='./simulate'):
        """
        コンストラクタ

        Params
        ---------
        predict: Predict
            シミュレーション用データを保持するPredictクラス

        
        profit_list: list
            シミュレーションに利用する、利確する価格リスト
        loss_cut_std_list: list
            シミュレーションに利用する、損切する下落標準偏差のリスト
        """
        self.__predict = predict
        self.__output_path = output_path

    def run(self, bgn_date: datetime, end_date: datetime, profit_list: list, loss_cut_std_list: list, count: int):
        """
        simulatorを起動する

        Params
        ---------
        bgn_date: datetime
            シミューレーション開始日
        end_date: datetime
            シミューレーション終了日
        profit_list: list
            利確を行う1株当たりの利益
        loss_cut_std_list: list
            損切を行う価格下落率(標準偏差ベース)のリスト
        count: シミュレーション回数(各profit, loss_cutの組合せ毎)
        """
        pool = Pool(cpu_count())
        
        # 結果出力先テーブルをtruncate
        truncate('simulation')
        truncate('simulation_detail')

        # simulatorに渡すパラメータ
        # 利益、損切率のすべての組み合わせ(直積)
        parameters = product([bgn_date], [end_date], [count], profit_list, loss_cut_std_list)

        results = pool.map(self.wrap_start_simulator, parameters)
        pool.close()
        pool.join()

        for result in results:
            self.output_result(*result)

    def wrap_start_simulator(self, prm):
        """
        タプルで受け取ったパラメータを展開し、simulatorを開始する

        Params
        ---------
        prm: パラメータのタプル
        """
        # MEMO: このメソッド自体デコレータ化できるかも
        return self.start_simulator(*prm)

    def start_simulator(self, bgn_date, end_date, count, profit, loss_std):
        """
        シミュレータを起動する

        Params
        ---------
        bgn_date: datetime.datetime
            シミュレーション開始日
        end_date: datetime.datetime
            シミュレーション終了日
        count: int
            シミュレーション回数
        profit: int
            利確価格
        loss_std: float
            損切り下落標準偏差
        """
        try:
            # シミュレーション結果のDataFrame
            result_df = pd.DataFrame()
            # 取引履歴のDataFrame (すべての取引履歴を連結)
            histories_df = pd.DataFrame()

            # # 合計損益のリスト([1回目結果, 2回目結果, 3回目結果, ...])
            # amount_profit_list = list()
            # # 残存銘柄のend_dateベースの評価損益のリスト([1回目結果, 2回目結果, 3回目結果, ...])
            # remain_profit_list = list()

            for idx in range(count):
                result = self.simulate(bgn_date, end_date, profit, loss_std)
                # シミュレーション結果
                result_sim = {
                    "take_profit": profit,
                    "losscut_rate": loss_std,
                    "time": idx,
                    "fixed_profit": result[1],
                    "unsettled_profit": result[2],
                    "sum_profit": result[1] + result[2]
                }
                result_df = result_df.append(result_sim, ignore_index=True)

                # 取引履歴
                history_df = result[0]
                history_df['time'] = idx
                histories_df = pd.concat([histories_df, history_df])

                # amount_profit_list.append(result[1])
                # remain_profit_list.append(result[2])

            return result_df, histories_df
        except Exception as e:
            print('==== {} ====\n{}'.format(getpid(), traceback.format_exc()))

    def simulate(self, bgn_date: datetime, end_date: datetime, profit: float, loss_std: float):
        """
        Params
        ---------
        bgn_date: datetime.datetime
            シミュレーション開始日
        end_date: datetime.datetime
            シミュレーション終了日
        profit: float
            利確基準(1株あたり利益)
        loss_std: float
            損切基準(下落標準偏差率)

        Returns
        ---------
        0: pd.DataFrame
            売買履歴のリスト(history_list)
            リストの各要素は辞書
                {'buy_date': 購入日,
                'code': 銘柄コード,
                'buy_price': 購入金額,
                'std': 購入時の始値の標準偏差,
                'settle_date': 決済日
                'benefit': 決済損益(1単元は仮で100株)}
        1: int
            決算損益の合計(1単元は仮で100株)
        2: int
            未決済銘柄のend_date時点での評価損益(1単元は仮で100株)
        """

        # 各銘柄の単元数
        # MEMO: (仮で100固定)
        unit = 100

        date = bgn_date
        bs_day = BusinessDay()

        history_df = pd.DataFrame()
        amount_profit = 0
        unsettled_profit = 0
        seq = 0

        while date <= end_date:
            # 買付
            exec = self.__predict.select_purchase_stock(bs_day.previous(date))
            if exec is None:
                # 買付条件に一致する銘柄がなかった場合、
                # 日付を進めて次ループへ
                date = bs_day.next(date)
                continue
            # 銘柄コード, 購入金額, 購入時の始め値の標準偏差
            code = exec[0]
            buy_price, std = self.__predict.get_price_and_std(date, code)
            history = {
                'take_profit': profit,
                'losscut_rate': loss_std,
                'seq': seq,
                'description_code': code,
                'buy_date': date.strftime('%Y/%m/%d'),
                'buy_price': buy_price,
                'std': std
            }

            # 売却
            while date <= end_date:
                # 損益
                now_profit = self.__predict.calc_profit(date, code, buy_price)
                if now_profit >= profit:
                    # 利確条件が満たされた場合
                    history['sell_date'] = date.strftime('%Y/%m/%d')
                    history['sell_price'] = buy_price + profit
                    history['profit'] = profit * unit  # あくまでprofitの価格で決済される(指値を想定)
                    amount_profit += profit * unit
                    # 日付を進める
                    date = bs_day.next(date)
                    break
                elif now_profit / std <= loss_std:
                    # 損切条件が満たされた場合
                    history['sell_date'] = date.strftime('%Y/%m/%d')
                    history['sell_price'] = buy_price + now_profit
                    # 現在の損益で損切
                    history['profit'] = now_profit * unit
                    amount_profit += now_profit * unit
                    # 日付を進める
                    date = bs_day.next(date)
                    break
                else:
                    # 日付を進める
                    date = bs_day.next(date)
            else:
                # end_dateまで売却条件が成立しなかった場合
                # history['settle_date'] = ''
                # history['benefit'] = 0
                # history['sell_price'] = ''
                unsettled_profit = now_profit * unit

            history_df = history_df.append(history, ignore_index=True)
            seq = seq + 1
        return history_df, amount_profit, unsettled_profit

    def output_result(self, result_df: pd.DataFrame, histories_df: pd.DataFrame):
        """
        シミューレーション結果を出力する
        各profit, margin_cut_stdの組にて、指定回数のシミュレーション終了後に呼ばれる

        Params
        ---------
        profit: int
            利確価格
        loss_std: float
            損切り下落標準偏差
        result_df: float
            結果を保持するDataFrame
        histories_df: pd.DataFrame
            売買履歴を保持するDataFrame
        """
        # resultの出力
        insert('simulation', result_df)

        # historyの出力
        insert('simulation_detail', histories_df)


def main(bgn_date, end_date, count):
    
    """
    1株あたりの利益額のリスト
    この額の利益が出た場合に決済する

    """
    profit_list = [20, 10]

    """
    損切りの値下がり率
    標準偏差いくつ分かで判断する

    """
    margin_cut_std_list = [-5.0, -4.0]

    injector = Injector([PredictDIModule()])
    simulation = injector.get(Simulation)

    simulation.run(bgn_date, end_date, profit_list, margin_cut_std_list, count)


if __name__ == '__main__':
    """
    Params
    ---------
    $1:     シミューレーション開始日(yyyymmdd)
    $2:     シミューレーション終了日(yyyymmdd)
    $3:     シミュレーション回数
              (利益額、損失率の各組合せ毎の実行回数)
    """

    args = sys.argv
    if len(args) < 4:
        print('parameters are required for simulation')
    else:
        bgn_date = datetime.strptime(args[1], '%Y%m%d')
        end_date = datetime.strptime(args[2], '%Y%m%d')
        count = int(args[3])
        main(bgn_date, end_date, count)
