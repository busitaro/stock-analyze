import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_control.data import calc_price_difference, calc_moving_average


def show_graph(code_data, columns, colors):
    """
    １銘柄の時系列データに関して、グラフを描画する

    Parameters
    -------------
    data : pd.Dataframe
        グラフ描画を行うデータ
    columns : list
        対象のカラムのリスト
    colors : list
        各カラムの描画色

    Returns
    ---------
        
    """
    plt.figure(figsize=(14, 8))
    plt.xlabel('date')
    plt.ylabel('value')
    graph_cnt = len(columns)
    
    for idx, (column, color) in enumerate(zip(columns, colors)):
        plt.subplot(graph_cnt, 1, idx + 1)
        plt.plot(code_data.index, code_data[column], color=color, label=column)
        plt.legend(loc='upper right')
    plt.show()


def show_graph_for_pred_test(code_data, pred_data, column, code_color, pred_color):
    """
    １銘柄の時系列データに関して、実際値、予測値のグラフを描画する

    Params
    ---------
    code_data: pd.Dataframe
        実際値のデータ
    pred_data: pd.Dataframe
        予測値のデータ
    column: str
        描画対象のカラム
    code_color: str
        実際値の描画色
    pred_color: str
        予測値の描画色
    """
    plt.figure(figsize=(14, 8))
    plt.xlabel('date')
    plt.ylabel('value')

    plt.plot(code_data.index, code_data[column], color=code_color, label='org_data')
    plt.plot(pred_data.index, pred_data[column], color=pred_color, label='forecast_data')
    
    plt.legend(loc='upper right')
    plt.show()
