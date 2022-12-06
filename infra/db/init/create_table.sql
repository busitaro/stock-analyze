/* 日足 */
DROP TABLE IF EXISTS daily_chart;
CREATE TABLE daily_chart (
    chart_date date NOT NULL   -- 日付
  , description_code int(5) NOT NULL   -- 銘柄コード
  , open float(8,1) NOT NULL   -- 始値
  , high float(8,1) NOT NULL   -- 高値
  , low float(8,1) NOT NULL   -- 安値
  , close float(8,1) NOT NULL   -- 終値
  , turnover int(10) NOT NULL   -- 出来高
  , vwap float(10, 3) NOT NULL   -- VWAP
  , execution_count int(10) NOT NULL   -- 約定回数
  , PRIMARY KEY (chart_date, description_code)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;

/* シミュレーション売買履歴 */
DROP TABLE IF EXISTS simulation_detail;
CREATE TABLE simulation_detail (
    take_profit int(4) NOT NULL   -- 利確利益
  , losscut_rate float(6,2) NOT NULL   -- 損切基準
  , time int(5) NOT NULL   -- 回
  , seq int(3) NOT NULL   -- 連番
  , description_code int(5) NOT NULL   -- 銘柄コード
  , buy_date date NOT NULL   -- 購入日
  , sell_date date    -- 売却日
  , buy_price float(8,1) NOT NULL   -- 購入金額
  , sell_price float(8,1)    -- 売却金額
  , std float(6,2) NOT NULL   -- 購入時標準偏差
  , profit float(16,1)    -- 損益額
  , PRIMARY KEY (take_profit, losscut_rate, time, seq)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;

/* シミューレーション */
DROP TABLE IF EXISTS simulation;
CREATE TABLE simulation (
    take_profit int(4) NOT NULL   -- 利確利益
  , losscut_rate float(6,2) NOT NULL   -- 損切基準
  , time int(5) NOT NULL   -- 回
  , fixed_profit int(8) NOT NULL   -- 確定利益
  , unsettled_profit int(8)    -- 未決済損益
  , sum_profit int(8) NOT NULL   -- 合計損益
  , PRIMARY KEY (take_profit, losscut_rate, time)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;
