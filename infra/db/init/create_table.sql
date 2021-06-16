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
