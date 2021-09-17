import pandas as pd
import numpy as np
import talib


# 最原始的backtesting的表是用"简单版的回测（后续需要扩展）"实现的，周六整理的时候把它整理出来，顺便把下面这个函数也放进去
def get_atr_breakout_price(table_15_min, threshold_num):  # 出场或进场都适用？
    every_bar_range = table_15_min.High - table_15_min.Low
    ema_bar_range = pd.Series(talib.EMA(every_bar_range, timeperiod=14)).shift(1)
    std_bar_range = every_bar_range.rolling(14).std().shift(1)
    table_15_min['is_atr_breakout'] = every_bar_range > ema_bar_range + threshold_num * std_bar_range  # 添加上的表示当前的bar有没有发生突破
    table_15_min['previous_close_plus_multiple_range'] = table_15_min.Open + ema_bar_range + threshold_num * std_bar_range - (table_15_min.Open-table_15_min.Low)  # 添加上的表示当前bar如果进场，要进场的点位
    table_15_min['previous_close_minus_multiple_range'] = table_15_min.Open - ema_bar_range - threshold_num * std_bar_range + (table_15_min.High-table_15_min.Open)
    return table_15_min
    #


if __name__ == '__main__':
    table_15_min = pd.read_csv(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\backtesting_data.csv')
    table_15_min = get_atr_breakout_price(table_15_min, 1.618)
    table_15_min.to_csv('debug_whether_in_range.csv')
    print(table_15_min)
