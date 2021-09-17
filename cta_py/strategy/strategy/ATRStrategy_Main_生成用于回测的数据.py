import pandas as pd
import numpy as np
import talib
import datetime
# import features.zigzag as fz
from backtesting.BackTesting.datahub.zigzag import *
from tools.indicators.Indicators import *
from tools.indicators.Bars import *
from strategy.functionsforATRRegression import *
# from strategy.ATRStrategy import *



def import_data_source():
    table_15_min_sub1 = pd.read_csv('C:/Users/Administrator/Desktop/pythonHistoricalTesting/data/SCData/YMH21-CBOT_15min_from_2015_01_01_to_2017_12_31.txt',
                               parse_dates={'Time': [0, 1]})
    table_15_min_sub2 = pd.read_csv('C:/Users/Administrator/Desktop/pythonHistoricalTesting/data/SCData/YMH21-CBOT_start_from_2018-01-01_15min.txt',
                               parse_dates={'Time': [0, 1]})
    table_15_min = pd.concat([table_15_min_sub1, table_15_min_sub2], axis=0)
    table_15_min = table_15_min.reset_index(drop=True)
    table_15_min.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'NumberOfTrades', 'BidVolume', 'AskVolume']
    return table_15_min



# EMA34 = talib.EMA(table_15_min.Close, timeperiod=34)
# EMA55 = talib.EMA(table_15_min.Close, timeperiod=55)
# table_15_min['EMA_type'] = (EMA34 - EMA55) > 0  # 如果是EMA多头排列，记为True，空头排列，记为False

# 直接导入SC的60min的rsi14数据
table_60_rsi14 = pd.read_csv(
    'C:/Users/Administrator/Desktop/pythonHistoricalTesting/data/SCData/YMH21-rsi14-60 Min.txt',
    parse_dates={'Time': [0, 1]})
# 下面这个函数是错的
def add_bar60rsi14(table_15_min):
    bar60rsi14 = []
    for time in table_15_min.Time:
        if (pd.to_datetime(time).strftime("%H:%M:%S") >= "08:30:00") & (
                pd.to_datetime(time).strftime("%H:%M:%S") < "17:00:00"):
            if pd.to_datetime(time).strftime("%M:%S") >= "30:00":
                bar60time = pd.to_datetime(time).strftime("%Y-%m-%d %H:30:00")
                bar60rsi14_sub = table_60_rsi14.loc[table_60_rsi14.Time == bar60time][' RSI'].values
            else:
                bar60time = (pd.to_datetime(time) - datetime.timedelta(seconds=3600)).strftime("%Y-%m-%d %H:30:00")
                bar60rsi14_sub = table_60_rsi14.loc[table_60_rsi14.Time == bar60time][' RSI'].values
        else:
            bar60time = pd.to_datetime(time).strftime("%Y-%m-%d %H:00:00")
            bar60rsi14_sub = table_60_rsi14.loc[table_60_rsi14.Time == bar60time][' RSI'].values

        if len(bar60rsi14_sub) > 0:
            bar60rsi14.append(bar60rsi14_sub[0])
        else:
            bar60rsi14.append(-1)  # 填充一个不可能的rsi，比如-1
    table_15_min["bar60rsi14"] = bar60rsi14

# Keltner_Channel(table_15_min.Open, table_15_min, 13, 8, 1.618, 1.618, ma="EMA")
#
# table_15_min["RSI"] = talib.RSI(table_15_min.Open, timeperiod=14)  # 这边统一都用open的数据了
# AddRSI_T(table_15_min, 2)
# AddRSI_T(table_15_min, 7)
#
# RSI_SMA_shortgo(table_15_min, 3)
# Todo 等程序跑起来之后再来处理
def zzg_related():
    zig = fz.ZigZag(table_15_min, 89, mode='amount')
    df_only_last = zig.fixed_zzps()
    df_his_listed = zig.fixed_zzps(add_once=True)  # 历史被擦除的也显示了
    df_only_last['High_or_Low'] = [str(df_only_last.type_.loc[i]) for i in range(len(df_only_last))]
    df_his_listed['High_or_Low'] = [str(df_his_listed.type_.loc[i]) for i in range(len(df_his_listed))]


    AddRSIDivergence(table_15_min, df_his_listed)

    # 需要根据table_15_min时间，找到上一个极值点处有没有发生背离
    Extreme_BottomDivergence = []
    Extreme_TopDivergence = []
    for i in range(len(table_15_min)):
        Extreme_BottomDivergence_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].RSI_BottomDivergence
        Extreme_TopDivergence_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].RSI_TopDivergence
        Extreme_BottomDivergence.append(Extreme_BottomDivergence_sub)
        Extreme_TopDivergence.append(Extreme_TopDivergence_sub)
    table_15_min['Extreme_BottomDivergence'] = Extreme_BottomDivergence
    table_15_min['Extreme_TopDivergence'] = Extreme_TopDivergence

    IdentifyPinBar(table_15_min)

    Extreme_UpPinBar = []
    Extreme_DownPinBar = []
    for i in range(len(table_15_min)):
        Extreme_UpPinBar_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].UpPinBar
        Extreme_DownPinBar_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].DownPinBar
        Extreme_UpPinBar.append(Extreme_UpPinBar_sub)
        Extreme_DownPinBar.append(Extreme_DownPinBar_sub)
    table_15_min['Extreme_UpPinBar'] = Extreme_UpPinBar
    table_15_min['Extreme_DownPinBar'] = Extreme_DownPinBar
    IdentifyTopBottomType(table_15_min)

    Extreme_TopType = []
    Extreme_BottomType = []
    for i in range(len(table_15_min)):
        Extreme_TopType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].TopType
        Extreme_BottomType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].BottomType
        Extreme_TopType.append(Extreme_TopType_sub)
        Extreme_BottomType.append(Extreme_BottomType_sub)
    table_15_min['Extreme_TopType'] = Extreme_TopType
    table_15_min['Extreme_BottomType'] = Extreme_BottomType
    IdentifyPregnantType(table_15_min)

    Extreme_UpPregnantType = []
    Extreme_DownPregnantType = []
    for i in range(len(table_15_min)):
        Extreme_UpPregnantType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].UpPregnantType
        Extreme_DownPregnantType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].DownPregnantType
        Extreme_UpPregnantType.append(Extreme_UpPregnantType_sub)
        Extreme_DownPregnantType.append(Extreme_DownPregnantType_sub)
    table_15_min['Extreme_UpPregnantType'] = Extreme_UpPregnantType
    table_15_min['Extreme_DownPregnantType'] = Extreme_DownPregnantType
    IdentifyTriplePregnantType(table_15_min)

    Extreme_UpTriplePregnantType = []
    Extreme_DownTriplePregnantType = []
    for i in range(len(table_15_min)):
        Extreme_UpTriplePregnantType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].UpTriplePregnantType
        Extreme_DownTriplePregnantType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].DownTriplePregnantType
        Extreme_UpTriplePregnantType.append(Extreme_UpTriplePregnantType_sub)
        Extreme_DownTriplePregnantType.append(Extreme_DownTriplePregnantType_sub)
    table_15_min['Extreme_UpTriplePregnantType'] = Extreme_UpTriplePregnantType
    table_15_min['Extreme_DownTriplePregnantType'] = Extreme_DownTriplePregnantType
    IdentifySwallowType(table_15_min)

    Extreme_UpSwallowType = []
    Extreme_DownSwallowType = []
    for i in range(len(table_15_min)):
        Extreme_UpSwallowType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].UpSwallowType
        Extreme_DownSwallowType_sub = table_15_min.loc[lastExtremeBar(i, df_his_listed)[0]].DownSwallowType
        Extreme_UpSwallowType.append(Extreme_UpSwallowType_sub)
        Extreme_DownSwallowType.append(Extreme_DownSwallowType_sub)
    table_15_min['Extreme_UpSwallowType'] = Extreme_UpSwallowType
    table_15_min['Extreme_DownSwallowType'] = Extreme_DownSwallowType
    TR_SMA_BBand(table_15_min, 14, 1)

    isHighorLowthan_n_sigma(table_15_min, 10, 2)  # 这边有个问题，文档里面写的是用sma，但是我这里还是用了ema，后续改一下

# Heikin_Ashi(table_15_min, SCCP="YES")

# 添加TR
# 添加dayopen的tag
def add_realtime_dayrange(table_15_min):
    table_15_min['hms'] = table_15_min.Time.apply(lambda x: x.strftime("%H:%M:%S"))
    open_bar = table_15_min[table_15_min.hms == "17:00:00"].index.values.tolist()  # 后续用iloc
    close_bar = (np.array(open_bar[1:] + [table_15_min.index[-1] + 1]) - 1).tolist()
    day_range = [0] * open_bar[0]
    for i, j in zip(open_bar, close_bar):
        day_high = table_15_min.iloc[i].High
        day_low = table_15_min.iloc[i].Low
        for m in range(i, j + 1):# 因为range是取不到最后一个的，所以这边要+1
            high = table_15_min.iloc[m].High
            low = table_15_min.iloc[m].Low

            if high > day_high:
                day_high = high

            if low < day_low:
                day_low = low
            day_range_sub = day_high - day_low
            day_range.append(day_range_sub)

    table_15_min['day_range'] = day_range
    return table_15_min

# 添加last_day_range
def add_last_day_range(table_15_min):
    open_bar = table_15_min[table_15_min.hms == "17:00:00"].index.values.tolist()
    last_day_range = [0] * open_bar[0]
    close_bar = (np.array(open_bar[1:] + [table_15_min.index[-1] + 1]) - 1).tolist()
    for i, j in zip(open_bar, close_bar):
        # 获取i的前一个值
        if i == 0:
            last_day_range_sub = [0] * (j - i + 1)
            last_day_range.extend(last_day_range_sub)
        else:
            last_day_range_sub = [table_15_min.loc[i - 1].day_range] * (j - i + 1)
            last_day_range.extend(last_day_range_sub)

    table_15_min['last_day_range'] = last_day_range
    return table_15_min

# 添加range_ratio
def add_dayrange_ratio(table_15_min):
    table_15_min['range_ratio'] = table_15_min.day_range / table_15_min.last_day_range
    find_19_range_ratio = table_15_min[table_15_min.hms == "17:00:00"].index.values

    open_bar = table_15_min[table_15_min.hms == "17:00:00"].index.values.tolist()  # 后续用iloc
    close_bar = (np.array(open_bar[1:] + [table_15_min.index[-1] + 1]) - 1).tolist()

    realtime_zscore = [0] * open_bar[0]
    for i, j in zip(open_bar, close_bar):
        for m in range(i, j + 1):
            if len(find_19_range_ratio[(find_19_range_ratio - m) < 0]) >= 19:
                range_ratio_list = [table_15_min.loc[m].range_ratio]
                for n in find_19_range_ratio[(find_19_range_ratio - m) < 0][-19:]:
                    range_ratio_list.append(table_15_min.loc[n].range_ratio)
                range_ratio_mean = np.mean(range_ratio_list)
                range_ratio_std = np.std(range_ratio_list)
                realtime_zscore_sub = (table_15_min.loc[m].range_ratio - range_ratio_mean) / range_ratio_std
                realtime_zscore.append(realtime_zscore_sub)
            else:
                realtime_zscore.append(0)
    table_15_min['realtime_zscore'] = realtime_zscore
    return table_15_min

# 添加last_zscore
def add_lastday_zscore(table_15_min):
    open_bar = table_15_min[table_15_min.hms == "17:00:00"].index.values.tolist()  # 后续用iloc
    close_bar = (np.array(open_bar[1:] + [table_15_min.index[-1] + 1]) - 1).tolist()
    last_zscore = [0] * open_bar[0]
    for i, j in zip(open_bar, close_bar):
        # 获取i的前一个值
        if i == 0:
            last_zscore_sub = [0] * (j - i + 1)
            last_zscore.extend(last_zscore_sub)
        else:
            last_zscore_sub = [table_15_min.loc[i - 1].realtime_zscore] * (j - i + 1)
            last_zscore.extend(last_zscore_sub)
    table_15_min['last_zscore'] = last_zscore
    return table_15_min

# 导入df_D
# 包含有Tag和Tag_tomorrow
# df_D = pd.read_csv(r"F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\stat_df.csv")
# his_table = ATRRegression(table_15_min, df_his_listed, df_D, max_lots=10)[0]
#
# his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
# his_table["Profits"] = [(ExitPrice - EntryPrice) * Lots if OrderType == "long" else (EntryPrice - ExitPrice) * Lots for
#                         EntryPrice, ExitPrice, OrderType, Lots in
#                         zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
#
# print(his_table)
# print(np.cumsum(his_table.Profits))



if __name__ == '__main__':
    table_15_min = import_data_source()
    table_15_min = Heikin_Ashi(table_15_min, SCCP="YES")
    table_15_min = add_realtime_dayrange(table_15_min)
    table_15_min = add_last_day_range(table_15_min)
    table_15_min = add_dayrange_ratio(table_15_min)
    table_15_min = add_lastday_zscore(table_15_min)
    print(table_15_min)
    table_15_min.to_csv('back.csv')
