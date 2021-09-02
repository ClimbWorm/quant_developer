import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import talib


# 筛选中用到的技术指标放在features.Bars中，未包含在此文档中
# 此文档的函数用于处理矢量化回测的预先数据准备
# def ATR_EMA_BBand(table_n_min,T,n):
#     '''
#     table_n_min在这里为15min的数据
#     n表示是几倍标准差
#     '''

#     TR = table_n_min.High - table_n_min.Low
#     ATR = talib.EMA(TR,timeperiod = T)
#     rolling_TR_std = pd.Series(TR).rolling(T).std()
#     ATR_bottom_line = ATR - n * rolling_TR_std
#     ATR_top_line = ATR + n * rolling_TR_std
#     return table_n_min.Time,TR,ATR_top_line,ATR_bottom_line

def TR_SMA_BBand(table_n_min, T, n):
    '''
    table_n_min在这里为15min的数据
    n表示是几倍标准差
    '''

    TR = table_n_min.High - table_n_min.Low
    SMA_price = talib.SMA(table_n_min.Open, T)
    # ATR = talib.EMA(TR,timeperiod = T)
    rolling_TR_std = pd.Series(TR).rolling(T).std()
    TR_bottom_line = SMA_price - n * rolling_TR_std
    TR_top_line = SMA_price + n * rolling_TR_std
    return table_n_min.Time, TR, TR_top_line, TR_bottom_line


# 下面这个函数比较的应该要是TR大于一倍TR标准差
# def isTRmorethan_n_sigma(table_n_min,T,n):
#     ATR_info = TR_SMA_BBand(table_n_min,T,n)
#     TR = ATR_info[1]
#     ATR_top_line = ATR_info[2].shift(1)
#     table_n_min['TR'] = TR
#     table_n_min['ATR_top_line'] = ATR_top_line
#     table_n_min['is_TR_mt_ATR_{}_top_line'.format(n)] = (TR - ATR_top_line) > 0#用今天的tr和昨天的top line做比较
#     return table_n_min

def isHighorLowthan_n_sigma(table_n_min, T, n):
    ATR_info = TR_SMA_BBand(table_n_min, T, n)
    TR = ATR_info[1]
    ATR_top_line = ATR_info[2]
    ATR_bottom_line = ATR_info[3]
    table_n_min['TR'] = TR
    table_n_min['ATR_{}_top_line'.format(n)] = ATR_top_line
    table_n_min['ATR_{}_bottom_line'.format(n)] = ATR_bottom_line
    table_n_min['is_High_mt_ATR_{}_top_line'.format(n)] = (table_n_min.High - ATR_top_line) > 0
    table_n_min['is_Low_lt_ATR_{}_bottom_line'.format(n)] = (ATR_bottom_line - table_n_min.Low) > 0
    return table_n_min


def Heikin_Ashi(table_n_min, SCCP="YES"):
    '''
    table_n_min为用1min拼接好的n_min的数据
    SCCP = Set Close to Current Price for Last Bar
    '''

    df_to_array = np.array(table_n_min.loc[:, ['Open', 'High', 'Low', 'Close']])
    if SCCP == "YES":
        HA_Close = np.append(np.mean(df_to_array, axis=1)[:-1], table_n_min.Close.iloc[-1])  # sc中用的是yes
    else:
        HA_Close = np.mean(df_to_array, axis=1)  # Set Close to Current Price for Last Bar为no的情况

    HA_Open = []
    HA_Open_sub = table_n_min.Open.iloc[0]
    for i in range(len(HA_Close)):
        HA_Open.append(HA_Open_sub)
        HA_Open_sub = np.mean([HA_Open_sub, HA_Close[i]])

    HA_High = np.max(np.array([table_n_min.High, HA_Open]), axis=0)
    HA_Low = np.min(np.array([table_n_min.Low, HA_Open]), axis=0)

    # OHLC = [(Time,Open,High,Low,Close) for Time,Open,High,Low,Close in zip(table_n_min.Time,HA_Open,HA_High,HA_Low,HA_Close)]
    # df_OHLC = DataFrame(OHLC,columns = ["Time","Open","High","Low","Close"])
    # return df_OHLC
    table_n_min["HA_Open"] = HA_Open
    table_n_min["HA_Close"] = HA_Close
    return table_n_min


# 在jupyter中又实现了一遍，记为EMA_type
# def EMA_short_long(table_n_min,t_short = 34,t_long = 55):
#     EMA_short_long = []
#     EMA_short = talib.EMA(table_n_min.Close,t_short)
#     EMA_long = talib.EMA(table_n_min.Close,t_long)
#     for i in range(len(EMA_short)):
#         if EMA_short > EMA_long:#EMA多头排列
#             EMA_short_long.append(1)
#         elif EMA_short < EMA_long:#EMA空头排列
#             EMA_short_long.append(-1)
#         else:
#             EMA_short_long.append(0)

#     table_n_min['EMA_short_long'] = EMA_short_long

def Keltner_Channel(X, table_n_min, nK, nTR, vT, vB, ma="SMA"):
    '''
    X的input可选Open,High,Low,Last,volume,#of trades,OHLC Avg,HLC Avg,HL Avg,Bid Volume,Ask Volume
    table_n_min为提供开高低收数据的dataframe
    '''
    sma_X = pd.Series(X).rolling(nK).mean()  # 这个根据定义一定是sma
    # 自己算TR
    # TR = np.max([(table_n_min.High - table_n_min.Low),abs(table_n_min.High - table_n_min.Close.shift(1)),abs(table_n_min.Low - table_n_min.Close.shift(1))],axis = 0)
    TR = table_n_min.High - table_n_min.Low
    if ma == "SMA":
        ATR = pd.Series(TR).rolling(nTR).mean()
    else:
        ATR = talib.EMA(TR, nTR)
    Top_Band = sma_X + vT * ATR
    Middle_Band = sma_X
    Bottom_Band = sma_X - vB * ATR
    # 返回的序列的索引为0,1,...,len
    table_n_min['KC_Bottom_Band'] = Bottom_Band
    table_n_min['KC_Top_Band'] = Top_Band
    return table_n_min
    # return {"Top_Band":Top_Band,"Middle_Band":Middle_Band,"Bottom_Band":Bottom_Band}


def compare_zzg_kc(IdxfromFunclastExtremeBar, ExtremefromFunclastExtremeBar, KCLine, emalongorshort):
    '''
    传入的是上一个赋值点的index（就是zzg的结果中的bar_num),是通过lastExtremeBar()得到的
    上一个赋值点的点位;
    KCLine为Keltner_Channel的结果
    emalongorshort表示当前判断出的ema排列情况
    ema多头排列时，前一个最近的赋值点
    '''
    if emalongorshort == "long":
        kc = KCLine['Bottom_Band']
        kc_price = kc[IdxfromFunclastExtremeBar]
        if ExtremefromFunclastExtremeBar < kc_price:
            return True
        else:
            return False
    else:
        kc = KCLine['Top_Band']
        kc_price = kc[IdxfromFunclastExtremeBar]
        if kc_price < ExtremefromFunclastExtremeBar:
            return True
        else:
            return False


# 回测中需要知道极值点是否发生了rsi背离
def AddRSIDivergence(table_n_min, df_his_listed):  # 有一个问题，我之前只对历史赋值点和最终赋值点判断了是否有背离
    '''

    找到前一个赋值点的index为previousIdx1，type1为高点还是低点，前前前赋值点的index为previousIdx2
    记录当前bar的index为curIdx

    # 下面的逻辑已经改变
    判断curIdx是否in df_his_listed，如果in，记录当前的（即前一个赋值点的，就是它自己）bar的rsi为rsi，
    如果当前的type1是高点，寻找previousIdx2到previousIdx1之间的最大的rsi_max,若rsi_max > rsi,为顶背离
    如果当前的type1是低点，寻找previousIdx2到previousIdx1之间的最小的rsi_min,若rsi_min < rsi,为底背离
    
    如果not in df_his_listed，计算curIdx - previousIdx1是否小于等于3，如果是
    前一个赋值点type1若为高点，寻找previousIdx2到previousIdx1之间的最大的rsi_max,若rsi_max > rsi,为顶背离
    前一个赋值点type1若为低点，寻找previousIdx2到previousIdx1之间的最小的rsi_min,若rsi_min < rsi,为底背离

    '''
    # 对df_his_listed处理，按照bar_num去重，保留第二个
    df = df_his_listed.copy()
    df = df.drop_duplicates(subset=['bar_num'], keep='last', inplace=False)

    RSI_BottomDivergence = []
    RSI_TopDivergence = []
    extreme_bar_num = np.array(df.bar_num)
    for i in range(len(table_n_min)):
        if i >= 364:  # 这里的数值要随着跑出来的bar_num改 Todo
            Idx = extreme_bar_num[(extreme_bar_num - i) <= 0][-3:]
            previousIdx1 = Idx[-1]
            type1 = df.loc[df.bar_num == previousIdx1].zzp_type.values
            #             print(type1)
            previousIdx2 = Idx[0]
            rsi1 = table_n_min.loc[previousIdx1].RSI
            rsi2 = table_n_min.loc[previousIdx2].RSI

            high1 = table_n_min.loc[previousIdx1].High
            high2 = table_n_min.loc[previousIdx2].High

            low1 = table_n_min.loc[previousIdx1].Low
            low2 = table_n_min.loc[previousIdx2].Low

            if i in df.bar_num.tolist():
                if (type1 == "ZZPT.ONCE_HIGH") or (type1 == "ZZPT.HIGH"):
                    RSI_BottomDivergence.append(0)
                    if (high1 > high2) & (rsi1 < rsi2):
                        RSI_TopDivergence.append(1)
                    elif (high1 < high2) & (rsi1 > rsi2):
                        RSI_TopDivergence.append(1)
                    else:
                        RSI_TopDivergence.append(0)
                else:
                    RSI_TopDivergence.append(0)
                    if (low1 > low2) & (rsi1 < rsi2):
                        RSI_BottomDivergence.append(1)
                    elif (low1 < low2) & (rsi1 > rsi2):
                        RSI_BottomDivergence.append(1)
                    else:
                        RSI_BottomDivergence.append(0)
            else:
                if (i - previousIdx1) <= 3:
                    if (type1 == "ZZPT.ONCE_HIGH") or (type1 == "ZZPT.HIGH"):
                        RSI_BottomDivergence.append(0)
                        if (high1 > high2) & (rsi1 < rsi2):
                            RSI_TopDivergence.append(1)
                        elif (high1 < high2) & (rsi1 > rsi2):
                            RSI_TopDivergence.append(1)
                        else:
                            RSI_TopDivergence.append(0)
                    else:
                        RSI_TopDivergence.append(0)
                        if (low1 > low2) & (rsi1 < rsi2):
                            RSI_BottomDivergence.append(1)
                        elif (low1 < low2) & (rsi1 > rsi2):
                            RSI_BottomDivergence.append(1)
                        else:
                            RSI_BottomDivergence.append(0)
                else:
                    RSI_BottomDivergence.append(None)
                    RSI_TopDivergence.append(None)
        else:
            RSI_BottomDivergence.append(None)
            RSI_TopDivergence.append(None)
    table_n_min['RSI_BottomDivergence'] = RSI_BottomDivergence
    table_n_min['RSI_TopDivergence'] = RSI_TopDivergence
    return table_n_min


# 往df中添加二次平滑的RSI
# 下面这个函数调用起来会报错？？为啥,因为原来没有写return！
def AddRSI_T(table_n_min, T):  ##这边有可能会用到未来函数，因为talib的rsi的公式中是包含了当天的，所以如果要去掉当天的，那只能索引上一个
    '''
    由于RSI14是在index = 14时才有，故rsi2应该在index = 15开始才有意义，rsi7应该在index = 20时才有意义
    '''
    RSI_T = table_n_min["RSI"].rolling(T).mean()
    table_n_min['RSI_{}'.format(T)] = RSI_T
    return table_n_min


def RSI_SMA_shortgo(table_n_min, T):  # 往table_n_min中添加rsi上穿和下穿的列
    '''
        df为包含二次平滑完毕的RSI的表
        T为需要研究的区间长度，只要在T时间内有上穿或者下穿就认为有穿
    '''

    RSIShortgodown = [0] * T
    RSIShortgoup = [0] * T

    for i in range(T, len(table_n_min)):  ########这边后面去考虑一下，是否要取到i（如果添加二次平滑的rsi列的时候已经shift过1，那这边就要取到i，否则可能会出现使用未来函数的风险）
        RSI_short = table_n_min.iloc[(i - T): (i + 1)]['RSI_2'].tolist()  ###这边要注意14是短期的平滑的rsi所在的位置，如果df表格不一样，这里需要修改
        RSI_long = table_n_min.iloc[(i - T): (i + 1)]['RSI_7'].tolist()

        # diff = RSI2 - RSI7#存在由正变负的就ok
        diff_short_go_down = [(m - n) for m, n in zip(RSI_short, RSI_long)]
        diff_short_go_up = [(m - n) for m, n in zip(RSI_long, RSI_short)]

        for j in range(len(diff_short_go_down) - 1):
            # print(j)
            before_minus_after_down = diff_short_go_down[j] - diff_short_go_down[j + 1]
            before_minus_after_up = diff_short_go_up[j] - diff_short_go_up[j + 1]
            rsi_short = RSI_short[j]
            rsi_long = RSI_long[j]
            # if (before_minus_after >= diff_short_go_down[j]) & (diff_short_go_down[j] >= 0) & (rsi_short >= 50) & (rsi_short <= 100)& (rsi_long >= 50) & (rsi_long <= 100):给rsi限定范围
            if (before_minus_after_down >= diff_short_go_down[j]) & (diff_short_go_down[j] >= 0):
                RSIShortgodown_sub = 1
            else:
                RSIShortgodown_sub = 0

            if (before_minus_after_up >= diff_short_go_up[j]) & (diff_short_go_up[j] >= 0):
                RSIShortgoup_sub = 1
            else:
                RSIShortgoup_sub = 0
        RSIShortgodown.append(RSIShortgodown_sub)
        RSIShortgoup.append(RSIShortgoup_sub)

    # print(len(RSIShortgodown))
    table_n_min['RSIShortgodown'] = RSIShortgodown
    table_n_min['RSIShortgoup'] = RSIShortgoup
    return table_n_min


def yesterdayZscore(df):  # 就是原来的DataProcess4SMA1(df,a,b,sigma = 0)函数
    '''
    df是天数据
    新增Ratio,zscore
    默认滚20天为周期
    '''
    df['Range'] = df.High - df.Low
    # 记录昨日range，为后面计算实时zscore提供方便
    df["yesterdayRange"] = df.Range.shift(1)

    Ratio = pd.Series(df.Range) / pd.Series(df.Range).shift(1)
    df["yesterdayRatio"] = Ratio.shift(1)

    ratio_mean = pd.Series(Ratio).rolling(20).mean()  # 向前滚20
    ratio_std = pd.Series(Ratio).rolling(20).std()
    zscore = (pd.Series(Ratio) - ratio_mean) / ratio_std

    df['yesterdayZscore'] = zscore.shift(1)
    return df


# # 下面这个可能会有bug，最终回测添加数据没有用这个函数
# def real_time_zscore(table_n_min, df):  # 这边直接合成的数据time是timestamp的格式，但是如果从合成的csv导入，time是str格式
#     '''
#     在table_n_min中添加real_time_zscore
#     以8:30为分界，
#     df为包含yesterdayRange的表格
#     '''
#     # 先计算real_time_range
#     realtimezscore = []
#     if type(table_n_min.Time[0]) == str:
#         table_n_min['Time'] = pd.to_datetime(table_n_min.Time)
#
#     # 初始化变量
#     day_high = 0
#     day_low = 0
#     day_range = 0
#     flag = 0
#     for i in range(len(table_n_min)):
#         # 提取时刻，遇到新的8:30，一切数值就要归零
#         if table_n_min.iloc[i].strftime("%H:%M:%S") == "08:30:00":
#             flag = 1
#         else:
#             dayrange.append(0)
#
#         if flag:
#             if table_n_min.iloc[i].strftime("%H:%M:%S") == "08:30:00":
#                 day_high = table_n_min.iloc[i].High
#                 day_low = table_n_min.iloc[i].Low
#             else:
#                 high = table_n_min.iloc[i].High
#                 low = table_n_min.iloc[i].Low
#                 if high > day_high:
#                     day_high = high
#
#                 if low < day_low:
#                     day_low = low
#             day_range = day_high - day_low
#         # 再计算real_time_ratio
#         # 获取当前bar所属日期
#         date = (table_n_min.iloc[i].Time - datetime.timedelta(seconds=8.5 * 3600)).strftime("%Y-%m-%d")
#         yes_range = df[df.DateTag == date]["yesterdayRange"].values  # 这边DateTag的字段有待商榷，不知道合成完的表格里面叫什么名字
#         real_time_ratio = day_range / yes_range
#         realtimeratio.append(real_time_ratio)
#         df_used_for_cal_zscore = df[df.DateTag <= date][-19:]  # 这边有可能会报错，因为数据记录不够20条,这边是-19是因为第20个数据是今日实时产生的
#         ratio_mean = np.mean(df_used_for_cal_zscore_.yesterdayRatio.tolist() + [real_time_ratio])
#         ratio_std = np.std(df_used_for_cal_zscore_.yesterdayRatio.tolist() + [real_time_ratio])
#         real_time_zscore = (real_time_ratio - ratio_mean) / ratio_std
#         realtimezscore.append(real_time_zscore)
#     table_n_min['real_time_zscore'] = realtimezscore.shift(1)  # 这样子跨日的时候会不会有问题？

#
# # DateTag可能有问题
# def AddyesterdayZscore(table_n_min):
#     Zscore_yesterday = []
#     for i in range(len(table_n_min)):
#         date = (table_n_min.iloc[i].Time - datetime.timedelta(seconds=8.5 * 3600)).strftime("%Y-%m-%d")
#         Zscore_yesterday.append(df[df.DateTag == date].yesterdayZscore.values)
#     table_n_min['Zscore_yesterday'] = Zscore_yesterday


def AddZscoreTags(dataseries):
    '''
    输入一列数据，根据其值判断它应该打什么标签
    '''
    tag = []
    for data in dataseries:
        if data >= float(2):
            tag_sub = 'G'
        elif data >= float(1):
            tag_sub = 'F'
        elif data >= float(0.5):
            tag_sub = 'E'
        elif data >= float(0):
            tag_sub = 'D'
        elif data >= float(-0.5):
            tag_sub = 'C'
        elif data >= float(-1):
            tag_sub = 'B'
        else:
            tag_sub = 'A'
        tag.append(tag_sub)

    tag_tomorrow = tag[1:] + [1]

    return tag


def ProcessImportedProbTable(readcsv):
    '''
    以今天的zscore的tag为index
    '''
    readcsv = readcsv[2:]
    readcsv.columns = ['tag_tomorrow', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return readcsv


def FindProb(yesterdayZscore, currentZscore, Prob_table):
    '''
    这里输入的Prob_table需要事先处理过
    输入的yesterdayZscore,currentZscore要为str
    '''
    return (Prob_table[(Prob_table.index == yesterdayZscore) & (Prob_table.tag_tomorrow == currentZscore)][
                "count"].apply(float) / Prob_table[Prob_table.index == yesterdayZscore]["count"].apply(float).sum())[0]


def calc_60min_rsi(df, t=14):
    """用15min的bar计算
    t为60min的周期，那么4*t为15min的周期"""
    bar60rsi14 = talib.RSI(df.Close, timeperiod=4 * t)
    df['bar60rsi14'] = bar60rsi14
    return df
