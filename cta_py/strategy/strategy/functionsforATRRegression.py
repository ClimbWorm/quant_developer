import numpy as np
import pandas as pd
import datetime


def isInAllowedTradingTime(time):
    '''
    time是table_n_min中输入的str格式的时间
    '''
    clock = pd.to_datetime(time).strftime("%H:%M:%S")
    forbiden = (clock >= '17:00:00') & (clock < '19:00:00')
    return not forbiden


# def isAlreadyTraded(bar_num,opened_bar):#这个可能只会出现在需要15min的数据去细分，用1min数据时
# #判断当前的bar是否已经开过仓，只要输入想要获取的barnum就能
# if bar_num in opened_bar:#这样就需要opened_bar这个list是随着遍历的进行逐渐加入index的
#     return True
# else:
#     return False

# def doShortorLong(table_n_min,curr_bar_idx,scheme):
# if scheme == 1:####
#     curr_price = table_n_min.loc[curr_bar_idx].Open
#     #这里又要实时计算一天里面的高点和低点（我这边先跳过吧）先把用zzg的方案做了
#     curr_day_high
#     curr_day_low
#     if curr_price <= (curr_day_high + curr_day_low)/2:#离低点近
#         return "long"
#     else:
#         return "short"

# else:
#     if lastExtremeBar[1] == -1:
#         return "long"
#     else:
#         return "short"

# def isRSIinrange(lowlim,highlim):
# if doShortorLong() == "long":
#     if RSI < lowlim:#这边只是一个框架，rsi具体怎么传进去怎么计算都要细化
#         return True
# else:
#     if RSI > highlim:
#         return True

def GetTag(zscore):
    if (zscore < -1):

        tag = 'A'

    elif (zscore >= -1 and zscore < -0.5):

        tag = 'B'

    elif (zscore >= -0.5 and zscore < 0):

        tag = 'C'

    elif (zscore >= 0 and zscore < 0.5):

        tag = 'D'

    elif (zscore >= 0.5 and zscore < 1):

        tag = 'E'

    elif (zscore >= 1 and zscore < 2):

        tag = 'F'

    else:

        tag = 'G'

    return tag


def SetLots(zscore_l, zscore_c, df_D, miniLots=0, maxLotsLimit=10):
    '''
    若当前时刻存在进场机会，此函数返回一个手数数值，赋值给即将建仓的订单的Lots参数

    参数:

        1. zscore_l: 前一交易日，日级别zscore
        2. zscore_c: 当前交易日，实时zscore
        3. df_D: 日级别行情数据 + 特征列
        4. standardLots: 标准建仓手数

    规则:
        1. 提取到前一交易日zscore的tag_i(范围)
        2. 提取到当前时刻zscore current的tag_j(范围)

            基于1,2获取到当前时刻对应的tag状态(tag_i, tag_j)

            N = df的总长度

        3. 获取到历史N个交易日中，tag_i发生的次数 M
        4. 获取到历史N个交易日中，tag_i发生后，下一交易日发生tag_j的次数 K

            基于3,4计算出 K/M 的数值

        5. 建仓手数为 标准建仓手数 * K/M * M/N
    '''

    tag_l = GetTag(zscore_l)
    # print(tag_l)
    tag_c = GetTag(zscore_c)
    # print(tag_c)
    if (tag_c == 'F') | (tag_c == 'G'):
        numberK = df_D.groupby(['tag', 'tag_tomorrow'])['Z-score'].count().max()
    else:
        numberK = len(df_D.loc[df_D.tag == tag_l].loc[df_D.tag_tomorrow == tag_c])

    totalTagNumer = len(df_D.tag)

    if (numberK > 0):

        lots = maxLotsLimit * (numberK / totalTagNumer)

    else:

        lots = miniLots

    return lots


def lastExtremeBar(curr_bar_idx, df_all_zzg):
    '''
    获取当前bar的前一个极值点,需要return索引值、方向、绝对数值（顺序一定要按照这样写，因为doShortorLong中写的是lastExtremeBar[1]
    direction表示开仓的方向
    '''
    high_or_low_barnum = np.array(df_all_zzg.bar_num)
    if len(high_or_low_barnum[(high_or_low_barnum - curr_bar_idx) <= 0]) > 0:
        zzg = high_or_low_barnum[(high_or_low_barnum - curr_bar_idx) <= 0][-1]  # 获取到上一个极值点的barnum
        zzg_direction = df_all_zzg[df_all_zzg.bar_num == zzg].zzp_type.values[0]
        if (zzg_direction == "ZZPT.HIGH") | (zzg_direction == "ZZPT.ONCE_HIGH"):
            zzg_extreme = df_all_zzg[df_all_zzg.bar_num == zzg].High.values[0]
        else:
            zzg_extreme = df_all_zzg[df_all_zzg.bar_num == zzg].Low.values[0]

        threshold = df_all_zzg[df_all_zzg.bar_num == zzg].threshold.values[0]
    else:  # 找不到上一个极值点的情况
        zzg = 0
        zzg_extreme = 0
        zzg_direction = 0
        threshold = 0

    last_extreme_bar_idx = zzg
    last_extreme_bar_low_or_high = zzg_direction
    last_extreme_value = zzg_extreme
    last_extreme_bar_threshold = threshold

    return last_extreme_bar_idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold


# def CorrespondingStartBar(Time,table_n_min):
#     '''
#     输入当前的1min，获取这个1min的bar对应的开始的15min的bar的时间和在15min的表格中的index
#     '''
#     table_n_min.Time


# def CheckOnceOpened(Time):
#     '''
#     输入当前的1min时间，调用CorrespondingStartBar(Time)函数的结果，去判断这根15min的bar是否曾经开过
#     由此衍生出，我记录一根bar是否开过，要标记在这个15min的bar开始的地方，
#     return 0 表示之前没有开过仓
#     '''


# def BarNumJumpTo(self):
# '''
# 使用情形：这个15min开始的1minbar的基础数据不满足第一层限制，就不必再遍历这个15min剩下的1min的bar了，直接跳到下一个15min的bar
# '''
# pass


# def ZscoreTag():#strftime截取日期后，去合成的日表格中寻找昨日的zscore，当日的zscore需要根据当前已经走出来的当日行情实时计算
# return 

# def Lots(max_lots):
# zscore_distribution_prob = 
# return int(max_lots)


def DataProcess4SMA1(df):
    '''
    这个函数用于返回一个处理完的dataframe，新增Ratio,zscore
    df为日数据
    默认滚20天为周期
    '''

    Ratio = (df.Range / df.Range.shift(1)).tolist()
    df1 = df.copy()
    df1["Ratio"] = Ratio

    # 判断当前bar是涨还是跌
    bar_up_or_down = (df1["Close"] - df1["Open"]) >= 0
    df1["bar_up_or_down"] = bar_up_or_down

    ratio_mean = df1.Range.rolling(20).mean()
    ratio_std = df1.Range.rolling(20).std()
    zscore = (df1.Range - ratio_mean) / ratio_std

    df1['Z-score'] = zscore
    df1['Z-score_tomorrow'] = zscore.shift(-1)

    return df1


def AddTags(df1):
    # 对df1中的数据列打标签tag
    df1['tag'] = None
    df1.loc[df1[df1['Z-score'] < float(-1)].index.tolist(), 'tag'] = 'A'
    df1.loc[df1[df1['Z-score'] >= float(-1)].index.tolist(), 'tag'] = 'B'
    df1.loc[df1[df1['Z-score'] >= float(-0.5)].index.tolist(), 'tag'] = 'C'
    df1.loc[df1[df1['Z-score'] >= float(0)].index.tolist(), 'tag'] = 'D'
    df1.loc[df1[df1['Z-score'] >= float(0.5)].index.tolist(), 'tag'] = 'E'
    df1.loc[df1[df1['Z-score'] >= float(1)].index.tolist(), 'tag'] = 'F'
    df1.loc[df1[df1['Z-score'] >= float(2)].index.tolist(), 'tag'] = 'G'

    tag_tomorrow = pd.Series(df1.tag).shift(-1)
    df1['tag_tomorrow'] = tag_tomorrow
    return df1


# 拼一个用来统计分布概率的dataframe，使得每一个tag都刚好出现了num次
def StatisticDF(df1, num=100):
    tag_A_start = df1[df1.tag == 'A'][-num:].index[0]
    tag_B_start = df1[df1.tag == 'B'][-num:].index[0]
    tag_C_start = df1[df1.tag == 'C'][-num:].index[0]
    tag_D_start = df1[df1.tag == 'D'][-num:].index[0]
    tag_E_start = df1[df1.tag == 'E'][-num:].index[0]
    tag_F_start = df1[df1.tag == 'F'][-num:].index[0]
    tag_G_start = df1[df1.tag == 'G'][-num:].index[0]
    start_index = np.min([tag_A_start, tag_B_start, tag_C_start, tag_D_start, tag_E_start, tag_F_start, tag_G_start])
    return df1.loc[start_index:].reset_index(drop=True)


def generate_df_D(current_time, df_Day):
    """
    current_time格式为xxxx/xx/xx
    """

    df_Day = df_Day[df_Day.DateTag < current_time].reset_index(drop=True)
    df_Day['Range'] = df_Day.High - df_Day.Low
    df = DataProcess4SMA1(df_Day)
    df = AddTags(df)
    stat_df = StatisticDF(df, num=100)
    return stat_df


# 需要记录Idx_start和Idx_low（Idx_low是通过找Idx_start和Idx_end之间的最小值得到的）
# 计算Drawdown的函数，在BackTestingEvaluation中被调用
def Calc_net_profit_versus_maximum_DrawDown(his_table):
    series = his_table.cumsum_Profits  # 是pandas的series类型
    if series.iloc[-1] < 0:  ##考虑特殊情况，第一个即为最大值，事先写一个判断，如果曲线一直是向下的，就直接赋值一个-1或者别的常数值
        # net_profit_versus_maximum_DD = -1
        return np.nan  # 最后是亏钱的，就直接返回nan

    max_value = series.min()
    Idx = []  # 记录出现新高的bar的index

    for Idx_, item in enumerate(series):
        if (item > max_value):
            max_value = item
            Idx.append(Idx_)
    Idx_start = Idx[:]
    Idx_end = Idx[1:] + [len(series) - 1]
    drawdown_list = []
    drawdown_ratio_list = []
    drawdown_last_times_list = []
    for start, end in zip(Idx_start, Idx_end):
        previous_max = series[start]
        min_ = series[start:end].min()
        min_index_ = series[start:end].index(min_)# 获取两个peak之间的最小值的出现的第一个位置
        drawdown_last_times = min_index_ - start - 1
        drawdown = previous_max - min_
        drawdown_ratio = drawdown/previous_max
        drawdown_list.append(drawdown)
        drawdown_ratio_list.append(drawdown_ratio)
        drawdown_last_times_list.append(drawdown_last_times)
    net_profit_versus_maximum_DD = series.iloc[-1] / np.max(drawdown_list)
    return net_profit_versus_maximum_DD


def optimize_trailing_stop_multiplier(df_zigzag, current_date, num=100):
    """CurrentTime是程序中传入的，str格式的，包含时刻信息的时间"""
    # 把df_zigzag中的datetime列转化为timestamp格式
    df_zigzag['datetime'] = pd.to_datetime(df_zigzag['datetime']) #df_zigzag.datetime.apply(lambda x: pd.to_datetime(x))
    # 记录当前日期
    CurrentDate = pd.to_datetime(current_date)
    startdate = CurrentDate - datetime.timedelta(days=num)
    df_series = df_zigzag[(df_zigzag.datetime >= startdate) & (df_zigzag.datetime < CurrentDate)].groupby(['m_Date'])['day_range'].max()
    if CurrentDate < pd.to_datetime('2018/1/6'):
        return 1
    else:
        # min_avg_before, today_used_avg = df_series.rolling(5).mean().min(), df_series.rolling(5).mean()[-1]
        mean_avg_before, today_used_avg = df_series.rolling(5).mean().mean(), df_series.rolling(5).mean()[-1]
        # return today_used_avg/min_avg_before
        return today_used_avg / mean_avg_before

