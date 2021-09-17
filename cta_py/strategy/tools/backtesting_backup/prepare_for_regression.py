import pandas as pd
import numpy as np
import datetime
import multiprocessing
from strategy.define_enum_constant import *

# df_zigzag包含了所有的点，包括不是赋值点的，后续处理后都用df_zzg_add_date
# df_all_zzg只包含once和最终的,df_final_zzg_low只包括最终的负数赋值点，df_final_zzg_high只包括最终的正数赋值点
# Date字段是处理过17：00的

epsilon = 1e-5


def generate_zzg_table_include_once_and_final(zzg_num):
    from datetime import timedelta
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_all_zzg = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_all_zzg["bar_num"] = df_all_zzg.index
    df_all_zzg = df_all_zzg.reset_index(drop=True)
    df_all_zzg['datetime'] = pd.to_datetime(df_all_zzg['datetime'])
    df_all_zzg['Date'] = pd.to_datetime(
        (pd.to_datetime(df_all_zzg['datetime']) - timedelta(hours=17)).apply(lambda x: x.strftime("%Y-%m-%d")))
    df_all_zzg['m_Date'] = pd.to_datetime(df_all_zzg['m_Date'], format="%Y-%m-%d")
    return df_all_zzg


def generate_zzg_table_only_include_final_high(zzg_num):
    from datetime import timedelta
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_final_zzg_high = df_zigzag[df_zigzag.zzp_type == "ZZPT.HIGH"]
    df_final_zzg_high["bar_num"] = df_final_zzg_high.index
    df_final_zzg_high = df_final_zzg_high.reset_index(drop=True)
    df_final_zzg_high['datetime'] = pd.to_datetime(df_final_zzg_high['datetime'])
    df_final_zzg_high['Date'] = pd.to_datetime(
        (pd.to_datetime(df_final_zzg_high['datetime']) - timedelta(hours=17)).apply(lambda x: x.strftime("%Y-%m-%d")))
    df_final_zzg_high['m_Date'] = pd.to_datetime(df_final_zzg_high['m_Date'], format="%Y-%m-%d")
    return df_final_zzg_high


def generate_zzg_table_only_include_final_low(zzg_num):
    from datetime import timedelta
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_final_zzg_low = df_zigzag[df_zigzag.zzp_type == "ZZPT.LOW"]
    df_final_zzg_low["bar_num"] = df_final_zzg_low.index
    df_final_zzg_low = df_final_zzg_low.reset_index(drop=True)
    df_final_zzg_low['datetime'] = pd.to_datetime(df_final_zzg_low['datetime'])
    df_final_zzg_low['Date'] = pd.to_datetime(
        (pd.to_datetime(df_final_zzg_low['datetime']) - timedelta(hours=17)).apply(lambda x: x.strftime("%Y-%m-%d")))
    df_final_zzg_low['m_Date'] = pd.to_datetime(df_final_zzg_low['m_Date'], format="%Y-%m-%d")
    return df_final_zzg_low


def generate_zzg_table_only_include_final_and_once_high(zzg_num):
    from datetime import timedelta
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_final_and_once_zzg_high = df_zigzag[
        (df_zigzag.zzp_type == "ZZPT.HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_HIGH")]
    df_final_and_once_zzg_high["bar_num"] = df_final_and_once_zzg_high.index
    df_final_and_once_zzg_high = df_final_and_once_zzg_high.reset_index(drop=True)
    df_final_and_once_zzg_high['datetime'] = pd.to_datetime(df_final_and_once_zzg_high['datetime'])
    df_final_and_once_zzg_high['Date'] = pd.to_datetime(
        (pd.to_datetime(df_final_and_once_zzg_high['datetime']) - timedelta(hours=17)).apply(
            lambda x: x.strftime("%Y-%m-%d")))
    df_final_and_once_zzg_high['m_Date'] = pd.to_datetime(df_final_and_once_zzg_high['m_Date'], format="%Y-%m-%d")
    return df_final_and_once_zzg_high


def generate_zzg_table_only_include_final_and_once_low(zzg_num):
    from datetime import timedelta
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_final_and_once_zzg_low = df_zigzag[(df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW")]
    df_final_and_once_zzg_low["bar_num"] = df_final_and_once_zzg_low.index
    df_final_and_once_zzg_low = df_final_and_once_zzg_low.reset_index(drop=True)
    df_final_and_once_zzg_low['datetime'] = pd.to_datetime(df_final_and_once_zzg_low['datetime'])
    df_final_and_once_zzg_low['Date'] = pd.to_datetime(
        (pd.to_datetime(df_final_and_once_zzg_low['datetime']) - timedelta(hours=17)).apply(
            lambda x: x.strftime("%Y-%m-%d")))
    df_final_and_once_zzg_low['m_Date'] = pd.to_datetime(df_final_and_once_zzg_low['m_Date'], format="%Y-%m-%d")
    return df_final_and_once_zzg_low


def add_Date_based_on_17(zzg_num):
    from datetime import timedelta
    df_zzg_add_date = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210331_{}.csv".format(
            zzg_num),
        index_col=0)
    df_zzg_add_date['datetime'] = pd.to_datetime(df_zzg_add_date['datetime'])
    df_zzg_add_date['Date'] = pd.to_datetime(
        (df_zzg_add_date['datetime'] - timedelta(hours=17)).apply(lambda x: x.strftime("%Y-%m-%d")))
    return df_zzg_add_date


# 给一个bar的idx就能获取它所属day的start idx和end idx
# 按照17:00作为一天的起点
def get_day_start_and_end_index_for_zzp(bar_idx, df_zzg_add_date):
    """这里输入的df_zzg_add_date是经过add_Date_based_on_17(zzg_num)处理过的"""

    df = df_zzg_add_date.__deepcopy__()
    new_day_index = pd.Series(df.drop_duplicates(['Date']).index.tolist())
    newday_start_index = new_day_index[(new_day_index - bar_idx) <= 0].iloc[-1]
    try:
        next_day_start_index = new_day_index[(new_day_index - bar_idx) > 0].iloc[0]
    except Exception as e:
        #         print(e)
        next_day_start_index = len(df)
    return newday_start_index, next_day_start_index - 1


def get_extreme_bar_info(curr_bar_idx, df_all_zzg, last_or_last_last):
    '''
    last_or_last_last可以get last和last last extrem bar index
    输入1表示last 输入2表示last last
    '''
    high_or_low_barnum = np.array(df_all_zzg.bar_num)
    series_ = high_or_low_barnum[(high_or_low_barnum - curr_bar_idx) <= 0]
    if len(series_) >= last_or_last_last:
        zzg = series_[-last_or_last_last]  # 获取到上一个或上上个极值点的barnum
        zzg_direction = df_all_zzg[df_all_zzg.bar_num == zzg].zzp_type.values[0]
        if (zzg_direction == "ZZPT.HIGH") | (zzg_direction == "ZZPT.ONCE_HIGH"):
            zzg_extreme = df_all_zzg[df_all_zzg.bar_num == zzg].High.values[0]
        else:
            zzg_extreme = df_all_zzg[df_all_zzg.bar_num == zzg].Low.values[0]

        threshold = df_all_zzg[df_all_zzg.bar_num == zzg].threshold.values[0]
    else:  # 找不到上上一个极值点的情况
        zzg = 0
        zzg_extreme = 0
        zzg_direction = 0
        threshold = 0

    extreme_bar_idx = zzg
    extreme_bar_low_or_high = zzg_direction
    extreme_value = zzg_extreme
    extreme_bar_threshold = threshold

    return extreme_bar_idx, extreme_bar_low_or_high, extreme_value, extreme_bar_threshold


def calc_average_day_range(current_date, num_of_days, df_zzg_add_date):
    """
    str格式的current_date
    这里输入的df_zzg_add_date是经过add_Date_based_on_17(zzg_num)处理过的
    """

    df_day_high_low = df_zzg_add_date.groupby(['Date'])[['High', 'Low']].agg({'High': np.max, 'Low': np.min})
    df_day_range = df_day_high_low.High - df_day_high_low.Low
    # Todo 这边需要注意index的格式是不是时间格式
    avg_day_range = df_day_range[df_day_range.index < current_date][-num_of_days:].mean()
    return avg_day_range


def calc_zscores_for_linear_regression(curr_bar_idx, entry_point, entry_direction, df_all_zzg, df_zzg_add_date):
    """这里在使用时需要提前调用df_all_zzg = generate_zzg_table_include_once_and_final(zzg_num)传入
    这里输入的df_zzg_add_date是经过add_Date_based_on_17(zzg_num)处理过的"""
    last_zzg_idx, _, extreme_value, _ = get_extreme_bar_info(curr_bar_idx, df_all_zzg, 1)
    day_start_idx, day_end_idx = get_day_start_and_end_index_for_zzp(last_zzg_idx, df_zzg_add_date)
    day_open = df_zzg_add_date.loc[day_start_idx].Open
    day_high = np.max(df_zzg_add_date.loc[day_start_idx:day_end_idx].High)
    day_low = np.min(df_zzg_add_date.loc[day_start_idx:day_end_idx].Low)
    if entry_direction == "TradeDirection.SHORT":  # 空单
        if entry_point < day_open:
            extreme_value = get_extreme_bar_info(curr_bar_idx, df_all_zzg, 2)[2]
            numerator = day_high - extreme_value
        else:
            numerator = extreme_value - day_low
    else:  # 多单 前赋值点为负数赋值点
        if entry_point < day_open:
            numerator = day_high - extreme_value
        else:
            extreme_value = get_extreme_bar_info(curr_bar_idx, df_all_zzg, 2)[2]
            numerator = extreme_value - day_low
    current_date = df_zzg_add_date.loc[day_start_idx].Date
    avg_5_day_range = calc_average_day_range(current_date, 5, df_zzg_add_date)
    return numerator / avg_5_day_range


def add_zscores_column_for_linear_regression(df_all_zzg, df_zzg_add_date, his_table):
    zscore_list = []
    for i in range(len(his_table)):
        entry_idx = his_table.iloc[i].entry_idx
        entry_price = his_table.iloc[i].EntryPrice
        order_direction = his_table.iloc[i].Direction
        zscore = calc_zscores_for_linear_regression(entry_idx, entry_price, order_direction, df_all_zzg,
                                                    df_zzg_add_date)
        zscore_list.append(zscore)

    if len(his_table) != len(zscore_list):
        raise Exception

    his_table['zscore'] = zscore_list
    return his_table


def calc_misopened_count(df_final_zzg_high, df_final_zzg_low, his_table):
    """his_table为只有开仓信息的表"""
    entry_idx_list = his_table.entry_idx
    direction_list = his_table.Direction
    final_high_barnum = np.array(df_final_zzg_high.bar_num)
    final_low_barnum = np.array(df_final_zzg_low.bar_num)
    misopened_count_list = []
    for i in range(len(his_table)):
        order_direction = direction_list.iloc[i]
        entry_idx = entry_idx_list.iloc[i]
        if order_direction == "TradeDirection.SHORT":
            # 找前一个最终的负数赋值点
            try:
                previous_final_zzp = final_low_barnum[(final_low_barnum - entry_idx) <= 0][-1]
            except Exception as e:
                misopened_count_list.append(0)
                continue
        else:
            try:
                previous_final_zzp = final_high_barnum[(final_high_barnum - entry_idx) <= 0][-1]
            except Exception as e:
                misopened_count_list.append(0)
                continue
        # print((previous_final_zzp <= his_table.entry_idx <= entry_idx).all())
        misopened_count = len(his_table.loc[
                                  (previous_final_zzp <= his_table.entry_idx) & (his_table.entry_idx <= entry_idx) & (
                                          his_table.Direction == order_direction)])
        misopened_count_list.append(misopened_count)

    if len(his_table) != len(misopened_count_list):
        raise Exception

    his_table['count'] = misopened_count_list

    # his_table.to_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_count_table\table_{}.csv'.format(
    #         zzg_num))

    return his_table


def calc_floating_profit_before_next_final_high_or_low_zzp(df_final_zzg_high, df_final_zzg_low, df_zzg_add_date,
                                                           his_table):
    """做多的order，如果不扫硬止损，就从进场一直遍历到下一个最终正数赋值点，否则就遍历到触及硬止损的那个bar，计算这期间的max floating profit"""
    entry_idx_list = his_table.entry_idx
    entry_price_list = his_table.EntryPrice
    direction_list = his_table.Direction
    final_high_barnum = np.array(df_final_zzg_high.bar_num)
    final_low_barnum = np.array(df_final_zzg_low.bar_num)
    max_floating_profit_list = []
    for i in range(len(his_table)):
        direction = direction_list.iloc[i]
        entry_idx = entry_idx_list.iloc[i]
        entry_price = entry_price_list.iloc[i]
        if direction == "TradeDirection.SHORT":
            # 找到前正数赋值点的high，作为硬止损
            try:
                previous_final_zzp = final_high_barnum[(final_high_barnum - entry_idx) <= 0][-1]
                later_final_zzp = final_low_barnum[(final_low_barnum - entry_idx) > 0][0]  # 找后一个的就不能加等号
                previous_final_zzp_value = df_zzg_add_date.iloc[previous_final_zzp].High
            except Exception as e:
                max_floating_profit_list.append(0)
                continue
            # 找到后面的high首次大于前正数赋值点的bar
            j = entry_idx
            while df_zzg_add_date.iloc[j].High < previous_final_zzp_value + epsilon:
                if j >= later_final_zzp:
                    break
                j += 1
            # 判断是这个bar先出现还是后一个最终的负数赋值点先出现，取min
            traverse_end_bar = min(later_final_zzp, j)
            # 从df_zzg_add_date中截取对应的df_sub，取low的min，计算其与entry price之间的距离
            max_floating_profit = entry_price - df_zzg_add_date.loc[entry_idx:traverse_end_bar].Low.min()
        else:  # 做多的订单
            # 找到前负数赋值点的low，作为硬止损
            try:
                previous_final_zzp = final_low_barnum[(final_low_barnum - entry_idx) <= 0][-1]
                later_final_zzp = final_high_barnum[(final_high_barnum - entry_idx) > 0][0]  # 找后一个的就不能加等号
                previous_final_zzp_value = df_zzg_add_date.iloc[previous_final_zzp].Low
            except Exception as e:
                max_floating_profit_list.append(0)
                continue
            # 找到后面的low首次小于前负数赋值点的bar
            j = entry_idx
            while df_zzg_add_date.iloc[j].Low > previous_final_zzp_value - epsilon:
                if j >= later_final_zzp:
                    break
                j += 1
            # min（后一个最终的正数赋值点，触及硬止损的bar）
            traverse_end_bar = min(later_final_zzp, j)
            #  从df_zzg_add_date中截取对应的df_sub，取high的max，计算其与entry price之间的距离
            max_floating_profit = df_zzg_add_date.loc[entry_idx:traverse_end_bar].High.max() - entry_price
        max_floating_profit_list.append(max_floating_profit)

    if len(his_table) != len(max_floating_profit_list):
        raise Exception

    his_table['max_floating_profit'] = max_floating_profit_list

    # his_table.to_csv(
    #     r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_count_and_floating_table\table_{}.csv'.format(
    #         zzg_num))
    return his_table


# 下面这个其实不需要了，因为原来的表格里面有last_zzp_value和entryprice，相减即可
# def add_floating_loss(zzg_num,df_all_zzg,df_final_and_once_zzg_low):
#     """若是做多的订单，找到其最近的负数赋值点（包括once），用负数赋值点的low-entryprice
#     若是做空的订单，找到其最近的正数赋值点（包括once），用entryprice-正数赋值点的high"""

# Todo step1：运行这个整个第一波表格
def whole_table_for_regression(zzg_num):
    his_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\table_{}.csv'.format(
            zzg_num), index_col=0)
    df_all_zzg = generate_zzg_table_include_once_and_final(zzg_num)
    df_final_zzg_high = generate_zzg_table_only_include_final_high(zzg_num)
    df_final_zzg_low = generate_zzg_table_only_include_final_low(zzg_num)
    df_zzg_add_date = add_Date_based_on_17(zzg_num)
    # 添加floating profit列
    his_table = calc_floating_profit_before_next_final_high_or_low_zzp(df_final_zzg_high, df_final_zzg_low,
                                                                       df_zzg_add_date, his_table)
    # 添加count列
    his_table = calc_misopened_count(df_final_zzg_high, df_final_zzg_low, his_table)
    # 添加zscore列
    his_table = add_zscores_column_for_linear_regression(df_all_zzg, df_zzg_add_date, his_table)
    his_table.to_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num))


def add_continuous_count(his_table):
    count_list = []
    count = 0
    current_direction = his_table.Direction[0]
    for i in range(len(his_table)):
        if his_table.Direction[i] != current_direction:
            count = 1
            current_direction = his_table.Direction[i]
        else:
            count += 1
        count_list.append(count)
    return count_list


# Todo step2
# 对原来的自变量和因变量(max floating profit/multiplier) 名义上的zscore和count进行zscore化
def add_standardized_zscore_and_count_and_Y(zzg_num):
    """这里的his_table是已经加上了count和zscore列的，该函数的作用是对变量进行标准化"""

    his_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # his_table = his_table.dropna(axis=0, how='any') 这句话不能加 因为里面的ExitTime都是nan
    # 去除multiplier为1的
    his_table = his_table.loc[his_table.loc[:, "multiplier"] != float(1), :].reset_index(drop=True)
    his_table['Y'] = his_table['max_floating_profit'] / his_table['multiplier']
    his_table['continuous_count'] = add_continuous_count(his_table)
    standardized_y = []
    standardized_count = []
    standardized_zscore = []
    for i in range(len(his_table)):
        direction = his_table.iloc[i].Direction
        entry_idx = his_table.iloc[i].entry_idx
        # 筛选出entry_idx小于当前bar并且同向的order
        sub_df = his_table.loc[
                 (his_table.loc[:, "entry_idx"] < entry_idx) & (his_table.loc[:, "Direction"] == direction), :]
        y_list = []
        count_list = []
        zscore_list = []
        try:
            for j in range(len(sub_df) - 1, 0, -1):
                if sub_df.iloc[j].continuous_count < sub_df.iloc[j - 1].continuous_count:
                    y_list.append(sub_df.iloc[j - 1].Y)
                    count_list.append(sub_df.iloc[j - 1]["count"])
                    zscore_list.append(sub_df.iloc[j - 1].zscore)
                    if len(y_list) == 55:
                        break
            y_mean_sub = np.mean(y_list)
            y_std_sub = np.std(y_list)

            count_mean_sub = np.mean(count_list)
            count_std_sub = np.std(count_list)

            zscore_mean_sub = np.mean(zscore_list)  # Todo 求均值和方差都没有包括当前的这个值，是否要包括？
            zscore_std_sub = np.std(zscore_list)
            # 下面可能会出现被0除的情况
            standardized_y.append((his_table.iloc[i].Y - y_mean_sub) / y_std_sub)
            standardized_count.append((his_table.iloc[i]["count"] - count_mean_sub) / count_std_sub)
            standardized_zscore.append((his_table.iloc[i].zscore - zscore_mean_sub) / zscore_std_sub)

        except Exception as e:
            print(e)
            standardized_y.append(np.nan)
            standardized_count.append(np.nan)
            standardized_zscore.append(np.nan)

    print(standardized_y)
    his_table["standardized_y"] = standardized_y
    his_table["standardized_count"] = standardized_count
    his_table["standardized_zscore"] = standardized_zscore

    his_table.to_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num))
    return his_table


def draw_regression_data_scatter(zzg_num):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    regression_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # 数据清洗
    sub_df = regression_table[
        ["standardized_y", "max_floating_profit", "multiplier", "standardized_count", "standardized_zscore",
         "last_zzp_value", "EntryPrice", "Direction"]]
    sub_df["Y"] = sub_df.max_floating_profit / sub_df.multiplier
    #     """若是做多的订单，找到其最近的负数赋值点（包括once），用负数赋值点的low-entryprice
    #     若是做空的订单，找到其最近的正数赋值点（包括once），用entryprice-正数赋值点的high"""
    sub_df['floating_loss'] = [sub_df.last_zzp_value.iloc[i] - sub_df.EntryPrice.iloc[i] if sub_df.Direction.iloc[
                                                                                                i] == "TradeDirection.LONG" else
                               sub_df.EntryPrice.iloc[i] - sub_df.last_zzp_value.iloc[i] for i in range(len(sub_df))]
    # 删除inf和-inf的
    sub_df = sub_df[~sub_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    y1 = sub_df.standardized_count.values
    y2 = sub_df.standardized_zscore.values
    x1 = sub_df.Y.values
    x2 = sub_df.floating_loss.values

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121)
    ax1.scatter(x2, y1, color='pink', alpha=0.5)
    ax1.set_ylabel('standardized_count')
    ax1.set_title('floating_loss')
    ax2 = ax1.twinx()
    ax2.scatter(x2, y2, color='green', alpha=0.5)
    ax2.set_ylabel('standardized_zscore')
    ax2.set_xlabel('standardized_y(floating_profit/multiplier)')

    ax3 = fig.add_subplot(122)
    ax3.scatter(x1, y1, color='pink', alpha=0.5)
    ax3.set_ylabel('standardized_count')
    ax3.set_title('floating_profit')
    ax4 = ax3.twinx()
    ax4.scatter(x1, y2, color='green', alpha=0.5)
    ax4.set_ylabel('standardized_zscore')
    ax4.set_xlabel('standardized_y(floating_profit/multiplier)')
    # plt.show()
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\standardized\floating_loss_and_profit_pic_{}.png'.format(
            zzg_num))


def draw_floating_loss_and_profit_pic(zzg_num):
    import matplotlib.pyplot as plt
    regression_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # 数据清洗
    sub_df = regression_table[
        ["standardized_y", "max_floating_profit", "multiplier", "standardized_count", "standardized_zscore",
         "last_zzp_value", "EntryPrice", "Direction"]]
    sub_df["Y"] = sub_df.max_floating_profit / sub_df.multiplier
    #     """若是做多的订单，找到其最近的负数赋值点（包括once），用负数赋值点的low-entryprice
    #     若是做空的订单，找到其最近的正数赋值点（包括once），用entryprice-正数赋值点的high"""
    sub_df['floating_loss'] = [sub_df.last_zzp_value.iloc[i] - sub_df.EntryPrice.iloc[i] if sub_df.Direction.iloc[
                                                                                                i] == "TradeDirection.LONG" else
                               sub_df.EntryPrice.iloc[i] - sub_df.last_zzp_value.iloc[i] for i in range(len(sub_df))]
    sub_df['floating_loss_divide_by_multiplier'] = sub_df.floating_loss / sub_df.multiplier
    # 删除inf和-inf的
    sub_df = sub_df[~sub_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    y = sub_df.Y.values
    x = sub_df.floating_loss_divide_by_multiplier.values

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color='pink')
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    plt.xlim(-1300, 100)
    plt.ylim(-200, 8100)
    ax.set_ylabel('floating_profit_divide_by_multiplier')
    ax.set_xlabel('floating_loss_divide_by_multiplier')
    ax.set_title('scatter of floating profit and loss')
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\standardized\floating_loss_and_profit_divide_by_multiplier_{}.png'.format(
            zzg_num))


def draw_3d_floating_loss_and_profit_distribution_under_timeframe(zzg_num):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import MultipleLocator

    regression_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # 数据清洗
    sub_df = regression_table[
        ["standardized_y", "max_floating_profit", "multiplier", "standardized_count", "standardized_zscore",
         "last_zzp_value", "EntryPrice", "EntryTime", "Direction"]]
    sub_df["Y"] = sub_df.max_floating_profit / sub_df.multiplier
    #     """若是做多的订单，找到其最近的负数赋值点（包括once），用负数赋值点的low-entryprice
    #     若是做空的订单，找到其最近的正数赋值点（包括once），用entryprice-正数赋值点的high"""
    sub_df['floating_loss'] = [sub_df.last_zzp_value.iloc[i] - sub_df.EntryPrice.iloc[i] if sub_df.Direction.iloc[
                                                                                                i] == "TradeDirection.LONG" else
                               sub_df.EntryPrice.iloc[i] - sub_df.last_zzp_value.iloc[i] for i in range(len(sub_df))]
    sub_df['floating_loss_divide_by_multiplier'] = sub_df.floating_loss / sub_df.multiplier

    sub_df['time'] = [int(regression_table.EntryTime.iloc[i].split(" ")[-1].split(":")[0]) for i in
                      range(len(regression_table))]
    # 删除inf和-inf的
    sub_df = sub_df[~sub_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # 用时间做一个groupby
    sub_df_long = sub_df.loc[sub_df.loc[:,"Direction"] == "TradeDirection.LONG"]
    sub_df_short = sub_df.loc[sub_df.loc[:,"Direction"] == "TradeDirection.SHORT"]
    df_draw_long = sub_df_long.groupby(['time'])["Y","floating_loss_divide_by_multiplier"].mean()
    df_draw_short = sub_df_short.groupby(['time'])["Y","floating_loss_divide_by_multiplier"].mean()

    x_long, y_long, z_long = df_draw_long.index.values, df_draw_long.floating_loss_divide_by_multiplier.values, df_draw_long.Y.values
    x_short, y_short, z_short = df_draw_short.index.values, df_draw_short.floating_loss_divide_by_multiplier.values, df_draw_short.Y.values


    fig = plt.figure(figsize=([42, 32]))

    gs = gridspec.GridSpec(4, 5, hspace=0.3)
    ax = {0: plt.subplot(gs[:2, :3], projection='3d'),
          1: plt.subplot(gs[0, 3], projection='3d'),
          2: plt.subplot(gs[1, 3], projection='3d'),
          3: plt.subplot(gs[0, 4], projection='3d'),
          4: plt.subplot(gs[1, 4], projection='3d'),
          5: plt.subplot(gs[2:, :3], projection='3d'),
          6: plt.subplot(gs[2, 3], projection='3d'),
          7: plt.subplot(gs[3, 3], projection='3d'),
          8: plt.subplot(gs[2, 4], projection='3d'),
          9: plt.subplot(gs[3, 4], projection='3d'),
          }
    view = [(30, 80), (0, 90), (90, 0), (45, 45), (0, 0), (30, 80), (0, 90), (90, 0), (45, 45), (0, 0),]

    for i in range(5):
        ax[i].plot_trisurf(x_long, y_long, z_long, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(1)
        ax[i].xaxis.set_major_locator(x_major_locator)
        y_major_locator = MultipleLocator(100)
        ax[i].yaxis.set_major_locator(y_major_locator)
        z_major_locator = MultipleLocator(1000)
        ax[i].zaxis.set_major_locator(z_major_locator)
        ax[i].set_title("long_avg")
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("hard_stoploss")
        ax[i].set_zlabel('floating_profit')
        ax[i].view_init(*view[i])

    for i in range(5,10):
        ax[i].plot_trisurf(x_short, y_short, z_short, cmap=plt.cm.Spectral, linewidth=0.1)
        x_major_locator = MultipleLocator(1)
        ax[i].xaxis.set_major_locator(x_major_locator)
        y_major_locator = MultipleLocator(100)
        ax[i].yaxis.set_major_locator(y_major_locator)
        z_major_locator = MultipleLocator(1000)
        ax[i].zaxis.set_major_locator(z_major_locator)
        ax[i].set_title("short_avg")
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("hard_stoploss")
        ax[i].set_zlabel('floating_profit')
        ax[i].view_init(*view[i])

    # Todo 下面这个地址在使用时要改
    fig.savefig(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\standardized\3d_floating_loss_and_profit_divide_by_multiplier_{}.png'.format(
            zzg_num))
    plt.close('all')


def linear_regression_all_about_conditions_between_zzp(zzg_num):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    regression_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # 数据清洗
    sub_df = regression_table[["standardized_y", "standardized_count", "standardized_zscore"]]
    # 删除inf和-inf的
    sub_df = sub_df[~sub_df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # sub_df = sub_df.dropna(axis=0, how='any')

    df_regression = sub_df.reset_index(
        drop=True)  # 清洗完后，只有用于回归的三列
    # 开始回归
    y = df_regression["standardized_y"]
    x = df_regression[["standardized_count", "standardized_zscore"]]
    # 将 y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
    y = y[:np.newaxis]

    model = LinearRegression()
    model.fit(x, y)
    predicts = model.predict(x)
    R2 = model.score(x, y)
    show_R2 = 'R2 = %.3f' % R2
    coef = model.coef_
    intercept = model.intercept_
    rst = f'线性回归   zzg_num: {zzg_num}', f'斜率为: count: {coef[0]}  zscore: {coef[1]}', f'截距为{intercept}', show_R2
    with open(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\standardized\linear_regression_result_{}.txt'.format(
                zzg_num), 'w') as f:
        for line in rst:
            f.write(line)
    return f'斜率为: count:{coef[0]} zscore:{coef[1]}', f'截距为{intercept}', show_R2


def linear_regression_all_about_conditions_between_zzp_with_timeframe(zzg_num):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import timedelta
    from sklearn.linear_model import LinearRegression
    regression_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
            zzg_num), index_col=0)
    # 数据清洗
    sub_df = regression_table[["standardized_y", "standardized_count", "standardized_zscore", "EntryTime"]]
    sub_df = sub_df.dropna(axis=0, how='any')
    # 去除multiplier为1的
    # sub_df = sub_df.loc[sub_df.loc[:, "multiplier"] != float(1), :]
    # sub_df['Y'] = sub_df['max_floating_profit'] / sub_df['multiplier']
    sub_df['Time'] = sub_df.EntryTime.apply(lambda x: x.split(" ")[-1])
    sub_df['Time'] = pd.to_datetime(sub_df['Time'])  # 转化后的时间自动加上程序运行时的年月日
    start_time = pd.to_datetime('00:00:00')
    end_time = pd.to_datetime('00:00:00') + timedelta(days=1)
    time = pd.date_range(start=start_time, end=end_time, freq='H')
    start_Time = time[:-1]
    end_Time = time[1:]
    df_regression = sub_df.reset_index(
        drop=True)  # 清洗完后，只有count，zscore，和Y（用floating profit/multiplier），都是标准化过的，和时间
    # 拆分成24张表
    rst_list = []
    for start, end in zip(start_Time, end_Time):
        sub_regression_df = df_regression.loc[
            (start <= df_regression.loc[:, "Time"]) & (df_regression.loc[:, "Time"] < end)]
        # 开始回归
        y = sub_regression_df["standardized_y"]
        x = sub_regression_df[["standardized_count", "standardized_zscore"]]
        # 将 y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
        y = y[:np.newaxis]

        model = LinearRegression()
        model.fit(x, y)
        predicts = model.predict(x)
        R2 = model.score(x, y)
        show_R2 = 'R2 = %.3f' % R2
        coef = model.coef_
        intercept = model.intercept_
        rst = (f'线性回归   zzg_num: {zzg_num}', f'斜率为: count: {coef[0]}  zscore: {coef[1]}', f'截距为{intercept}', show_R2)
        rst_list.append(rst)
    with open(
            r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\linear_regression_result_timeframe_{}.txt'.format(
                zzg_num), 'w') as f:
        for items in rst_list:
            for line in items:
                f.write(line)


# def logistic_regression_all_about_conditions_between_zzp(zzg_num):
#     import pandas as pd
#     import numpy as np
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import classification_report, confusion_matrix
#     from sklearn.linear_model import LogisticRegression
#
#     regression_table = pd.read_csv(
#         r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\table_{}.csv'.format(
#             zzg_num), index_col=0)
#     # 数据清洗
#     sub_df = regression_table[["multiplier", "max_floating_profit", "count", "zscore"]]
#     sub_df = sub_df.dropna(axis=0, how='any')
#     # 去除multiplier为1的
#     sub_df = sub_df.loc[sub_df.loc[:, "multiplier"] != float(1), :]
#     sub_df['Y'] = sub_df['max_floating_profit'] / sub_df['multiplier']
#     df_regression = sub_df.loc[:, "count":"Y"].reset_index(
#         drop=True)  # 清洗完后，只有count，zscore，和Y（用floating profit/multiplier）
#     # 开始回归
#     y = df_regression["Y"]
#     x = df_regression[["count", "zscore"]]
#     # 将 y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
#     y = y[:np.newaxis]
#
#     model = LinearRegression()
#     model.fit(x, y)
#     predicts = model.predict(x)
#     R2 = model.score(x, y)
#     show_R2 = 'R2 = %.3f' % R2
#     coef = model.coef_
#     intercept = model.intercept_
#     rst = f'逻辑回归   zzg_num: {zzg_num}', f'斜率为: count: {coef[0]}  zscore: {coef[1]}', f'截距为{intercept}', show_R2
#     with open(
#             r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\only_entry_info\add_regression_indicators_table\linear_regression_result_{}.txt'.format(
#                     zzg_num), 'w') as f:
#         for line in rst:
#             f.write(line)
#     return f'斜率为: count:{coef[0]} zscore:{coef[1]}', f'截距为{intercept}', show_R2

# Todo 下面要改成60个交易日 下面关于每个订单的multiplier我可以在生成订单的时候就记录在案，不用后续计算
# def get_multiplier(df_zigzag, current_date, num=100):
#     """CurrentTime是程序中传入的，str格式的，包含时刻信息的时间"""
#     # 把df_zigzag中的datetime列转化为timestamp格式
#     df_zigzag['datetime'] = pd.to_datetime(
#         df_zigzag['datetime'])  # df_zigzag.datetime.apply(lambda x: pd.to_datetime(x))
#     # 记录当前日期
#     CurrentDate = pd.to_datetime(current_date)
#     startdate = CurrentDate - datetime.timedelta(days=num)
#     df_series = df_zigzag[(df_zigzag.datetime >= startdate) & (df_zigzag.datetime < CurrentDate)].groupby(['m_Date'])[
#         'day_range'].max()
#     if CurrentDate < pd.to_datetime('2018/1/6'):
#         return 1
#     else:
#         # min_avg_before, today_used_avg = df_series.rolling(5).mean().min(), df_series.rolling(5).mean()[-1]
#         mean_avg_before, today_used_avg = df_series.rolling(5).mean().mean(), df_series.rolling(5).mean()[-1]
#         # return today_used_avg/min_avg_before
#         return today_used_avg / mean_avg_before


# Todo same direction用不到了？
# def add_next_same_direction_order_entry_idx_column(his_table):
#     """传入的his_table为csv"""
#     sub_his_table = his_table.filter(items=['Direction', 'entry_idx'])
#     sub_his_table_long = sub_his_table.query('Direction == ["TradeDirection.LONG"]')
#     sub_his_table_short = sub_his_table.query('Direction == ["TradeDirection.SHORT"]')
#     sub_his_table_long['next_same_direction_bar_idx'] = pd.array(sub_his_table_long.entry_idx.shift(-1),
#                                                                  dtype=pd.Int64Dtype())
#     sub_his_table_short['next_same_direction_bar_idx'] = pd.array(sub_his_table_short.entry_idx.shift(-1),
#                                                                   dtype=pd.Int64Dtype())
#
#     df = pd.concat([sub_his_table_long, sub_his_table_short], axis=0)
#     df = df.sort_values(by='entry_idx', ascending=True)
#     his_table['next_same_direction_bar_idx'] = df.next_same_direction_bar_idx
#     return his_table


# def calc_floating_distance(his_table, df_zigzag):
#     """
#     long：如果后续价格的low没有回落至前负数赋值点，就计算在出现下一个long单之前的这段时间内的floating
#     如果后续价格的low回落至了前负数赋值点，那么就计算进场时间点到回落点之间出现的floating
#     # multiplier是5天/30天
#     该函数的return值之后还要÷每个订单的multiplier作为回归的因变量
#     """
#     his_table = add_next_same_direction_order_entry_idx_column(his_table)
#     floating_distance = []
#     for i in range(len(his_table)):
#         # Todo 下面有没有简洁的写法？
#         open_direction = his_table.loc[i].Direction
#         entry_index = his_table.loc[i].entry_idx
#         exit_index = his_table.loc[i].exit_idx
#         last_zzp_value = his_table.loc[i].last_zzp_value
#         floating_profit = his_table.loc[i].floating_profit
#         entry_price = his_table.loc[i].EntryPrice
#         next_same_direction_bar_idx = his_table.loc[i].next_same_direction_bar_idx
#         if open_direction == TradeDirection.SHORT:
#             if df_zigzag.loc[entry_index:exit_index].High.max() >= last_zzp_value:
#                 # 有回落，就计算前一个zzp到回落点之间的最低点距离last_zzp_value的距离
#                 floating_distance_sub = floating_profit
#             else:  # 没有回落，就计算在下一个空单之间的
#                 floating_distance_sub = entry_price - df_zigzag.loc[
#                                                       entry_index:next_same_direction_bar_idx].low.min()  # Todo 这边要包括下一个进场的bar吗？要-1吗
#         else:
#             if df_zigzag.loc[entry_index:exit_index].Low.min() <= last_zzp_value:
#                 floating_distance_sub = floating_profit
#             else:
#                 floating_distance_sub = df_zigzag.loc[
#                                         entry_index:next_same_direction_bar_idx].High.max() - entry_price  # Todo 这边要包括下一个进场的bar吗？要-1吗
#
#         floating_distance.append(floating_distance_sub)
#     his_table['floating_distance'] = floating_distance
#     return his_table


if __name__ == '__main__':
    # linear regression

    arg_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    print('.' * 30, '优化开始', '.' * 30)
    pool = multiprocessing.Pool(9)
    i = 1
    for zzg_num in arg_list:
        # pool.apply_async(whole_table_for_regression, (zzg_num,))
        # pool.apply_async(linear_regression_all_about_conditions_between_zzp, (zzg_num,))
        # rst = pool.apply_async(draw_regression_data_scatter, (zzg_num,))
        # rst.get()
        # rst = pool.apply_async(draw_floating_loss_and_profit_pic,(zzg_num,))
        # rst.get()
        rst = pool.apply_async(draw_3d_floating_loss_and_profit_distribution_under_timeframe, (zzg_num,))
        rst.get()
        # pool.apply_async(add_standardized_zscore_and_count_and_Y,(zzg_num,))
        # pool.apply_async(linear_regression_all_about_conditions_between_zzp_with_timeframe, (zzg_num,))
        print('i: ', i)
        i += 1
    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')
