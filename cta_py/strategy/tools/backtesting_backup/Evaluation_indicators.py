#! /c/Users/Lorrie/AppData/Local/Microsoft/WindowsApps/python3
# _*_ coding:utf-8 _*_
# author: Lorrie
# date: 2021/2/22 22:01
# filename: Evaluation_indicators.py
# develop_tool: PyCharm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from enum import Enum

"""
此文档为实现systematic trading中evaluation的复现
"""


# class TradeDirection(Enum):
#     LONG = 'long'
#     SHORT = 'short'
#     NONE = None


def average_trade(his_table):
    total_net_profit = np.sum(his_table.Profits)  # 这里的profits还未减去slippage和commission
    return total_net_profit / len(his_table)


def percentage_of_profitable_trades(his_table):
    num_of_profitable_trades = len(his_table[his_table.Profits > 0])  # 这里的profits还未减去slippage和commission
    return num_of_profitable_trades / len(his_table)


def profit_factor(his_table):
    gross_profit = np.sum(his_table[his_table.Profits > 0].Profits)
    gross_loss = abs(np.sum(his_table[his_table.Profits < 0].Profits))
    return gross_profit / gross_loss  # 2 is a healthy one


# 这是一个bug满满的function
def drawdown(his_table, df, mode):
    """
    df为15min bar的数据表
    回测代码中记录入场bar和出场bar的index，用在这里会方便很多
    mode: 1: end trade drawdown：以long为例，入场后价格的最高点和最终退出点之间的距离，就是我们损失的可能盈利部分
          2: close trade drawdown : abs(entry - exit)
          3: start trade drawdown : the trade went against us after entry before it started to go our way
                以long为例 需要记录入场点之后，回到入场点价位之前，的最低价，该价格-入场价
    """
    if mode == 1:
        drawdown_list = []
        for i in range(len(his_table)):
            start_index, end_index = his_table.loc[i].entry_index, his_table.loc[i].exit_index - 1
            df_sub = df.loc[start_index, end_index]
            if his_table.loc[i].OrderType == "long":
                highest_price_bf_exit = np.max(df_sub.High)
                drawdown_sub = highest_price_bf_exit - his_table.loc[i].ExitPrice
            else:
                lowest_price_bf_exit = np.min(df_sub.Low)
                drawdown_sub = his_table.loc[i].ExitPrice - lowest_price_bf_exit
            drawdown_list.append(drawdown_sub)
    elif mode == 2:
        drawdown_list = abs(his_table.PointsChanged).tolist()

    else:  # mode == 3
        # 这里好像不应该设置end_index
        drawdown_list = []
        for i in range(len(his_table)):
            entry_point, start_index, end_index = his_table.loc[i].EntryPrice, his_table.loc[i].entry_index, \
                                                  his_table.loc[i].exit_index
            df_sub = df.loc[start_index + 1, end_index]
            bar_price_equal_to_entry_point = df_sub[(df_sub.High > entry_point) & (df_sub.Low < entry_point)].index
            if his_table.loc[i].OrderType == "long":
                if (df.loc[start_index].Low < entry_point) & (df.loc[start_index].High > entry_point):
                    drawdown_sub = entry_point - df.loc[start_index].Low
                elif (df.loc[start_index].Low < entry_point) & (df.loc[start_index].High <= entry_point):
                    if len(bar_price_equal_to_entry_point) == 0:  # 说明是单向行情
                        bar_index = end_index
                    else:
                        bar_index = bar_price_equal_to_entry_point[0]

                    negative_change_before_go_our_way = np.min(
                        df_sub.loc[start_index:bar_index].Low)
                    drawdown_sub = entry_point - negative_change_before_go_our_way
                else:  # 一开始就按照我们的方向走
                    drawdown_sub = 0
            else:
                if (df.loc[start_index].High > entry_point) & (df.loc[start_index].Low < entry_point):
                    drawdown_sub = df.loc[start_index].High - entry_point
                elif (df.loc[start_index].High > entry_point) & (df.loc[start_index].Low >= entry_point):
                    if len(bar_price_equal_to_entry_point) == 0:  # 说明是单向行情
                        bar_index = end_index
                    else:
                        bar_index = bar_price_equal_to_entry_point[0]

                    negative_change_before_go_our_way = np.max(
                        df_sub.loc[start_index:bar_index].High)
                    drawdown_sub = negative_change_before_go_our_way - entry_point
                else:  # 一开始就按照我们的方向走
                    drawdown_sub = 0
            drawdown_list.append(drawdown_sub)
    return drawdown_list


# def draw_underwater_equity_line(his_table):

def EquityLine(his_table):
    equity_line_series = np.cumsum(his_table.EntryPrice * his_table.Lots) + np.cumsum(his_table.Profits)
    return equity_line_series


def calcBiggestChange(his_table, df):
    change_list = []
    for i in range(len(his_table)):
        entry_point, start_index, end_index = his_table.loc[i].EntryPrice, his_table.loc[i].entry_index, \
                                              his_table.loc[i].exit_index
        df_sub = df.loc[start_index, end_index]
        if his_table.loc[i].OrderType == "long":
            change_list.append((entry_point - np.min(df_sub.Low)) / entry_point)
        else:
            change_list.append((np.max(df_sub.High) - entry_point) / entry_point)
    return change_list


def calcHighandLow_during_holding(his_table, df):
    max_profit_list = []
    max_loss_list = []
    for i in range(len(his_table)):
        entry_index, exit_index, entry_point, exit_point, direction = his_table.loc[i].entry_index, his_table.loc[
            i].exit_index, \
                                                                      his_table.loc[i].EntryPrice, his_table.loc[
                                                                          i].ExitPrice, his_table.loc[i].OrderType
        # 改成动态更新浮盈浮亏
        df_sub = df.loc[entry_index:exit_index]
        temp_high = 0
        temp_low = 100000

        for j in range(len(df_sub)):
            if direction.value == "long":
                if df_sub.iloc[j].High > temp_high:
                    temp_high = df_sub.iloc[j].High

                if df_sub.iloc[j].Low < temp_low:
                    temp_low = df_sub.iloc[j].Low

                if temp_low >= entry_point:
                    max_loss = 0
                else:
                    max_loss = entry_point - temp_low

                if temp_high <= entry_point:
                    max_profit = 0
                else:
                    max_profit = temp_high - entry_point

            else:
                if df_sub.iloc[j].High > temp_high:
                    temp_high = df_sub.iloc[j].High

                if df_sub.iloc[j].Low < temp_low:
                    temp_low = df_sub.iloc[j].Low

                if temp_high <= entry_point:
                    max_loss = 0
                else:
                    max_loss = temp_high - entry_point

                if temp_low >= entry_point:
                    max_profit = 0
                else:
                    max_profit = entry_point - temp_low

        max_loss_list.append(max_loss)
        max_profit_list.append(max_profit)

    return max_profit_list, max_loss_list
    #
    # if entry_index != exit_index:
    #     df_sub = df.loc[entry_index:exit_index-1]
    #     highest = np.max(df_sub.High)
    #     lowest = np.min(df_sub.Low)
    #     if direction.value == "long":
    #         if his_table.loc[i].Profits < 0:#止损出场
    #             max_profit_list.append(highest-entry_point)
    #             max_loss_list.append(entry_point - exit_point)
    #         else:
    #             max_profit_list.append(highest - entry_point)
    #             max_loss_list.append(entry_point - lowest)
    #     else:
    #         if his_table.loc[i].Profits < 0:#止损出场
    #             max_profit_list.append(entry_point-lowest)
    #             max_loss_list.append(exit_point - entry_point)
    #         else:
    #             max_profit_list.append(entry_point - lowest)
    #             max_loss_list.append(highest - entry_point)
    # else:# 同一根bar进出，一定是止损出场
    #     df_sub = df.loc[entry_index]
    #     highest = df_sub.High
    #     lowest = df_sub.Low
    #     if direction.value == "long":
    #         max_profit_list.append(highest - entry_point)
    #         max_loss_list.append(entry_point - exit_point)
    #     else:
    #         max_profit_list.append(entry_point - lowest)
    #         max_loss_list.append(exit_point - entry_point)


def calc_underwater_equityline(his_table):
    pnl = his_table.cumsum_Profits
    new_high = 0  ####
    new_high_list = []
    for eq in pnl:
        if eq > new_high:
            new_high = eq
            new_high_list.append(eq)
        else:
            new_high_list.append(new_high)
    return pnl / new_high_list


def calc_drawdown_ratio_list(his_table):
    pnl = his_table.cumsum_Profits.tolist()
    if pnl[-1] < 0:  # 一路向下的pnl曲线，或者最终是负的
        return 0
    new_high = 0  ####
    new_high_list = []
    new_high_index = []
    for index in range(len(pnl)):
        if pnl[index] > new_high:
            new_high = pnl[index]
            new_high_list.append(new_high)
            new_high_index.append(index)
    start_high_index = new_high_index[:-1]
    end_high_index = new_high_index[1:]
    new_low_list = []
    for i, j in zip(start_high_index, end_high_index):
        new_low = np.min(pnl[i:j])
        new_low_list.append(new_low)
    return np.max((np.array(new_high_list[:-1]) - np.array(new_low_list)) / np.array(new_high_list[:-1]))


# 在回测程序内部被调用的，直接传入his_table即可
def trade_group_by_daytime_frame(his_table, zzg_num, loss_mode_4_percent_fixed):
    df = pd.DataFrame(columns=['Time', 'Profits'])
    df['Time'] = his_table.EntryTime.apply(lambda x: x.split(" ")[-1])
    df['Profits'] = his_table.Profits
    result = df.groupby(['Time'])['Profits'].sum()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    # result.index = result.index.applymap(lambda x: x.strftime("%H:%m"))
    result.index = pd.Series(result.index).apply(lambda x: x.strftime("%H:%M:%S"))
    plt.figure(figsize=(30, 12))
    plt.plot(result)
    plt.xticks(rotation=90)
    plt.savefig(
        '../code_generated_csv/simplest_HA_without_stop_loss_and_profit/daytime_frame_trade_{}_{}.png'.format(zzg_num,
                                                                                                              loss_mode_4_percent_fixed))
    # return plt.show()


# 在外部传参调用，需要把上面的his_table换掉
# def trade_group_by_daytime_frame(zzg_num, fixed_stop_loss_num):
#     his_table = pd.read_excel(
#         '../code_generated_csv/900_zzg_num_and_stoploss_fixed/table_{}_{}.xlsx'.format(zzg_num, fixed_stop_loss_num),
#         index_col=0)
#     df = pd.DataFrame(columns=['Time', 'Profits'])
#     df['Time'] = his_table.EntryTime.apply(lambda x: x.split(" ")[-1])
#     df['Profits'] = his_table.Profits
#     result = df.groupby(['Time'])['Profits'].sum()
#     result.index = pd.to_datetime(result.index)
#     result = result.sort_index()
#     # result.index = result.index.applymap(lambda x: x.strftime("%H:%m"))
#     result.index = pd.Series(result.index).apply(lambda x: x.strftime("%H:%M:%S"))
#     plt.figure(figsize=(30, 12))
#     plt.plot(result)
#     plt.xticks(rotation=90)
#     plt.savefig(
#         '../code_generated_csv/900_zzg_num_and_stoploss_fixed/daytime_frame_trade_{}_{}.png'.format(zzg_num,fixed_stop_loss_num))

# Todo 下面三个函数的xy轴是这俩吗？ x轴：日内的时间段，y轴：trailing stop的值（这个值要用原先的值还是弄上乘数的值？）
def average_trade_net_profit(his_table):
    ...


def number_of_total_trades_generated(his_table):
    ...


def maximum_intraday_drawdown(zzg_num, trailing_stop_num):
    '''
    计算准确度不是很高，因为出场的价格往往不是exit bar的close，是触发到某个价格才离场的
    获取每一笔订单进场和出场的index，然后去zigzag表格中依次判断在这段index范围内high和low怎么办，实时计算每天的drawdown，
    最后取最大值
    '''
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210301_{}.csv".format(
            zzg_num),
        index_col=0)
    # 下面的地址要修改
    his_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\mean_multiplier_added_integrate_csv\table_{}_{}.csv'.format(
            zzg_num, trailing_stop_num), index_col=0)
    every_order_drawdown_list = []
    # Todo 目前是按照8:30到第二天8:30界定为一天
    for start, end, direction, entry_price in zip(his_table['entry_index'], his_table['exit_index'],
                                                  his_table['OrderType'], his_table['EntryPrice']):
        # 获取到这个start对应的那天的起始和结束的bar的index
        _, next_day_start_index = get_day_start_and_end_index_for_each_order(start, df_zigzag)
        if direction == 'TradeDirection.LONG':
            if end < next_day_start_index:  # 在当天就出场
                drawdown_sub = np.min(df_zigzag.loc[start:end].Low)
                every_order_drawdown_list.append(entry_price - drawdown_sub)

            else:  # 跨天持仓，就取到上一天结束的bar
                drawdown_sub = np.min(df_zigzag.loc[start:next_day_start_index - 1].Low)
                every_order_drawdown_list.append(entry_price - drawdown_sub)


        elif direction == 'TradeDirection.SHORT':
            if end < next_day_start_index:  # 在当天就出场
                drawdown_sub = np.max(df_zigzag.loc[start:end].High)
                every_order_drawdown_list.append(entry_price - drawdown_sub)

            else:  # 跨天持仓，就取到上一天结束的bar
                drawdown_sub = np.max(df_zigzag.loc[start:next_day_start_index - 1].High)
                every_order_drawdown_list.append(entry_price - drawdown_sub)

        else:
            pass
    return np.min(every_order_drawdown_list)


# 获取订单进场的那天日期在zigzag表中当天开始和结束的index
def get_day_start_and_end_index_for_each_order(order_entry_index, df_zigzag):
    from datetime import timedelta

    df = df_zigzag.__deepcopy__()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] - timedelta(hours=8.5)
    df['Date'] = df['datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))

    new_day_index = pd.Series(df.drop_duplicates(['Date']).index.tolist())

    newday_start_index = new_day_index[(new_day_index - order_entry_index) <= 0].iloc[-1]

    try:
        next_day_start_index = new_day_index[(new_day_index - order_entry_index) > 0].iloc[0]
    except Exception as e:
        #         print(e)
        next_day_start_index = len(df) - 1

    return newday_start_index, next_day_start_index



def by_day_peak_ratio_and_period(zzg_num, trailing_stop_num):
    """
    peak_list的层层筛选：
    一层：先把出现的所有新peak都记录下来---->在这一层中，如果只记录到一个高点或者0个高点，输出min_ratio和corresponding_period都为None
    二层：根据1.1剔除一些peak的值
    根据得到的peak算ratio----->如果得到的ratio全部为1,输出min_ratio和corresponding_period分别为None和0
    （全部为1的可能原因：①每一笔order都盈利，使得两个peak都是相邻的peak，ratio就肯定为1
    ②出现连续的几笔盈利后，就不再出现新高，导致peak_list不再扩展，ratio也就全部都为初始开始计算的1）
    """
    from datetime import timedelta
    his_table = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\code_generated_csv\zzg_num_and_trailing_stop\mean_multiplier_added_integrate_csv\table_{}_{}.csv'.format(
            zzg_num, trailing_stop_num), index_col=0)
    his_table['EntryTime'] = pd.to_datetime(his_table['EntryTime'])
    date_list = his_table['EntryTime'] - timedelta(hours=8.5)
    his_table['Date'] = date_list.apply(lambda x: x.strftime("%Y-%m-%d"))
    profit_list = his_table.groupby(['Date'])['Profits'].sum()

    df = pd.DataFrame(columns=['Date', 'Profits', 'cum_Profits'])

    df['Date'] = profit_list.index
    df['Profits'] = profit_list.values
    df['cum_Profits'] = np.cumsum(df['Profits'])
    #
    high = -10000
    idx_list = []
    peak_value = []
    for idx, cum_profits in enumerate(df['cum_Profits']):
        if cum_profits >= high:
            high = cum_profits
            idx_list.append(idx)
            peak_value.append(cum_profits)

    if len(peak_value) <= 1:
        return None, None

    peak1 = idx_list
    peak2 = idx_list[1:len(df)]

    # print(peak2)

    min_idx = []
    min_value = []
    for i, j in zip(peak1, peak2):
        sub_df = df.loc[i:j]
        data = sub_df.cum_Profits
        min_idx.append(sub_df[sub_df.cum_Profits == data.min()].index[0])
        min_value.append(data.min())
    if len(min_value) != len(peak_value):
        min_idx.append(peak2[-1])
        min_value.append(peak_value[-1])
    ratio = pd.Series(min_value) / pd.Series(peak_value)
    period = pd.Series(min_idx) - pd.Series(peak1)
    initial_capital = his_table.EntryPrice.iloc[0] * his_table.Lots.iloc[0]
    filtered_ratio = []
    filtered_period = []
    for peak_value, ratio, period in zip(peak_value, ratio, period):
        if peak_value > 0.1 * initial_capital:
            filtered_ratio.append(ratio)
            filtered_period.append(period)
    # print(filtered_ratio)

    if len(filtered_ratio) != 0:
        array_ = np.array(filtered_ratio)
        if all(array_ == float(1)):# list中所有数都是1 返回True
            # print("进入这里啦！")
            min_ratio = None
            corresponding_period = 0
        else:
            min_ratio = np.min(filtered_ratio)
            corresponding_period = filtered_period[np.argmin(filtered_ratio)]
    else:
        min_ratio = None
        corresponding_period = None

    # print(min_ratio, corresponding_period)

    return min_ratio, corresponding_period

def generate_df(zzg_num_list,hard_loss_num):
    max_drawdown = maximum_intraday_drawdown(zzg_num_list, hard_loss_num)
    lock.acquire()
    dict_ = {'zzg_num': zzg_num_list, 'trailing_stop_multiplied': hard_loss_num, 'max_drawdown': max_drawdown}
    row_list.append(dict_)
    lock.release()

if __name__ == '__main__':
    # his_and_ohlc = maximum_intraday_drawdown(0.5, 0.001)
    # his_and_ohlc.to_csv('his_and_ohlc.csv')
    import pandas as pd
    import multiprocessing
    row_list = []
    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]
    hard_loss_num = np.arange(0.0005, 0.01, 0.00025)

    # 计算byday的drawdown
    # for i in zzg_num_list:
    #     for j in hard_loss_num:
    #         min_ratio, corresponding_period = by_day_peak_ratio_and_period(i, j)
    #         dict_ = {'zzg_num': i, 'trailing_stop_multiplied': j, 'min_ratio': min_ratio,
    #                    'corresponding_period': corresponding_period}
    #         row_list.append(dict_)
    # df = pd.DataFrame(row_list, columns=['zzg_num', 'trailing_stop_multiplied', 'min_ratio', 'corresponding_period'])
    #
    # df.to_csv('C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/drawdown_ratio_and_period.csv')
    pool = multiprocessing.Pool(10)
    lock = multiprocessing.Lock()



    # 计算每个订单进场日的drawdown
    n = 1
    for i in zzg_num_list:
        for j in hard_loss_num:
            pool.apply_async(generate_df,(i,j,))
            print('n: ', n)
            n += 1

    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')

    df = pd.DataFrame(row_list, columns=['zzg_num', 'trailing_stop_multiplied', 'max_drawdown'])

    df.to_csv('C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/mean_multiplier_added_integrate_csv/day_drawdown_when_order_entered.csv')