import pandas as pd
import numpy as np
import os
from enum import Enum
from pandas.core.frame import DataFrame
from strategy.functionsforATRRegression import *
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tools.Evaluation_indicators import calc_underwater_equityline, calcHighandLow_during_holding, \
    trade_group_by_daytime_frame
import matplotlib.pyplot as plt
import threading


# 定义宏变量
class TradeDirection(Enum):
    LONG = 'long'
    SHORT = 'short'
    NONE = None


def ATRRegression_HA_donot_limit_open(df_long, zzg_num,CaS):  # , loss_mode_4_percent_trailing,loss_mode_4_percent_fixed):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    '''
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backup/BackTesting/zigzag_20210301_{}.csv".format(
            zzg_num),
        index_col=0)
    df_only_last = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_only_last["bar_num"] = df_only_last.index
    df_only_last = df_only_last.reset_index(drop=True)

    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
    Commissions_and_slippage = CaS
    OpenedOrders = 0

    last_zzg_bar_Idx_list = []

    for i in range(1, len(df_long)):
        CurrentTime = df_long.iloc[i].Time  # 得到str格式的时间
        # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold = lastExtremeBar(
            i, df_only_last)
        if last_zzg_bar_Idx == 0:  # 没有找到上一个极值点
            continue
        distance = i - last_zzg_bar_Idx
        if distance > 5:
            continue
        else:
            # 进一步判断是否在允许的时间范围内
            # if isInAllowedTradingTime(CurrentTime):#调用一下外部函数
            if (last_extreme_bar_low_or_high == "ZZPT.ONCE_HIGH") | (
                    last_extreme_bar_low_or_high == "ZZPT.HIGH"):
                OrderType = TradeDirection.SHORT
            else:
                OrderType = TradeDirection.LONG

            def existed_orders_direction() -> TradeDirection:
                """获取之前订单的方向"""
                try:
                    return Direction[-1]
                    # return TradeDirection.LONG if Direction[-1] == "long" else TradeDirection.SHORT
                except:
                    return TradeDirection.NONE

            def can_open_long(df_long_: pd.DataFrame,  # 输入的总的回测数据
                              i_: int,  # 当前读到哪一行
                              ) -> bool:
                """是否能开多"""
                return (df_long_.loc[i_ - 1].HA_Close - df_long_.loc[i_ - 1].HA_Open) > 0

            def can_open_short(df_long_: pd.DataFrame,  # 输入的总的回测数据
                               i_: int,  # 当前读到哪一行
                               ) -> bool:
                """是否能开空"""
                return (df_long_.loc[i_ - 1].HA_Close - df_long_.loc[i_ - 1].HA_Open) < 0

            # 根据OrderType来判断开仓
            open_direction = TradeDirection.NONE  # 开仓方向
            need_flat: bool = False  # 是否要平仓
            if OrderType == TradeDirection.LONG:
                if existed_orders_direction() == TradeDirection.NONE:# 当前没有存在的订单，那就直接开仓
                    if can_open_long(df_long, i):
                        open_direction = TradeDirection.LONG
                elif existed_orders_direction() == TradeDirection.LONG:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_long(df_long, i):
                            open_direction = TradeDirection.LONG
                else:  # 原来存在空单
                    if can_open_long(df_long, i):
                        open_direction = TradeDirection.LONG
                        need_flat = True

            else:  # SHORT
                if existed_orders_direction() == TradeDirection.NONE:# 当前没有存在的订单，那就直接开仓
                    if can_open_short(df_long, i):
                        open_direction = TradeDirection.SHORT
                elif existed_orders_direction() == TradeDirection.SHORT:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_short(df_long, i):
                            open_direction = TradeDirection.SHORT
                else:  # 原来存在多单
                    if can_open_short(df_long, i):
                        open_direction = TradeDirection.SHORT
                        need_flat = True

            # 执行操作 （平仓，开仓）
            if open_direction == TradeDirection.NONE:  # 未出现开仓信号，去判断是否满足出场条件
                pass
            else:  # 开空或者开多信号出现
                if need_flat:
                    ExitTime.extend([CurrentTime] * OpenedOrders)  # 平掉之前所有订单
                    exit_index.extend([i] * OpenedOrders)
                    ExitPrice.extend([df_long.loc[i].Open] * OpenedOrders)
                    OpenedOrders = 0
                # 新开仓
                OpenedOrders += 1
                last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                EntryTime_sub = CurrentTime
                EntryPrice_sub = df_long.loc[i].Open
                OrderType_sub: TradeDirection = open_direction
                Lots_sub = 1
                EntryTime.append(EntryTime_sub)
                entry_index.append(i)
                EntryPrice.append(EntryPrice_sub)
                Direction.append(OrderType_sub)
                Lots.append(Lots_sub)

    if len(EntryTime) != len(ExitTime):
        ExitTime.extend([df_long.Time.iloc[-1]] * OpenedOrders)
        exit_index.extend([df_long.index[-1]] * OpenedOrders)
        ExitPrice.extend([df_long.Open.iloc[-1]] * OpenedOrders)

    # print(len(ExitTime))
    his_table['EntryTime'] = EntryTime
    his_table['entry_index'] = entry_index
    his_table['EntryPrice'] = EntryPrice
    his_table['ExitTime'] = ExitTime
    his_table['exit_index'] = exit_index
    his_table['ExitPrice'] = ExitPrice
    his_table['OrderType'] = Direction
    his_table['Lots'] = Lots
    his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
    his_table["Profits"] = [
        (ExitPrice - Commissions_and_slippage - EntryPrice) * Lots if OrderType == TradeDirection.LONG else (
                                                                                                                    EntryPrice - ExitPrice - Commissions_and_slippage) * Lots
        for EntryPrice, ExitPrice, OrderType, Lots in
        zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list


def save_result(table_15_min, zzg_num, CaS):
    '''
    要对回测的结果执行的任何二次计算都可以放到这里，然后保存到一个excel的不同sheet中
    '''
    his_table, last_zzg_bar_Idx_list = ATRRegression_HA_donot_limit_open(table_15_min, zzg_num, CaS)
    max_profit_list, max_loss_list = calcHighandLow_during_holding(his_table, table_15_min)
    # trade_group_by_daytime_frame(his_table, zzg_num)
    his_table['max_profit'] = max_profit_list
    his_table['max_loss'] = max_loss_list
    fig, axs = plt.subplots(2)
    fig.suptitle('pnl(up)&underwater_equity_line(down)')
    axs[0].plot(his_table.cumsum_Profits)
    axs[1].plot(calc_underwater_equityline(his_table))
    fig.savefig(
        '../code_generated_csv/first_atr_strategy_testing_result_only_HA/pic_{}.png'.format(zzg_num))
    folder = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\first_atr_strategy_testing_result_only_HA"
    with pd.ExcelWriter(folder + '\\table_{}.xlsx'.format(zzg_num)) as writer:
        his_table.to_excel(writer, sheet_name='his_table')
        DataFrame(last_zzg_bar_Idx_list).to_excel(writer, sheet_name='last_zzg_bar_Idx_list')
        sheet = writer.book.add_worksheet('pnl&underwater')
        sheet.insert_image('A1',
                           '../code_generated_csv/first_atr_strategy_testing_result_only_HA/pic_{}.png'.format(zzg_num))


if __name__ == '__main__':
    table_15_min = pd.read_csv('../code_generated_csv/backtesting_data.csv')
    # save_result(table_15_min, 2, 3)

    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]

    # 多进程方式
    print('.' * 30, '优化开始', '.' * 30)
    pool = multiprocessing.Pool(5)

    i = 1
    for params in zzg_num_list:
        pool.apply_async(save_result, (table_15_min, params, 3))
        print('i: ', i)
        i += 1

    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')
