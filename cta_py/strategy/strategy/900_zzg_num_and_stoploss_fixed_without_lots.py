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


def ATRRegression_HA_donot_limit_open(df_long, zzg_num,
                                      CaS,loss_mode_4_percent_fixed):  # , loss_mode_4_percent_trailing,loss_mode_4_percent_fixed):
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
                if existed_orders_direction() == TradeDirection.NONE:
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
                if existed_orders_direction() == TradeDirection.NONE:
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

        # 仅固定止损
        touch_loss_flag = 0
        if OpenedOrders != 0:  # (new_open == 0) & (OpenedOrders != 0):
            if existed_orders_direction() == TradeDirection.LONG:
                stoploss_point = EntryPrice[-1] * (1 - loss_mode_4_percent_fixed)
            else:
                stoploss_point = EntryPrice[-1] * (1 - loss_mode_4_percent_fixed)  # 如果这边有多个订单在场呢？

            if open_direction == TradeDirection.LONG:
                if df_long.loc[i].Low <= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1
            else:
                if df_long.loc[i].High >= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1

            if touch_loss_flag == 1:
                ExitTime.extend([ExitTime_sub] * OpenedOrders)  # 平掉之前所有订单
                exit_index.extend([i] * OpenedOrders)
                ExitPrice.extend([ExitPrice_sub] * OpenedOrders)
                OpenedOrders = 0



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
    his_table["Profits"] = [(ExitPrice - Commissions_and_slippage - EntryPrice) if OrderType == TradeDirection.LONG else (
                            EntryPrice - ExitPrice - Commissions_and_slippage)
                            for EntryPrice, ExitPrice, OrderType in
                            zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list


def save_result(table_15_min, zzg_num, CaS,
                loss_mode_4_percent_fixed):  #, loss_mode_4_percent_trailing, loss_mode_4_percent_fixed,
    '''
    要对回测的结果执行的任何二次计算都可以放到这里，然后保存到一个excel的不同sheet中
    '''
    his_table, last_zzg_bar_Idx_list = ATRRegression_HA_donot_limit_open(table_15_min, zzg_num, CaS,loss_mode_4_percent_fixed
                                                                         )#
    # loss_mode_4_percent_trailing,
    # loss_mode_4_percent_fixed)
    max_profit_list, max_loss_list = calcHighandLow_during_holding(his_table, table_15_min)
    # trade_group_by_daytime_frame(his_table, zzg_num,loss_mode_4_percent_fixed)
    his_table['max_profit'] = max_profit_list
    his_table['max_loss'] = max_loss_list
    fig, axs = plt.subplots(2)
    fig.suptitle('pnl(up)&underwater_equity_line(down)')
    axs[0].plot(his_table.cumsum_Profits)
    axs[1].plot(calc_underwater_equityline(his_table))
    fig.savefig(
        '../code_generated_csv/900_zzg_num_and_stoploss_fixed_without_lots/pic_{}_{}.png'.format(zzg_num,loss_mode_4_percent_fixed))#
    folder = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\900_zzg_num_and_stoploss_fixed_without_lots"
    with pd.ExcelWriter(folder + '\\table_{}_{}.xlsx'.format(zzg_num, loss_mode_4_percent_fixed)) as writer:#
        his_table.to_excel(writer, sheet_name='his_table')
        DataFrame(last_zzg_bar_Idx_list).to_excel(writer, sheet_name='last_zzg_bar_Idx_list')
        sheet = writer.book.add_worksheet('pnl&underwater')
        sheet.insert_image('A1',
                           '../code_generated_csv/900_zzg_num_and_stoploss_fixed_without_lots/pic_{}_{}.png'.format(zzg_num,loss_mode_4_percent_fixed))#
        # writer.save()  # 保存
        # writer.close()  # 关闭


if __name__ == '__main__':
    table_15_min = pd.read_csv('../code_generated_csv/backtesting_data.csv')
    # df_D = pd.read_csv("../code_generated_csv/stat_df.csv")

    # trailing_percent_list = np.arange(0.0001, 0.015, 0.001)  # np.arange(0.0001, 0.015, 0.001)
    fixed_percent_list = np.arange(0.011, 0.031, 0.001)  # np.arange(0.0001, 0.0101, 0.005)
    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]

    addressList = []
    for num in zzg_num_list:
        for fixed_loss_percent in fixed_percent_list:
            # addressList.append({"table_15_min": table_15_min, "df_his_listed": df_his_listed, "CaS": 3,
            #                     "loss_mode_4_percent_trailing": trailing_loss_percent,
            #                     "loss_mode_4_percent_fixed": fixed_loss_percent})
            addressList.append({"zzg_num": num, "loss_mode_4_percent_fixed": fixed_loss_percent})
    # addressList = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]

    # print('*' * 10, "开始", '*' * 10)
    # 多线程方式
    # i = 1
    # for pct in percent_list[20:]:
    #     # t = threading.Thread(target=save_result,args=(table_15_min, df_his_listed, df_D, 3, pct))
    #     # t.start()
    #     save_result(table_15_min, df_his_listed, df_D, CaS=3, loss_mode_4_percent=pct)
    #     print('i: ', i)
    #     i += 1
    #
    # print('*' * 10, "结束", '*' * 10)

    # 多进程方式
    print('.' * 30, '优化开始', '.' * 30)

    pool = multiprocessing.Pool(10)

    i = 1
    for params in addressList:
        # save_result(table_15_min, df_his_listed, 3, list(params.values())[0], list(params.values())[1])
        # pool.apply_async(save_result,
        #                  (table_15_min, df_his_listed, 3, list(params.values())[0], list(params.values())[1]))

        pool.apply_async(save_result, (table_15_min, list(params.values())[0], 3, list(params.values())[1]))
        print('i: ', i)
        i += 1

    # 不添加任何止盈止损的多进程
    # i = 1
    # for params in zzg_num_list:
    #     # save_result(table_15_min, df_his_listed, 3, list(params.values())[0], list(params.values())[1])
    #     # pool.apply_async(save_result,
    #     #                  (table_15_min, df_his_listed, 3, list(params.values())[0], list(params.values())[1]))
    #
    #     pool.apply_async(save_result, (table_15_min, params, 3))
    #     print('i: ', i)
    #     i += 1
    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')
    # save_result(table_15_min, zzg_num, 3)
