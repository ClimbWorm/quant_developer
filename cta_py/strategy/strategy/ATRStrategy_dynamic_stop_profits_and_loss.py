import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from strategy.functionsforATRRegression import *
import features.ProfitAndLoss as pl
from concurrent.futures import ProcessPoolExecutor
from tools.Evaluation_indicators import calc_underwater_equityline, calcHighandLow_during_holding


def ATRRegression(df_long, df_only_last, df_D, CaS, loss_mode_4_percent, profit_mode, loss_mode, sigma_stopprofit, max_lots=10,
                  multiple_bolling_stoploss=0.5, ATR_n=20):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    CaS表示commission and slippage
    '''
    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
    Commissions_and_slippage = CaS
    OpenedOrders = 0

    last_zzg_bar_Idx_list = []
    simultaneously_touch_profit_loss = []  # 记录同时触发止盈止损的时间
    for i in range(1, len(df_long)):

        CurrentTime = df_long.iloc[i].Time  # 得到str格式的时间
        # if OpenedOrders != 1:
        # 初始化触发止盈止损的标记
        touch_profit_flag = 0
        touch_loss_flag = 0
        # 标记是否本次有新开仓
        new_open = 0
        # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold = lastExtremeBar(i, df_only_last)

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
                OrderType = "short"
            else:
                OrderType = "long"

            if OrderType == "long":
                if (df_long.loc[i - 1].HA_Close - df_long.loc[i - 1].HA_Open) > 0:
                    # if (df_long.loc[i - 1].RSIShortgoup == 1) | (df_long.loc[i - 1].RSI_BottomDivergence == 1):
                        # | (
                        # (df_long.loc[i - 1].bar60rsi14 > 0) & (df_long.loc[i - 1].bar60rsi14 < 20)):
                    if OpenedOrders == 0:
                        OpenedOrders += 1
                        # 新开仓
                        new_open = 1
                        last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                        EntryTime_sub = CurrentTime
                        EntryPrice_sub = df_long.loc[i].Open
                        OrderType_sub = "long"
                        Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                        EntryTime.append(EntryTime_sub)
                        entry_index.append(i)
                        EntryPrice.append(EntryPrice_sub)
                        Direction.append(OrderType_sub)
                        Lots.append(Lots_sub)

                    elif Direction[-1] != OrderType:
                        # 添加出场信息
                        ExitTime.extend([CurrentTime] * OpenedOrders)  # 把当前不同方向的仓位全平
                        exit_index.extend([i] * OpenedOrders)
                        ExitPrice.extend([df_long.loc[i].Open] * OpenedOrders)
                        # 新开仓
                        OpenedOrders = 1
                        new_open = 1
                        last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                        EntryTime_sub = CurrentTime
                        EntryPrice_sub = df_long.loc[i].Open
                        OrderType_sub = "long"
                        Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                        EntryTime.append(EntryTime_sub)
                        entry_index.append(i)
                        EntryPrice.append(EntryPrice_sub)
                        Direction.append(OrderType_sub)
                        Lots.append(Lots_sub)
                    else:  # 本来有订单，但是是同向的
                        if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                            OpenedOrders += 1
                            new_open = 1
                            last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                            EntryTime_sub = CurrentTime
                            EntryPrice_sub = df_long.loc[i].Open
                            OrderType_sub = "long"
                            Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                            EntryTime.append(EntryTime_sub)
                            entry_index.append(i)
                            EntryPrice.append(EntryPrice_sub)
                            Direction.append(OrderType_sub)
                            Lots.append(Lots_sub)
                        else:
                            pass
            else:
                if (df_long.loc[i - 1].HA_Close - df_long.loc[i - 1].HA_Open) < 0:
                    # if (df_long.loc[i - 1].RSIShortgodown == 1) | (df_long.loc[i - 1].RSI_TopDivergence == 1):
                        # | (
                        # (df_long.loc[i - 1].bar60rsi14 > 80) & (df_long.loc[i - 1].bar60rsi14 < 100)):

                    if OpenedOrders == 0:
                        OpenedOrders += 1
                        # 新开仓
                        new_open = 1
                        last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                        EntryTime_sub = CurrentTime
                        EntryPrice_sub = df_long.loc[i].Open
                        OrderType_sub = "short"
                        Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                        EntryTime.append(EntryTime_sub)
                        entry_index.append(i)
                        EntryPrice.append(EntryPrice_sub)
                        Direction.append(OrderType_sub)
                        Lots.append(Lots_sub)
                    elif Direction[-1] != OrderType:
                        # 添加出场信息
                        ExitTime.extend([CurrentTime] * OpenedOrders)  # 把当前不同方向的仓位全平
                        exit_index.extend([i] * OpenedOrders)
                        ExitPrice.extend([df_long.loc[i].Open] * OpenedOrders)
                        # 新开仓
                        OpenedOrders = 1
                        new_open = 1
                        last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                        EntryTime_sub = CurrentTime
                        entry_index.append(i)
                        EntryPrice_sub = df_long.loc[i].Open
                        OrderType_sub = "short"
                        Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                        EntryTime.append(EntryTime_sub)
                        EntryPrice.append(EntryPrice_sub)
                        Direction.append(OrderType_sub)
                        Lots.append(Lots_sub)
                    else:  # 本来有订单，但是是同向的
                        if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                            OpenedOrders += 1
                            new_open = 1
                            last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                            EntryTime_sub = CurrentTime
                            EntryPrice_sub = df_long.loc[i].Open
                            OrderType_sub = "short"
                            Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                            EntryTime.append(EntryTime_sub)
                            entry_index.append(i)
                            EntryPrice.append(EntryPrice_sub)
                            Direction.append(OrderType_sub)
                            Lots.append(Lots_sub)
                        else:
                            pass
        # new_open这边可以不加，就算有新开的订单也可以在当前bar判断有没有出场可能性
        if (new_open == 0) & (OpenedOrders != 0):  # OpenedOrders = 1 说明有订单在场，就考虑是否满足离场条件

            # 判断止盈条件是否触及
            # profit_mode: 1:布林带止盈 2：绝对数值止盈
            if profit_mode == 1:
                is_touch, touch_value = pl.profit_reach_sigma(df_long, i, direction=OrderType_sub,
                                                              sigma=sigma_stopprofit)
                if is_touch:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = touch_value
                    touch_profit_flag = 1
            else:
                currentpoints = df_long.loc[i].Open
                is_touch = pl.absnum_stopprofit(EntryPrice_sub, currentpoints, amount=95, unit=5)
                if is_touch:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = currentpoints
                    touch_profit_flag = 1

            # 判断止损条件是否触及
            # loss_mode： 1：止盈*系数 2：与实时atr相关的止损 3：low/high止损 4: threshold trailing stop,达到threshold的阈值才触发移动止损，否则就是固定止损

            if loss_mode == 1:
                stoploss_point = pl.stoploss_related_to_profits_dynamic(df_long, i, direction=OrderType_sub,
                                                                        sigma=sigma_stopprofit,
                                                                        multiple=multiple_bolling_stoploss)
            elif loss_mode == 2:
                stoploss_point = pl.stoploss_related_to_atr_trailing(df_long, i, EntryTime_sub, direction=OrderType_sub,
                                                                     atr_n=ATR_n, percentage=0.05)
            elif loss_mode == 3:
                stoploss_point = pl.stoploss_related_to_low_or_high_trailing(df_long, EntryTime_sub,
                                                                             direction=OrderType_sub)
            else:  # 前赋值点为负数赋值点，若后续high大于前赋值点值+阈值，触发移动止损；前赋值点为正数赋值点，若后续low小于前赋值点值-阈值，触发移动止损
                if Direction[-1] == "long":
                    if df_long.loc[i].High - last_extreme_value >= last_extreme_bar_threshold:
                        # 触发移动止损
                        stoploss_point = df_long.loc[i].High * (1 - loss_mode_4_percent)
                    else:
                        stoploss_point = EntryPrice[-1] * (1 - loss_mode_4_percent)
                else:
                    if last_extreme_value - df_long.loc[i].Low >= last_extreme_bar_threshold:
                        # 触发移动止损
                        stoploss_point = df_long.loc[i].High * (1 - loss_mode_4_percent)
                    else:
                        stoploss_point = EntryPrice[-1] * (1 - loss_mode_4_percent)  # 如果这边有多个订单在场呢？


            if OrderType_sub == "long":
                if df_long.loc[i].Low <= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1
            else:
                if df_long.loc[i].High >= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1

            if (touch_loss_flag == 1) & (touch_profit_flag == 1):  # 同时触发止盈止损
                simultaneously_touch_profit_loss.append(ExitTime_sub)
                ExitTime.extend([ExitTime_sub] * OpenedOrders)
                exit_index.extend([i] * OpenedOrders)
                ExitPrice.extend([ExitPrice_sub] * OpenedOrders)
                OpenedOrders = 0
            elif (touch_loss_flag == 0) & (touch_profit_flag == 0):  # 并没有产生任何触发信号
                continue
            else:  # 有触发其中之一
                ExitTime.extend([ExitTime_sub] * OpenedOrders)
                exit_index.extend([i] * OpenedOrders)
                ExitPrice.extend([ExitPrice_sub] * OpenedOrders)
                OpenedOrders = 0

    if len(EntryTime) != len(ExitTime):
        ExitTime.extend([df_long.Time.iloc[-1]] * OpenedOrders)
        exit_index.extend([df_long.index[-1]] * OpenedOrders)
        ExitPrice.extend([df_long.Open.iloc[-1]] * OpenedOrders)

    his_table['EntryTime'] = EntryTime
    his_table['entry_index'] = entry_index
    his_table['EntryPrice'] = EntryPrice
    his_table['ExitTime'] = ExitTime
    his_table['exit_index'] = exit_index
    his_table['ExitPrice'] = ExitPrice
    his_table['OrderType'] = Direction
    his_table['Lots'] = Lots
    his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
    his_table["Profits"] = [(ExitPrice - Commissions_and_slippage - EntryPrice) * Lots if OrderType == "long" else (
                                                                                                                           EntryPrice - ExitPrice - Commissions_and_slippage) * Lots
                            for EntryPrice, ExitPrice, OrderType, Lots in
                            zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list, simultaneously_touch_profit_loss


def save_result(table_15_min, df_his_listed, df_D, CaS, parameters):
    '''
    要对回测的结果执行的任何二次计算都可以放到这里，然后保存到一个excel的不同sheet中
    '''
    his_table, last_zzg_bar_Idx_list, simultaneously_touch_profit_loss = ATRRegression(table_15_min, df_his_listed,
                                                                                       df_D, CaS,
                                                                                       **parameters)
    max_profit_list, max_loss_list = calcHighandLow_during_holding(his_table, table_15_min)
    his_table['max_profit'] = max_profit_list
    his_table['max_loss'] = max_loss_list
    fig, axs = plt.subplots(2)
    fig.suptitle('pnl(up)&underwater_equity_line(down)')
    axs[0].plot(his_table.cumsum_Profits)
    axs[1].plot(calc_underwater_equityline(his_table))
    fig.savefig(
        '../code_generated_csv/HA_touch_threshold_trailing_stop_and_bband_stopprofit/pic_{}_{}_{}_{}.png'.format(
            parameters['profit_mode'], parameters['loss_mode'],
            parameters['sigma_stopprofit'],
            parameters['loss_mode_4_percent']))
    folder = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\HA_touch_threshold_trailing_stop_and_bband_stopprofit"
    with pd.ExcelWriter(folder + '\\test_{}_{}_{}_{}.xlsx'.format(parameters['profit_mode'], parameters['loss_mode'],
                                                                  parameters['sigma_stopprofit'],
                                                                  parameters['loss_mode_4_percent'])) as writer:
        his_table.to_excel(writer, sheet_name='his_table')
        DataFrame(last_zzg_bar_Idx_list).to_excel(writer, sheet_name='last_zzg_bar_Idx_list')
        DataFrame(simultaneously_touch_profit_loss).to_excel(writer, sheet_name='simultaneously_touch')
        sheet = writer.book.add_worksheet('pnl&underwater')
        sheet.insert_image('A1',
                           '../code_generated_csv/HA_touch_threshold_trailing_stop_and_bband_stopprofit/pic_{}_{}_{}_{}.png'.format(
                               parameters['profit_mode'], parameters['loss_mode'],
                               parameters['sigma_stopprofit'],
                               parameters['loss_mode_4_percent']))
        # writer.save()  # 保存
        # writer.close()  # 关闭


if __name__ == '__main__':
    table_15_min = pd.read_csv('../code_generated_csv/backtesting_data.csv')
    df_D = pd.read_csv("../code_generated_csv/stat_df.csv")
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backup/BackTesting/zigzag_YM_starts_from_2018-01-01.csv",
        index_col=0)
    df_his_listed = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_his_listed["bar_num"] = df_his_listed.index
    df_his_listed = df_his_listed.reset_index(drop=True)

    profit_mode = [1]
    loss_mode = [4]
    loss_mode_4_percent = np.arange(0.0001, 0.015, 0.0001)
    sigma_stopprofit = [1, 2]
    max_lots = 10
    multiple_bolling_stoploss = [1] # 因为选择方案4止损不会用到这个变量，故全设为1，减少不必要的循环
    ATR_n = 20

    Args = []
    for a1 in profit_mode:
        for a2 in loss_mode:
            for a3 in sigma_stopprofit:
                for a4 in multiple_bolling_stoploss:
                    for a5 in loss_mode_4_percent:
                        Args.append({'loss_mode_4_percent': a5, 'profit_mode': a1, 'loss_mode': a2, 'sigma_stopprofit': a3, 'max_lots': 10,
                                     'multiple_bolling_stoploss': a4, 'ATR_n': 20})


    print('.' * 30, '优化开始')

    i = 1
    with ProcessPoolExecutor(max_workers=11) as executor:
        for parameters in Args:
        # df_D后的1表示commission and slippage
            executor.submit(save_result(table_15_min, df_his_listed, df_D, 3, parameters))
            print('i: ', i)
            i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[1]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[2]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[3]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[4]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[5]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[6]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[7]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[8]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[9]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[10]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[11]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[12]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[13]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[14]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D,1, Args[15]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[16]))
        # print('i: ', i)
        # i += 1
        # executor.submit(save_result(table_15_min, df_his_listed, df_D, 1,Args[17]))
        # print('i: ', i)
        # i += 1
    print('.' * 30, '优化结束')
