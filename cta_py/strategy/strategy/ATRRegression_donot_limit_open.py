import pandas as pd
import numpy as np
import os
from pandas.core.frame import DataFrame
from strategy.functionsforATRRegression import *
import features.ProfitAndLoss as pl
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def ATRRegression(df_long, df_only_last, df_D, profit_mode, loss_mode, sigma_stopprofit, max_lots=10,
                  multiple_bolling_stoploss=0.5, ATR_n=20):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    '''
    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
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
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value = lastExtremeBar(i, df_only_last)

        if last_zzg_bar_Idx == 0:
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
                    # if (df_long.loc[i - 1].EMA_type is True):
                    # if not ((df_long.loc[i - 1].EMA_type is False) & (
                    #         df_long.loc[last_zzg_bar_Idx].Low > df_long.loc[
                    #     last_zzg_bar_Idx].KC_Bottom_Band)):  # True表示是EMA多头
                    # if (df_long.loc[i - 1].RSIShortgoup == 1) | (
                    #         df_long.loc[i - 1].RSI_BottomDivergence == 1) | (
                    #         (df_long.loc[i - 1].bar60rsi14 > 0) & (df_long.loc[i - 1].bar60rsi14 < 20)):
                    #     """(df_long.loc[i - 1].RSIShortgoup == 1) | (
                    #         df_long.loc[i - 1].RSI_BottomDivergence == 1) | (
                    #         (df_long.loc[i - 1].bar60rsi14 > 0) & (df_long.loc[i - 1].bar60rsi14 < 20))"""
                    #     """(df_long.loc[i - 1].Extreme_UpPinBar == 1) | (
                    #         df_long.loc[i - 1].Extreme_BottomType == 1) | (
                    #         df_long.loc[i - 1].Extreme_UpPregnantType == 1) | (
                    #         df_long.loc[i - 1].Extreme_UpTriplePregnantType == 1) | (
                    #         df_long.loc[i - 1].Extreme_UpSwallowType == 1):"""

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

                    elif (OpenedOrders == 1) & (Direction[-1] != OrderType):

                        # 添加出场信息
                        ExitTime.append(CurrentTime)
                        exit_index.append(i)
                        ExitPrice.append(df_long.loc[i].Open)
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
                    else:  # 本来有订单，但是是同向的 Todo 或许可以在这里再加if判断前赋值点是不是同一个
                        pass
            else:
                if (df_long.loc[i - 1].HA_Close - df_long.loc[i - 1].HA_Open) < 0:
                    # if (df_long.loc[i - 1].EMA_type is False):
                    # if not ((df_long.loc[i - 1].EMA_type is True) & (
                    #         df_long.loc[last_zzg_bar_Idx].High < df_long.loc[
                    #     last_zzg_bar_Idx].KC_Top_Band)):  # False表示是EMA空头
                    # if (df_long.loc[i - 1].RSIShortgodown == 1) | (
                    #         df_long.loc[i - 1].RSI_TopDivergence == 1) | (
                    #         (df_long.loc[i - 1].bar60rsi14 > 80) & (df_long.loc[i - 1].bar60rsi14 < 100)):
                    #     """(df_long.loc[i - 1].RSIShortgodown == 1) | (
                    #         df_long.loc[i - 1].RSI_TopDivergence == 1) | (
                    #         (df_long.loc[i - 1].bar60rsi14 > 80) & (df_long.loc[i - 1].bar60rsi14 < 100))"""
                    #     """(df_long.loc[i - 1].Extreme_DownPinBar == 1) | (
                    #         df_long.loc[i - 1].Extreme_TopType == 1) | (
                    #         df_long.loc[i - 1].Extreme_DownPregnantType == 1) | (
                    #         df_long.loc[i - 1].Extreme_DownTriplePregnantType == 1) | (
                    #         df_long.loc[i - 1].Extreme_DownSwallowType == 1):"""

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
                    elif (OpenedOrders == 1) & (Direction[-1] != OrderType):
                        # 添加出场信息
                        ExitTime.append(CurrentTime)
                        exit_index.append(i)
                        ExitPrice.append(df_long.loc[i].Open)
                        # 新开仓
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
                        pass

        if (new_open == 0) & (OpenedOrders != 0):  # OpenedOrders = 1 说明有订单在场，就考虑是否满足离场条件

            # 判断止盈条件是否触及
            # profit_mode: 1:布林带止盈 2：绝对数值止盈
            if profit_mode == 1:
                is_touch, touch_value = pl.profit_reach_sigma(df_long, i - 1, direction=OrderType_sub,
                                                              sigma=sigma_stopprofit)
                if is_touch:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = touch_value
                    OpenedOrders = 0
                    touch_profit_flag = 1
            else:
                currentpoints = df_long.loc[i].Open
                is_touch = pl.absnum_stopprofit(EntryPrice_sub, currentpoints, amount=95, unit=5)
                if is_touch:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = currentpoints
                    OpenedOrders = 0
                    touch_profit_flag = 1

            # 判断止损条件是否触及
            # loss_mode： 1：止盈*系数 2：与实时atr相关的止损 3：low/high止损

            if loss_mode == 1:
                stoploss_point = pl.stoploss_related_to_profits(df_long, EntryTime_sub, EntryPrice_sub,
                                                                direction=OrderType_sub, sigma=sigma_stopprofit,
                                                                multiple=multiple_bolling_stoploss)
            elif loss_mode == 2:
                stoploss_point = pl.stoploss_related_to_atr(df_long, i - 1, EntryTime_sub, direction=OrderType_sub,
                                                            atr_n=ATR_n, percentage=0.05)
            else:
                stoploss_point = pl.stoploss_related_to_low_or_high(df_long, EntryTime_sub, direction=OrderType_sub)

            if OrderType_sub == "long":
                if df_long.loc[i].Low <= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    OpenedOrders = 0
                    touch_loss_flag = 1
            else:
                if df_long.loc[i].High >= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    OpenedOrders = 0
                    touch_loss_flag = 1

            if (touch_loss_flag == 1) & (touch_profit_flag == 1):  # 同时触发止盈止损
                simultaneously_touch_profit_loss.append(ExitTime_sub)
                ExitTime.append(ExitTime_sub)
                exit_index.append(i)
                ExitPrice.append(ExitPrice_sub)
            elif (touch_loss_flag == 0) & (touch_profit_flag == 0):  # 并没有产生任何触发信号
                continue
            else:  # 有触发其中之一
                ExitTime.append(ExitTime_sub)
                exit_index.append(i)
                ExitPrice.append(ExitPrice_sub)

    if len(EntryTime) != len(ExitTime): # 确保最后所有单子都是平掉的
        ExitTime.append(df_long.Time.iloc[-1])
        exit_index.append(df_long.index[-1])
        ExitPrice.append(df_long.Open.iloc[-1])

    his_table['EntryTime'] = EntryTime
    his_table['entry_index'] = entry_index
    his_table['EntryPrice'] = EntryPrice
    his_table['ExitTime'] = ExitTime
    his_table['exit_index'] = exit_index
    his_table['ExitPrice'] = ExitPrice
    his_table['OrderType'] = Direction
    his_table['Lots'] = Lots
    return his_table, last_zzg_bar_Idx_list, simultaneously_touch_profit_loss


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
    his_table, last_zzg_bar_Idx_list, simultaneously_touch_profit_loss = ATRRegression(table_15_min, df_his_listed,
                                                                                       df_D, profit_mode=1, loss_mode=1,
                                                                                       sigma_stopprofit=1, max_lots=10,
                                                                                       multiple_bolling_stoploss=1,
                                                                                       ATR_n=20)
    his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
    his_table["Profits"] = [(ExitPrice - EntryPrice) * Lots if OrderType == "long" else (EntryPrice - ExitPrice) * Lots
                            for EntryPrice, ExitPrice, OrderType, Lots in
                            zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    print(his_table)
