import pandas as pd
import numpy as np
import os
from enum import Enum
from pandas.core.frame import DataFrame
from strategy.functionsforATRRegression import *
import multiprocessing
from tools.backtesting_backup.Evaluation_indicators import calc_underwater_equityline, calcHighandLow_during_holding, \
    trade_group_by_daytime_frame
import matplotlib.pyplot as plt
import talib
import threading
# import autopep8
from tools.backtesting_backup.get_specific_enter_price import get_atr_breakout_price


# 定义宏变量
class TradeDirection(Enum):
    LONG = 'long'
    SHORT = 'short'
    NONE = None

# Todo 发现下面整篇代码里的巨大bug，date list从来没有更新过、新增过date
def ATRRegression_HA_donot_limit_open(df_long, zzg_num, CaS, loss_mode_4_percent_trailing):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    '''
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210301_{}.csv".format(
            zzg_num),
        index_col=0)
    df_only_last = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_only_last["bar_num"] = df_only_last.index
    df_only_last = df_only_last.reset_index(drop=True)

    df_Day = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\data\SCData\YMH21-CBOT_day_from_2015_01_01.txt')
    df_Day = df_Day.loc[:,
             ['Date', ' AskVolume', ' BidVolume', ' High', ' Low', ' NumberOfTrades', ' Volume', ' Open', ' Last']]
    df_Day.columns = ["DateTag", "AskVolume", "BidVolume", "High", "Low", "NumberOfTrades", "Volume", "Open", "Close"]
    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
    Commissions_and_slippage = CaS
    OpenedOrders = 0

    last_zzg_bar_Idx_list = []

    date_list = [df_long.iloc[0].Time.split(" ")[0]]
    df_D = generate_df_D(date_list[0], df_Day)

    for i in range(1, len(df_long)):
        CurrentTime = df_long.iloc[i].Time  # 得到str格式的时间
        current_date = CurrentTime.split(" ")[0]
        # if OpenedOrders != 1:
        # 初始化触发止盈止损的标记
        # touch_profit_flag = 0
        # touch_loss_flag = 0
        # 标记是否本次有新开仓
        new_open = 0
        # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold = lastExtremeBar(
            i, df_only_last)

        if last_zzg_bar_Idx == 0:  # 没有找到上一个极值点
            continue
        # print(last_extreme_value)
        distance = i - last_zzg_bar_Idx
        if ((zzg_num < 0.886) & (distance > 5)) | ((zzg_num >= 0.886) & (distance > 10)):
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
                open_exist_direction = existed_orders_direction()
                if open_exist_direction == TradeDirection.NONE:
                    if can_open_long(df_long, i):
                        open_direction = TradeDirection.LONG
                        # Direction.append(open_direction)
                elif open_exist_direction == TradeDirection.LONG:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_long(df_long, i):
                            open_direction = TradeDirection.LONG
                            # Direction.append(open_direction)
                else:  # 原来存在空单
                    if can_open_long(df_long, i):
                        open_direction = TradeDirection.LONG
                        need_flat = True

            else:  # SHORT
                open_exist_direction = existed_orders_direction()
                if open_exist_direction == TradeDirection.NONE:
                    if can_open_short(df_long, i):
                        open_direction = TradeDirection.SHORT
                        # Direction.append(open_direction)
                elif open_exist_direction == TradeDirection.SHORT:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_short(df_long, i):
                            open_direction = TradeDirection.SHORT
                            # Direction.append(open_direction)
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
                if open_direction == TradeDirection.LONG:
                    EntryPrice_sub = np.min(df_long.loc[i].High, df_long.loc[i].previous_close_plus_multiple_range)
                else:
                    EntryPrice_sub = np.max(df_long.loc[i].Low, df_long.loc[i].previous_close_minus_multiple_range)

                OrderType_sub: TradeDirection = open_direction
                if current_date != date_list[-1]:
                    df_D = generate_df_D(current_date, df_Day)
                    date_list.append(current_date)
                Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                EntryTime.append(EntryTime_sub)
                entry_index.append(i)
                EntryPrice.append(EntryPrice_sub)
                Direction.append(OrderType_sub)
                Lots.append(Lots_sub)

        # 移动止损 + 固定止损（设置在前赋值点的low或者high）
        touch_loss_flag = 0
        loss_exist_direction = existed_orders_direction()
        if OpenedOrders != 0:
            if loss_exist_direction == TradeDirection.LONG:
                if df_long.loc[i].High - last_extreme_value >= last_extreme_bar_threshold: # Todo 用到last_extreme_bar_threshold的这边需要改，因为这个其实并不是用了五日的平均值
                    # 触发移动止损
                    if df_long.loc[i].High > df_long.loc[i - 1].High:
                        stoploss_point = df_long.loc[i].High * (1 - loss_mode_4_percent_trailing)
                    else: #不更新stoploss
                        pass
                else:
                    stoploss_point = last_extreme_value - float(0.01)  # 防止极值点进去的就在当前bar出了

            else:
                if last_extreme_value - df_long.loc[i].Low >= last_extreme_bar_threshold:
                    # 触发移动止损
                    if df_long.loc[i].Low < df_long.loc[i - 1].Low:
                        stoploss_point = df_long.loc[i].Low * (1 + loss_mode_4_percent_trailing)
                    else:
                        pass
                else:
                    stoploss_point = last_extreme_value + float(0.01)


            if open_direction == TradeDirection.LONG:
                if df_long.loc[i].Low <= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1
            elif open_direction == TradeDirection.SHORT:
                if df_long.loc[i].High >= stoploss_point:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                    touch_loss_flag = 1
            elif open_direction == TradeDirection.NONE:  # 去找前一个订单的方向
                if loss_exist_direction == TradeDirection.LONG:
                    if df_long.loc[i].Low <= stoploss_point:
                        ExitTime_sub = df_long.loc[i].Time
                        ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                        touch_loss_flag = 1
                elif loss_exist_direction == TradeDirection.SHORT:
                    if df_long.loc[i].High >= stoploss_point:
                        ExitTime_sub = df_long.loc[i].Time
                        ExitPrice_sub = stoploss_point  # Todo：出场点位问老板
                        touch_loss_flag = 1
                else:
                    pass
            else:
                pass


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
    his_table["Profits"] = [(ExitPrice - Commissions_and_slippage - EntryPrice) * Lots if OrderType == TradeDirection.LONG else (
                            EntryPrice - ExitPrice - Commissions_and_slippage) * Lots
                            for EntryPrice, ExitPrice, OrderType, Lots in
                            zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list

# 只开多单
def long_ATRRegression_HA_donot_limit_open(df_long, zzg_num, CaS, loss_mode_4_percent_trailing):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    '''
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210301_{}.csv".format(
            zzg_num),
        index_col=0)
    df_only_last = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_only_last["bar_num"] = df_only_last.index
    df_only_last = df_only_last.reset_index(drop=True)

    df_Day = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\data\SCData\YMH21-CBOT_day_from_2015_01_01.txt')
    df_Day = df_Day.loc[:,
             ['Date', ' AskVolume', ' BidVolume', ' High', ' Low', ' NumberOfTrades', ' Volume', ' Open', ' Last']]
    df_Day.columns = ["DateTag", "AskVolume", "BidVolume", "High", "Low", "NumberOfTrades", "Volume", "Open", "Close"]


    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
    Commissions_and_slippage = CaS
    OpenedOrders = 0

    last_zzg_bar_Idx_list = []
    stoploss_point_list = [df_long.loc[0].High * (1 - loss_mode_4_percent_trailing)]

    date_list = [df_long.iloc[0].Time.split(" ")[0]]
    multiplier = optimize_trailing_stop_multiplier(df_zigzag, df_long.iloc[0].Time.split(" ")[0], num=100)
    multiplied_trailing_stop_list = []
    df_D = generate_df_D(date_list[0], df_Day)

    for i in range(1, len(df_long)):
        CurrentTime = df_long.iloc[i].Time  # 得到str格式的时间
        PreviousTime = df_long.iloc[i-1].Time
        current_date = CurrentTime.split(" ")[0]
        # if OpenedOrders != 1:
        # 初始化触发止盈止损的标记
        # touch_profit_flag = 0
        # touch_loss_flag = 0
        # 标记是否本次有新开仓
        new_open = 0
        # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold = lastExtremeBar(
            i, df_only_last)

        if last_zzg_bar_Idx == 0:  # 没有找到上一个极值点
            continue
        # print(last_extreme_value)
        distance = i - last_zzg_bar_Idx
        if ((zzg_num < 0.886) & (distance > 5)) | ((zzg_num >= 0.886) & (distance > 10)):
            continue
        else:
            # 进一步判断是否在允许的时间范围内
            # if isInAllowedTradingTime(CurrentTime):#调用一下外部函数
            if (last_extreme_bar_low_or_high == "ZZPT.ONCE_LOW") | (
                    last_extreme_bar_low_or_high == "ZZPT.LOW"):
                OrderType = TradeDirection.LONG
            else:
                continue

            def existed_orders_direction() -> TradeDirection:
                """获取之前订单的方向"""
                try:
                    return Direction[-1]
                except:
                    return TradeDirection.NONE

            def can_open_long(df_long_: pd.DataFrame,  # 输入的总的回测数据
                              i_: int,  # 当前读到哪一行
                              ) -> bool:
                """是否能开多"""
                return (df_long_.loc[i_ - 1].HA_Close - df_long_.loc[i_ - 1].HA_Open) > 0

            # 根据OrderType来判断开仓
            open_direction = TradeDirection.NONE  # 开仓方向
            if OrderType == TradeDirection.LONG:
                open_exist_direction = existed_orders_direction()
                if open_exist_direction == TradeDirection.NONE:
                    if can_open_long(df_long, i):
                        open_direction = TradeDirection.LONG
                        multiplier = optimize_trailing_stop_multiplier(df_zigzag, current_date, num=100)
                        multiplied_trailing_stop_list.append(multiplier * loss_mode_4_percent_trailing)
                elif open_exist_direction == TradeDirection.LONG:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_long(df_long, i):
                            open_direction = TradeDirection.LONG
                            multiplier = optimize_trailing_stop_multiplier(df_zigzag, current_date, num=100)
                            multiplied_trailing_stop_list.append(multiplier * loss_mode_4_percent_trailing)
                else:  # 原来存在空单
                    pass

            else:  # 因为这边不可能有空单
                pass

            can_be_optimized_flag = 0

            # 执行操作 （平仓，开仓）
            if open_direction == TradeDirection.NONE:  # 未出现开仓信号，去判断是否满足出场条件
                pass

            else:  # 开多信号出现
                if open_direction == TradeDirection.LONG:
                    if df_long.loc[i-1].is_atr_breakout:  # Todo 这里修改过，外加阳线的条件
                        EntryTime_sub = PreviousTime
                        can_be_optimized_flag = 1  # 可以在前一根进场
                        if df_long.loc[i-1].Close > df_long.loc[i-1].Open:
                            EntryPrice_sub = np.min([df_long.loc[i-1].High, df_long.loc[i-1].previous_close_plus_multiple_range])
                        else:
                            EntryPrice_sub = df_long.loc[i-1].Open
                    else:
                        EntryPrice_sub = df_long.loc[i].Open
                        EntryTime_sub = CurrentTime
                else:
                    pass
                    # EntryPrice_sub = np.max(df_long.loc[i].Low, df_long.loc[i].previous_close_minus_multiple_range)

                # 新开仓
                OpenedOrders += 1
                if OpenedOrders == 1:  # 记录第一次进场订单的止损线
                    if can_be_optimized_flag:
                        stoploss_point_list.append(df_long.loc[i-1].High * (1 - multiplied_trailing_stop_list[-1]))  # Todo 这边要不要改成-2，因为是在前一根进场，就要用前一根bar的数据（但目测差别不大）
                    else:
                        stoploss_point_list.append(df_long.loc[i].High * (1 - multiplied_trailing_stop_list[-1]))
                else:
                    pass
                last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)
                OrderType_sub: TradeDirection = open_direction

                if current_date != date_list[-1]:
                    df_D = generate_df_D(current_date, df_Day)
                    date_list.append(current_date)

                Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)

                EntryTime.append(EntryTime_sub)
                if can_be_optimized_flag:
                    entry_index.append(i-1)
                else:
                    entry_index.append(i)
                EntryPrice.append(EntryPrice_sub)
                Direction.append(OrderType_sub)
                Lots.append(Lots_sub)

        # 移动止损
        touch_loss_flag = 0
        loss_exist_direction = existed_orders_direction()
        if OpenedOrders != 0:
            if loss_exist_direction == TradeDirection.LONG:
                # if df_long.loc[i].High - last_extreme_value >= last_extreme_bar_threshold:
                #     # 触发移动止损
                if df_long.loc[i].High > df_long.loc[i - 1].High: # Todo 这边是不是也要分类，如果有优化空间的，初始止损位置要定在当前那个bar的high下面
                    stoploss_point = df_long.loc[i].High * (1 - multiplied_trailing_stop_list[-1])
                    stoploss_point_list.append(stoploss_point)
                else:  # 不更新stoploss
                    pass
                # else:
                #     stoploss_point = last_extreme_value - float(0.01)  # 防止极值点进去的就在当前bar出了

            else:
                pass

            if open_direction == TradeDirection.LONG:
                if df_long.loc[i].Low <= stoploss_point_list[-1]:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point_list[-1]
                    touch_loss_flag = 1

            elif open_direction == TradeDirection.NONE:  # 去找前一个订单的方向
                if loss_exist_direction == TradeDirection.LONG:
                    if df_long.loc[i].Low <= stoploss_point_list[-1]:
                        ExitTime_sub = df_long.loc[i].Time
                        ExitPrice_sub = stoploss_point_list[-1]
                        touch_loss_flag = 1

            else:
                pass

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
    his_table['trailing_stop_loss'] = multiplied_trailing_stop_list
    his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
    his_table["Profits"] = [
        (ExitPrice - Commissions_and_slippage - EntryPrice) * Lots if OrderType == TradeDirection.LONG else (
                                                                                                                    EntryPrice - ExitPrice - Commissions_and_slippage) * Lots
        for EntryPrice, ExitPrice, OrderType, Lots in
        zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list

# 只开空单
def short_ATRRegression_HA_donot_limit_open(df_long, zzg_num, CaS, loss_mode_4_percent_trailing):
    '''
    程序还是按照15min的框架来写吧
    df_long为15min的数据
    df_only_last记录了最终赋值点的情况，好像改成了记录的是所有赋值点
    根据前赋值点是正数还是负数确定开空还是开多
    profit_mode有1,2,3种取值：1表示大于1sigma，2表示大于2sigma，
    '''
    df_zigzag = pd.read_csv(
        "C:/Users/Administrator/Desktop/pythonHistoricalTesting/backtesting/BackTesting/zigzag_20210301_{}.csv".format(
            zzg_num),
        index_col=0)
    df_only_last = df_zigzag[(df_zigzag.zzp_type == "ZZPT.ONCE_HIGH") | (df_zigzag.zzp_type == "ZZPT.ONCE_LOW") | (
            df_zigzag.zzp_type == "ZZPT.LOW") | (df_zigzag.zzp_type == "ZZPT.HIGH")]
    df_only_last["bar_num"] = df_only_last.index
    df_only_last = df_only_last.reset_index(drop=True)

    df_Day = pd.read_csv(
        r'F:\pythonHistoricalTesting\pythonHistoricalTesting\data\SCData\YMH21-CBOT_day_from_2015_01_01.txt')
    df_Day = df_Day.loc[:,
             ['Date', ' AskVolume', ' BidVolume', ' High', ' Low', ' NumberOfTrades', ' Volume', ' Open', ' Last']]
    df_Day.columns = ["DateTag", "AskVolume", "BidVolume", "High", "Low", "NumberOfTrades", "Volume", "Open", "Close"]
    # 初始化historytable
    his_table = pd.DataFrame(
        columns=['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'OrderType', 'Lots', 'PointsChanged',
                 'Commissions_and_slippage', 'Profits'])  # 后面三个事后再计算
    EntryTime, entry_index, EntryPrice, ExitTime, exit_index, ExitPrice, Direction, Lots = [], [], [], [], [], [], [], []
    Commissions_and_slippage = CaS
    OpenedOrders = 0

    last_zzg_bar_Idx_list = []
    stoploss_point_list = [df_long.loc[0].Low * (1 + loss_mode_4_percent_trailing)]

    date_list = [df_long.iloc[0].Time.split(" ")[0]]
    multiplier = optimize_trailing_stop_multiplier(df_zigzag, df_long.iloc[0].Time.split(" ")[0], num=100)
    multiplied_trailing_stop_list = []
    df_D = generate_df_D(date_list[0], df_Day)

    for i in range(1, len(df_long)):
        CurrentTime = df_long.iloc[i].Time  # 得到str格式的时间
        PreviousTime = df_long.iloc[i-1].Time
        current_date = CurrentTime.split(" ")[0]
        # if OpenedOrders != 1:
        # 初始化触发止盈止损的标记
        # touch_profit_flag = 0
        # touch_loss_flag = 0
        # 标记是否本次有新开仓
        new_open = 0
        # 计算当前bar距离上一个极值点的距离,获取到上一个极值点的类型，是高点还是低点及其数值
        last_zzg_bar_Idx, last_extreme_bar_low_or_high, last_extreme_value, last_extreme_bar_threshold = lastExtremeBar(
            i, df_only_last)

        if last_zzg_bar_Idx == 0:  # 没有找到上一个极值点
            continue
        # print(last_extreme_value)
        distance = i - last_zzg_bar_Idx
        if ((zzg_num < 0.886) & (distance > 5)) | ((zzg_num >= 0.886) & (distance > 10)):
            continue
        else:
            # 进一步判断是否在允许的时间范围内
            # if isInAllowedTradingTime(CurrentTime):#调用一下外部函数
            if (last_extreme_bar_low_or_high == "ZZPT.ONCE_HIGH") | (
                    last_extreme_bar_low_or_high == "ZZPT.HIGH"):
                OrderType = TradeDirection.SHORT
            else:
                continue

            def existed_orders_direction() -> TradeDirection:
                """获取之前订单的方向"""
                try:
                    return Direction[-1]
                except:
                    return TradeDirection.NONE

            def can_open_short(df_long_: pd.DataFrame,  # 输入的总的回测数据
                               i_: int,  # 当前读到哪一行
                               ) -> bool:
                """是否能开空"""
                return (df_long_.loc[i_ - 1].HA_Close - df_long_.loc[i_ - 1].HA_Open) < 0

            # 根据OrderType来判断开仓
            open_direction = TradeDirection.NONE  # 开仓方向
            if OrderType == TradeDirection.SHORT:
                open_exist_direction = existed_orders_direction()
                if open_exist_direction == TradeDirection.NONE:
                    if can_open_short(df_long, i):
                        open_direction = TradeDirection.SHORT
                        if current_date != date_list[-1]:
                            multiplier = optimize_trailing_stop_multiplier(df_zigzag, current_date, num=100)
                        multiplied_trailing_stop_list.append(multiplier * loss_mode_4_percent_trailing)
                elif open_exist_direction == TradeDirection.SHORT:
                    if last_zzg_bar_Idx != last_zzg_bar_Idx_list[-1]:
                        if can_open_short(df_long, i):
                            open_direction = TradeDirection.SHORT
                            if current_date != date_list[-1]:
                                multiplier = optimize_trailing_stop_multiplier(df_zigzag, current_date, num=100)
                            multiplied_trailing_stop_list.append(multiplier * loss_mode_4_percent_trailing)
                else:
                    pass

            else:  # 因为这边不可能有空单
                pass

            can_be_optimized_flag = 0

            # 执行操作 （平仓，开仓）
            if open_direction == TradeDirection.NONE:  # 未出现开仓信号，去判断是否满足出场条件
                pass
            else:  # 开多信号出现
                if open_direction == TradeDirection.SHORT:
                    if df_long.loc[i-1].is_atr_breakout:
                        can_be_optimized_flag = 1
                        EntryTime_sub = PreviousTime
                        if df_long.loc[i-1].Close < df_long.loc[i-1].Open:
                            EntryPrice_sub = np.max([df_long.loc[i-1].Low, df_long.loc[i-1].previous_close_minus_multiple_range])
                        else:
                            EntryPrice_sub = df_long.loc[i-1].Open
                    else:
                        EntryPrice_sub = df_long.loc[i].Open
                        EntryTime_sub = CurrentTime
                else:
                    pass


                # 新开仓
                OpenedOrders += 1
                if OpenedOrders == 1:  # 记录第一次进场订单的止损线
                    if can_be_optimized_flag:
                        stoploss_point_list.append(df_long.loc[i-1].Low * (1 + multiplied_trailing_stop_list[-1]))
                    else:
                        stoploss_point_list.append(df_long.loc[i].Low * (1 + multiplied_trailing_stop_list[-1]))

                else:
                    pass
                last_zzg_bar_Idx_list.append(last_zzg_bar_Idx)

                OrderType_sub: TradeDirection = open_direction
                if current_date != date_list[-1]:
                    df_D = generate_df_D(current_date, df_Day)
                    date_list.append(current_date)

                Lots_sub = SetLots(df_long.loc[i].last_zscore, df_long.loc[i].realtime_zscore, df_D)
                EntryTime.append(EntryTime_sub)
                if can_be_optimized_flag:
                    entry_index.append(i-1)
                else:
                    entry_index.append(i)
                EntryPrice.append(EntryPrice_sub)
                Direction.append(OrderType_sub)
                Lots.append(Lots_sub)


        # 移动止损
        touch_loss_flag = 0
        loss_exist_direction = existed_orders_direction()
        if OpenedOrders != 0:
            if loss_exist_direction == TradeDirection.SHORT:
                # if last_extreme_value - df_long.loc[i].Low >= last_extreme_bar_threshold:
                #     # 触发移动止损


                if df_long.loc[i].Low < df_long.loc[i - 1].Low:
                    stoploss_point = df_long.loc[i].Low * (1 + multiplied_trailing_stop_list[-1])
                    stoploss_point_list.append(stoploss_point)
                else:
                    pass
            # else:
            #     stoploss_point = last_extreme_value + float(0.01)

            else:
                pass

            if open_direction == TradeDirection.SHORT:
                if df_long.loc[i].High >= stoploss_point_list[-1]:
                    ExitTime_sub = df_long.loc[i].Time
                    ExitPrice_sub = stoploss_point_list[-1]
                    touch_loss_flag = 1

            elif open_direction == TradeDirection.NONE:  # 去找前一个订单的方向
                if loss_exist_direction == TradeDirection.SHORT:
                    if df_long.loc[i].High >= stoploss_point_list[-1]:
                        ExitTime_sub = df_long.loc[i].Time
                        ExitPrice_sub = stoploss_point_list[-1]
                        touch_loss_flag = 1
            else:
                pass

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
    his_table['trailing_stop_loss'] = multiplied_trailing_stop_list
    his_table["PointsChanged"] = his_table.ExitPrice - his_table.EntryPrice
    his_table["Profits"] = [
        (ExitPrice - Commissions_and_slippage - EntryPrice) * Lots if OrderType == TradeDirection.LONG else (
                                                                                                                    EntryPrice - ExitPrice - Commissions_and_slippage) * Lots
        for EntryPrice, ExitPrice, OrderType, Lots in
        zip(his_table.EntryPrice, his_table.ExitPrice, his_table.OrderType, his_table.Lots)]
    his_table['Commissions_and_slippage'] = [Commissions_and_slippage] * len(his_table)
    his_table["cumsum_Profits"] = np.cumsum(his_table.Profits)
    return his_table, last_zzg_bar_Idx_list



def save_result(table_15_min, zzg_num, CaS, loss_mode_4_percent_trailing):  #, loss_mode_4_percent_trailing, loss_mode_4_percent_fixed,
    '''
    要对回测的结果执行的任何二次计算都可以放到这里，然后保存到一个excel的不同sheet中
    '''
    his_table, last_zzg_bar_Idx_list = short_ATRRegression_HA_donot_limit_open(table_15_min, zzg_num, CaS,loss_mode_4_percent_trailing)
    # loss_mode_4_percent_trailing,
    # loss_mode_4_percent_fixed)
    max_profit_list, max_loss_list = calcHighandLow_during_holding(his_table, table_15_min)
    # trade_group_by_daytime_frame(his_table, zzg_num, loss_mode_4_percent_trailing)
    his_table['max_profit'] = max_profit_list
    his_table['max_loss'] = max_loss_list
    fig, axs = plt.subplots(2)
    fig.suptitle('pnl(up)&underwater_equity_line(down)')
    axs[0].plot(his_table.cumsum_Profits)
    axs[1].plot(calc_underwater_equityline(his_table))
    fig.savefig(
        'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/short_order_only_with_trailing_stop_multiplier/pic_{}_{}.png'.format(zzg_num,loss_mode_4_percent_trailing))
    folder = "F:\\pythonHistoricalTesting\\pythonHistoricalTesting\\code_generated_csv\\zzg_num_and_trailing_stop\\short_order_only_with_trailing_stop_multiplier"
    with pd.ExcelWriter(folder + '\\table_{}_{}.xlsx'.format(zzg_num, loss_mode_4_percent_trailing)) as writer:
        his_table.to_excel(writer, sheet_name='his_table')
        DataFrame(last_zzg_bar_Idx_list).to_excel(writer, sheet_name='last_zzg_bar_Idx_list')
        sheet = writer.book.add_worksheet('pnl&underwater')
        sheet.insert_image('A1',
                           'C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/zzg_num_and_trailing_stop/short_order_only_with_trailing_stop_multiplier/pic_{}_{}.png'.format(zzg_num,loss_mode_4_percent_trailing))#
        # writer.save()  # 保存
        # writer.close()  # 关闭


if __name__ == '__main__':
    table_15_min = pd.read_csv('C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/backtesting_data.csv')
    table_15_min = get_atr_breakout_price(table_15_min, 1.618)
    # table_15_min.to_csv('./table.csv')


    # print(table_15_min)
    #
    # long_ATRRegression_HA_donot_limit_open(table_15_min, 0.382, 3, 0.0005)

    #
    # df_D = pd.read_csv("C:/Users/Administrator/Desktop/pythonHistoricalTesting/code_generated_csv/stat_df.csv")
    # loss_mode_4_percent_trailing = np.arange(0.0005, 0.00425, 0.00025)





    loss_mode_4_percent_trailing = np.arange(0.0005, 0.01, 0.00025)
    zzg_num_list = [0.382, 0.5, 0.618, 0.782, 0.886, 1, 1.236, 1.5, 2]

    addressList = []
    for num in zzg_num_list:
        for trailing_loss_percent in loss_mode_4_percent_trailing:
            addressList.append({"zzg_num": num, "loss_mode_4_percent_trailing": trailing_loss_percent})

    print('.' * 30, '优化开始', '.' * 30)
    # save_result(table_15_min, 0.382, 3, 0.0005)

    pool = multiprocessing.Pool(10)

    i = 1
    for params in addressList:
        pool.apply_async(save_result, (table_15_min, list(params.values())[0], 3, list(params.values())[1]))
        # rst.get()
        print('i: ', i)
        i += 1

    print('.' * 30, '程序正在进行......')
    pool.close()
    pool.join()
    print('.' * 30, '程序运行结束')


